"""Review-only bounded reader for the Site07 5 km sample audit.

This reader deliberately supports only the verified layout needed for the
audit: classic TIFF, uncompressed float32, one full-width row per strip, and
contiguous strip bytes. It does not calculate terrain horizons.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.terrain_provider import TerrainWindow


URL = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site07/Site07_final_adj_5mpp_surf.tif"
OUT = Path("reports/review/site07_5km_row_slab_coverage_review_v1.json")
CAP = 26 * 1024 * 1024
PREFIX_SIZE = 65_536
EXPECTED_FILE_SIZE = 40_980_808
PRIOR_ATTEMPT_BODY_BYTES = 65_536
WIDTH, HEIGHT = 3200, 3200
ROW0, ROW1 = 600, 2601
COL0, COL1 = 600, 2601
SLAB_ROWS = 32
CX, CY, RADIUS = 30000.19238525766, -20000.152817410097, 5000.0
X_ORIGIN, Y_ORIGIN, PIXEL = 22000.0, -12000.0, 5.0


class AuditError(RuntimeError):
    pass


class Budget:
    """Count response bodies cumulatively and reject reads over a hard cap."""

    def __init__(self, limit: int):
        self.limit = limit
        self.received = 0

    def consume(self, stream: BinaryIO, content_length: int | None) -> bytes:
        remaining = self.limit - self.received
        if remaining <= 0:
            raise AuditError("Aggregate HTTP response-body cap exhausted.")
        if content_length is not None and content_length > remaining:
            raise AuditError("Response Content-Length would breach aggregate byte cap; body not read.")
        chunks: list[bytes] = []
        body_size = 0
        while True:
            allowance = self.limit - self.received
            if allowance <= 0:
                raise AuditError("Aggregate HTTP response-body cap reached while reading response.")
            chunk = stream.read(min(65536, allowance))
            if not chunk:
                return b"".join(chunks)
            self.received += len(chunk)
            body_size += len(chunk)
            chunks.append(chunk)
            if len(chunk) > allowance:
                raise AuditError("Aggregate HTTP response-body cap breached.")
            if self.received == self.limit:
                # Exact-cap EOF is acceptable only if it can be established without
                # consuming another response byte. Otherwise fail conservatively.
                if content_length == body_size:
                    return b"".join(chunks)
                raise AuditError("Aggregate HTTP response-body cap reached before EOF was established.")


class RangeClient:
    def __init__(self, url: str, budget: Budget):
        self.url, self.budget = url, budget
        self.ranges: list[dict[str, int]] = []
        self.etag_values: list[str] = []
        self.expected_etag: str | None = None
        self.opener = urllib.request.build_opener(_NoRedirectHandler())

    def get(self, start: int, end: int, total: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.expected_etag is not None:
            headers["If-Match"] = self.expected_etag
        request = urllib.request.Request(self.url, headers=headers, method="GET")
        try:
            response = self.opener.open(request, timeout=60)
        except urllib.error.HTTPError as exc:
            # Do not consume an unexpected response body.
            raise AuditError(f"Unexpected HTTP status {exc.code} for bytes {start}-{end}.") from exc
        except Exception as exc:
            raise AuditError(f"HTTP range request failed for bytes {start}-{end}: {exc}") from exc
        with response:
            if response.status != 206:
                raise AuditError(f"Expected HTTP 206, received {response.status}; response body not read.")
            etag = response.headers.get("ETag")
            if not _usable_strong_etag(etag):
                raise AuditError("Missing or unusable strong ETag; response body not read.")
            if self.expected_etag is not None and etag != self.expected_etag:
                raise AuditError("Response ETag changed during the bounded read; response body not read.")
            expected_range = f"bytes {start}-{end}/{total}"
            if response.headers.get("Content-Range") != expected_range:
                raise AuditError(
                    f"Unexpected Content-Range {response.headers.get('Content-Range')!r}; "
                    f"expected {expected_range!r}; response body not read."
                )
            cl = response.headers.get("Content-Length")
            content_length = int(cl) if cl is not None else None
            if content_length is not None and content_length != end - start + 1:
                raise AuditError("Content-Length does not match requested byte range; body not read.")
            if self.expected_etag is None:
                self.expected_etag = etag
            body = self.budget.consume(response, content_length)
            if len(body) != end - start + 1:
                raise AuditError("Response body ended before the requested range was complete.")
            self.ranges.append({"start": start, "end": end, "body_bytes": len(body)})
            self.etag_values.append(etag)
            return body


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not let a range request silently contact a redirect destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _usable_strong_etag(value: str | None) -> bool:
    """If-Match requires a syntactically quoted, non-weak entity tag."""
    if value is None or len(value) < 2 or value.startswith("W/"):
        return False
    if not (value.startswith('"') and value.endswith('"')):
        return False
    opaque = value[1:-1]
    return all((ord(char) == 0x21) or (0x23 <= ord(char) <= 0x7E) or (0x80 <= ord(char) <= 0xFF) for char in opaque)


TIFF_TYPE_BYTES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def parse_classic_tiff(prefix: bytes) -> tuple[str, dict[int, list[int] | bytes]]:
    if len(prefix) < 8 or prefix[:2] not in (b"II", b"MM"):
        raise AuditError("Unsupported or malformed TIFF byte order/header.")
    endian = "<" if prefix[:2] == b"II" else ">"
    magic, = struct.unpack_from(endian + "H", prefix, 2)
    if magic != 42:
        raise AuditError("Only classic TIFF is supported; BigTIFF refused.")
    ifd_offset, = struct.unpack_from(endian + "I", prefix, 4)
    count, = struct.unpack_from(endian + "H", prefix, ifd_offset)
    tags: dict[int, list[int] | bytes] = {}
    relevant_tags = {256, 257, 258, 259, 273, 277, 278, 279, 339, 42113}
    for i in range(count):
        pos = ifd_offset + 2 + i * 12
        tag, typ, n = struct.unpack_from(endian + "HHI", prefix, pos)
        if tag not in relevant_tags:
            continue
        size = TIFF_TYPE_BYTES.get(typ)
        if size is None:
            continue
        nbytes = size * n
        value_pos = pos + 8 if nbytes <= 4 else struct.unpack_from(endian + "I", prefix, pos + 8)[0]
        if value_pos + nbytes > len(prefix):
            raise AuditError(f"TIFF tag {tag} data lies outside the bounded header prefix.")
        if typ == 2:
            value: list[int] | bytes = prefix[value_pos:value_pos + nbytes]
        else:
            formats = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f"}
            if typ not in formats:
                raise AuditError(f"Unsupported TIFF type {typ} in relevant tag {tag}.")
            value = list(struct.unpack_from(endian + formats[typ] * n, prefix, value_pos))
        tags[tag] = value
    return endian, tags


def assemble_terrain_window_from_slabs(
    slabs: Iterable[tuple[int, np.ndarray]],
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    center_x_m: float,
    center_y_m: float,
    affine: tuple[float, float, float, float, float, float],
) -> TerrainWindow:
    """Assemble ordered native float32 row slabs into one bounded window.

    The iterator may fetch slabs lazily; only the requested subwindow is kept.
    Missing rows, duplicate/out-of-order slabs, and shape mismatches are fatal.
    """
    if not (0 <= row_start < row_stop and 0 <= column_start < column_stop):
        raise ValueError("Requested row and column bounds must be increasing and non-negative.")
    height, width = row_stop - row_start, column_stop - column_start
    if abs(affine[1]) > 1e-12 or abs(affine[3]) > 1e-12:
        raise ValueError("TerrainWindow coordinate vectors require the audited axis-aligned raster affine.")
    elevation = np.full((height, width), np.nan, dtype=np.float32)
    next_row = row_start
    for slab_start, slab in slabs:
        values = np.asarray(slab)
        if values.ndim != 2 or values.shape[1] != width or values.shape[0] < 1:
            raise ValueError("Each slab must be a non-empty 2D array spanning the requested columns.")
        slab_stop = slab_start + values.shape[0]
        if slab_start != next_row or slab_stop > row_stop:
            raise ValueError("Slabs must cover the requested rows once, contiguously, and in order.")
        elevation[slab_start - row_start:slab_stop - row_start, :] = values
        next_row = slab_stop
    if next_row != row_stop:
        raise ValueError(f"Slabs ended at row {next_row}; expected row {row_stop}.")

    a, b, c, d, e, f = affine
    columns = np.arange(column_start, column_stop, dtype=np.float64)
    rows = np.arange(row_start, row_stop, dtype=np.float64)
    x_centers = c + (columns + 0.5) * a + (rows[0] + 0.5) * b
    y_centers = f + (columns[0] + 0.5) * d + (rows + 0.5) * e
    return TerrainWindow(
        requested_center_x_m=float(center_x_m),
        requested_center_y_m=float(center_y_m),
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        elevation_m=elevation,
        nodata_mask=np.isnan(elevation),
        row_offset=row_start,
        column_offset=column_start,
    )


def read_site07_5km_terrain_window() -> tuple[TerrainWindow, dict[str, object]]:
    """Read the audited Site07 square window with one fresh 26 MiB body cap.

    This function performs network I/O when called. Importing this module or
    using the pure assembler is offline.
    """
    client = RangeClient(URL, Budget(CAP))
    budget = client.budget
    prefix = client.get(0, PREFIX_SIZE - 1, EXPECTED_FILE_SIZE)
    endian, tags = parse_classic_tiff(prefix)

    def scalar(tag: int) -> int:
        value = tags.get(tag)
        if not isinstance(value, list) or len(value) != 1:
            raise AuditError(f"Missing or unsupported scalar TIFF tag {tag}.")
        return int(value[0])

    width, height, bits = scalar(256), scalar(257), tags.get(258)
    compression, rows_per_strip = scalar(259), scalar(278)
    samples_per_pixel, sample_format = scalar(277), tags.get(339, [1])
    offsets, byte_counts = tags.get(273), tags.get(279)
    if (width, height, bits, compression, rows_per_strip, samples_per_pixel, sample_format) != (
        WIDTH, HEIGHT, [32], 1, 1, 1, [3]
    ):
        raise AuditError("TIFF layout differs from the audited uncompressed float32 row-strip layout.")
    if not isinstance(offsets, list) or not isinstance(byte_counts, list) or len(offsets) != HEIGHT or len(byte_counts) != HEIGHT:
        raise AuditError("Strip offset/byte-count arrays are absent or malformed.")
    if any(int(byte_counts[i]) != WIDTH * 4 for i in range(HEIGHT)):
        raise AuditError("Unexpected strip byte count; refusing to decode this TIFF layout.")
    if any(int(offsets[i + 1]) != int(offsets[i]) + int(byte_counts[i]) for i in range(HEIGHT - 1)):
        raise AuditError("Strips are not contiguous; refusing grouped range reads.")
    if int(offsets[0]) < 0 or int(offsets[-1]) + int(byte_counts[-1]) > EXPECTED_FILE_SIZE:
        raise AuditError("TIFF strip table extends outside the known file size.")

    def strip_slabs():
        boundaries = list(range(ROW0, ROW1, SLAB_ROWS)) + [ROW1]
        for r0, r1 in zip(boundaries[:-1], boundaries[1:]):
            start = int(offsets[r0])
            end = int(offsets[r1 - 1]) + int(byte_counts[r1 - 1]) - 1
            blob = client.get(start, end, EXPECTED_FILE_SIZE)
            full_rows = np.frombuffer(blob, dtype=np.dtype(endian + "f4")).reshape((r1 - r0, WIDTH))
            yield r0, full_rows[:, COL0:COL1]

    window = assemble_terrain_window_from_slabs(
        strip_slabs(), row_start=ROW0, row_stop=ROW1,
        column_start=COL0, column_stop=COL1,
        center_x_m=CX, center_y_m=CY,
        affine=(PIXEL, 0.0, X_ORIGIN, 0.0, -PIXEL, Y_ORIGIN),
    )
    return window, {
        "response_body_bytes_received": budget.received,
        "aggregate_response_body_cap_bytes": CAP,
        "request_ranges": client.ranges,
        "etag_values_observed": list(dict.fromkeys(client.etag_values)),
        "tiff_layout": {
            "classic_tiff": True, "byte_order": "little" if endian == "<" else "big",
            "width": width, "height": height, "bits_per_sample": bits,
            "compression": compression, "rows_per_strip": rows_per_strip,
            "strips_contiguous": True,
        },
    }


def write_report(document: dict[str, object]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)


def offline_self_test() -> None:
    from io import BytesIO
    from email.message import Message

    class FakeResponse(BytesIO):
        def __init__(self, status: int, headers: dict[str, str], body: bytes):
            super().__init__(body)
            self.status = status
            self.headers = Message()
            self.read_calls = 0
            for key, value in headers.items():
                self.headers[key] = value

        def read(self, size: int = -1) -> bytes:
            self.read_calls += 1
            return super().read(size)

    class FakeOpener:
        def __init__(self, *responses: FakeResponse):
            self.responses = list(responses)
            self.requests = []

        def open(self, request, timeout):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            return self.responses.pop(0)

    b = Budget(10)
    assert b.consume(BytesIO(b"123456"), 6) == b"123456"
    assert b.received == 6
    assert b.consume(BytesIO(b"7890"), 4) == b"7890"
    assert b.received == 10
    b = Budget(10)
    bad_response = FakeResponse(200, {"Content-Length": "6", "ETag": '"v1"'}, b"ignored")
    client = RangeClient(URL, b)
    client.opener = FakeOpener(bad_response)  # type: ignore[assignment]
    try:
        client.get(0, 5, 10)
    except AuditError:
        pass
    else:
        raise AssertionError("Unexpected HTTP status must stop the range reader.")
    assert bad_response.read_calls == 0 and b.received == 0
    bad_response = FakeResponse(206, {"Content-Range": "bytes 0-5/10", "Content-Length": "6", "ETag": '"v1"'}, b"ignored")
    client.opener = FakeOpener(bad_response)  # type: ignore[assignment]
    try:
        client.get(0, 5, 11)
    except AuditError:
        pass
    else:
        raise AssertionError("Unexpected Content-Range must stop the reader.")
    assert bad_response.read_calls == 0 and b.received == 0
    missing_initial_etag = FakeResponse(206, {"Content-Range": "bytes 0-3/10", "Content-Length": "4"}, b"data")
    client.opener = FakeOpener(missing_initial_etag)  # type: ignore[assignment]
    try:
        client.get(0, 3, 10)
    except AuditError:
        pass
    else:
        raise AssertionError("The initial prefix must provide an ETag.")
    assert missing_initial_etag.read_calls == 0 and client.expected_etag is None
    weak_initial_etag = FakeResponse(206, {"Content-Range": "bytes 0-3/10", "Content-Length": "4", "ETag": 'W/"v1"'}, b"data")
    client.opener = FakeOpener(weak_initial_etag)  # type: ignore[assignment]
    try:
        client.get(0, 3, 10)
    except AuditError:
        pass
    else:
        raise AssertionError("A weak ETag cannot be used for If-Match.")
    assert weak_initial_etag.read_calls == 0 and client.expected_etag is None
    initial = FakeResponse(206, {"Content-Range": "bytes 0-3/10", "Content-Length": "4", "ETag": '"v1"'}, b"abcd")
    mismatch = FakeResponse(206, {"Content-Range": "bytes 4-7/10", "Content-Length": "4", "ETag": '"v2"'}, b"efgh")
    opener = FakeOpener(initial, mismatch)
    client.opener = opener  # type: ignore[assignment]
    assert client.get(0, 3, 10) == b"abcd"
    try:
        client.get(4, 7, 10)
    except AuditError:
        pass
    else:
        raise AssertionError("A changed ETag must stop the read before its response body is consumed.")
    assert opener.requests[1].get_header("If-match") == '"v1"'
    assert mismatch.read_calls == 0 and client.budget.received == 4
    wrong_length = FakeResponse(206, {"Content-Range": "bytes 4-7/10", "Content-Length": "3", "ETag": '"v1"'}, b"efg")
    opener = FakeOpener(wrong_length)
    client.opener = opener  # type: ignore[assignment]
    try:
        client.get(4, 7, 10)
    except AuditError:
        pass
    else:
        raise AssertionError("A mismatched Content-Length must stop the read.")
    assert opener.requests[0].get_header("If-match") == '"v1"'
    assert wrong_length.read_calls == 0
    missing_later = FakeResponse(206, {"Content-Range": "bytes 4-7/10", "Content-Length": "4"}, b"efgh")
    client.opener = FakeOpener(missing_later)  # type: ignore[assignment]
    try:
        client.get(4, 7, 10)
    except AuditError:
        pass
    else:
        raise AssertionError("A later range without an ETag must stop the read.")
    assert missing_later.read_calls == 0
    b = Budget(10)
    b.consume(BytesIO(b"123456"), 6)
    b.consume(BytesIO(b"7890"), 4)
    try:
        b.consume(BytesIO(b"1"), 1)
    except AuditError:
        pass
    else:
        raise AssertionError("Repeated response bodies must not exceed the aggregate cap.")
    assert b.received == 10
    b = Budget(10)
    try:
        b.consume(BytesIO(b"01234567890"), 11)
    except AuditError:
        pass
    else:
        raise AssertionError("A response whose Content-Length exceeds the cap must be rejected.")
    assert b.received == 0
    boundaries = list(range(ROW0, ROW1, SLAB_ROWS)) + [ROW1]
    slabs = list(zip(boundaries[:-1], boundaries[1:]))
    assert len(slabs) == 63 and slabs[-1] == (2584, 2601)
    rows = np.arange(ROW0, ROW1, dtype=np.float64)
    cols = np.arange(COL0, COL1, dtype=np.float64)
    px = X_ORIGIN + (cols + 0.5) * PIXEL
    py = Y_ORIGIN - (rows + 0.5) * PIXEL
    circle_count = int((((px[None, :] - CX) ** 2 + (py[:, None] - CY) ** 2) <= RADIUS ** 2).sum())
    assert circle_count == 3_141_633
    print("offline reader checks passed: cumulative byte cap, strong ETag and If-Match consistency, missing/weak/changed ETag rejection, unexpected status/range rejection, slab layout, circle count")


def run_audit() -> None:
    offline_self_test()
    budget, client = Budget(CAP), None
    # The preceding approved attempt consumed the metadata prefix before its
    # parser stopped. Count that body again against this audit's single cap.
    budget.received = PRIOR_ATTEMPT_BODY_BYTES
    report: dict[str, object] = {
        "schema_version": "terrain-sample-coverage-review/1.0",
        "review_status": "incomplete-bounded-read",
        "site_id": "Site07",
        "source_url": URL,
        "requested_radius_m": RADIUS,
        "window_rows_half_open": [ROW0, ROW1],
        "window_columns_half_open": [COL0, COL1],
        "slab_rows": SLAB_ROWS,
        "aggregate_response_body_cap_bytes": CAP,
        "prior_attempt_response_body_bytes": PRIOR_ATTEMPT_BODY_BYTES,
        "prior_attempt": {
            "status": "stopped-during-local TIFF header parsing before elevation reads",
            "request": "GET Range bytes=0-65535",
            "response_body_bytes": PRIOR_ATTEMPT_BODY_BYTES,
            "parser_issue": "Initial parser rejected unrelated GeoTIFF DOUBLE tag 33550; parser now ignores unrelated tags.",
        },
        "response_body_bytes_received": PRIOR_ATTEMPT_BODY_BYTES,
        "completed_slabs": [],
        "expected_circle_pixel_centers": 3_141_633,
        "valid_circle_samples": 0,
        "nodata_or_nonfinite_circle_samples": 0,
        "failed_slab": None,
        "horizon_generated": False,
    }
    try:
        client = RangeClient(URL, budget)
        prefix = client.get(0, PREFIX_SIZE - 1, EXPECTED_FILE_SIZE)
        endian, tags = parse_classic_tiff(prefix)

        def scalar(tag: int) -> int:
            value = tags.get(tag)
            if not isinstance(value, list) or len(value) != 1:
                raise AuditError(f"Missing or unsupported scalar TIFF tag {tag}.")
            return int(value[0])

        width, height, bits = scalar(256), scalar(257), tags.get(258)
        compression, rows_per_strip = scalar(259), scalar(278)
        samples_per_pixel = scalar(277)
        sample_format = tags.get(339, [1])
        offsets, byte_counts = tags.get(273), tags.get(279)
        if (width, height, bits, compression, rows_per_strip, samples_per_pixel, sample_format) != (
            WIDTH, HEIGHT, [32], 1, 1, 1, [3]
        ):
            raise AuditError("TIFF layout differs from the explicitly supported uncompressed float32 row-strip layout.")
        if not isinstance(offsets, list) or not isinstance(byte_counts, list) or len(offsets) != HEIGHT or len(byte_counts) != HEIGHT:
            raise AuditError("Strip offset/byte-count arrays are absent or malformed.")
        if any(int(byte_counts[i]) != WIDTH * 4 for i in range(HEIGHT)):
            raise AuditError("Unexpected strip byte count; refusing to decode this TIFF layout.")
        if any(int(offsets[i + 1]) != int(offsets[i]) + int(byte_counts[i]) for i in range(HEIGHT - 1)):
            raise AuditError("Strips are not contiguous; refusing grouped range reads.")
        if int(offsets[0]) + sum(map(int, byte_counts)) > EXPECTED_FILE_SIZE:
            raise AuditError("Strip table exceeds the known file size.")

        report["tiff_layout_validated"] = {
            "classic_tiff": True, "byte_order": "little" if endian == "<" else "big",
            "width": width, "height": height, "bits_per_sample": bits,
            "compression": compression, "rows_per_strip": rows_per_strip,
            "strip_count": len(offsets), "strip_byte_count": WIDTH * 4,
            "strips_contiguous": True,
        }
        nodata_raw = tags.get(42113)
        nodata_text = nodata_raw.decode("ascii", errors="strict").strip("\x00 \t\r\n") if isinstance(nodata_raw, bytes) else None
        if nodata_text is not None and nodata_text.lower() != "nan":
            try:
                nodata_value = float(nodata_text)
            except ValueError as exc:
                raise AuditError(f"Unrecognized GDAL_NODATA value {nodata_text!r}.") from exc
        else:
            nodata_value = math.nan
        report["nodata_encoding"] = nodata_text or "NaN as recorded in the bounded metadata review"
        report["no_data_value"] = nodata_text or "NaN"
        boundaries = list(range(ROW0, ROW1, SLAB_ROWS)) + [ROW1]
        for slab_index, (r0, r1) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            start = int(offsets[r0])
            end = int(offsets[r1 - 1]) + int(byte_counts[r1 - 1]) - 1
            try:
                blob = client.get(start, end, EXPECTED_FILE_SIZE)
                block = np.frombuffer(blob, dtype=np.dtype(endian + "f4")).reshape((r1 - r0, WIDTH))
                rows = np.arange(r0, r1, dtype=np.float64)
                cols = np.arange(COL0, COL1, dtype=np.float64)
                xs = X_ORIGIN + (cols + 0.5) * PIXEL
                ys = Y_ORIGIN - (rows + 0.5) * PIXEL
                circle = (ys[:, None] - CY) ** 2 + (xs[None, :] - CX) ** 2 <= RADIUS ** 2
                values = block[:, COL0:COL1]
                invalid = ~np.isfinite(values)
                if math.isfinite(nodata_value):
                    invalid |= values == nodata_value
                circle_count = int(circle.sum())
                valid_count = int((circle & ~invalid).sum())
                invalid_count = int((circle & invalid).sum())
                report["valid_circle_samples"] = int(report["valid_circle_samples"]) + valid_count
                report["nodata_or_nonfinite_circle_samples"] = int(report["nodata_or_nonfinite_circle_samples"]) + invalid_count
                completed = report["completed_slabs"]
                assert isinstance(completed, list)
                completed.append({"slab_index": slab_index, "rows_half_open": [r0, r1], "circle_centers": circle_count,
                                  "valid_circle_samples": valid_count, "nodata_or_nonfinite_circle_samples": invalid_count,
                                  "response_body_bytes_cumulative": budget.received})
                report["response_body_bytes_received"] = budget.received
                write_report(report)
                del block, values, circle, invalid
            except Exception as exc:
                report["failed_slab"] = {"slab_index": slab_index, "rows_half_open": [r0, r1], "error": str(exc)}
                raise
        report["review_status"] = "bounded-sample-audit-complete"
        report["usability_assessment"] = "Coverage counts describe circle pixel centers only; they do not establish horizon validity."
    except Exception as exc:
        report["review_status"] = "incomplete-bounded-read"
        report["failure"] = str(exc)
    finally:
        report["response_body_bytes_received"] = budget.received
        report["request_ranges"] = client.ranges if client else []
        report["horizon_generated"] = False
        write_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        offline_self_test()
    else:
        run_audit()
