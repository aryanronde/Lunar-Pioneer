"""Versioned, provenance-aware terrain-horizon report schema.

This module only validates and serializes report data. It performs no raster,
SPICE, network, horizon, or visibility calculations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "1.0"
FRAME_EVIDENCE_STATUSES = {"independently_verified", "code_assigned", "unverified"}
WINDOW_KEYS = {"rows_half_open", "columns_half_open"}


def summarize_raster_window(elevation: Any, nodata_mask: Any) -> dict[str, Any]:
    """Summarize a local raster window without interpreting or filling samples."""
    values = np.asarray(elevation)
    missing_mask = np.asarray(nodata_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("Elevation window must be a two-dimensional array.")
    if missing_mask.shape != values.shape:
        raise ValueError("NoData mask must have the same shape as the elevation window.")
    missing_mask = missing_mask | ~np.isfinite(values)
    valid = np.asarray(values, dtype=np.float64)[~missing_mask]
    return {
        "shape_rows_columns": [int(values.shape[0]), int(values.shape[1])],
        "valid_pixel_count": int(valid.size),
        "nodata_pixel_count": int(missing_mask.sum()),
        "minimum_elevation_m": float(np.min(valid)) if valid.size else None,
        "maximum_elevation_m": float(np.max(valid)) if valid.size else None,
        "median_elevation_m": float(np.median(valid)) if valid.size else None,
    }


def make_raster_source_record(
    *,
    source_id: str,
    source_url: str | None,
    product_id: str | None,
    source_hash: str | None,
    resolution_m: Sequence[float] | None,
    raster_dimensions_rows_columns: Sequence[int] | None,
    dtype: str | None,
    crs_wkt: str | None,
    affine: Sequence[float] | None,
    pixel_registration: str | None,
    elevation_units: str | None,
    scale: float | None,
    offset: float | None,
    nodata_value: float | str | None,
    nodata_handling: str,
    elevation_units_evidence: Mapping[str, Any] | None,
    elevation: Any,
    nodata_mask: Any,
    coverage_counts: Mapping[str, int],
    requested_window: Mapping[str, Sequence[int]] | None,
    returned_window: Mapping[str, Sequence[int]] | None,
    source_frame: str | None,
    frame_evidence_status: str,
    frame_evidence: Mapping[str, Any] | None,
    unavailable: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one source record; callers must explain each unavailable value."""
    record = {
        "source_id": source_id,
        "source_url": source_url,
        "product_id": product_id,
        "source_hash": source_hash,
        "resolution_m": list(resolution_m) if resolution_m is not None else None,
        "raster_dimensions_rows_columns": (
            list(raster_dimensions_rows_columns) if raster_dimensions_rows_columns is not None else None
        ),
        "dtype": dtype,
        "crs_wkt": crs_wkt,
        "affine": list(affine) if affine is not None else None,
        "pixel_registration": pixel_registration,
        "elevation_units": elevation_units,
        "scale": scale,
        "offset": offset,
        "nodata_value": nodata_value,
        "nodata_handling": nodata_handling,
        "elevation_units_evidence": dict(elevation_units_evidence) if elevation_units_evidence is not None else None,
        "window_statistics": summarize_raster_window(elevation, nodata_mask),
        "coverage_counts": dict(coverage_counts),
        "requested_window": _copy_window(requested_window),
        "returned_window": _copy_window(returned_window),
        "source_frame": {
            "value": source_frame,
            "evidence_status": frame_evidence_status,
            "evidence": dict(frame_evidence) if frame_evidence is not None else None,
        },
        "unavailable": dict(unavailable or {}),
    }
    validate_raster_source_record(record)
    return record


def make_horizon_report(
    *,
    generated_at_utc: str,
    site: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    radius_m: float,
    coverage_status: str,
    azimuth_bins: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    profile_angles_deg: Sequence[float | None],
    sample_counts_by_bin: Sequence[int],
    contributing_source_ids_by_bin: Sequence[Sequence[str]] | None,
    unavailable: Mapping[str, str] | None = None,
    composite_rule: str | None = None,
    comparison_profiles: Sequence[Mapping[str, Any]] = (),
    software: Mapping[str, Any] | None = None,
    spice_kernels: Sequence[Mapping[str, Any]] | None = None,
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    """Construct and validate a schema-1.0 report from explicit inputs."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "site": dict(site),
        "sources": [dict(source) for source in sources],
        "horizon": {
            "radius_m": radius_m,
            "coverage_status": coverage_status,
            "azimuth_bins": dict(azimuth_bins),
            "algorithm": dict(algorithm),
            "profile_angles_deg": list(profile_angles_deg),
            "sample_counts_by_bin": list(sample_counts_by_bin),
            "contributing_source_ids_by_bin": (
                [list(ids) for ids in contributing_source_ids_by_bin]
                if contributing_source_ids_by_bin is not None
                else None
            ),
            "composite_rule": composite_rule,
            "comparison_profiles": [dict(profile) for profile in comparison_profiles],
        },
        "software": dict(software or {}),
        "spice_kernels": [dict(kernel) for kernel in spice_kernels] if spice_kernels is not None else None,
        "limitations": list(limitations),
        "unavailable": dict(unavailable or {}),
    }
    validate_horizon_report(report)
    return report


def validate_raster_source_record(record: Mapping[str, Any]) -> None:
    """Raise ValueError when a source record is incomplete or inconsistent."""
    required = {
        "source_id", "source_url", "product_id", "source_hash", "resolution_m",
        "raster_dimensions_rows_columns", "dtype", "crs_wkt", "affine", "pixel_registration",
        "elevation_units", "elevation_units_evidence", "scale", "offset", "nodata_value",
        "nodata_handling", "coverage_counts", "window_statistics",
        "requested_window", "returned_window", "source_frame", "unavailable",
    }
    _require_keys(record, required, "source")
    if not isinstance(record["source_id"], str) or not record["source_id"].strip():
        raise ValueError("source_id must be a non-empty string.")
    resolution = record["resolution_m"]
    if resolution is not None and (len(resolution) != 2 or any(_positive_number(x) is False for x in resolution)):
        raise ValueError("resolution_m must contain two positive finite values or be null.")
    dimensions = record["raster_dimensions_rows_columns"]
    if dimensions is not None and (len(dimensions) != 2 or any(not _positive_int(x) for x in dimensions)):
        raise ValueError("Raster dimensions must contain two positive integers or be null.")
    affine = record["affine"]
    if affine is not None and (len(affine) != 6 or not all(_finite_number(x) for x in affine)):
        raise ValueError("Affine must contain six finite coefficients or be null.")
    for key in ("scale", "offset"):
        if record[key] is not None and not _finite_number(record[key]):
            raise ValueError(f"{key} must be finite or null.")
    _validate_window(record["requested_window"], "requested_window")
    _validate_window(record["returned_window"], "returned_window")
    frame = record["source_frame"]
    if not isinstance(frame, Mapping):
        raise ValueError("source_frame must contain value, evidence_status, and evidence.")
    _require_keys(frame, {"value", "evidence_status", "evidence"}, "source_frame")
    if frame["evidence_status"] not in FRAME_EVIDENCE_STATUSES:
        raise ValueError("source_frame evidence_status is invalid.")
    if frame["evidence_status"] == "independently_verified" and not frame["evidence"]:
        raise ValueError("Independently verified frame labels require evidence.")
    stats = record["window_statistics"]
    _require_keys(stats, {
        "shape_rows_columns", "valid_pixel_count", "nodata_pixel_count",
        "minimum_elevation_m", "maximum_elevation_m", "median_elevation_m",
    }, "window_statistics")
    _validate_counts(stats)
    coverage = record["coverage_counts"]
    _require_keys(coverage, {"requested_area_pixel_count", "valid_pixel_count", "nodata_pixel_count", "other_invalid_pixel_count", "out_of_coverage_pixel_count"}, "coverage_counts")
    if any(not _nonnegative_int(v) for v in coverage.values()):
        raise ValueError("Coverage counts must be non-negative integers.")
    if coverage["valid_pixel_count"] + coverage["nodata_pixel_count"] + coverage["other_invalid_pixel_count"] + coverage["out_of_coverage_pixel_count"] != coverage["requested_area_pixel_count"]:
        raise ValueError("Valid, NoData, other-invalid, and out-of-coverage counts must sum to requested area pixels.")
    if not isinstance(record["nodata_handling"], str) or not record["nodata_handling"].strip():
        raise ValueError("nodata_handling must describe the mask rule.")
    if record["returned_window"] is not None:
        rows = record["returned_window"]["rows_half_open"]
        cols = record["returned_window"]["columns_half_open"]
        if [rows[1] - rows[0], cols[1] - cols[0]] != stats["shape_rows_columns"]:
            raise ValueError("Returned window dimensions must match the summarized elevation array.")
    _validate_null_explanations(record, record["unavailable"], "source")


def validate_horizon_report(report: Mapping[str, Any]) -> None:
    """Validate structure, dimensions, bin arrays, counts, and null explanations."""
    required = {
        "schema_version", "generated_at_utc", "site", "sources", "horizon",
        "software", "spice_kernels", "limitations", "unavailable",
    }
    _require_keys(report, required, "report")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported horizon report schema version: {report['schema_version']!r}.")
    _validate_timestamp(report["generated_at_utc"])
    if not isinstance(report["site"], Mapping) or not isinstance(report["site"].get("id"), str):
        raise ValueError("site must be an object with a string id.")
    sources = report["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list of separate raster records.")
    for source in sources:
        validate_raster_source_record(source)
    source_ids = [source["source_id"] for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id values must be unique.")
    horizon = report["horizon"]
    _require_keys(horizon, {
        "radius_m", "coverage_status", "azimuth_bins", "algorithm",
        "profile_angles_deg", "sample_counts_by_bin", "contributing_source_ids_by_bin",
        "composite_rule", "comparison_profiles",
    }, "horizon")
    if not _positive_number(horizon["radius_m"]):
        raise ValueError("horizon.radius_m must be a positive finite number.")
    if horizon["coverage_status"] not in {"limited-range", "insufficient-coverage", "complete"}:
        raise ValueError("horizon.coverage_status is invalid.")
    bins = horizon["azimuth_bins"]
    _require_keys(bins, {"count", "width_deg", "centers_deg", "convention", "assignment"}, "azimuth_bins")
    count = bins["count"]
    if not _positive_int(count) or not _positive_number(bins["width_deg"]):
        raise ValueError("Azimuth bin count and width must be positive.")
    centers = bins["centers_deg"]
    if len(centers) != count or any(not _finite_number(x) for x in centers):
        raise ValueError("Azimuth centers must contain one finite value per bin.")
    if len(horizon["profile_angles_deg"]) != count or len(horizon["sample_counts_by_bin"]) != count:
        raise ValueError("Profile and per-bin sample-count arrays must match azimuth bin count.")
    if any(value is not None and not _finite_number(value) for value in horizon["profile_angles_deg"]):
        raise ValueError("Profile angles must be finite numbers or null for empty bins.")
    if any(not _nonnegative_int(value) for value in horizon["sample_counts_by_bin"]):
        raise ValueError("Per-bin sample counts must be non-negative integers.")
    contributors = horizon["contributing_source_ids_by_bin"]
    if contributors is not None:
        if len(contributors) != count:
            raise ValueError("Per-bin contributors must match azimuth bin count.")
        if any(not isinstance(ids, list) or any(source_id not in source_ids for source_id in ids) for ids in contributors):
            raise ValueError("Each contributing source id must reference a declared source.")
    for comparison in horizon["comparison_profiles"]:
        _require_keys(comparison, {"profile_id", "source_ids", "radius_m", "profile_angles_deg", "sample_counts_by_bin"}, "comparison_profile")
        if any(source_id not in source_ids for source_id in comparison["source_ids"]):
            raise ValueError("Comparison profile source ids must reference declared sources.")
        if len(comparison["profile_angles_deg"]) != count or len(comparison["sample_counts_by_bin"]) != count:
            raise ValueError("Comparison profile arrays must match azimuth bin count.")
    report_values = {key: value for key, value in report.items() if key not in {"sources", "unavailable"}}
    _validate_null_explanations(report_values, report["unavailable"], "report")


def serialize_horizon_report(report: Mapping[str, Any], path: str | Path | None = None) -> str:
    """Validate and return deterministic JSON; optionally write to a caller path."""
    validate_horizon_report(report)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def collect_runtime_provenance(script_path: str | Path, spice_dir: str | Path, kernel_names: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Collect installed package versions and local kernel hashes, without network access."""
    software: dict[str, Any] = {"python": sys.version.split()[0], "hash_algorithm": "SHA-256"}
    unavailable: dict[str, str] = {}
    for distribution in ("numpy", "rasterio", "pyproj", "spiceypy"):
        try:
            software[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            software[distribution] = None
            unavailable[f"/software/{distribution}"] = "Package version is not available in the local environment."
    script = Path(script_path)
    try:
        software["generator_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    except OSError:
        software["generator_sha256"] = None
        unavailable["/software/generator_sha256"] = "Generator source file could not be read for hashing."
    kernels = []
    root = Path(spice_dir)
    for index, name in enumerate(kernel_names):
        path = root / name
        kernel_unavailable: dict[str, str] = {"/version": "Kernel version is not separately recorded by the project."}
        record: dict[str, Any] = {"name": name, "path": str(path), "version": None, "sha256": None}
        unavailable[f"/spice_kernels/{index}/version"] = "Kernel version is not separately recorded by the project."
        try:
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            unavailable[f"/spice_kernels/{index}/sha256"] = "Kernel file is not available locally for hashing."
        record["unavailable"] = kernel_unavailable
        kernels.append(record)
    return software, kernels, unavailable


def _copy_window(window: Mapping[str, Sequence[int]] | None) -> dict[str, list[int]] | None:
    if window is None:
        return None
    return {key: list(window[key]) for key in WINDOW_KEYS if key in window}


def _validate_window(window: Any, name: str) -> None:
    if window is None:
        return
    if not isinstance(window, Mapping) or set(window) != WINDOW_KEYS:
        raise ValueError(f"{name} must contain rows_half_open and columns_half_open, or be null.")
    for axis in WINDOW_KEYS:
        pair = window[axis]
        if len(pair) != 2 or any(not _nonnegative_int(v) for v in pair) or pair[1] < pair[0]:
            raise ValueError(f"{name}.{axis} must be a non-negative half-open index pair.")


def _validate_counts(stats: Mapping[str, Any]) -> None:
    shape = stats["shape_rows_columns"]
    if len(shape) != 2 or any(not _positive_int(v) for v in shape):
        raise ValueError("Raster window shape must contain two positive dimensions.")
    valid, missing = stats["valid_pixel_count"], stats["nodata_pixel_count"]
    if not _nonnegative_int(valid) or not _nonnegative_int(missing):
        raise ValueError("Raster valid and NoData counts must be non-negative integers.")
    if valid + missing != shape[0] * shape[1]:
        raise ValueError("Valid plus NoData counts must equal the raster window pixel count.")
    for key in ("minimum_elevation_m", "maximum_elevation_m", "median_elevation_m"):
        value = stats[key]
        if valid and (value is None or not _finite_number(value)):
            raise ValueError(f"{key} must be finite when valid pixels exist.")
        if not valid and value is not None:
            raise ValueError(f"{key} must be null when no valid pixels exist.")
    if valid and not (stats["minimum_elevation_m"] <= stats["median_elevation_m"] <= stats["maximum_elevation_m"]):
        raise ValueError("Raster elevation statistics are inconsistent.")


def _validate_null_explanations(value: Any, explanations: Mapping[str, str], root: str) -> None:
    if not isinstance(explanations, Mapping) or any(not isinstance(v, str) or not v.strip() for v in explanations.values()):
        raise ValueError(f"{root}.unavailable must map JSON Pointer paths to non-empty explanations.")
    null_paths: set[str] = set()

    def visit(item: Any, path: str) -> None:
        if item is None:
            # Empty azimuth bins are valid scientific output when their sample
            # count is zero; they are not missing provenance metadata.
            if "/profile_angles_deg/" not in path:
                null_paths.add(path)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                if key != "unavailable":
                    visit(child, f"{path}/{_escape_pointer(str(key))}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(value, "")
    missing = [path for path in null_paths if path not in explanations]
    if missing:
        raise ValueError("Every null value requires an unavailable explanation: " + ", ".join(sorted(missing)))


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("generated_at_utc must be an ISO-8601 UTC timestamp ending in Z.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("generated_at_utc is not a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError("generated_at_utc must use UTC.")


def _require_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(sorted(missing))}.")


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
