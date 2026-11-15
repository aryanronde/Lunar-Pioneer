"""Check or provision the exact SPICE kernels configured by Lunar Pioneer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import re
import sys
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import URLError
from urllib.request import urlopen

import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPICE_DIR = ROOT / "data" / "spice"
NAIF = "https://naif.jpl.nasa.gov/pub/naif"
CLPS_CHECKSUMS_URL = f"{NAIF}/pds/pds4/clps/clps_spice/miscellaneous/checksum/checksum_v004.tab"
DE440_MD5_URL = f"{NAIF}/generic_kernels/spk/planets/aa_checksums.txt"
DE440_MD5 = "c9d581bfd84209dbeee8b1583939b148"


@dataclass(frozen=True)
class KernelSpec:
    filename: str
    url: str
    checksum_manifest: str | None = None
    required_md5: str | None = None


KERNEL_SPECS = (
    KernelSpec("de440.bsp", f"{NAIF}/generic_kernels/spk/planets/de440.bsp", DE440_MD5_URL, DE440_MD5),
    KernelSpec("moon_de440_250416.tf", f"{NAIF}/generic_kernels/fk/satellites/moon_de440_250416.tf", CLPS_CHECKSUMS_URL),
    KernelSpec("moon_pa_de440_200625.bpc", f"{NAIF}/generic_kernels/pck/moon_pa_de440_200625.bpc", CLPS_CHECKSUMS_URL),
    KernelSpec("naif0012.tls", f"{NAIF}/generic_kernels/lsk/naif0012.tls", CLPS_CHECKSUMS_URL),
    KernelSpec("pck00011.tpc", f"{NAIF}/generic_kernels/pck/pck00011.tpc", CLPS_CHECKSUMS_URL),
    KernelSpec("moon_080317.tf", f"{NAIF}/generic_kernels/fk/satellites/moon_080317.tf"),
    KernelSpec("moon_pa_de421_1900-2050.bpc", f"{NAIF}/generic_kernels/pck/moon_pa_de421_1900-2050.bpc"),
)


class KernelSetupError(RuntimeError):
    """A required kernel could not be downloaded or validated."""


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected: str) -> bool:
    """Compare a local file with an official NAIF/PDS MD5 value."""
    if not re.fullmatch(r"[0-9a-fA-F]{32}", expected):
        raise ValueError("Expected MD5 must contain exactly 32 hexadecimal characters.")
    return _md5(path).lower() == expected.lower()


def _fetch_text(url: str, opener: Callable = urlopen) -> str:
    with opener(url, timeout=60) as response:
        return response.read().decode("ascii", errors="replace")


def _manifest_md5(text: str, filename: str) -> str | None:
    """Find a published checksum line containing the exact kernel basename."""
    for line in text.splitlines():
        if filename.casefold() not in line.casefold():
            continue
        match = re.search(r"\b([0-9a-fA-F]{32})\b", line)
        if match:
            return match.group(1).lower()
    return None


def _validate_spice_file(path: Path) -> str | None:
    """Return a validation problem, or None when SPICE can furnish the file."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty file"
    try:
        spice.kclear()
        spice.furnsh(str(path))
        spice.kclear()
    except SpiceyError as exc:
        spice.kclear()
        return f"SPICE rejected the file: {exc}"
    return None


def check_kernels(
    spice_dir: Path = DEFAULT_SPICE_DIR,
    specs: Iterable[KernelSpec] = KERNEL_SPECS,
) -> list[str]:
    """Check all configured local files without making any network requests."""
    problems: list[str] = []
    for spec in specs:
        path = spice_dir / spec.filename
        problem = _validate_spice_file(path)
        if problem:
            problems.append(f"{spec.filename}: {problem}")
        elif spec.required_md5 and not verify_md5(path, spec.required_md5):
            problems.append(f"{spec.filename}: official NAIF MD5 mismatch")
    return problems


def _download(url: str, destination: Path, opener: Callable) -> None:
    with opener(url, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def provision_kernels(
    spice_dir: Path = DEFAULT_SPICE_DIR,
    *,
    opener: Callable = urlopen,
    specs: Iterable[KernelSpec] = KERNEL_SPECS,
) -> list[str]:
    """Download missing/invalid files from NAIF and atomically replace invalid ones."""
    selected = tuple(specs)
    spice_dir.mkdir(parents=True, exist_ok=True)
    manifest_cache: dict[str, str] = {}
    manifest_errors: dict[str, Exception] = {}
    messages: list[str] = []

    for manifest_url in {spec.checksum_manifest for spec in selected if spec.checksum_manifest}:
        try:
            manifest_cache[manifest_url] = _fetch_text(manifest_url, opener)
        except (OSError, URLError, TimeoutError) as exc:
            manifest_errors[manifest_url] = exc

    for spec in selected:
        destination = spice_dir / spec.filename
        problem = _validate_spice_file(destination)
        expected_md5 = spec.required_md5
        manifest_text = manifest_cache.get(spec.checksum_manifest) if spec.checksum_manifest else None
        if expected_md5 is None and manifest_text is not None:
            expected_md5 = _manifest_md5(manifest_text, spec.filename)
        if not problem and expected_md5 and not verify_md5(destination, expected_md5):
            problem = "official MD5 mismatch"
        if not problem:
            messages.append(f"OK (kept existing): {spec.filename}")
            continue
        if spec.checksum_manifest and expected_md5 is None:
            if spec.checksum_manifest in manifest_errors:
                exc = manifest_errors[spec.checksum_manifest]
                raise KernelSetupError(f"Could not retrieve official checksum manifest for {spec.filename}: {exc}") from exc
            raise KernelSetupError(f"Official checksum manifest has no entry for required kernel {spec.filename}.")

        temporary = destination.with_name(destination.name + ".download")
        try:
            _download(spec.url, temporary, opener)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise KernelSetupError(f"NAIF returned an empty download for {spec.filename}.")
            if expected_md5 and not verify_md5(temporary, expected_md5):
                raise KernelSetupError(f"Official MD5 verification failed for {spec.filename}.")
            validation = _validate_spice_file(temporary)
            if validation:
                raise KernelSetupError(f"Downloaded {spec.filename} is invalid: {validation}.")
            temporary.replace(destination)
        except (OSError, URLError, TimeoutError, SpiceyError) as exc:
            raise KernelSetupError(
                f"Could not provision {spec.filename} from its official NAIF source: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        messages.append(f"Downloaded and verified: {spec.filename}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spice-dir", type=Path, default=DEFAULT_SPICE_DIR, help="Kernel directory (default: data/spice).")
    parser.add_argument("--check", action="store_true", help="Validate local kernels only; never use the network.")
    args = parser.parse_args(argv)
    try:
        if args.check:
            problems = check_kernels(args.spice_dir)
            if problems:
                print("SPICE kernel check failed:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                print("Run: python scripts/setup_spice_kernels.py", file=sys.stderr)
                return 1
            print(f"All {len(KERNEL_SPECS)} required SPICE kernels are present and loadable.")
            return 0
        for message in provision_kernels(args.spice_dir):
            print(message)
        problems = check_kernels(args.spice_dir)
        if problems:
            raise KernelSetupError("Post-install check failed: " + "; ".join(problems))
        print("SPICE kernel setup complete.")
        return 0
    except KernelSetupError as exc:
        print(f"SPICE kernel setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
