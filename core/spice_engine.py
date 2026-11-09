"""SPICE based inertial Sun/Earth geometry relative to the Moon."""

from pathlib import Path

import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError


class SpiceGeometryEngine:
    """Load project kernels and compute body positions relative to the Moon.

    Positions are returned in kilometers in the J2000 inertial frame. The
    class does not transform them into a lunar surface horizon frame.
    """

    KERNEL_NAMES = (
        "de440.bsp",
        "moon_de440_250416.tf",
        "moon_pa_de440_200625.bpc",
        "naif0012.tls",
        "pck00011.tpc",
    )
    REFERENCE_FRAME = "J2000"
    # This class-2 frame is defined by the lunar FK and its orientation comes
    # from the loaded DE440 binary PCK. It is not the IAU_MOON frame.
    MOON_FIXED_FRAME = "MOON_PA_DE440"
    ABERRATION_CORRECTION = "NONE"

    def __init__(self, spice_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.spice_dir = Path(spice_dir) if spice_dir is not None else project_root / "data" / "spice"
        self._loaded_kernels: list[Path] = []

    def load_kernels(self) -> None:
        """Load all five configured kernels, reporting missing or invalid files."""
        kernel_paths = [self.spice_dir / name for name in self.KERNEL_NAMES]
        missing = [path for path in kernel_paths if not path.is_file()]
        if missing:
            details = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Required SPICE kernel(s) not found: {details}")

        try:
            for path in kernel_paths:
                spice.furnsh(str(path))
                self._loaded_kernels.append(path)
        except SpiceyError as exc:
            self.close()
            raise RuntimeError(f"SPICE could not load kernel {path}: {exc}") from exc

    def close(self) -> None:
        """Unload kernels furnished by this engine, preserving unrelated kernels."""
        for path in reversed(self._loaded_kernels):
            try:
                spice.unload(str(path))
            except SpiceyError as exc:
                raise RuntimeError(f"SPICE could not unload kernel {path}: {exc}") from exc
            finally:
                self._loaded_kernels.remove(path)

    def __enter__(self) -> "SpiceGeometryEngine":
        self.load_kernels()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def utc_to_et(self, utc_timestamp: str) -> float:
        """Convert a UTC timestamp to SPICE ephemeris time (ET/TDB seconds)."""
        self._require_loaded()
        try:
            return float(spice.str2et(utc_timestamp))
        except SpiceyError as exc:
            raise ValueError(f"SPICE could not parse UTC timestamp {utc_timestamp!r}: {exc}") from exc

    def sun_position_km(self, et: float) -> tuple[float, float, float]:
        """Return Sun relative to Moon in J2000, in kilometers."""
        return self._position_relative_to_moon("SUN", et)

    def earth_position_km(self, et: float) -> tuple[float, float, float]:
        """Return Earth relative to Moon in J2000, in kilometers."""
        return self._position_relative_to_moon("EARTH", et)

    def j2000_to_moon_fixed(
        self, vector_j2000: tuple[float, float, float], et: float
    ) -> tuple[float, float, float]:
        """Rotate a J2000 vector into the DE440 Moon PA frame at ET.

        This is a pure frame rotation: it does not add a translation. Supply
        a vector already expressed relative to the Moon (as returned by the
        position methods) when transforming Sun/Earth directions.
        """
        self._require_loaded()
        if len(vector_j2000) != 3:
            raise ValueError("A Cartesian vector must have exactly three components.")
        try:
            rotation = spice.pxform(self.REFERENCE_FRAME, self.MOON_FIXED_FRAME, et)
            transformed = spice.mxv(rotation, vector_j2000)
        except SpiceyError as exc:
            raise RuntimeError(
                f"SPICE could not rotate a vector from {self.REFERENCE_FRAME} "
                f"to {self.MOON_FIXED_FRAME} at ET {et}: {exc}"
            ) from exc
        return tuple(float(component) for component in transformed)

    def _position_relative_to_moon(self, target: str, et: float) -> tuple[float, float, float]:
        self._require_loaded()
        try:
            # spkpos evaluates target minus observer; NONE requests geometric
            # positions without light-time or stellar-aberration corrections.
            position, _light_time = spice.spkpos(
                target,
                et,
                self.REFERENCE_FRAME,
                self.ABERRATION_CORRECTION,
                "MOON",
            )
        except SpiceyError as exc:
            raise RuntimeError(f"SPICE could not compute {target} relative to Moon at ET {et}: {exc}") from exc
        return tuple(float(component) for component in position)

    def _require_loaded(self) -> None:
        if len(self._loaded_kernels) != len(self.KERNEL_NAMES):
            raise RuntimeError("All five SPICE kernels must be loaded before geometry operations.")
