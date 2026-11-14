"""Explicit SPICE bridge between LOLA's DE421 ME and project DE440 PA frames."""

from math import isfinite
from pathlib import Path

import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError


class LunarFrameBridge:
    """Transform vectors between ``MOON_ME_DE421`` and ``MOON_PA_DE440``.

    The bridge loads only the two DE421 frame kernels it needs. The DE440
    FK/PCK and leap-seconds kernel must already be furnished (normally by
    ``SpiceGeometryEngine``). Frames are specified by their DE-specific names
    so generic ``MOON_ME``/``MOON_PA`` aliases cannot silently select a
    different ephemeris realization.
    """

    DEM_FRAME = "MOON_ME_DE421"
    GEOMETRY_FRAME = "MOON_PA_DE440"
    KERNEL_NAMES = ("moon_080317.tf", "moon_pa_de421_1900-2050.bpc")

    def __init__(self, spice_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.spice_dir = Path(spice_dir) if spice_dir is not None else project_root / "data" / "spice"
        self._loaded_kernels: list[Path] = []

    def load_kernels(self) -> None:
        """Furnish the DE421 FK and orientation PCK, with explicit missing-file errors."""
        paths = [self.spice_dir / name for name in self.KERNEL_NAMES]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required DE421 frame kernel(s) not found: " + ", ".join(map(str, missing)))
        try:
            for path in paths:
                spice.furnsh(str(path))
                self._loaded_kernels.append(path)
        except SpiceyError as exc:
            self.close()
            raise RuntimeError(f"SPICE could not load DE421 kernel {path}: {exc}") from exc

    def close(self) -> None:
        """Unload this bridge's kernels without clearing the shared SPICE pool."""
        while self._loaded_kernels:
            path = self._loaded_kernels[-1]
            try:
                spice.unload(str(path))
            except SpiceyError as exc:
                raise RuntimeError(f"SPICE could not unload DE421 kernel {path}: {exc}") from exc
            self._loaded_kernels.pop()

    def __enter__(self) -> "LunarFrameBridge":
        self.load_kernels()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def frame_transform_matrix(
        self,
        et: float,
        from_frame: str = DEM_FRAME,
        to_frame: str = GEOMETRY_FRAME,
    ) -> tuple[tuple[float, float, float], ...]:
        """Return the SPICE rotation matrix from ``from_frame`` to ``to_frame`` at ET."""
        self._require_loaded()
        try:
            matrix = spice.pxform(from_frame, to_frame, et)
        except SpiceyError as exc:
            raise RuntimeError(f"SPICE could not transform {from_frame} to {to_frame} at ET {et}: {exc}") from exc
        return tuple(tuple(float(value) for value in row) for row in matrix)

    def vector_dem_to_geometry(self, vector_dem: tuple[float, float, float], et: float) -> tuple[float, float, float]:
        """Rotate a vector from ``MOON_ME_DE421`` into ``MOON_PA_DE440``."""
        return self._rotate(vector_dem, et, self.DEM_FRAME, self.GEOMETRY_FRAME)

    def vector_geometry_to_dem(
        self, vector_geometry: tuple[float, float, float], et: float
    ) -> tuple[float, float, float]:
        """Rotate a vector from ``MOON_PA_DE440`` into ``MOON_ME_DE421``."""
        return self._rotate(vector_geometry, et, self.GEOMETRY_FRAME, self.DEM_FRAME)

    def _rotate(
        self, vector: tuple[float, float, float], et: float, from_frame: str, to_frame: str
    ) -> tuple[float, float, float]:
        if len(vector) != 3 or not all(isfinite(float(value)) for value in vector):
            raise ValueError("A frame-bridge vector must have exactly three finite components.")
        matrix = self.frame_transform_matrix(et, from_frame, to_frame)
        try:
            rotated = spice.mxv(matrix, vector)
        except SpiceyError as exc:
            raise RuntimeError(f"SPICE could not rotate a vector from {from_frame} to {to_frame}: {exc}") from exc
        return tuple(float(value) for value in rotated)

    def _require_loaded(self) -> None:
        if len(self._loaded_kernels) != len(self.KERNEL_NAMES):
            raise RuntimeError("Load both DE421 frame kernels before using the lunar frame bridge.")
