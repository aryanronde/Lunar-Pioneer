"""SPICE Sun/Earth directions in the audited Site06 local terrain frame.

This module stops at geometric direction angles. It does not read a DEM or
compare the Sun/Earth against a terrain obstruction profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import spiceypy as spice

from .lunar_frame_bridge import LunarFrameBridge
from .lunar_horizon import LocalHorizonFrame, LunarSite
from .spice_engine import SpiceGeometryEngine
from .terrain_provider import MOON_ME_DE421


SITE06_OBSERVER_POSITION_ME_KM = (
    110.21143659666053,
    84.16382863817336,
    -1737.8192860716285,
)
SITE06_TERRAIN_UP_ME = (
    0.12396688856499274,
    0.30047987214444133,
    -0.9456976562176598,
)
SITE06_FRAME_REFERENCE_UTC = "2026-10-05T00:00:00"


@dataclass(frozen=True)
class CelestialDirection:
    """Geometric target direction relative to the Moon and Site06 horizon."""

    target: str
    utc_timestamp: str
    et_tdb_seconds: float
    frame: str
    position_vector_moon_me_de421_km: tuple[float, float, float]
    unit_vector_moon_me_de421: tuple[float, float, float]
    range_km: float
    azimuth_deg: float
    elevation_deg: float


class Site06CelestialDirectionEngine:
    """Compute geometric Sun/Earth angles in the audited Site06 basis.

    The target vector is computed by SPICE in J2000, rotated through
    MOON_PA_DE440, then converted with the existing frame bridge to
    MOON_ME_DE421. The fixed Site06 basis reuses LocalHorizonFrame's V0.2A
    North at the documented reference epoch, transforms that North into ME,
    projects it onto the measured terrain tangent plane, and sets
    East = North x Up.
    """

    def __init__(self, spice_dir: Path | None = None) -> None:
        self.geometry = SpiceGeometryEngine(spice_dir)
        self.bridge = LunarFrameBridge(spice_dir)
        self._entered = False
        self._north_me: np.ndarray | None = None
        self._east_me: np.ndarray | None = None
        self._up_me: np.ndarray | None = None

    def __enter__(self) -> "Site06CelestialDirectionEngine":
        self.geometry.load_kernels()
        try:
            self.bridge.load_kernels()
            self._build_local_basis()
        except Exception:
            self.bridge.close()
            self.geometry.close()
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.bridge.close()
        finally:
            self.geometry.close()
            self._entered = False
            self._north_me = None
            self._east_me = None
            self._up_me = None

    @property
    def local_basis(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Return (North, East, Up) unit axes in MOON_ME_DE421."""
        self._require_entered()
        assert self._north_me is not None and self._east_me is not None and self._up_me is not None
        return tuple(tuple(float(x) for x in axis) for axis in (self._north_me, self._east_me, self._up_me))

    def celestial_direction(self, utc_timestamp: str, target: str) -> CelestialDirection:
        """Return the geometric Sun or Earth direction for a supplied UTC time."""
        self._require_entered()
        canonical_target = target.strip().upper()
        if canonical_target not in {"SUN", "EARTH"}:
            raise ValueError("target must be 'SUN' or 'EARTH'.")
        et = self.geometry.utc_to_et(utc_timestamp)
        inertial = (
            self.geometry.sun_position_km(et)
            if canonical_target == "SUN"
            else self.geometry.earth_position_km(et)
        )
        pa_vector = self.geometry.j2000_to_moon_fixed(inertial, et)
        me_vector = self.bridge.vector_geometry_to_dem(pa_vector, et)
        vector = np.asarray(me_vector, dtype=np.float64)
        range_km = float(np.linalg.norm(vector))
        if not math.isfinite(range_km) or range_km <= 0.0:
            raise ValueError(f"SPICE returned an invalid {canonical_target} position vector.")
        unit = vector / range_km
        assert self._north_me is not None and self._east_me is not None and self._up_me is not None
        east_component = float(np.dot(unit, self._east_me))
        north_component = float(np.dot(unit, self._north_me))
        up_component = float(np.dot(unit, self._up_me))
        azimuth = math.degrees(math.atan2(east_component, north_component)) % 360.0
        elevation = math.degrees(math.atan2(up_component, math.hypot(east_component, north_component)))
        return CelestialDirection(
            target=canonical_target,
            utc_timestamp=utc_timestamp,
            et_tdb_seconds=et,
            frame=MOON_ME_DE421,
            position_vector_moon_me_de421_km=tuple(float(x) for x in vector),
            unit_vector_moon_me_de421=tuple(float(x) for x in unit),
            range_km=range_km,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
        )

    def _build_local_basis(self) -> None:
        """Project the transformed, audited V0.2A North against terrain Up."""
        ref_et = self.geometry.utc_to_et(SITE06_FRAME_REFERENCE_UTC)
        observer_me = np.asarray(SITE06_OBSERVER_POSITION_ME_KM, dtype=np.float64)
        observer_pa = np.asarray(self.bridge.vector_dem_to_geometry(tuple(observer_me), ref_et))
        radii = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
        ray = observer_pa / np.linalg.norm(observer_pa)
        surface_point = ray / math.sqrt(float(np.sum((ray / radii) ** 2)))
        geodetic_normal_pa = surface_point / (radii * radii)
        geodetic_normal_pa /= np.linalg.norm(geodetic_normal_pa)
        latitude = math.degrees(
            math.atan2(geodetic_normal_pa[2], math.hypot(geodetic_normal_pa[0], geodetic_normal_pa[1]))
        )
        longitude = math.degrees(math.atan2(geodetic_normal_pa[1], geodetic_normal_pa[0])) % 360.0
        v02a = LocalHorizonFrame.at_site(LunarSite(latitude, longitude))
        north_reference_me = np.asarray(
            self.bridge.vector_geometry_to_dem(v02a.north, ref_et), dtype=np.float64
        )

        up = np.asarray(SITE06_TERRAIN_UP_ME, dtype=np.float64)
        up /= np.linalg.norm(up)
        north_projected = north_reference_me - float(np.dot(north_reference_me, up)) * up
        north_length = float(np.linalg.norm(north_projected))
        if north_length <= 1e-12:
            raise ValueError("V0.2A North is parallel to Site06 terrain Up; local azimuth is undefined.")
        north = north_projected / north_length
        east = np.cross(north, up)
        east_length = float(np.linalg.norm(east))
        if east_length <= 1e-12:
            raise ValueError("Could not construct Site06 East from North x Up.")
        east /= east_length
        self._north_me, self._east_me, self._up_me = north, east, up

    def _require_entered(self) -> None:
        if not self._entered:
            raise RuntimeError("Use Site06CelestialDirectionEngine as a context manager before computing directions.")


def celestial_direction(utc_timestamp: str, target: str) -> CelestialDirection:
    """Convenience API that loads local kernels, calculates one direction, and closes them."""
    with Site06CelestialDirectionEngine() as engine:
        return engine.celestial_direction(utc_timestamp, target)
