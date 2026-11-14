"""Geometric site coordinates and local horizon calculations for the Moon."""

from dataclasses import dataclass
import math

import numpy as np


def _vector3(vector: tuple[float, float, float], label: str) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain exactly three finite components.")
    return result


@dataclass(frozen=True)
class LunarSite:
    """A site in geodetic latitude and east-positive longitude, in degrees.

    Longitude is normalized to [0, 360); latitude is in [-90, 90].
    """

    latitude_deg: float
    longitude_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude_deg) or not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("Latitude must be finite and within [-90, 90] degrees.")
        if not math.isfinite(self.longitude_deg) or not 0.0 <= self.longitude_deg < 360.0:
            raise ValueError("East-positive longitude must be finite and within [0, 360) degrees.")

    def surface_point_km(self, radii_km: tuple[float, float, float]) -> tuple[float, float, float]:
        """Return the ellipsoid surface point whose outward normal is this geodetic coordinate.

        The ellipsoid equation is x²/a² + y²/b² + z²/c² = 1. Geodetic
        latitude/longitude specify the outward normal direction, not a
        planetocentric radius vector.
        """
        radii = _vector3(radii_km, "Ellipsoid radii")
        if np.any(radii <= 0.0):
            raise ValueError("Ellipsoid radii must be positive.")
        lat = math.radians(self.latitude_deg)
        lon = math.radians(self.longitude_deg)
        normal = np.array((math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)))
        squared_radii = radii * radii
        point = squared_radii * normal / math.sqrt(float(np.dot(squared_radii, normal * normal)))
        return tuple(float(component) for component in point)


@dataclass(frozen=True)
class LocalHorizonFrame:
    """Orthonormal local axes represented in Moon-fixed Cartesian components.

    East points toward increasing east longitude; North points toward
    increasing geodetic latitude; Up is the outward ellipsoid normal.
    """

    east: tuple[float, float, float]
    north: tuple[float, float, float]
    up: tuple[float, float, float]

    @classmethod
    def at_site(cls, site: LunarSite) -> "LocalHorizonFrame":
        lat = math.radians(site.latitude_deg)
        lon = math.radians(site.longitude_deg)
        east = (-math.sin(lon), math.cos(lon), 0.0)
        north = (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat))
        up = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
        return cls(east=east, north=north, up=up)

    def components(self, vector_moon_fixed: tuple[float, float, float]) -> tuple[float, float, float]:
        """Resolve a Moon-fixed vector into (East, North, Up) components."""
        vector = _vector3(vector_moon_fixed, "Moon-fixed vector")
        return tuple(float(np.dot(axis, vector)) for axis in (self.east, self.north, self.up))

    def direction_angles(self, vector_moon_fixed: tuple[float, float, float]) -> tuple[float, float]:
        """Return (azimuth, elevation) in degrees.

        Azimuth is clockwise from North: 0° North, 90° East, 180° South,
        270° West. Elevation is +90° at Up, 0° on the ideal horizon, and
        negative below it. The zero vector has no defined direction.
        """
        east, north, up = self.components(vector_moon_fixed)
        if east == 0.0 and north == 0.0 and up == 0.0:
            raise ValueError("Azimuth and elevation are undefined for the zero vector.")
        azimuth = math.degrees(math.atan2(east, north)) % 360.0
        elevation = math.degrees(math.atan2(up, math.hypot(east, north)))
        return azimuth, elevation
