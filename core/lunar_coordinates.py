"""Explicit lunar site records and planetocentric Cartesian conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin

from .terrain_provider import MOON_ME_DE421


@dataclass(frozen=True)
class LunarSiteRecord:
    """Published coordinates with their interpretation and provenance attached."""

    name: str
    latitude_deg: float
    longitude_deg: float
    latitude_type: str
    longitude_convention: str
    source_frame: str
    source_ephemeris: str
    coordinate_source: str


SITE06_NOBILE_RIM_1 = LunarSiteRecord(
    name="Site06 — Nobile Rim 1",
    latitude_deg=-85.4381,
    longitude_deg=+37.36667,
    latitude_type="planetocentric",
    longitude_convention="east_positive",
    source_frame=MOON_ME_DE421,
    source_ephemeris="DE421",
    coordinate_source="NASA LOLA South Pole site documentation",
)


def planetocentric_latlon_to_moon_me_de421(
    latitude_deg: float,
    longitude_deg: float,
    radius_km: float,
) -> tuple[float, float, float]:
    """Convert planetocentric latitude/east-positive longitude to ME/DE421 XYZ.

    The spherical angles are in degrees and the Moon-centered position radius
    is in kilometers. The returned components are kilometers in MOON_ME_DE421.
    """
    latitude_deg = float(latitude_deg)
    longitude_deg = float(longitude_deg)
    radius_km = float(radius_km)
    if not all(isfinite(value) for value in (latitude_deg, longitude_deg, radius_km)):
        raise ValueError("Latitude, longitude, and radius must be finite.")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("Planetocentric latitude must be between -90 and +90 degrees.")
    if radius_km <= 0.0:
        raise ValueError("Radius must be positive and expressed in kilometers.")

    latitude_rad = radians(latitude_deg)
    longitude_rad = radians(longitude_deg)
    cos_latitude = cos(latitude_rad)
    return (
        radius_km * cos_latitude * cos(longitude_rad),
        radius_km * cos_latitude * sin(longitude_rad),
        radius_km * sin(latitude_rad),
    )
