"""Lunar occultation and optional site-horizon visibility checks.

All positions are Moon-centred vectors in km and must share the same tagged
frame. Terrain profiles are site-specific angular horizons; limited-range
profiles never establish complete terrain visibility beyond their coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from core.terrain_visibility import interpolate_horizon_elevation

from .lumen_optics import PositionFrame, TaggedLunarPosition, _norm, _subtract


HorizonStatus = Literal["complete", "limited-range", "insufficient-coverage"]


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


@dataclass(frozen=True)
class SiteHorizonProfile:
    """A 360-bin site horizon and its local frame/coverage metadata.

    The profile must be transformed to the same frame as its position and ray
    endpoints before use. ``range_km`` is the terrain sampling range, not a
    claim that the entire lunar horizon is known.
    """

    profile_deg: tuple[float | None, ...]
    site_position_km: tuple[float, float, float]
    frame: PositionFrame
    east_axis: tuple[float, float, float]
    north_axis: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    range_km: float
    status: HorizonStatus
    site_name: str

    def __post_init__(self) -> None:
        profile = tuple(self.profile_deg)
        if len(profile) != 360:
            raise ValueError("A terrain horizon profile must contain 360 azimuth bins.")
        normalized_profile: list[float | None] = []
        for index, elevation in enumerate(profile):
            if elevation is None:
                normalized_profile.append(None)
            else:
                value = _finite(elevation, f"Horizon bin {index}")
                if not -90.0 <= value <= 90.0:
                    raise ValueError("Terrain horizon elevations must be in [-90, 90] degrees.")
                normalized_profile.append(value)
        object.__setattr__(self, "profile_deg", tuple(normalized_profile))
        if self.frame not in ("J2000", "MOON_PA_DE440", "MOON_ME_DE421"):
            raise ValueError("frame must be J2000, MOON_PA_DE440, or MOON_ME_DE421.")
        if self.status not in ("complete", "limited-range", "insufficient-coverage"):
            raise ValueError("status must be complete, limited-range, or insufficient-coverage.")
        range_km = _finite(self.range_km, "Terrain range")
        if range_km <= 0.0:
            raise ValueError("Terrain range_km must be positive.")
        object.__setattr__(self, "range_km", range_km)
        object.__setattr__(self, "site_position_km", _vector(self.site_position_km, "Site position"))
        axes = tuple(_vector(axis, "Local axis") for axis in (self.east_axis, self.north_axis, self.up_axis))
        for axis in axes:
            if abs(_norm(axis) - 1.0) > 1e-6:
                raise ValueError("Local horizon axes must be unit length.")
        if any(abs(_dot(axes[i], axes[j])) > 1e-6 for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Local horizon axes must be orthogonal.")
        object.__setattr__(self, "east_axis", axes[0])
        object.__setattr__(self, "north_axis", axes[1])
        object.__setattr__(self, "up_axis", axes[2])
        if not self.site_name.strip():
            raise ValueError("site_name must not be empty.")


@dataclass(frozen=True)
class RayVisibility:
    """Visibility result for one segment; terrain status may remain unknown."""

    ray_name: str
    geometrically_clear: bool
    lunar_occulted: bool
    terrain_clearance_deg: float | None
    terrain_visible: bool | None
    terrain_status: str
    terrain_range_km: float | None
    terrain_site: str | None

    @property
    def visible(self) -> bool | None:
        """Return false for a known block, true only with sufficient evidence."""
        if not self.geometrically_clear:
            return False
        if self.terrain_status == "not-applicable":
            return True
        if self.terrain_visible is False:
            return False
        if self.terrain_status.startswith("unknown") or self.terrain_status == "insufficient-coverage":
            return None
        if self.terrain_visible is True:
            return True
        return None


@dataclass(frozen=True)
class LumenVisibilityResult:
    """Per-ray geometry and terrain results, retaining frame and Moon radius."""

    frame: str
    lunar_reference_radius_km: float
    sun_to_mirror: RayVisibility
    mirror_to_station: RayVisibility
    mirror_to_rover: RayVisibility


def segment_occludes_lunar_sphere(
    start_km: tuple[float, float, float],
    end_km: tuple[float, float, float],
    lunar_radius_km: float = 1_737.4,
    *,
    tolerance_km: float = 1e-7,
) -> bool:
    """Return whether an open line segment intersects the reference sphere.

    Endpoint neighborhoods are excluded so a station/rover on the reference
    surface does not self-occlude. A tangent segment is treated as clear
    within the numerical tolerance.
    """
    start = _vector(start_km, "Ray start")
    end = _vector(end_km, "Ray end")
    radius = _finite(lunar_radius_km, "Lunar reference radius")
    tolerance = _finite(tolerance_km, "Intersection tolerance")
    if radius <= 0 or tolerance < 0:
        raise ValueError("Lunar radius must be positive and tolerance nonnegative.")
    delta = _subtract(end, start)
    length_sq = _dot(delta, delta)
    if length_sq <= 1e-24:
        raise ValueError("Ray endpoints must be distinct.")
    t = -_dot(start, delta) / length_sq
    t = min(1.0 - 1e-12, max(1e-12, t))
    closest = (start[0] + t * delta[0], start[1] + t * delta[1], start[2] + t * delta[2])
    return _norm(closest) < radius - tolerance


def evaluate_lumen_visibility(
    sun: TaggedLunarPosition,
    mirror: TaggedLunarPosition,
    station: TaggedLunarPosition,
    rover: TaggedLunarPosition,
    *,
    lunar_reference_radius_km: float = 1_737.4,
    station_horizon: SiteHorizonProfile | None = None,
    rover_horizon: SiteHorizonProfile | None = None,
) -> LumenVisibilityResult:
    """Check lunar occultation and optional terrain horizons for LUMEN rays.

    Sun, mirror, station and rover vectors must share one lunar-centred frame.
    A missing profile yields ``unknown-terrain``. A limited-range profile
    reports angular clearance only within its declared range and leaves full
    ray visibility unknown when the target lies beyond that range.
    """
    positions = (sun, mirror, station, rover)
    if any(not isinstance(item, TaggedLunarPosition) for item in positions):
        raise TypeError("Sun, mirror, station and rover must be TaggedLunarPosition values.")
    frames = {item.frame for item in positions}
    if len(frames) != 1:
        raise ValueError("Sun, mirror, station and rover must use the same coordinate frame.")
    frame = mirror.frame
    radius = _finite(lunar_reference_radius_km, "Lunar reference radius")
    if radius <= 0:
        raise ValueError("lunar_reference_radius_km must be positive.")
    for profile in (station_horizon, rover_horizon):
        if profile is not None and profile.frame != frame:
            raise ValueError("Terrain profile and ray positions must use the same coordinate frame.")

    sun_blocked = segment_occludes_lunar_sphere(mirror.position_km, sun.position_km, radius)
    station_blocked = segment_occludes_lunar_sphere(mirror.position_km, station.position_km, radius)
    rover_blocked = segment_occludes_lunar_sphere(mirror.position_km, rover.position_km, radius)
    return LumenVisibilityResult(
        frame=frame,
        lunar_reference_radius_km=radius,
        sun_to_mirror=RayVisibility(
            "sun-to-mirror", not sun_blocked, sun_blocked, None, None,
            "not-applicable", None, None,
        ),
        mirror_to_station=_terrain_ray_result(
            "mirror-to-station", station, mirror, station_blocked, station_horizon
        ),
        mirror_to_rover=_terrain_ray_result(
            "mirror-to-rover", rover, mirror, rover_blocked, rover_horizon
        ),
    )


def _terrain_ray_result(
    name: str,
    site_position: TaggedLunarPosition,
    target_position: TaggedLunarPosition,
    occulted: bool,
    profile: SiteHorizonProfile | None,
) -> RayVisibility:
    if profile is None:
        return RayVisibility(name, not occulted, occulted, None, None, "unknown-terrain", None, None)
    _assert_profile_site_matches(profile, site_position)
    ray = _subtract(target_position.position_km, site_position.position_km)
    east = _dot(ray, profile.east_axis)
    north = _dot(ray, profile.north_axis)
    up = _dot(ray, profile.up_axis)
    horizontal = math.hypot(east, north)
    if _norm(ray) <= 1e-12:
        raise ValueError(f"{name} has coincident site and target positions.")
    azimuth = math.degrees(math.atan2(east, north)) % 360.0
    elevation = math.degrees(math.atan2(up, horizontal))
    horizon = interpolate_horizon_elevation(azimuth, profile.profile_deg)
    clearance = elevation - horizon
    range_km = _norm(ray)
    terrain_visible = clearance > 0.0
    if profile.status == "insufficient-coverage":
        status = "insufficient-coverage"
        terrain_visible = None
    elif profile.status == "limited-range":
        status = "limited-range-clearance-only"
        if range_km > profile.range_km:
            status = "unknown-beyond-profile-range"
            terrain_visible = None
    else:
        status = "complete-profile-clearance"
    return RayVisibility(
        name,
        not occulted,
        occulted,
        clearance,
        terrain_visible,
        status,
        profile.range_km,
        profile.site_name,
    )


def _assert_profile_site_matches(profile: SiteHorizonProfile, position: TaggedLunarPosition) -> None:
    separation = _norm(_subtract(profile.site_position_km, position.position_km))
    if separation > 1e-6:
        raise ValueError(f"Terrain profile site {profile.site_name!r} does not match the ray's surface site.")


def _vector(values: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} must have three components.")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} components must be finite.")
    return result


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
