"""Compare Site06 celestial directions with the saved limited-range horizon.

This is only a terrain-relative angular clearance comparison. It does not
claim complete lunar visibility because its saved horizon profile extends
only 1 km from Site06.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from .celestial_direction import CelestialDirection, celestial_direction
from .terrain_provider import MOON_ME_DE421


SITE06_HORIZON_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "site06_1km_5m_20m_horizon_crosscheck.json"
)
SITE06_HORIZON_RANGE_KM = 1.0
LIMITED_RANGE_STATUS = "limited-range"


@dataclass(frozen=True)
class TerrainRelativeVisibility:
    """Celestial elevation compared with one azimuth bin of terrain horizon."""

    target: str
    azimuth_deg: float
    celestial_elevation_deg: float
    terrain_horizon_elevation_deg: float
    clearance_deg: float
    visible: bool
    horizon_range_km: float
    horizon_status: str


def load_site06_horizon_profile(
    profile_path: str | Path = SITE06_HORIZON_PROFILE_PATH,
) -> tuple[float | None, ...]:
    """Load the saved 5 m, 1 km profile without accessing any DEM source."""
    path = Path(profile_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise FileNotFoundError(f"Saved Site06 terrain profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved Site06 terrain profile is not valid JSON: {path}") from exc
    if report.get("source_frame") != MOON_ME_DE421:
        raise ValueError(f"Saved terrain profile must be in {MOON_ME_DE421}.")
    if float(report.get("requested_radius_m", -1.0)) != 1000.0:
        raise ValueError("Saved Site06 profile must be the validated 1 km limited-range profile.")
    values = report.get("profile_bins_deg")
    if not isinstance(values, list) or len(values) != 360:
        raise ValueError("Saved Site06 terrain profile must contain exactly 360 one-degree bins.")
    parsed: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            parsed.append(None)
            continue
        angle = float(value)
        if not math.isfinite(angle):
            raise ValueError(f"Terrain horizon value in azimuth bin {index} is not finite.")
        parsed.append(angle)
    return tuple(parsed)


def azimuth_to_horizon_bin(azimuth_deg: float) -> int:
    """Return the lower 1-degree sample index, wrapping 360° to bin zero."""
    azimuth = float(azimuth_deg)
    if not math.isfinite(azimuth):
        raise ValueError("Azimuth must be finite.")
    return math.floor(azimuth % 360.0)


def interpolate_horizon_elevation(
    azimuth_deg: float,
    horizon_profile_deg: Sequence[float | None],
) -> float:
    """Circularly linearly interpolate neighboring 1-degree profile samples."""
    if len(horizon_profile_deg) != 360:
        raise ValueError("Terrain horizon profile must contain 360 one-degree samples.")
    azimuth = float(azimuth_deg)
    if not math.isfinite(azimuth):
        raise ValueError("Azimuth must be finite.")
    wrapped = azimuth % 360.0
    lower_bin = math.floor(wrapped)
    upper_bin = (lower_bin + 1) % 360
    fraction = wrapped - lower_bin
    lower = horizon_profile_deg[lower_bin]
    if lower is None or not math.isfinite(float(lower)):
        raise ValueError(f"Terrain horizon is unavailable in azimuth bin {lower_bin}°.")
    lower_value = float(lower)
    if fraction == 0.0:
        return lower_value
    upper = horizon_profile_deg[upper_bin]
    if upper is None or not math.isfinite(float(upper)):
        raise ValueError(f"Terrain horizon is unavailable in azimuth bin {upper_bin}°.")
    return lower_value + fraction * (float(upper) - lower_value)


def compare_with_terrain_horizon(
    direction: CelestialDirection,
    horizon_profile_deg: Sequence[float | None],
) -> TerrainRelativeVisibility:
    """Compute terrain-relative clearance by circular profile interpolation."""
    if direction.frame != MOON_ME_DE421:
        raise ValueError(f"Celestial direction must be in {MOON_ME_DE421}.")
    horizon_elevation = interpolate_horizon_elevation(direction.azimuth_deg, horizon_profile_deg)
    clearance = float(direction.elevation_deg) - horizon_elevation
    return TerrainRelativeVisibility(
        target=direction.target,
        azimuth_deg=float(direction.azimuth_deg),
        celestial_elevation_deg=float(direction.elevation_deg),
        terrain_horizon_elevation_deg=horizon_elevation,
        clearance_deg=clearance,
        visible=clearance > 0.0,
        horizon_range_km=SITE06_HORIZON_RANGE_KM,
        horizon_status=LIMITED_RANGE_STATUS,
    )


def site06_terrain_visibility(utc_timestamp: str, target: str) -> TerrainRelativeVisibility:
    """Evaluate Sun/Earth angular clearance against the saved 1 km profile."""
    profile = load_site06_horizon_profile()
    direction = celestial_direction(utc_timestamp, target)
    return compare_with_terrain_horizon(direction, profile)
