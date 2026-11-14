"""Evidence-based LUMEN charging-station site feasibility assessments.

This module keeps solar illumination, reflector access, terrain support, slope
evidence and rover access as separate criteria. PASS means only that the
explicitly evaluated criterion met its supplied bounds; it is not a site
selection or mission-readiness finding. Missing, incompatible, limited-range,
or NoData evidence remains UNKNOWN unless it proves a known obstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from typing import Literal, Protocol

import numpy as np

from ..lunar_coordinates import planetocentric_latlon_to_moon_me_de421
from ..lunar_frame_bridge import LunarFrameBridge
from ..lunar_horizon import LunarSite
from ..spice_engine import SpiceGeometryEngine
from ..terrain_provider import (
    MOON_ME_DE421,
    MOON_REFERENCE_RADIUS_M,
    TaggedMoonPosition,
    TerrainProvider,
    TerrainWindow,
)
from ..terrain_region import regional_horizon_status
from ..terrain_visibility import interpolate_horizon_elevation
from .lumen_optics import MirrorOpticsAssumptions, TaggedLunarPosition, calculate_mirror_optics
from .lumen_orbit import CircularLunarOrbit, ORBIT_MODEL_LABEL
from .lumen_visibility import SiteHorizonProfile, segment_occludes_lunar_sphere


UTC = timezone.utc
VALID_FRAMES = {"MOON_ME_DE421", "MOON_PA_DE440"}


class CriterionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class _TimeGeometry(Protocol):
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, traceback): ...
    def utc_to_et(self, utc_timestamp: str) -> float: ...
    def sun_position_km(self, et: float) -> tuple[float, float, float]: ...
    def j2000_to_moon_fixed(self, vector, et: float) -> tuple[float, float, float]: ...


class _FrameConversion(Protocol):
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, traceback): ...
    def vector_dem_to_geometry(self, vector, et: float) -> tuple[float, float, float]: ...


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _positive(value: float, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _utc(value: str, label: str = "timestamp") -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp with UTC offset or Z.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset or Z.")
    return parsed.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CandidateSite:
    """Immutable candidate coordinate and evidence record.

    Latitude is planetocentric degrees, longitude east-positive degrees in
    [0, 360), elevation meters above the specified product datum when known.
    Terrain radius/resolution are meters and remain ``None`` when unknown.
    """

    identifier: str
    name: str
    latitude_deg: float
    longitude_east_deg: float
    coordinate_frame: str
    coordinate_provenance: str
    elevation_m: float | None
    elevation_source: str | None
    terrain_coverage_radius_m: float | None
    terrain_resolution_m: float | None
    explicit_assumptions: tuple[str, ...] = ()
    explicit_unknowns: tuple[str, ...] = ()
    source_version_consistent: bool | None = None
    experimental: bool = False

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.name.strip() or not self.coordinate_provenance.strip():
            raise ValueError("Candidate identifier, name, and coordinate provenance are required.")
        latitude = _finite(self.latitude_deg, "Latitude")
        longitude = _finite(self.longitude_east_deg, "East-positive longitude")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("Latitude must be in [-90, 90] degrees.")
        if not 0.0 <= longitude < 360.0:
            raise ValueError("East-positive longitude must be in [0, 360) degrees.")
        if self.coordinate_frame not in VALID_FRAMES:
            raise ValueError("Coordinate frame must be MOON_ME_DE421 or MOON_PA_DE440.")
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "longitude_east_deg", longitude)
        if self.elevation_m is None:
            if self.elevation_source is not None:
                raise ValueError("An elevation source cannot be supplied when elevation_m is unknown.")
        else:
            object.__setattr__(self, "elevation_m", _finite(self.elevation_m, "Elevation"))
            if not self.elevation_source or not self.elevation_source.strip():
                raise ValueError("A known elevation requires its source.")
        for name in ("terrain_coverage_radius_m", "terrain_resolution_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))
        if self.source_version_consistent not in (True, False, None):
            raise ValueError("source_version_consistent must be True, False, or None.")
        object.__setattr__(self, "explicit_assumptions", tuple(self.explicit_assumptions))
        object.__setattr__(self, "explicit_unknowns", tuple(self.explicit_unknowns))


SITE06_NOBILE_RIM_1 = CandidateSite(
    identifier="Site06",
    name="Nobile Rim 1",
    latitude_deg=-85.4381,
    longitude_east_deg=37.36667,
    coordinate_frame=MOON_ME_DE421,
    coordinate_provenance="NASA LOLA South Pole site documentation; published planetocentric latitude and east-positive longitude.",
    elevation_m=None,
    elevation_source=None,
    terrain_coverage_radius_m=None,
    terrain_resolution_m=None,
    explicit_assumptions=("Site position projection may use the LOLA 1,737,400 m reference sphere; this does not supply site elevation." ,),
    explicit_unknowns=("Station-scale elevation, slope, illumination, terrain support and rover access require assessment." ,),
)


SITE07_PEAK_NEAR_SHACKLETON_EXPERIMENTAL = CandidateSite(
    identifier="Site07",
    name="Peak Near Shackleton",
    latitude_deg=-88.811,
    longitude_east_deg=123.6901,
    coordinate_frame=MOON_ME_DE421,
    coordinate_provenance="Proposed NASA Lunar Surface Data Book Table A2-2 coordinates, interpreted using LOLA planetocentric/east-positive ME/DE421 conventions; source record is review-only.",
    elevation_m=None,
    elevation_source=None,
    terrain_coverage_radius_m=None,
    terrain_resolution_m=5.0,
    explicit_assumptions=("Nominal 5 m resolution is documented for the dedicated Site07 DEM; exact raster metadata is unverified." ,),
    explicit_unknowns=(
        "Site07 DEM source-version consistency is unresolved and blocks production terrain generation.",
        "The shared 20 m DEM suitability at Site07 is unverified.",
        "Published coordinates are proposed source-record values, not a production site configuration.",
    ),
    source_version_consistent=None,
    experimental=True,
)


@dataclass(frozen=True)
class CriterionAssessment:
    criterion: str
    status: CriterionStatus
    evidence: tuple[str, ...] = ()
    missing_data: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoverAccessPlan:
    """Explicit operating-area and polyline inputs in DEM projected meters."""

    operating_area_radius_m: float
    route_points_xy_m: tuple[tuple[float, float], ...]
    corridor_width_m: float
    maximum_slope_deg: float
    route_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operating_area_radius_m", _positive(self.operating_area_radius_m, "Operating area radius"))
        object.__setattr__(self, "corridor_width_m", _positive(self.corridor_width_m, "Route corridor width"))
        angle = _finite(self.maximum_slope_deg, "Maximum route slope")
        if not 0.0 <= angle <= 90.0:
            raise ValueError("Maximum route slope must be in [0, 90] degrees.")
        object.__setattr__(self, "maximum_slope_deg", angle)
        points = tuple(tuple(_finite(v, "Route coordinate") for v in point) for point in self.route_points_xy_m)
        if len(points) < 2 or any(len(point) != 2 for point in points):
            raise ValueError("An explicit route must have at least two projected (x_m, y_m) points.")
        if not self.route_name.strip():
            raise ValueError("route_name is required; routes are never generated implicitly.")
        object.__setattr__(self, "route_points_xy_m", points)


@dataclass(frozen=True)
class TerrainAssessment:
    """Bounded DEM evidence; NoData and incomplete coverage remain explicit."""

    coverage: CriterionAssessment
    slope: CriterionAssessment
    rover_access: CriterionAssessment
    source_frame: str | None
    requested_radius_m: float
    terrain_resolution_m: tuple[float, float] | None
    coverage_radius_m: float | None
    coverage_fraction: float | None
    valid_sample_count: int
    selected_sample_count: int
    nodata_sample_count: int
    elevation_min_m: float | None
    elevation_max_m: float | None
    elevation_range_m: float | None
    roughness_rms_m: float | None
    slope_max_deg: float | None
    slope_median_deg: float | None
    slope_sample_count: int
    route_covered_fraction: float | None
    route_max_slope_deg: float | None


@dataclass(frozen=True)
class IlluminationSample:
    timestamp_utc: str
    sun_elevation_deg: float | None
    terrain_horizon_deg: float | None
    solar_clearance_deg: float | None
    lunar_occulted: bool | None
    status: CriterionStatus
    terrain_status: str


@dataclass(frozen=True)
class SampledVisibilityInterval:
    start_utc: str
    end_utc: str
    sample_count: int
    boundary_semantics: str = "Sample-based timestamps; not exact continuous interval boundaries."


@dataclass(frozen=True)
class IlluminationAssessment:
    criterion: CriterionAssessment
    samples: tuple[IlluminationSample, ...]
    visibility_intervals: tuple[SampledVisibilityInterval, ...]
    cadence_s: float
    terrain_radius_km: float | None
    horizon_status: str


@dataclass(frozen=True)
class ReflectorSample:
    timestamp_utc: str
    spacecraft_position_km: tuple[float, float, float] | None
    frame: str
    sun_to_mirror_clear: bool | None
    mirror_to_site_clear: bool | None
    lunar_occulted: bool | None
    terrain_clearance_deg: float | None
    terrain_status: str
    mirror_incidence_angle_deg: float | None
    receiver_incidence_angle_deg: float | None
    reflected_optical_power_w: float | None
    status: CriterionStatus


@dataclass(frozen=True)
class ReflectorAssessment:
    criterion: CriterionAssessment
    samples: tuple[ReflectorSample, ...]
    valid_illumination_intervals: tuple[SampledVisibilityInterval, ...]
    cadence_s: float
    terrain_radius_km: float | None
    horizon_status: str
    orbit_model: str
    geometry_provenance: tuple[str, ...]


@dataclass(frozen=True)
class SiteFeasibilityReport:
    candidate: CandidateSite
    solar_illumination: CriterionAssessment
    reflector_visibility: CriterionAssessment
    terrain_coverage: CriterionAssessment
    slope_evidence: CriterionAssessment
    rover_access: CriterionAssessment
    overall_status: CriterionStatus
    mission_readiness_claim: bool = False

    def __post_init__(self) -> None:
        expected = combine_criteria((
            self.solar_illumination, self.reflector_visibility, self.terrain_coverage,
            self.slope_evidence, self.rover_access,
        ))
        if self.overall_status != expected:
            raise ValueError("overall_status must match the five individual criteria.")
        if self.mission_readiness_claim:
            raise ValueError("Site criteria do not establish mission readiness.")


@dataclass(frozen=True)
class SiteComparison:
    reports: tuple[SiteFeasibilityReport, ...]
    comparison_status: CriterionStatus
    selection_made: bool
    selection_reason: str

    def __post_init__(self) -> None:
        if self.selection_made:
            raise ValueError("This evidence comparison does not automatically select or rank a site.")
        expected = (
            CriterionStatus.UNKNOWN
            if any(_report_has_unknown_critical_criterion(report) for report in self.reports)
            else CriterionStatus.PASS
        )
        if self.comparison_status != expected:
            raise ValueError("comparison_status must preserve unknown critical criteria.")


def combine_criteria(criteria: tuple[CriterionAssessment, ...]) -> CriterionStatus:
    """Combine critical criteria conservatively: known failure, then unknown."""
    if not criteria:
        return CriterionStatus.UNKNOWN
    if any(item.status == CriterionStatus.FAIL for item in criteria):
        return CriterionStatus.FAIL
    if any(item.status == CriterionStatus.UNKNOWN for item in criteria):
        return CriterionStatus.UNKNOWN
    return CriterionStatus.PASS


def candidate_me_position(site: CandidateSite, *, elevation_m: float | None = None) -> tuple[float, float, float]:
    """Return ME/DE421 Moon-centred km; radius datum is explicit in provenance.

    An unknown site elevation remains unknown in the record. If omitted here,
    the DEM reference sphere is used only for projection/geometry, and callers
    must retain that assumption in their report.
    """
    if site.coordinate_frame != MOON_ME_DE421:
        raise ValueError("Existing candidate coordinate conversion requires MOON_ME_DE421.")
    elevation = site.elevation_m if elevation_m is None else _finite(elevation_m, "Elevation override")
    radial_km = (MOON_REFERENCE_RADIUS_M + (elevation or 0.0)) / 1000.0
    return planetocentric_latlon_to_moon_me_de421(
        site.latitude_deg, site.longitude_east_deg, radial_km
    )


def assess_terrain(
    site: CandidateSite,
    provider: TerrainProvider,
    *,
    evaluation_radius_m: float,
    maximum_station_slope_deg: float | None = None,
    rover_access_plan: RoverAccessPlan | None = None,
    maximum_window_side_px: int = 2048,
) -> TerrainAssessment:
    """Assess bounded DEM support, elevation variation, slope and route evidence.

    The provider must already be open. Regions outside its raster or containing
    NoData remain uncovered; no interpolation fills gaps. Slope uses only
    center cells with four valid cardinal neighbors.
    """
    radius = _positive(evaluation_radius_m, "Evaluation radius")
    if not isinstance(site, CandidateSite):
        raise TypeError("site must be CandidateSite.")
    metadata = provider.metadata
    if site.coordinate_frame != metadata.source_frame:
        return _unknown_terrain(radius, "Candidate coordinates and DEM source frame are incompatible.", metadata.source_frame)
    if metadata.source_frame != MOON_ME_DE421:
        return _unknown_terrain(radius, "Existing DEM terrain reconstruction supports MOON_ME_DE421 only.", metadata.source_frame)
    if site.source_version_consistent is False:
        return _unknown_terrain(radius, "Candidate DEM source-version consistency is known to be invalid.", metadata.source_frame, fail=True)
    if site.source_version_consistent is None and site.experimental:
        return _unknown_terrain(radius, "Experimental candidate DEM source-version consistency is unresolved.", metadata.source_frame)
    a, b, _, d, e, _ = metadata.affine
    if abs(b) > 1e-12 or abs(d) > 1e-12:
        return _unknown_terrain(radius, "Rotated/sheared DEM grids are not evaluated by this axis-aligned slope/corridor assessment.", metadata.source_frame)
    px, py = metadata.pixel_size_m
    if px <= 0 or py <= 0:
        return _unknown_terrain(radius, "DEM pixel resolution is unavailable or invalid.", metadata.source_frame)
    try:
        me_xyz = candidate_me_position(site)
        tagged = TaggedMoonPosition(*me_xyz, frame=MOON_ME_DE421)
        center_x, center_y = provider.site_position_to_dem_xy(tagged)
    except (ValueError, RuntimeError, OSError) as exc:
        return _unknown_terrain(radius, f"Site coordinate could not be projected to the DEM: {exc}", metadata.source_frame)

    requested_radius = max(radius, rover_access_plan.operating_area_radius_m if rover_access_plan else 0.0)
    if rover_access_plan:
        route_radius = max(
            math.hypot(x - center_x, y - center_y)
            for x, y in rover_access_plan.route_points_xy_m
        ) + rover_access_plan.corridor_width_m / 2.0
        requested_radius = max(requested_radius, route_radius)
    left, bottom, right, top = metadata.bounds_m
    if (
        center_x - requested_radius < left or center_x + requested_radius > right
        or center_y - requested_radius < bottom or center_y + requested_radius > top
    ):
        return _unknown_terrain(
            radius,
            "The requested terrain circle or explicit route corridor extends beyond DEM bounds.",
            metadata.source_frame,
            fail=True,
            radius_available=None,
        )
    width = int(math.ceil(2.0 * requested_radius / px)) + 2
    height = int(math.ceil(2.0 * requested_radius / py)) + 2
    if width > maximum_window_side_px or height > maximum_window_side_px:
        return _unknown_terrain(radius, "Requested analysis window exceeds the configured bounded-read limit.", metadata.source_frame)
    try:
        window = provider.read_window(center_x, center_y, width, height)
    except (ValueError, RuntimeError, OSError) as exc:
        return _unknown_terrain(radius, f"DEM window could not be read: {exc}", metadata.source_frame)

    return _terrain_from_window(
        site, metadata, window, (center_x, center_y), requested_radius,
        maximum_station_slope_deg=maximum_station_slope_deg,
        rover_access_plan=rover_access_plan,
        requested_radius_m=radius,
    )


def _terrain_from_window(site, metadata, window, center, window_radius, *, maximum_station_slope_deg, rover_access_plan, requested_radius_m):
    xg, yg = np.meshgrid(window.x_centers_m, window.y_centers_m)
    in_circle = (xg - center[0]) ** 2 + (yg - center[1]) ** 2 <= window_radius**2
    left, bottom, right, top = metadata.bounds_m
    in_bounds = (xg >= left) & (xg < right) & (yg > bottom) & (yg <= top)
    nodata = np.asarray(window.nodata_mask, dtype=bool)
    elevations = np.asarray(window.elevation_m, dtype=float)
    valid = in_bounds & ~nodata & np.isfinite(elevations)
    selected = in_circle
    selected_count = int(np.count_nonzero(selected))
    valid_selected = selected & valid
    valid_count = int(np.count_nonzero(valid_selected))
    nodata_count = int(np.count_nonzero(selected & ~valid))
    fraction = valid_count / selected_count if selected_count else None
    window_covers_circle = (
        len(window.x_centers_m) > 0 and len(window.y_centers_m) > 0
        and min(window.x_centers_m) - metadata.pixel_size_m[0] / 2 <= center[0] - window_radius
        and max(window.x_centers_m) + metadata.pixel_size_m[0] / 2 >= center[0] + window_radius
        and min(window.y_centers_m) - metadata.pixel_size_m[1] / 2 <= center[1] - window_radius
        and max(window.y_centers_m) + metadata.pixel_size_m[1] / 2 >= center[1] + window_radius
    )
    if not selected_count or not window_covers_circle:
        coverage_status = CriterionStatus.UNKNOWN
        coverage_reason = "Returned raster window does not fully sample the requested region."
    elif nodata_count:
        coverage_status = CriterionStatus.UNKNOWN
        coverage_reason = f"{nodata_count} selected cells are NoData or non-finite; gaps were preserved."
    else:
        coverage_status = CriterionStatus.PASS
        coverage_reason = "All selected pixel centers in the bounded evaluation region contain valid samples."

    if valid_count:
        selected_elevations = elevations[valid_selected]
        emin, emax = float(np.min(selected_elevations)), float(np.max(selected_elevations))
        erange = emax - emin
        residuals = _local_roughness_grid(elevations, valid)
        usable_roughness = selected & np.isfinite(residuals)
        roughness = float(np.sqrt(np.mean(residuals[usable_roughness] ** 2))) if np.any(usable_roughness) else None
    else:
        emin = emax = erange = roughness = None

    slopes = _slope_grid(elevations, valid, metadata.pixel_size_m)
    selected_slopes = selected & np.isfinite(slopes)
    slope_values = slopes[selected_slopes]
    slope_count = int(slope_values.size)
    max_slope = float(np.max(slope_values)) if slope_count else None
    median_slope = float(np.median(slope_values)) if slope_count else None
    if maximum_station_slope_deg is None:
        slope_assessment = CriterionAssessment(
            "slope evidence", CriterionStatus.UNKNOWN,
            evidence=((f"Measured maximum slope is {max_slope:.3f} degrees." if max_slope is not None else "No valid slope stencil is available."),),
            missing_data=("An explicit station slope limit is required for PASS/FAIL." ,),
            reasons=("Slope measurement is evidence only; no suitability threshold was supplied." ,),
        )
    else:
        limit = _finite(maximum_station_slope_deg, "Maximum station slope")
        if not 0 <= limit <= 90:
            raise ValueError("maximum_station_slope_deg must be in [0, 90].")
        if max_slope is None:
            slope_assessment = CriterionAssessment(
                "slope evidence", CriterionStatus.UNKNOWN,
                missing_data=("No valid four-neighbor slope stencils in the analyzed region." ,),
            )
        else:
            slope_assessment = CriterionAssessment(
                "slope evidence",
                CriterionStatus.PASS if max_slope <= limit else CriterionStatus.FAIL,
                evidence=(f"Measured maximum slope {max_slope:.3f} degrees; supplied limit {limit:.3f} degrees.",),
                reasons=() if max_slope <= limit else ("Measured slope exceeds the supplied limit.",),
            )

    coverage_assessment = CriterionAssessment(
        "terrain coverage", coverage_status,
        evidence=(coverage_reason, f"Valid selected samples: {valid_count}/{selected_count}."),
        missing_data=("Terrain beyond the requested bounded radius was not evaluated.",),
        assumptions=(f"Requested bounded radius: {window_radius:g} m; DEM pixel size {metadata.pixel_size_m} m.",),
    )
    route_status, route_fraction, route_max, route_evidence, route_missing = _route_assessment(
        rover_access_plan, window, elevations, valid, slopes, metadata.pixel_size_m, center
    )
    access_assessment = CriterionAssessment(
        "rover access", route_status, evidence=route_evidence, missing_data=route_missing,
        assumptions=("Route assessment uses only the explicit projected polyline and sampled DEM; elevation does not establish traversability.",),
        reasons=("Slope or coverage evidence is insufficient for the supplied route criterion.",) if route_status == CriterionStatus.UNKNOWN else (),
    )
    if site.source_version_consistent is None and site.experimental:
        coverage_assessment = CriterionAssessment(
            "terrain coverage", CriterionStatus.UNKNOWN,
            evidence=coverage_assessment.evidence,
            missing_data=coverage_assessment.missing_data + ("Candidate source-version consistency remains unresolved.",),
            assumptions=coverage_assessment.assumptions,
            reasons=("Experimental source data cannot support a production coverage finding.",),
        )
        slope_assessment = CriterionAssessment(
            "slope evidence", CriterionStatus.UNKNOWN,
            evidence=slope_assessment.evidence,
            missing_data=slope_assessment.missing_data + ("Source-version consistency is unresolved.",),
        )
        access_assessment = CriterionAssessment(
            "rover access", CriterionStatus.UNKNOWN,
            evidence=access_assessment.evidence,
            missing_data=access_assessment.missing_data + ("Source-version consistency is unresolved.",),
        )
    return TerrainAssessment(
        coverage=coverage_assessment,
        slope=slope_assessment,
        rover_access=access_assessment,
        source_frame=metadata.source_frame,
        requested_radius_m=requested_radius_m,
        terrain_resolution_m=metadata.pixel_size_m,
        coverage_radius_m=window_radius,
        coverage_fraction=fraction,
        valid_sample_count=valid_count,
        selected_sample_count=selected_count,
        nodata_sample_count=nodata_count,
        elevation_min_m=emin,
        elevation_max_m=emax,
        elevation_range_m=erange,
        roughness_rms_m=roughness,
        slope_max_deg=max_slope,
        slope_median_deg=median_slope,
        slope_sample_count=slope_count,
        route_covered_fraction=route_fraction,
        route_max_slope_deg=route_max,
    )


def _slope_grid(elevations: np.ndarray, valid: np.ndarray, pixel_size_m: tuple[float, float]) -> np.ndarray:
    rows, cols = elevations.shape
    slopes = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return slopes
    stencil = (
        valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
        & valid[:-2, 1:-1] & valid[2:, 1:-1]
    )
    dx, dy = pixel_size_m
    dzdx = (elevations[1:-1, 2:] - elevations[1:-1, :-2]) / (2.0 * dx)
    dzdy = (elevations[2:, 1:-1] - elevations[:-2, 1:-1]) / (2.0 * dy)
    gradient = np.hypot(dzdx, dzdy)
    slopes[1:-1, 1:-1][stencil] = np.degrees(np.arctan(gradient[stencil]))
    return slopes


def _local_roughness_grid(elevations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return 3x3 detrended local residuals; incomplete stencils stay unknown."""
    rows, cols = elevations.shape
    residuals = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return residuals
    values = np.lib.stride_tricks.sliding_window_view(elevations, (3, 3))
    masks = np.lib.stride_tricks.sliding_window_view(valid, (3, 3))
    complete = np.all(masks, axis=(-2, -1))
    means = np.mean(values, axis=(-2, -1))
    center = elevations[1:-1, 1:-1]
    residuals[1:-1, 1:-1][complete] = (center - means)[complete]
    return residuals


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    return math.hypot(point[0] - (start[0] + fraction * dx), point[1] - (start[1] + fraction * dy))


def _route_assessment(plan, window, elevations, valid, slopes, pixel_size, center):
    if plan is None:
        return CriterionStatus.UNKNOWN, None, None, (), ("No explicit operating area or rover route was supplied.",)
    dx = float(np.median(np.diff(window.x_centers_m))) if len(window.x_centers_m) > 1 else pixel_size[0]
    dy = float(np.median(np.diff(window.y_centers_m))) if len(window.y_centers_m) > 1 else -pixel_size[1]
    route_samples: list[tuple[float, float]] = []
    spacing = min(pixel_size) / 2.0
    for start, end in zip(plan.route_points_xy_m, plan.route_points_xy_m[1:]):
        distance = math.dist(start, end)
        count = max(1, int(math.ceil(distance / spacing)))
        route_samples.extend(
            (start[0] + (end[0] - start[0]) * i / count, start[1] + (end[1] - start[1]) * i / count)
            for i in range(count)
        )
    route_samples.append(plan.route_points_xy_m[-1])
    if any(math.hypot(x - center[0], y - center[1]) > plan.operating_area_radius_m for x, y in route_samples):
        return CriterionStatus.FAIL, 0.0, None, (), ("Explicit route leaves the declared operating area.",)
    # Check every raster cell whose center lies inside the explicit corridor,
    # not only the route centerline. A corridor intersecting NoData is unknown.
    corridor_half = plan.corridor_width_m / 2.0
    xg, yg = np.meshgrid(window.x_centers_m, window.y_centers_m)
    corridor_mask = np.zeros(elevations.shape, dtype=bool)
    for start, end in zip(plan.route_points_xy_m, plan.route_points_xy_m[1:]):
        corridor_mask |= np.vectorize(_point_segment_distance, signature="(n),(n),(n)->()")(
            np.stack((xg, yg), axis=-1), np.asarray(start), np.asarray(end)
        ) <= corridor_half
    corridor_cells = int(np.count_nonzero(corridor_mask))
    corridor_valid = int(np.count_nonzero(corridor_mask & valid))
    corridor_fraction = corridor_valid / corridor_cells if corridor_cells else 0.0
    covered = 0
    route_slopes: list[float] = []
    for x, y in route_samples:
        col = int(np.argmin(np.abs(window.x_centers_m - x)))
        row = int(np.argmin(np.abs(window.y_centers_m - y)))
        nearest = math.hypot(window.x_centers_m[col] - x, window.y_centers_m[row] - y)
        if nearest > math.hypot(pixel_size[0], pixel_size[1]) / 2.0 or not valid[row, col]:
            continue
        # At least the centerline sample must be valid; inspect every raster
        # center in the requested corridor cross-section and retain NoData.
        covered += 1
        if math.isfinite(slopes[row, col]):
            route_slopes.append(float(slopes[row, col]))
    fraction = covered / len(route_samples) if route_samples else 0.0
    if corridor_fraction < 1.0 or fraction < 1.0 or len(route_slopes) < max(1, int(0.5 * len(route_samples))):
        return (
            CriterionStatus.UNKNOWN,
            fraction,
            max(route_slopes) if route_slopes else None,
            (f"Explicit route centerline coverage: {fraction:.1%}; corridor raster coverage: {corridor_fraction:.1%}.",),
            ("Route corridor contains uncovered/NoData cells or too few valid slope stencils.",),
        )
    maximum = max(route_slopes)
    if maximum > plan.maximum_slope_deg:
        status = CriterionStatus.FAIL
        reasons = (f"Measured route slope {maximum:.3f} degrees exceeds supplied limit {plan.maximum_slope_deg:.3f} degrees.",)
    else:
        status = CriterionStatus.PASS
        reasons = ()
    return (
        status,
        fraction,
        maximum,
        (f"Explicit route centerline sampled at <= {spacing:g} m; corridor raster coverage {corridor_fraction:.1%}; maximum measured slope {maximum:.3f} degrees.",),
        reasons,
    )


def _unknown_terrain(radius, reason, frame=None, *, fail=False, radius_available=None):
    status = CriterionStatus.FAIL if fail else CriterionStatus.UNKNOWN
    common = CriterionAssessment(
        "terrain coverage", status,
        missing_data=(reason,), reasons=(reason,),
    )
    return TerrainAssessment(
        coverage=common,
        slope=CriterionAssessment("slope evidence", CriterionStatus.UNKNOWN, missing_data=(reason,)),
        rover_access=CriterionAssessment("rover access", CriterionStatus.UNKNOWN, missing_data=("Terrain coverage unavailable.",)),
        source_frame=frame,
        requested_radius_m=radius,
        terrain_resolution_m=None,
        coverage_radius_m=radius_available,
        coverage_fraction=None,
        valid_sample_count=0,
        selected_sample_count=0,
        nodata_sample_count=0,
        elevation_min_m=None,
        elevation_max_m=None,
        elevation_range_m=None,
        roughness_rms_m=None,
        slope_max_deg=None,
        slope_median_deg=None,
        slope_sample_count=0,
        route_covered_fraction=None,
        route_max_slope_deg=None,
    )


def assess_solar_illumination(
    site: CandidateSite,
    start_utc: str,
    duration_s: float,
    cadence_s: float,
    horizon_profile: SiteHorizonProfile | None,
    *,
    lunar_reference_radius_km: float = MOON_REFERENCE_RADIUS_M / 1000.0,
    geometry_engine_factory=SpiceGeometryEngine,
    frame_bridge_factory=LunarFrameBridge,
) -> IlluminationAssessment:
    """Sample direct solar geometry; missing/incompatible horizons are UNKNOWN."""
    if not isinstance(site, CandidateSite):
        raise TypeError("site must be CandidateSite.")
    start = _utc(start_utc, "start_utc")
    duration = _positive(duration_s, "duration_s")
    cadence = _positive(cadence_s, "cadence_s")
    radius = _positive(lunar_reference_radius_km, "lunar_reference_radius_km")
    if horizon_profile is not None and horizon_profile.frame not in VALID_FRAMES:
        raise ValueError("Horizon profile has an unsupported lunar frame.")
    times = _sample_times(start, duration, cadence)
    samples: list[IlluminationSample] = []
    with geometry_engine_factory() as geometry, frame_bridge_factory() as bridge:
        for timestamp in times:
            text = _utc_text(timestamp)
            et = float(geometry.utc_to_et(text[:-1]))
            sun_j2000 = geometry.sun_position_km(et)
            sun_pa = geometry.j2000_to_moon_fixed(sun_j2000, et)
            site_me = candidate_me_position(site)
            site_pa = _to_pa(site_me, MOON_ME_DE421, et, bridge)
            profile_pa = _profile_to_pa(horizon_profile, et, bridge)
            if profile_pa is not None and _distance(profile_pa.site_position_km, site_pa) > 1e-5:
                samples.append(IlluminationSample(text, None, None, None, None, CriterionStatus.UNKNOWN, "profile-site-mismatch"))
                continue
            sun_delta = _subtract(sun_pa, site_pa)
            sun_range = _norm(sun_delta)
            local = _local_angles(sun_delta, profile_pa)
            elevation = local[1] if local else None
            occulted = segment_occludes_lunar_sphere(site_pa, sun_pa, radius)
            if profile_pa is None:
                samples.append(IlluminationSample(text, elevation, None, None, occulted, CriterionStatus.UNKNOWN, "missing-horizon-profile"))
                continue
            try:
                horizon = interpolate_horizon_elevation(local[0], profile_pa.profile_deg)
            except (ValueError, TypeError):
                samples.append(IlluminationSample(text, elevation, None, None, occulted, CriterionStatus.UNKNOWN, "missing-horizon-bin"))
                continue
            clearance = elevation - horizon
            if occulted or clearance <= 0.0:
                state = CriterionStatus.FAIL
                terrain_status = "known-blocked"
            elif profile_pa.status == "complete":
                state = CriterionStatus.PASS
                terrain_status = "complete-profile"
            else:
                state = CriterionStatus.UNKNOWN
                terrain_status = "limited-range" if profile_pa.status == "limited-range" else "insufficient-coverage"
            samples.append(IlluminationSample(text, elevation, horizon, clearance, occulted, state, terrain_status))
    intervals = _sampled_intervals(samples, cadence)
    statuses = [item.status for item in samples]
    if CriterionStatus.PASS in statuses:
        criterion_status = CriterionStatus.PASS
        reason = "One or more sampled epochs have terrain-supported solar clearance."
    elif statuses and all(item == CriterionStatus.FAIL for item in statuses):
        criterion_status = CriterionStatus.FAIL
        reason = "All sampled epochs are known geometrically or terrain blocked."
    else:
        criterion_status = CriterionStatus.UNKNOWN
        reason = "No complete compatible horizon evidence establishes direct solar visibility."
    return IlluminationAssessment(
        criterion=CriterionAssessment(
            "solar illumination", criterion_status,
            evidence=(reason, f"Evaluated {len(samples)} sample times at {cadence:g} s cadence."),
            missing_data=() if criterion_status != CriterionStatus.UNKNOWN else ("Complete compatible horizon profile and bins are required for a positive result.",),
            assumptions=("SPICE Sun vectors are geometric; sampled visibility is not a continuous interval guarantee.",),
        ),
        samples=tuple(samples),
        visibility_intervals=intervals,
        cadence_s=cadence,
        terrain_radius_km=(horizon_profile.range_km if horizon_profile else None),
        horizon_status=(horizon_profile.status if horizon_profile else "unknown"),
    )


def assess_reflector_visibility(
    site: CandidateSite,
    orbit: CircularLunarOrbit,
    start_utc: str,
    duration_s: float,
    cadence_s: float,
    horizon_profile: SiteHorizonProfile | None,
    optics_assumptions: MirrorOpticsAssumptions,
    *,
    receiver_normal_local_enu: tuple[float, float, float] | None,
    lunar_reference_radius_km: float = MOON_REFERENCE_RADIUS_M / 1000.0,
    geometry_engine_factory=SpiceGeometryEngine,
    frame_bridge_factory=LunarFrameBridge,
) -> ReflectorAssessment:
    """Sample assumed-orbit reflector-to-site geometry and local terrain evidence."""
    if not isinstance(site, CandidateSite) or not isinstance(orbit, CircularLunarOrbit):
        raise TypeError("site and orbit must be CandidateSite and CircularLunarOrbit.")
    if not isinstance(optics_assumptions, MirrorOpticsAssumptions):
        raise TypeError("optics_assumptions must be MirrorOpticsAssumptions.")
    if receiver_normal_local_enu is not None:
        normal = tuple(_finite(value, "Receiver normal component") for value in receiver_normal_local_enu)
        if len(normal) != 3 or abs(_norm(normal) - 1.0) > 1e-6:
            raise ValueError("receiver_normal_local_enu must be a unit (east, north, up) vector.")
        receiver_normal_local_enu = normal
    start = _utc(start_utc, "start_utc")
    duration, cadence = _positive(duration_s, "duration_s"), _positive(cadence_s, "cadence_s")
    radius = _positive(lunar_reference_radius_km, "lunar_reference_radius_km")
    times = _sample_times(start, duration, cadence)
    samples: list[ReflectorSample] = []
    with geometry_engine_factory() as geometry, frame_bridge_factory() as bridge:
        epoch_et = float(geometry.utc_to_et(orbit.epoch_utc))
        for timestamp in times:
            text = _utc_text(timestamp)
            et = float(geometry.utc_to_et(text[:-1]))
            sun_j2000 = geometry.sun_position_km(et)
            sun_pa = geometry.j2000_to_moon_fixed(sun_j2000, et)
            mirror_j2000 = orbit.position_at_et(et, epoch_et)
            mirror_pa = geometry.j2000_to_moon_fixed(mirror_j2000, et)
            site_pa = _to_pa(candidate_me_position(site), MOON_ME_DE421, et, bridge)
            profile_pa = _profile_to_pa(horizon_profile, et, bridge)
            occult_sun = segment_occludes_lunar_sphere(mirror_pa, sun_pa, radius)
            occult_site = segment_occludes_lunar_sphere(mirror_pa, site_pa, radius)
            sun_clear = not occult_sun
            site_clear = not occult_site
            clearance = None
            terrain_status = "missing-horizon-profile"
            if profile_pa is not None and _distance(profile_pa.site_position_km, site_pa) <= 1e-5:
                ray_site_to_mirror = _subtract(mirror_pa, site_pa)
                local = _local_angles(ray_site_to_mirror, profile_pa)
                if local:
                    try:
                        terrain_horizon = interpolate_horizon_elevation(local[0], profile_pa.profile_deg)
                        clearance = local[1] - terrain_horizon
                        if clearance <= 0:
                            terrain_status = "known-blocked"
                            site_clear = False
                        elif profile_pa.status == "complete":
                            terrain_status = "complete-profile"
                        elif _norm(ray_site_to_mirror) / 1000.0 > profile_pa.range_km:
                            terrain_status = "unknown-beyond-limited-range"
                            site_clear = None
                        else:
                            terrain_status = "limited-range-clearance-only"
                            site_clear = None
                    except ValueError:
                        terrain_status = "missing-horizon-bin"
                        site_clear = None
            elif profile_pa is not None:
                terrain_status = "profile-site-mismatch"
                site_clear = None

            optics = None
            mirror_incidence = receiver_incidence = optical_power = None
            try:
                optics = calculate_mirror_optics(
                    TaggedLunarPosition(sun_pa, "MOON_PA_DE440"),
                    TaggedLunarPosition(mirror_pa, "MOON_PA_DE440"),
                    TaggedLunarPosition(site_pa, "MOON_PA_DE440"),
                    optics_assumptions,
                    geometry_origin="assumed",
                )
                mirror_incidence = optics.incidence_angle_deg
                optical_power = optics.reflected_optical_power_w
                if receiver_normal_local_enu is not None and profile_pa is not None:
                    direction_local = _local_components(_subtract(mirror_pa, site_pa), profile_pa)
                    beam_dot = sum(direction_local[i] * receiver_normal_local_enu[i] for i in range(3))
                    receiver_incidence = math.degrees(math.acos(max(-1.0, min(1.0, beam_dot))))
            except ValueError:
                pass
            if occult_sun or occult_site or (clearance is not None and clearance <= 0):
                state = CriterionStatus.FAIL
            elif not sun_clear or site_clear is not True or receiver_incidence is None:
                state = CriterionStatus.UNKNOWN
            else:
                state = CriterionStatus.PASS
            samples.append(ReflectorSample(
                timestamp_utc=text,
                spacecraft_position_km=tuple(float(v) for v in mirror_pa),
                frame="MOON_PA_DE440",
                sun_to_mirror_clear=sun_clear,
                mirror_to_site_clear=site_clear,
                lunar_occulted=bool(occult_sun or occult_site),
                terrain_clearance_deg=clearance,
                terrain_status=terrain_status,
                mirror_incidence_angle_deg=mirror_incidence,
                receiver_incidence_angle_deg=receiver_incidence,
                reflected_optical_power_w=optical_power,
                status=state,
            ))
    intervals = _sampled_intervals(samples, cadence)
    states = [sample.status for sample in samples]
    if CriterionStatus.PASS in states:
        status = CriterionStatus.PASS
        reason = "One or more sampled epochs have geometrically and terrain-supported reflector access with receiver incidence defined."
    elif states and all(value == CriterionStatus.FAIL for value in states):
        status = CriterionStatus.FAIL
        reason = "All sampled reflector rays are known to be occulted or terrain blocked."
    else:
        status = CriterionStatus.UNKNOWN
        reason = "Reflector access is not established for all required evidence (terrain range, geometry or receiver attitude may be unknown)."
    return ReflectorAssessment(
        criterion=CriterionAssessment(
            "reflector visibility", status,
            evidence=(reason, f"{len(samples)} sampled orbital epochs; assumed orbit period {orbit.period_s:.3f} s."),
            missing_data=() if status != CriterionStatus.UNKNOWN else ("Complete compatible horizon coverage and explicit receiver normal are needed for a positive access finding.",),
            assumptions=("Orbit positions are assumed circular kinematics, not SPICE propagation or stability analysis.", "Flat mirror redirects but does not focus sunlight."),
        ),
        samples=tuple(samples),
        valid_illumination_intervals=intervals,
        cadence_s=cadence,
        terrain_radius_km=horizon_profile.range_km if horizon_profile else None,
        horizon_status=horizon_profile.status if horizon_profile else "unknown",
        orbit_model=ORBIT_MODEL_LABEL,
        geometry_provenance=("SPICE Sun vector and J2000→MOON_PA_DE440 rotation.", "Assumed circular orbit parameters and phase.", "Site coordinates transformed ME→PA with LunarFrameBridge where required."),
    )


def build_site_feasibility_report(
    site: CandidateSite,
    illumination: IlluminationAssessment,
    reflector: ReflectorAssessment,
    terrain: TerrainAssessment,
) -> SiteFeasibilityReport:
    """Build a consistent five-criterion site report without weighted scoring."""
    return SiteFeasibilityReport(
        candidate=site,
        solar_illumination=illumination.criterion,
        reflector_visibility=reflector.criterion,
        terrain_coverage=terrain.coverage,
        slope_evidence=terrain.slope,
        rover_access=terrain.rover_access,
        overall_status=combine_criteria((
            illumination.criterion, reflector.criterion, terrain.coverage,
            terrain.slope, terrain.rover_access,
        )),
    )


def compare_site_reports(reports: tuple[SiteFeasibilityReport, ...]) -> SiteComparison:
    """Return side-by-side evidence status; never weight or auto-select sites."""
    values = tuple(reports)
    if not values:
        raise ValueError("At least one site report is required for comparison.")
    ids = [report.candidate.identifier for report in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Site comparison identifiers must be unique.")
    unknown = any(_report_has_unknown_critical_criterion(report) for report in values)
    return SiteComparison(
        reports=values,
        comparison_status=CriterionStatus.UNKNOWN if unknown else CriterionStatus.PASS,
        selection_made=False,
        selection_reason=(
            "No site selected. One or more critical criteria remain UNKNOWN; compare evidence without weighted scoring."
            if unknown else
            "No site selected. Criteria are comparable, but this evaluator does not automatically choose a candidate."
        ),
    )


def _report_has_unknown_critical_criterion(report: SiteFeasibilityReport) -> bool:
    return any(item.status == CriterionStatus.UNKNOWN for item in (
        report.solar_illumination, report.reflector_visibility, report.terrain_coverage,
        report.slope_evidence, report.rover_access,
    ))


def _sample_times(start: datetime, duration_s: float, cadence_s: float) -> tuple[datetime, ...]:
    values = [start + timedelta(seconds=offset) for offset in np.arange(0.0, duration_s + 1e-9, cadence_s)]
    end = start + timedelta(seconds=duration_s)
    if not values or values[-1] < end:
        values.append(end)
    return tuple(values)


def _site_local_axes(profile: SiteHorizonProfile | None, site: CandidateSite):
    if profile is not None:
        return profile.east_axis, profile.north_axis, profile.up_axis
    frame = LunarSite(site.latitude_deg, site.longitude_east_deg)
    return frame.east, frame.north, frame.up


def _profile_to_pa(profile, et, bridge):
    if profile is None:
        return None
    if profile.frame == "MOON_PA_DE440":
        return profile
    if profile.frame != MOON_ME_DE421:
        return None
    return SiteHorizonProfile(
        profile_deg=profile.profile_deg,
        site_position_km=bridge.vector_dem_to_geometry(profile.site_position_km, et),
        frame="MOON_PA_DE440",
        east_axis=bridge.vector_dem_to_geometry(profile.east_axis, et),
        north_axis=bridge.vector_dem_to_geometry(profile.north_axis, et),
        up_axis=bridge.vector_dem_to_geometry(profile.up_axis, et),
        range_km=profile.range_km,
        status=profile.status,
        site_name=profile.site_name,
    )


def _to_pa(vector, frame, et, bridge):
    if frame == "MOON_PA_DE440":
        return tuple(vector)
    if frame == MOON_ME_DE421:
        return bridge.vector_dem_to_geometry(vector, et)
    raise ValueError(f"Unsupported lunar frame {frame!r}.")


def _local_angles(vector, profile):
    if profile is None:
        return None
    east, north, up = _local_components(vector, profile)
    if math.hypot(east, north) <= 1e-15 and abs(up) <= 1e-15:
        return None
    return math.degrees(math.atan2(east, north)) % 360.0, math.degrees(math.atan2(up, math.hypot(east, north)))


def _local_components(vector, profile):
    axes = (profile.east_axis, profile.north_axis, profile.up_axis)
    return tuple(sum(vector[i] * axis[i] for i in range(3)) for axis in axes)


def _sampled_intervals(samples, cadence):
    visible = [item for item in samples if item.status == CriterionStatus.PASS]
    if not visible:
        return ()
    runs: list[list[object]] = []
    for item in visible:
        stamp = _utc(item.timestamp_utc)
        if not runs or (stamp - _utc(runs[-1][-1].timestamp_utc)).total_seconds() > cadence + 1e-6:
            runs.append([item])
        else:
            runs[-1].append(item)
    return tuple(
        SampledVisibilityInterval(
            start_utc=run[0].timestamp_utc,
            end_utc=run[-1].timestamp_utc,
            sample_count=len(run),
        )
        for run in runs
    )


def _subtract(a, b):
    return tuple(float(a[i]) - float(b[i]) for i in range(3))


def _norm(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _distance(a, b):
    return _norm(_subtract(a, b))
