"""Core calculations for the CLPS Lunar Mission Browser."""

from .spice_engine import SpiceGeometryEngine
from .lunar_horizon import LocalHorizonFrame, LunarSite
from .terrain_provider import LOLA20mTerrainProvider, TaggedMoonPosition, TerrainMetadata, TerrainProvider, TerrainWindow
from .lunar_coordinates import LunarSiteRecord, SITE06_NOBILE_RIM_1, planetocentric_latlon_to_moon_me_de421
from .terrain_surface import (
    LunarTerrainSurface,
    TerrainNormalField,
    calculate_terrain_normals,
    reconstruct_lunar_terrain,
)
from .terrain_horizon import TerrainHorizonProfile, calculate_terrain_horizon_profile
from .celestial_direction import CelestialDirection, Site06CelestialDirectionEngine, celestial_direction
from .terrain_visibility import (
    TerrainRelativeVisibility,
    azimuth_to_horizon_bin,
    compare_with_terrain_horizon,
    interpolate_horizon_elevation,
    load_site06_horizon_profile,
    site06_terrain_visibility,
)
from .visibility_timeseries import (
    VisibilitySample,
    VisibilityTimeSeries,
    VisibilityWindow,
    generate_timestamps,
    group_both_visible_windows,
    group_visibility_windows,
    run_site06_visibility_timeseries,
)
from .terrain_region import (
    circle_pixel_coverage,
    enclosing_pixel_range,
    radius_window_indices,
    regional_horizon_status,
    regional_sample_masks,
)
from .mission_analysis import (
    MissionAnalysisResult,
    MissionSample,
    MissionVisibilityWindow,
    TargetMissionSummary,
    TerrainHorizonConfiguration,
    analyze_mission_window,
    load_site06_10km_horizon_configuration,
)

__all__ = [
    "LOLA20mTerrainProvider",
    "LunarSiteRecord",
    "LunarTerrainSurface",
    "LocalHorizonFrame",
    "LunarSite",
    "SpiceGeometryEngine",
    "SITE06_NOBILE_RIM_1",
    "TaggedMoonPosition",
    "TerrainMetadata",
    "TerrainHorizonProfile",
    "TerrainNormalField",
    "TerrainProvider",
    "TerrainWindow",
    "planetocentric_latlon_to_moon_me_de421",
    "calculate_terrain_normals",
    "calculate_terrain_horizon_profile",
    "reconstruct_lunar_terrain",
    "CelestialDirection",
    "Site06CelestialDirectionEngine",
    "celestial_direction",
    "TerrainRelativeVisibility",
    "azimuth_to_horizon_bin",
    "compare_with_terrain_horizon",
    "interpolate_horizon_elevation",
    "load_site06_horizon_profile",
    "site06_terrain_visibility",
    "VisibilitySample",
    "VisibilityTimeSeries",
    "VisibilityWindow",
    "generate_timestamps",
    "group_both_visible_windows",
    "group_visibility_windows",
    "run_site06_visibility_timeseries",
    "circle_pixel_coverage",
    "enclosing_pixel_range",
    "radius_window_indices",
    "regional_horizon_status",
    "regional_sample_masks",
    "MissionAnalysisResult",
    "MissionSample",
    "MissionVisibilityWindow",
    "TargetMissionSummary",
    "TerrainHorizonConfiguration",
    "analyze_mission_window",
    "load_site06_10km_horizon_configuration",
]
