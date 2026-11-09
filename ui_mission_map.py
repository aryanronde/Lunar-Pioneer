"""Presentation helpers for the local Site06 south-polar mission map."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from core.lunar_coordinates import LunarSiteRecord, planetocentric_latlon_to_moon_me_de421
from core.terrain_provider import (
    LOLA20M_URL,
    LOLA20M_SOUTH_POLAR_CRS,
    LOLA20mTerrainProvider,
    MOON_REFERENCE_RADIUS_M,
    MOON_ME_DE421,
    TaggedMoonPosition,
    TerrainMetadata,
    TerrainWindow,
    project_moon_me_de421_position,
)


LOLA20M_MAP_LABEL = "NASA LOLA 20 m DEM — MOON_ME / DE421"
TERRAIN_MARGIN_M = 5_000.0


@dataclass(frozen=True)
class SiteMapGeometry:
    """Projected site center and analysis-radius circle in LOLA map coordinates."""

    center_x_m: float
    center_y_m: float
    radius_m: float
    circle_x_m: tuple[float, ...]
    circle_y_m: tuple[float, ...]


@dataclass(frozen=True)
class SiteTerrainMapData:
    """One cached provider result shared by the map and diagnostics."""

    window: TerrainWindow
    metadata: TerrainMetadata
    source: str
    status: str = "Bounded terrain window read completed."


@lru_cache(maxsize=8)
def build_site_map_geometry(site: LunarSiteRecord, radius_m: float) -> SiteMapGeometry:
    """Prepare a 10 km-style circle using the shared LOLA coordinate transform."""
    radius_m = float(radius_m)
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Analysis radius must be positive and finite meters.")
    if site.source_frame != MOON_ME_DE421:
        raise ValueError(f"Map site must be tagged in {MOON_ME_DE421}.")
    if site.latitude_type != "planetocentric" or site.longitude_convention != "east_positive":
        raise ValueError("Map site must use planetocentric latitude and east-positive longitude.")

    position_km = planetocentric_latlon_to_moon_me_de421(
        site.latitude_deg,
        site.longitude_deg,
        MOON_REFERENCE_RADIUS_M / 1000.0,
    )
    tagged_position = TaggedMoonPosition(*position_km, frame=site.source_frame)
    center_x_m, center_y_m = project_moon_me_de421_position(
        tagged_position,
        LOLA20M_SOUTH_POLAR_CRS,
    )

    # Circle points are offsets in the established projected meter grid.
    angles = tuple(2.0 * math.pi * index / 360.0 for index in range(361))
    circle_x_m = tuple(center_x_m + radius_m * math.cos(angle) for angle in angles)
    circle_y_m = tuple(center_y_m + radius_m * math.sin(angle) for angle in angles)
    return SiteMapGeometry(center_x_m, center_y_m, radius_m, circle_x_m, circle_y_m)


@lru_cache(maxsize=8)
def _read_cached_site_terrain_window(
    source: str,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
) -> SiteTerrainMapData:
    """Read and cache one bounded native-resolution DEM window for a site map."""
    with LOLA20mTerrainProvider(source) as provider:
        metadata = provider.metadata
        if metadata.source_frame != MOON_ME_DE421:
            raise ValueError(f"Map DEM must use {MOON_ME_DE421}.")
        pixel_x_m, pixel_y_m = metadata.pixel_size_m
        if pixel_x_m <= 0 or pixel_y_m <= 0:
            raise ValueError("Map DEM pixel sizes must be positive.")
        half_width_m = radius_m + TERRAIN_MARGIN_M
        width_px = math.ceil(2.0 * half_width_m / pixel_x_m)
        height_px = math.ceil(2.0 * half_width_m / pixel_y_m)
        if max(width_px, height_px) > provider.MAX_WINDOW_SIDE_PX:
            raise ValueError(
                "Requested Site06 terrain view exceeds the provider's "
                f"{provider.MAX_WINDOW_SIDE_PX}-pixel side limit."
            )
        window = provider.read_window(center_x_m, center_y_m, width_px, height_px)
        return SiteTerrainMapData(window, metadata, provider.source)


def load_site_terrain_window(
    site: LunarSiteRecord,
    radius_m: float,
    *,
    source: str = LOLA20M_URL,
) -> SiteTerrainMapData:
    """Load a cached, bounded DEM window centered on the projected map site."""
    geometry = build_site_map_geometry(site, radius_m)
    return _read_cached_site_terrain_window(
        str(source), geometry.center_x_m, geometry.center_y_m, geometry.radius_m
    )


def build_terrain_diagnostics(
    geometry: SiteMapGeometry,
    terrain_data: SiteTerrainMapData,
) -> dict[str, object]:
    """Summarize the exact cached map window without changing its samples."""
    window = terrain_data.window
    elevation = np.asarray(window.elevation_m, dtype=np.float64)
    nodata_mask = np.asarray(window.nodata_mask, dtype=bool)
    if elevation.ndim != 2 or nodata_mask.shape != elevation.shape:
        raise ValueError("Terrain elevation and NoData mask must be matching two-dimensional arrays.")
    missing = nodata_mask | ~np.isfinite(elevation)
    valid_values = elevation[~missing]
    x_centers = np.asarray(window.x_centers_m, dtype=np.float64)
    y_centers = np.asarray(window.y_centers_m, dtype=np.float64)
    if x_centers.size != elevation.shape[1] or y_centers.size != elevation.shape[0]:
        raise ValueError("Terrain coordinate vectors do not match the elevation array dimensions.")
    center_bounds = (
        float(np.min(x_centers)),
        float(np.min(y_centers)),
        float(np.max(x_centers)),
        float(np.max(y_centers)),
    )
    left, bottom, right, top = center_bounds
    site_inside = left <= geometry.center_x_m <= right and bottom <= geometry.center_y_m <= top
    a, b, _c, d, e, _f = terrain_data.metadata.affine
    col_dx, col_dy = float(a), float(d)
    row_dx, row_dy = float(b), float(e)

    def direction(dx: float, dy: float) -> str:
        horizontal = "+X" if dx > 0 else "−X" if dx < 0 else "no X change"
        vertical = "+Y" if dy > 0 else "−Y" if dy < 0 else "no Y change"
        return f"{horizontal}, {vertical}"

    source = terrain_data.source
    configured_nasa_source = source.lower().startswith(("https://", "/vsicurl/")) and "pgda.gsfc.nasa.gov" in source.lower()
    return {
        "status": terrain_data.status,
        "source": source,
        "provenance": (
            "Configured NASA LOLA COG source; provider read completed, but the remote content was not independently authenticated."
            if configured_nasa_source
            else "Provider source recorded; NASA provenance is not established."
        ),
        "shape": tuple(int(value) for value in elevation.shape),
        "valid_count": int(valid_values.size),
        "nodata_count": int(missing.sum()),
        "minimum_m": float(np.min(valid_values)) if valid_values.size else None,
        "maximum_m": float(np.max(valid_values)) if valid_values.size else None,
        "median_m": float(np.median(valid_values)) if valid_values.size else None,
        "site_x_y_m": (float(geometry.center_x_m), float(geometry.center_y_m)),
        "window_center_bounds_m": center_bounds,
        "raster_bounds_m": tuple(float(value) for value in terrain_data.metadata.bounds_m),
        "site_inside_window_centers": bool(site_inside),
        "pixel_spacing_m": tuple(float(value) for value in terrain_data.metadata.pixel_size_m),
        "column_step_m": (col_dx, col_dy),
        "row_step_m": (row_dx, row_dy),
        "column_orientation": direction(col_dx, col_dy),
        "row_orientation": direction(row_dx, row_dy),
        "axis_aligned": bool(math.isclose(b, 0.0, abs_tol=1e-12) and math.isclose(d, 0.0, abs_tol=1e-12)),
    }


def _plot_elevation(window: TerrainWindow) -> np.ndarray:
    """Return float elevations with all NoData and non-finite samples missing."""
    elevation = np.asarray(window.elevation_m, dtype=np.float64).copy()
    invalid = np.asarray(window.nodata_mask, dtype=bool) | ~np.isfinite(elevation)
    if invalid.shape != elevation.shape:
        raise ValueError("Terrain elevation and NoData mask must have matching shapes.")
    elevation[invalid] = np.nan
    return elevation


def build_site_map_figure(
    site: LunarSiteRecord,
    radius_m: float,
    *,
    terrain_data: SiteTerrainMapData | None = None,
    terrain_window: TerrainWindow | None = None,
):
    """Create an interactive map with a bounded NASA LOLA elevation layer."""
    import plotly.graph_objects as go

    geometry = build_site_map_geometry(site, radius_m)
    if terrain_data is None and terrain_window is None:
        terrain_data = load_site_terrain_window(site, radius_m)
    if terrain_data is not None:
        terrain_window = terrain_data.window
    assert terrain_window is not None
    circle_x_km = [x / 1000.0 for x in geometry.circle_x_m]
    circle_y_km = [y / 1000.0 for y in geometry.circle_y_m]
    center_x_km = geometry.center_x_m / 1000.0
    center_y_km = geometry.center_y_m / 1000.0
    radius_km = geometry.radius_m / 1000.0
    extent_km = max(1.5 * radius_km, radius_km + TERRAIN_MARGIN_M / 1000.0)

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            x=np.asarray(terrain_window.x_centers_m, dtype=np.float64) / 1000.0,
            y=np.asarray(terrain_window.y_centers_m, dtype=np.float64) / 1000.0,
            z=_plot_elevation(terrain_window),
            name=LOLA20M_MAP_LABEL,
            colorscale="Earth",
            colorbar={
                "title": {"text": "Elevation (m)", "font": {"color": "#10223d", "size": 13}},
                "tickfont": {"color": "#10223d", "size": 11},
                "outlinecolor": "#52657d",
                "x": 1.02,
                "xanchor": "left",
            },
            connectgaps=False,
            hovertemplate="X: %{x:.3f} km<br>Y: %{y:.3f} km<br>Elevation: %{z:.1f} m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=circle_x_km,
            y=circle_y_km,
            mode="lines",
            name=f"{radius_km:g} km limited-range radius",
            line={"color": "#e6533b", "width": 2, "dash": "dash"},
            hovertemplate="10 km limited-range terrain analysis<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[center_x_km],
            y=[center_y_km],
            mode="markers",
            name=site.name,
            marker={"color": "#e6533b", "size": 13, "line": {"color": "#ffffff", "width": 2}},
            customdata=[[site.latitude_deg, site.longitude_deg, geometry.center_x_m, geometry.center_y_m]],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Planetocentric latitude: %{customdata[0]:.5f}°"
                "<br>East longitude: %{customdata[1]:.5f}°<br>LOLA X: %{customdata[2]:,.1f} m"
                "<br>LOLA Y: %{customdata[3]:,.1f} m<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#f6f9fd",
        plot_bgcolor="#edf3fa",
        margin={"l": 22, "r": 110, "t": 22, "b": 96},
        height=520,
        dragmode="pan",
        showlegend=True,
        font={"color": "#10223d"},
        legend={
            "font": {"color": "#10223d", "size": 12},
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#9aacc1",
            "borderwidth": 1,
            "orientation": "h",
            "x": 0.5,
            "y": -0.18,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis={
            "title": "South-polar stereographic X (km; +X ≈ 90°E)",
            "range": [center_x_km - extent_km, center_x_km + extent_km],
            "showgrid": True,
            "gridcolor": "#d7e1ec",
            "zeroline": False,
            "tickfont": {"color": "#10223d", "size": 11},
            "linecolor": "#52657d",
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "title": "South-polar stereographic Y (km; +Y toward 0° longitude)",
            "range": [center_y_km - extent_km, center_y_km + extent_km],
            "showgrid": True,
            "gridcolor": "#d7e1ec",
            "zeroline": False,
            "tickfont": {"color": "#10223d", "size": 11},
            "linecolor": "#52657d",
        },
        annotations=[
            {
                "x": 0.02,
                "y": 0.96,
                "xref": "paper",
                "yref": "paper",
                "text": "↑ +Y · 0° lon<br>→ +X · 90°E",
                "showarrow": False,
                "align": "left",
                "font": {"color": "#10223d", "size": 12},
                "bgcolor": "rgba(255,255,255,0.96)",
                "bordercolor": "#9aacc1",
                "borderpad": 5,
            }
        ],
    )
    figure.layout.xaxis.title.font = {"color": "#10223d", "size": 13}
    figure.layout.yaxis.title.font = {"color": "#10223d", "size": 13}
    return figure
