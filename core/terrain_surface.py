"""Spherical LOLA terrain point reconstruction and local surface normals.

This layer creates Moon-centered Cartesian terrain points and finite-difference
surface normals. It deliberately does not calculate horizons or visibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

from .terrain_provider import MOON_ME_DE421, MOON_REFERENCE_RADIUS_M, TerrainMetadata, TerrainWindow


REFERENCE_RADIUS_KM = MOON_REFERENCE_RADIUS_M / 1000.0
MOON_ME_SPHERICAL_CRS = CRS.from_proj4(
    f"+proj=longlat +R={MOON_REFERENCE_RADIUS_M:g} +no_defs +type=crs"
)


@dataclass(frozen=True)
class LunarTerrainSurface:
    """Moon-centered terrain points in km, explicitly tagged by source frame."""

    positions_km: np.ndarray
    valid_mask: np.ndarray
    pixel_size_m: tuple[float, float]
    row_offset: int
    column_offset: int
    source_frame: str = MOON_ME_DE421


@dataclass(frozen=True)
class TerrainNormalField:
    """Unit surface-normal vectors in the same body-fixed frame as the surface."""

    normals: np.ndarray
    valid_mask: np.ndarray
    source_frame: str


def reconstruct_lunar_terrain(
    window: TerrainWindow,
    metadata: TerrainMetadata,
    reference_radius_km: float = REFERENCE_RADIUS_KM,
) -> LunarTerrainSurface:
    """Reconstruct each valid DEM cell as a Moon-centered MOON_ME_DE421 point.

    The polar-stereographic pixel centers are inverse-projected using the DEM
    CRS. Their planetocentric east-positive lon/lat directions are placed at
    radius ``reference_radius_km + elevation_m / 1000`` because the LOLA DEM
    heights are meters above/below its spherical reference surface.
    """
    if metadata.source_frame != MOON_ME_DE421:
        raise ValueError(
            f"Terrain DEM frame must be {MOON_ME_DE421}; got {metadata.source_frame!r}. "
            "Transform frames explicitly with the existing SPICE bridge before combining data."
        )
    if not np.isfinite(reference_radius_km) or reference_radius_km <= 0.0:
        raise ValueError("Spherical reference radius must be a positive finite value in kilometers.")
    if window.elevation_m.ndim != 2 or window.nodata_mask.shape != window.elevation_m.shape:
        raise ValueError("Terrain window elevations and NoData mask must be matching two-dimensional arrays.")
    if len(window.x_centers_m) != window.elevation_m.shape[1]:
        raise ValueError("Terrain window X centers do not match its column count.")
    if len(window.y_centers_m) != window.elevation_m.shape[0]:
        raise ValueError("Terrain window Y centers do not match its row count.")

    transformer = Transformer.from_crs(CRS.from_wkt(metadata.crs), MOON_ME_SPHERICAL_CRS, always_xy=True)
    x_grid_m, y_grid_m = np.meshgrid(window.x_centers_m, window.y_centers_m)
    try:
        longitude_deg, latitude_deg = transformer.transform(x_grid_m, y_grid_m, errcheck=True)
    except Exception as exc:
        raise ValueError(f"Could not inverse-project LOLA DEM X/Y into Moon coordinates: {exc}") from exc

    longitude_rad = np.deg2rad(longitude_deg)
    latitude_rad = np.deg2rad(latitude_deg)
    radius_km = reference_radius_km + np.asarray(window.elevation_m, dtype=np.float64) / 1000.0
    cos_latitude = np.cos(latitude_rad)
    positions = np.stack(
        (
            radius_km * cos_latitude * np.cos(longitude_rad),
            radius_km * cos_latitude * np.sin(longitude_rad),
            radius_km * np.sin(latitude_rad),
        ),
        axis=-1,
    )
    valid = (
        ~np.asarray(window.nodata_mask, dtype=bool)
        & np.isfinite(window.elevation_m)
        & np.isfinite(longitude_deg)
        & np.isfinite(latitude_deg)
        & np.all(np.isfinite(positions), axis=-1)
        & (radius_km > 0.0)
    )
    positions[~valid] = np.nan
    pixel_size_x, pixel_size_y = metadata.pixel_size_m
    if pixel_size_x <= 0.0 or pixel_size_y <= 0.0:
        raise ValueError("DEM pixel sizes must be positive.")
    return LunarTerrainSurface(
        positions_km=positions,
        valid_mask=valid,
        pixel_size_m=(float(pixel_size_x), float(pixel_size_y)),
        row_offset=window.row_offset,
        column_offset=window.column_offset,
        source_frame=MOON_ME_DE421,
    )


def calculate_terrain_normals(surface: LunarTerrainSurface) -> TerrainNormalField:
    """Calculate outward unit normals with centered 3D finite differences.

    For each interior cell, centered tangents approximate dP/dX and dP/dY:
    adjacent positions two pixels apart are divided by their map distance.
    Raster rows increase toward decreasing DEM Y, so the northward tangent is
    formed from row-1 minus row+1. The normal is ``cross(dP/dX, dP/dY)``;
    it is flipped if needed so its dot product with the Moon-centered point is
    positive. Edges and stencils touching NoData have invalid (NaN) normals.
    """
    if surface.source_frame != MOON_ME_DE421:
        raise ValueError(f"Surface frame must be {MOON_ME_DE421}; got {surface.source_frame!r}.")
    if surface.positions_km.ndim != 3 or surface.positions_km.shape[-1] != 3:
        raise ValueError("Terrain Cartesian positions must have shape (rows, columns, 3).")
    if surface.valid_mask.shape != surface.positions_km.shape[:2]:
        raise ValueError("Terrain position validity mask must match the raster grid.")
    rows, columns = surface.valid_mask.shape
    normals = np.full_like(surface.positions_km, np.nan, dtype=np.float64)
    normal_valid = np.zeros((rows, columns), dtype=bool)
    if rows < 3 or columns < 3:
        return TerrainNormalField(normals, normal_valid, surface.source_frame)

    center_valid = surface.valid_mask[1:-1, 1:-1]
    stencil_valid = (
        center_valid
        & surface.valid_mask[1:-1, :-2]
        & surface.valid_mask[1:-1, 2:]
        & surface.valid_mask[:-2, 1:-1]
        & surface.valid_mask[2:, 1:-1]
    )
    dx_m, dy_m = surface.pixel_size_m
    # DEM X increases with columns; DEM Y increases opposite to raster row number.
    tangent_x = (
        surface.positions_km[1:-1, 2:] - surface.positions_km[1:-1, :-2]
    ) / (2.0 * dx_m / 1000.0)
    tangent_y = (
        surface.positions_km[:-2, 1:-1] - surface.positions_km[2:, 1:-1]
    ) / (2.0 * dy_m / 1000.0)
    cross = np.cross(tangent_x, tangent_y)
    magnitude = np.linalg.norm(cross, axis=-1)
    stencil_valid &= np.isfinite(magnitude) & (magnitude > 0.0)
    unit = np.full_like(cross, np.nan, dtype=np.float64)
    unit[stencil_valid] = cross[stencil_valid] / magnitude[stencil_valid, None]

    radial = surface.positions_km[1:-1, 1:-1]
    inward = np.sum(unit * radial, axis=-1) < 0.0
    unit[inward & stencil_valid] *= -1.0
    normals[1:-1, 1:-1] = unit
    normal_valid[1:-1, 1:-1] = stencil_valid
    return TerrainNormalField(normals, normal_valid, surface.source_frame)
