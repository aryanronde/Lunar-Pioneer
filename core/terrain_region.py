"""Bounded radius windows and coverage masks for regional lunar DEM reads."""

from __future__ import annotations

import math

import numpy as np

from .terrain_provider import TerrainMetadata, TerrainWindow


def enclosing_pixel_range(low: float, high: float, origin: float, step: float) -> tuple[int, int]:
    """Return half-open pixel indices whose cells enclose the coordinate span."""
    if not all(math.isfinite(float(value)) for value in (low, high, origin, step)) or step == 0:
        raise ValueError("Pixel range coordinates and nonzero step must be finite.")
    first_pixel = (min(low, high) - origin) / step
    last_pixel = (max(low, high) - origin) / step
    start = math.floor(min(first_pixel, last_pixel))
    stop = math.ceil(max(first_pixel, last_pixel))
    return start, stop


def radius_window_indices(
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    metadata: TerrainMetadata,
    *,
    max_window_side_px: int,
) -> tuple[int, int, int, int]:
    """Return [row_start,row_stop), [col_start,col_stop) enclosing a circle.

    Raises when the enclosing rectangle leaves raster bounds or exceeds the
    provider's bounded-read limit; partial coverage is assessed separately by
    the returned-window coverage mask.
    """
    if not all(math.isfinite(float(value)) for value in (center_x_m, center_y_m, radius_m)):
        raise ValueError("Center and radius must be finite.")
    if radius_m <= 0:
        raise ValueError("Radius must be positive.")
    a, _, c, _, e, f = metadata.affine
    col_start, col_stop = enclosing_pixel_range(center_x_m - radius_m, center_x_m + radius_m, c, a)
    row_start, row_stop = enclosing_pixel_range(center_y_m - radius_m, center_y_m + radius_m, f, e)
    if row_start < 0 or col_start < 0 or row_stop > metadata.height or col_stop > metadata.width:
        raise ValueError("Radius window's enclosing rectangle extends beyond the DEM raster bounds.")
    width, height = col_stop - col_start, row_stop - row_start
    if width > max_window_side_px or height > max_window_side_px:
        raise ValueError(
            f"Radius window {width} x {height} exceeds provider limit of {max_window_side_px} pixels per side."
        )
    return row_start, row_stop, col_start, col_stop


def regional_sample_masks(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    bounds_m: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return circular-region and raster-coverage masks for pixel centers."""
    in_circle = (x_grid - center_x_m) ** 2 + (y_grid - center_y_m) ** 2 <= radius_m**2
    left, bottom, right, top = bounds_m
    in_coverage = (
        (x_grid >= left) & (x_grid < right) & (y_grid > bottom) & (y_grid <= top)
    )
    return in_circle, in_coverage


def regional_horizon_status(
    in_circle: np.ndarray,
    in_coverage: np.ndarray,
    nodata_mask: np.ndarray,
    elevation_m: np.ndarray,
) -> str:
    """Classify sampled regional support without claiming a complete horizon.

    A bounded profile is ``limited-range`` when all selected circular samples
    are covered and valid. Missing coverage or NoData within the requested
    circle makes it ``insufficient-coverage``. This regional layer never calls
    its profile globally ``complete``.
    """
    if not (in_circle.shape == in_coverage.shape == nodata_mask.shape == elevation_m.shape):
        raise ValueError("Circle, coverage, NoData, and elevation arrays must have matching shapes.")
    if not np.any(in_circle):
        return "insufficient-coverage"
    valid = in_coverage & ~nodata_mask & np.isfinite(elevation_m)
    if np.any(in_circle & ~valid):
        return "insufficient-coverage"
    return "limited-range"


def circle_pixel_coverage(
    window: TerrainWindow,
    metadata: TerrainMetadata,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return circle, in-bounds, and valid elevation masks for a raster window."""
    x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
    in_circle, in_coverage = regional_sample_masks(
        x_grid, y_grid, center_x_m, center_y_m, radius_m, metadata.bounds_m
    )
    valid = in_coverage & ~window.nodata_mask & np.isfinite(window.elevation_m)
    return in_circle, in_coverage, valid

