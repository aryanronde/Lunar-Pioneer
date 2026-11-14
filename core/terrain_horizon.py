"""Sample-based terrain horizon profiles from Moon-centered Cartesian points.

This module estimates terrain obstruction only. It does not calculate target
body directions, sunlight, Earth visibility, or whether a horizon profile is
complete around the Moon.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .terrain_provider import MOON_ME_DE421


@dataclass(frozen=True)
class TerrainHorizonProfile:
    """Per-azimuth maximum sampled terrain elevation relative to local tangent."""

    azimuth_centers_deg: np.ndarray
    max_obstruction_elevation_deg: np.ndarray
    sample_count: np.ndarray
    source_frame: str
    north_axis: tuple[float, float, float]
    east_axis: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    coverage_note: str = "Maximum is over supplied valid terrain samples only; profile completeness is not inferred."


def calculate_terrain_horizon_profile(
    observer_position_km: tuple[float, float, float] | np.ndarray,
    observer_outward_normal: tuple[float, float, float] | np.ndarray,
    terrain_points_km: np.ndarray,
    *,
    observer_frame: str,
    terrain_frame: str,
    reference_north_km: tuple[float, float, float] | np.ndarray,
    valid_mask: np.ndarray | None = None,
    observer_sample_index: int | tuple[int, ...] | None = None,
    azimuth_bin_count: int = 360,
) -> TerrainHorizonProfile:
    """Return the maximum sampled terrain angle in each azimuth bin.

    All vectors are Moon-centered Cartesian km in MOON_ME_DE421. The observer
    outward terrain normal supplies local Up only. ``reference_north_km`` must
    be the actual V0.2A North vector, transformed into this same frame (for
    example with ``LunarFrameBridge``); it is projected onto the plane
    perpendicular to terrain Up and normalized. East is explicitly
    ``North cross Up``. Azimuth is clockwise from North: 0 North, 90 East,
    180 South, 270 West. With this definition, North x East = -Up and
    East x North = Up; these signs follow directly from the cross-product
    identity and establish the handedness unambiguously.

    For observer P and terrain sample Q, the exact Moon-centered chord is
    V = Q - P. Its local components are e=V·East, n=V·North, u=V·Up. The
    sample azimuth is atan2(e,n), and its obstruction/elevation angle relative
    to the observer's local terrain tangent plane is
    atan2(u, hypot(e,n)). Computing V in 3D Moon-centered coordinates includes
    lunar curvature; no flat elevation-over-horizontal-range approximation is
    used. Each bin retains the greatest angle among its valid samples.

    An explicit observer_sample_index excludes the observer's raster sample;
    any exact duplicate position (zero-length observer-to-sample vector) is
    also discarded. NaN/Inf and masked samples are ignored. Empty bins contain
    NaN and a sample count of zero. This is only a profile of supplied samples:
    a small terrain window is not considered a complete lunar horizon.
    """
    if observer_frame != MOON_ME_DE421 or terrain_frame != MOON_ME_DE421:
        raise ValueError(
            "Observer and terrain points must both be explicitly in MOON_ME_DE421. "
            "Use LunarFrameBridge with SPICE pxform before this calculation if frames differ."
        )
    if observer_frame != terrain_frame:
        raise ValueError("Observer and terrain samples must use the same reference frame.")
    if not isinstance(azimuth_bin_count, int) or azimuth_bin_count < 1:
        raise ValueError("azimuth_bin_count must be a positive integer.")

    observer = np.asarray(observer_position_km, dtype=np.float64)
    up = np.asarray(observer_outward_normal, dtype=np.float64)
    reference_north = np.asarray(reference_north_km, dtype=np.float64)
    points = np.asarray(terrain_points_km, dtype=np.float64)
    if observer.shape != (3,) or not np.all(np.isfinite(observer)) or np.linalg.norm(observer) == 0.0:
        raise ValueError("Observer position must be a finite, nonzero three-component vector in km.")
    if up.shape != (3,) or not np.all(np.isfinite(up)):
        raise ValueError("Observer outward normal must contain three finite components.")
    up_magnitude = float(np.linalg.norm(up))
    if not np.isclose(up_magnitude, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Observer outward normal must be a unit vector.")
    up = up / up_magnitude
    if float(np.dot(up, observer)) <= 0.0:
        raise ValueError("Observer normal must point outward, away from the Moon center.")
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("terrain_points_km must have shape (..., 3), with at least one sample axis.")

    sample_shape = points.shape[:-1]
    flat_points = points.reshape((-1, 3))
    if valid_mask is None:
        flat_valid = np.ones(flat_points.shape[0], dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != sample_shape:
            raise ValueError("valid_mask must have the same shape as the terrain sample grid.")
        flat_valid = mask.ravel().copy()
    flat_valid &= np.all(np.isfinite(flat_points), axis=1)

    if observer_sample_index is not None:
        if isinstance(observer_sample_index, tuple):
            if len(observer_sample_index) != len(sample_shape):
                raise ValueError("Tuple observer_sample_index must have one index per terrain sample axis.")
            try:
                observer_flat_index = int(np.ravel_multi_index(observer_sample_index, sample_shape))
            except (ValueError, IndexError) as exc:
                raise ValueError("observer_sample_index is outside the terrain sample grid.") from exc
        else:
            observer_flat_index = int(observer_sample_index)
            if observer_flat_index < 0:
                observer_flat_index += flat_points.shape[0]
            if not 0 <= observer_flat_index < flat_points.shape[0]:
                raise ValueError("observer_sample_index is outside the terrain sample array.")
        flat_valid[observer_flat_index] = False

    if reference_north.shape != (3,) or not np.all(np.isfinite(reference_north)):
        raise ValueError("reference_north_km must contain three finite components in the terrain frame.")
    reference_length = float(np.linalg.norm(reference_north))
    if reference_length <= 1e-12:
        raise ValueError("reference_north_km must be nonzero.")
    north_reference = reference_north / reference_length
    north = north_reference - float(np.dot(north_reference, up)) * up
    north_length = float(np.linalg.norm(north))
    if north_length <= 1e-12:
        raise ValueError("Reference North is parallel to terrain Up; local azimuth is undefined.")
    north /= north_length
    east = np.cross(north, up)
    east_length = float(np.linalg.norm(east))
    if east_length <= 1e-12:
        raise ValueError("Local East is undefined for the supplied observer normal.")
    east /= east_length

    vectors = flat_points - observer
    ranges = np.linalg.norm(vectors, axis=1)
    flat_valid &= np.isfinite(ranges) & (ranges > 0.0)
    azimuth_centers = np.arange(azimuth_bin_count, dtype=np.float64) * (360.0 / azimuth_bin_count)
    maxima = np.full(azimuth_bin_count, np.nan, dtype=np.float64)
    counts = np.zeros(azimuth_bin_count, dtype=np.int64)
    if np.any(flat_valid):
        local_east = vectors[flat_valid] @ east
        local_north = vectors[flat_valid] @ north
        local_up = vectors[flat_valid] @ up
        horizontal = np.hypot(local_east, local_north)
        directional = np.isfinite(horizontal) & (horizontal > 0.0)
        if np.any(directional):
            azimuth = np.degrees(np.arctan2(local_east[directional], local_north[directional])) % 360.0
            elevation = np.degrees(np.arctan2(local_up[directional], horizontal[directional]))
            bin_width = 360.0 / azimuth_bin_count
            # Assign each sample to the nearest bin center; indices wrap at 360°.
            bin_index = np.floor((azimuth + 0.5 * bin_width) / bin_width).astype(np.int64) % azimuth_bin_count
            counts = np.bincount(bin_index, minlength=azimuth_bin_count).astype(np.int64)
            maxima[:] = -np.inf
            np.maximum.at(maxima, bin_index, elevation)
            maxima[counts == 0] = np.nan

    return TerrainHorizonProfile(
        azimuth_centers_deg=azimuth_centers,
        max_obstruction_elevation_deg=maxima,
        sample_count=counts,
        source_frame=MOON_ME_DE421,
        north_axis=tuple(float(value) for value in north),
        east_axis=tuple(float(value) for value in east),
        up_axis=tuple(float(value) for value in up),
    )
