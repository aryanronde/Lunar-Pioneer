"""Bounded, radius-parameterized regional terrain-window validation for Site06.

This reads only the enclosing native 20 m raster window from the NASA COG.
The resulting profile is explicitly limited-range validation, not a complete
lunar horizon and is not used for Sun/Earth visibility.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import spiceypy as spice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lunar_coordinates import SITE06_NOBILE_RIM_1, planetocentric_latlon_to_moon_me_de421
from core.horizon_report import collect_runtime_provenance, make_horizon_report, make_raster_source_record, serialize_horizon_report, validate_horizon_report
from core.lunar_frame_bridge import LunarFrameBridge
from core.lunar_horizon import LocalHorizonFrame, LunarSite
from core.spice_engine import SpiceGeometryEngine
from core.terrain_horizon import calculate_terrain_horizon_profile
from core.terrain_provider import LOLA20mTerrainProvider, MOON_ME_DE421, TaggedMoonPosition
from core.terrain_region import (
    enclosing_pixel_range,
    radius_window_indices,
    regional_horizon_status,
    regional_sample_masks,
)
from core.terrain_surface import reconstruct_lunar_terrain


PIXEL_SIZE_M = 20.0
SITE06_1KM_CROSSCHECK = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "site06_1km_5m_20m_horizon_crosscheck.json"
)
SITE06_1KM_VERSIONED_CROSSCHECK = (
    Path(__file__).resolve().parents[1] / "reports" / "review" / "site06_1km_5m_20m_horizon_v1.json"
)
REVIEW_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "review"
SITE06_OBSERVER_KM = np.array(
    (110.21143659666053, 84.16382863817336, -1737.8192860716285), dtype=np.float64
)
SITE06_UP_ME = np.array(
    (0.12396688856499274, 0.30047987214444133, -0.9456976562176598), dtype=np.float64
)


def _combined_multiresolution_profile(
    regional_angles: np.ndarray,
    high_resolution_report_path: Path = SITE06_1KM_CROSSCHECK,
) -> np.ndarray:
    """Union saved 5 m inner profile with the native 20 m regional profile."""
    report = json.loads(high_resolution_report_path.read_text(encoding="utf-8-sig"))
    if float(report.get("requested_radius_m", -1.0)) != 1000.0:
        raise ValueError("Saved high-resolution profile must cover the validated 1 km Site06 region.")
    if report.get("source_frame") != MOON_ME_DE421:
        raise ValueError(f"Saved 5 m profile must use {MOON_ME_DE421}.")
    high_values = report.get("profile_bins_deg")
    if not isinstance(high_values, list) or len(high_values) != 360:
        raise ValueError("Saved 5 m profile must contain 360 one-degree bins.")
    high = np.asarray([np.nan if value is None else float(value) for value in high_values], dtype=np.float64)
    if regional_angles.shape != high.shape:
        raise ValueError("Regional and high-resolution horizon profiles must have matching bins.")
    return np.fmax(regional_angles, high)


def _load_versioned_5m_profile(path: Path = SITE06_1KM_VERSIONED_CROSSCHECK) -> tuple[dict, np.ndarray, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_horizon_report(report)
    if float(report["horizon"]["radius_m"]) != 1000.0:
        raise ValueError("Versioned 5 m input must cover exactly the validated 1 km region.")
    frame = report["site"].get("frame", {})
    if frame.get("value") != MOON_ME_DE421:
        raise ValueError(f"Versioned Site06 profile must use {MOON_ME_DE421}.")
    five_m = np.asarray([np.nan if value is None else float(value) for value in report["horizon"]["profile_angles_deg"]])
    counts = np.asarray(report["horizon"]["sample_counts_by_bin"], dtype=np.int64)
    if len(five_m) != 360:
        raise ValueError("Versioned 5 m input must contain 360 one-degree bins.")
    return report, five_m, counts


def _union_with_contributors(regional: np.ndarray, five_m: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
    """Return the unchanged per-bin maximum and all sources tied at that maximum."""
    combined = np.fmax(regional, five_m)
    contributors: list[list[str]] = []
    for a, b, result in zip(five_m, regional, combined):
        ids = []
        if np.isfinite(result):
            if np.isfinite(a) and a == result:
                ids.append("lola_5m")
            if np.isfinite(b) and b == result:
                ids.append("lola_20m")
        contributors.append(ids)
    return combined, contributors


def _nodata_value(value):
    return "NaN" if value is not None and math.isnan(float(value)) else value


def _make_provider_source_record(provider, metadata, window, row_bounds, col_bounds, counts):
    unavailable = {"/source_hash": "Remote COG content was not hashed; hashing it would require transferring the full raster."}
    if metadata.band_unit is None:
        unavailable["/elevation_units"] = "The raster band exposes no unit tag; the existing reconstruction interprets raw heights as meters."
        unavailable["/elevation_units_evidence/value"] = unavailable["/elevation_units"]
    if metadata.nodata is None:
        unavailable["/nodata_value"] = "The opened raster exposes no NoData value; non-finite values are also masked."
    if metadata.scale is None:
        unavailable["/scale"] = "The opened raster exposes no scale metadata."
    if metadata.offset is None:
        unavailable["/offset"] = "The opened raster exposes no offset metadata."
    if not metadata.pixel_registration_observed:
        unavailable["/pixel_registration"] = "The raster AREA_OR_POINT tag was absent; returned arrays still use affine pixel centers."
    return make_raster_source_record(
        source_id="lola_20m",
        source_url=provider.source,
        product_id=provider.source.split("?", 1)[0].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1],
        source_hash=None,
        resolution_m=metadata.pixel_size_m,
        raster_dimensions_rows_columns=(metadata.height, metadata.width),
        dtype=metadata.dtype,
        crs_wkt=metadata.crs,
        affine=metadata.affine,
        pixel_registration=metadata.pixel_registration if metadata.pixel_registration_observed else None,
        elevation_units=metadata.band_unit,
        scale=metadata.scale,
        offset=metadata.offset,
        nodata_value=_nodata_value(metadata.nodata),
        nodata_handling="Mask exact raster NoData value and all non-finite samples; no fill/interpolation.",
        elevation_units_evidence={"status": "raster_metadata" if metadata.band_unit else "unverified",
                                  "value": metadata.band_unit,
                                  "calculation_assumption": "Existing terrain reconstruction interprets raw elevations as meters.",
                                  "scale_offset_applied_by_generator": False},
        elevation=window.elevation_m,
        nodata_mask=window.nodata_mask,
        coverage_counts=counts,
        requested_window={"rows_half_open": row_bounds, "columns_half_open": col_bounds},
        returned_window={"rows_half_open": row_bounds, "columns_half_open": col_bounds},
        source_frame=metadata.source_frame,
        frame_evidence_status="code_assigned",
        frame_evidence={"detail": "TerrainMetadata.source_frame is assigned by provider code, not read from a raster frame tag."},
        unavailable=unavailable,
    )


def build_versioned_report(*, radius_m, status, site_record, sources, five_m_angles, five_m_counts,
                           regional_profile, software, kernels, unavailable):
    combined, contributors = _union_with_contributors(
        regional_profile.max_obstruction_elevation_deg, five_m_angles
    )
    explanations = dict(unavailable)
    explanations["/horizon/algorithm/version"] = "The existing horizon implementation has no semantic version; implementation source hash is recorded."
    algorithm_path = Path(__file__).resolve().parents[1] / "core" / "terrain_horizon.py"
    software = dict(software)
    software["horizon_algorithm_sha256"] = hashlib.sha256(algorithm_path.read_bytes()).hexdigest()
    return make_horizon_report(
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        site=site_record,
        sources=sources,
        radius_m=radius_m,
        coverage_status=status,
        azimuth_bins={"count": 360, "width_deg": 1.0, "centers_deg": list(range(360)),
                      "convention": "0=N, 90=E, 180=S, 270=W; clockwise",
                      "assignment": "nearest 1 degree center with circular wrap"},
        algorithm={"name": "calculate_terrain_horizon_profile", "version": None,
                   "implementation_sha256": software["horizon_algorithm_sha256"]},
        profile_angles_deg=[float(v) if np.isfinite(v) else None for v in combined],
        sample_counts_by_bin=[int(a + b) for a, b in zip(five_m_counts, regional_profile.sample_count)],
        contributing_source_ids_by_bin=contributors,
        composite_rule="Per-bin maximum union of the saved native 5 m 1 km profile and valid native 20 m samples within the requested radius; ties list both sources.",
        unavailable=explanations,
        software=software,
        spice_kernels=kernels,
        limitations=["Finite-radius horizon is limited-range and is not a complete lunar horizon.",
                     "The 5 m source metadata describes its separately read 1 km window; frame evidence remains code-assigned unless independently established."],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radius-km", type=float, default=5.0,
        help="Requested centered terrain-horizon radius in km (bounded by provider window size).",
    )
    args = parser.parse_args(argv)
    five_m_report, five_m_angles, five_m_counts = _load_versioned_5m_profile()
    if not math.isfinite(args.radius_km) or args.radius_km <= 0:
        parser.error("--radius-km must be positive and finite")
    radius_m = args.radius_km * 1000.0
    site_xyz = planetocentric_latlon_to_moon_me_de421(
        SITE06_NOBILE_RIM_1.latitude_deg,
        SITE06_NOBILE_RIM_1.longitude_deg,
        1737.4,
    )
    tagged_site = TaggedMoonPosition(*site_xyz, frame=MOON_ME_DE421)
    with LOLA20mTerrainProvider() as provider:
        metadata = provider.metadata
        center_x, center_y = provider.site_position_to_dem_xy(tagged_site)
        # The smallest cell-aligned rectangle containing the requested circle.
        top_row, bottom_row, left_col, right_col = radius_window_indices(
            center_x,
            center_y,
            radius_m,
            metadata,
            max_window_side_px=provider.MAX_WINDOW_SIDE_PX,
        )
        width = right_col - left_col
        height = bottom_row - top_row
        # Use the pixel coordinate of Site06 to anchor the provider window, then
        # assert that this exact enclosing window agrees with the affine-derived bounds.
        window_center_x = metadata.affine[2] + (left_col + width / 2.0) * metadata.affine[0]
        window_center_y = metadata.affine[5] + (top_row + height / 2.0) * metadata.affine[4]
        window = provider.read_window(window_center_x, window_center_y, width, height)
        actual_rows = (window.row_offset, window.row_offset + window.elevation_m.shape[0])
        actual_cols = (window.column_offset, window.column_offset + window.elevation_m.shape[1])
        if actual_rows != (top_row, bottom_row) or actual_cols != (left_col, right_col):
            raise RuntimeError(
                f"Provider returned rows {actual_rows}, columns {actual_cols}; expected "
                f"rows {(top_row, bottom_row)}, columns {(left_col, right_col)}."
            )

        x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
        within_circle, in_coverage = regional_sample_masks(
            x_grid,
            y_grid,
            center_x,
            center_y,
            radius_m,
            metadata.bounds_m,
        )
        covered = within_circle & in_coverage
        valid = covered & ~window.nodata_mask & np.isfinite(window.elevation_m)
        nodata = covered & (window.nodata_mask | ~np.isfinite(window.elevation_m))
        out_of_coverage = within_circle & ~in_coverage
        coverage_status = regional_horizon_status(within_circle, in_coverage, window.nodata_mask, window.elevation_m)
        if not valid.any():
            raise RuntimeError("The bounded Site06 region contained no valid elevation samples.")

        surface = reconstruct_lunar_terrain(window, metadata)
        surface_valid = surface.valid_mask & valid
        positions = surface.positions_km.copy()
        positions[~surface_valid] = np.nan

        # Recover the actual V0.2A axes at Site06 from the same PCK ellipsoid
        # construction used in the local-frame audit, then bridge the exact
        # North vector into the DEM's ME/DE421 frame.
        with SpiceGeometryEngine():
            et = spice.str2et("2026-10-05T00:00:00")
            with LunarFrameBridge() as bridge:
                observer_pa = np.asarray(bridge.vector_dem_to_geometry(tuple(SITE06_OBSERVER_KM), et))
                radii = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
                ray = observer_pa / np.linalg.norm(observer_pa)
                ellipsoid_point = ray / math.sqrt(float(np.sum((ray / radii) ** 2)))
                pck_normal = ellipsoid_point / (radii * radii)
                pck_normal /= np.linalg.norm(pck_normal)
                lat = math.degrees(math.atan2(pck_normal[2], math.hypot(pck_normal[0], pck_normal[1])))
                lon = math.degrees(math.atan2(pck_normal[1], pck_normal[0])) % 360.0
                v02a = LocalHorizonFrame.at_site(LunarSite(lat, lon))
                north_me = bridge.vector_geometry_to_dem(v02a.north, et)

        # Use Site06's previously validated terrain point and measured terrain
        # normal as observer origin and Up. Exclude the containing 20 m pixel;
        # its center is not identical to the 5 m Site06 observer point.
        site_row, site_col = 9699, 19400
        center_pixel_x = metadata.affine[2] + (site_col + 0.5) * metadata.affine[0]
        center_pixel_y = metadata.affine[5] + (site_row + 0.5) * metadata.affine[4]
        center_pixel_distance_m = math.hypot(center_pixel_x - center_x, center_pixel_y - center_y)
        local_index = (site_row - top_row, site_col - left_col)
        observer_index = local_index if 0 <= local_index[0] < height and 0 <= local_index[1] < width else None
        profile = calculate_terrain_horizon_profile(
            SITE06_OBSERVER_KM,
            SITE06_UP_ME,
            positions,
            observer_frame=MOON_ME_DE421,
            terrain_frame=MOON_ME_DE421,
            reference_north_km=north_me,
            valid_mask=surface_valid,
            observer_sample_index=observer_index,
        )
        combined_profile, contributors = _union_with_contributors(
            profile.max_obstruction_elevation_deg, five_m_angles
        )

        bounds_left, bounds_bottom, bounds_right, bounds_top = metadata.bounds_m
        bound_distances_m = {
            "left": center_x - bounds_left,
            "right": bounds_right - center_x,
            "bottom": center_y - bounds_bottom,
            "top": bounds_top - center_y,
        }
        rectangular_radius_limit_m = min(
            bound_distances_m.values(),
        )
        # The named LDEM_80S product's nominal polar-cap edge is 80 degrees S.
        # Stereographic radius from the pole on the product sphere is
        # 2 R tan(5 degrees); this is a geometric upper bound, not a statement
        # that every edge pixel has valid elevation.
        eighty_s_radius_m = 2.0 * 1_737_400.0 * math.tan(math.radians(5.0))
        site_projected_radius_m = math.hypot(center_x, center_y)
        polar_cap_radius_limit_m = max(0.0, eighty_s_radius_m - site_projected_radius_m)
        geometric_radius_limit_m = min(rectangular_radius_limit_m, polar_cap_radius_limit_m)

        report = {
            "classification": f"{args.radius_km:g} km {coverage_status} terrain horizon validation; NOT a complete lunar horizon",
            "source": provider.source,
            "source_frame": metadata.source_frame,
            "resolution_m": metadata.pixel_size_m,
            "site_dem_xy_m": [center_x, center_y],
            "site_pixel_row_column": [site_row, site_col],
            "site_center_pixel_present": observer_index is not None,
            "site_pixel_center_xy_m": [center_pixel_x, center_pixel_y],
            "site_to_pixel_center_distance_m": center_pixel_distance_m,
            "requested_window_rows_half_open": [top_row, bottom_row],
            "requested_window_columns_half_open": [left_col, right_col],
            "window_dimensions_rows_columns": [height, width],
            "returned_window_rows_half_open": list(actual_rows),
            "returned_window_columns_half_open": list(actual_cols),
            "requested_radius_m": radius_m,
            "horizon_status": coverage_status,
            "coverage_note": "A finite DEM radius is limited-range even when the requested circle is fully covered; no global completeness is inferred.",
            "product_bounds_m": list(metadata.bounds_m),
            "site_distance_to_each_product_bound_m": bound_distances_m,
            "site_projected_distance_from_pole_m": site_projected_radius_m,
            "site_to_nearest_rectangular_bound_m": rectangular_radius_limit_m,
            "nominal_80S_projected_radius_m": eighty_s_radius_m,
            "site_to_nominal_80S_edge_radius_limit_m": polar_cap_radius_limit_m,
            "geometric_centered_circle_radius_limit_m": geometric_radius_limit_m,
            "current_provider_max_centered_radius_m": (provider.MAX_WINDOW_SIDE_PX * PIXEL_SIZE_M - PIXEL_SIZE_M) / 2.0,
            "circle_sample_cells": int(within_circle.sum()),
            "valid_elevation_samples": int(valid.sum()),
            "nodata_or_nan_samples": int(nodata.sum()),
            "out_of_coverage_samples": int(out_of_coverage.sum()),
            "requested_square_window_pixel_count": int(width * height),
            "elevation_min_max_mean_m": [
                float(np.min(window.elevation_m[valid])),
                float(np.max(window.elevation_m[valid])),
                float(np.mean(window.elevation_m[valid], dtype=np.float64)),
            ],
            "observer_pixel_excluded": observer_index is not None,
            "horizon_sample_count": int(profile.sample_count.sum()),
            "max_obstruction_angle_deg": float(np.nanmax(profile.max_obstruction_elevation_deg)),
            "maximum_bins": [
                {"azimuth_deg": float(profile.azimuth_centers_deg[i]), "elevation_deg": float(profile.max_obstruction_elevation_deg[i])}
                for i in np.flatnonzero(
                    np.isclose(profile.max_obstruction_elevation_deg, np.nanmax(profile.max_obstruction_elevation_deg), atol=1e-12)
                )
            ],
            "profile_bins_1deg_maximum_obstruction_deg": [
                None if not np.isfinite(angle) else float(angle)
                for angle in profile.max_obstruction_elevation_deg
            ],
            "profile_bins_1deg_5m_1km_plus_20m_regional_deg": [
                None if not np.isfinite(angle) else float(angle)
                for angle in combined_profile
            ],
            "profile_union_method": "Per-bin maximum union of the versioned saved native-resolution 5 m 1 km profile and all valid native 20 m samples within the requested radius.",
        }
        five_m_source = next(source for source in five_m_report["sources"] if source["source_id"] == "lola_5m")
        counts = {
            "requested_area_pixel_count": int(within_circle.sum()),
            "valid_pixel_count": int(valid.sum()),
            "nodata_pixel_count": int(nodata.sum()),
            "other_invalid_pixel_count": 0,
            "out_of_coverage_pixel_count": int(out_of_coverage.sum()),
        }
        source20 = _make_provider_source_record(
            provider, metadata, window, (top_row, bottom_row), (left_col, right_col), counts
        )
        site_record = {
            "id": SITE06_NOBILE_RIM_1.name,
            "geographic": {"latitude_deg": SITE06_NOBILE_RIM_1.latitude_deg,
                           "longitude_deg": SITE06_NOBILE_RIM_1.longitude_deg,
                           "latitude_type": SITE06_NOBILE_RIM_1.latitude_type,
                           "longitude_convention": SITE06_NOBILE_RIM_1.longitude_convention,
                           "coordinate_source": SITE06_NOBILE_RIM_1.coordinate_source},
            "projected_xy_m": [center_x, center_y],
            "projected_xy_method": "Site planetocentric latitude/longitude -> MOON_ME_DE421 Cartesian position at 1,737.4 km -> opened raster CRS via pyproj.",
            "observer_xyz_km": SITE06_OBSERVER_KM.tolist(),
            "observer_up": SITE06_UP_ME.tolist(),
            "frame": {"value": MOON_ME_DE421, "evidence_status": "code_assigned",
                      "evidence": {"kind": "application_coordinate_record", "coordinate_source": SITE06_NOBILE_RIM_1.coordinate_source,
                                   "detail": "Site/observer coordinates are tagged in application code; source frame is recorded per raster separately."}},
            "north_source": "V0.2A LocalHorizonFrame North transformed MOON_PA_DE440 to MOON_ME_DE421 with local SPICE kernels.",
        }
        source_records = [five_m_source, source20]
        spice_names = tuple(dict.fromkeys(SpiceGeometryEngine.KERNEL_NAMES + LunarFrameBridge.KERNEL_NAMES))
        software, kernels, unavailable = collect_runtime_provenance(
            Path(__file__), Path(__file__).resolve().parents[1] / "data" / "spice", spice_names
        )
        versioned = build_versioned_report(
            radius_m=radius_m, status=coverage_status, site_record=site_record,
            sources=source_records, five_m_angles=five_m_angles, five_m_counts=five_m_counts,
            regional_profile=profile, software=software, kernels=kernels, unavailable=unavailable,
        )
        report_path = REVIEW_REPORT_DIR / f"site06_{args.radius_km:g}km_limited_horizon_v1.json"
        serialize_horizon_report(versioned, report_path)
        report["versioned_review_report"] = str(report_path)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
