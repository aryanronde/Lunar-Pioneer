"""Verify the map and 10 km horizon against one shared Site06 TerrainWindow.

This performs one bounded 15 km dashboard-window read from the configured
20 m LOLA COG. The map renderer and existing horizon calculation receive the
same TerrainWindow object. It writes only versioned review outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import spiceypy as spice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.horizon_report import (
    collect_runtime_provenance,
    serialize_horizon_report,
    validate_horizon_report,
)
from core.lunar_frame_bridge import LunarFrameBridge
from core.lunar_horizon import LocalHorizonFrame, LunarSite
from core.spice_engine import SpiceGeometryEngine
from core.terrain_horizon import calculate_terrain_horizon_profile
from core.terrain_provider import MOON_ME_DE421
from core.terrain_region import regional_horizon_status, regional_sample_masks
from core.terrain_surface import reconstruct_lunar_terrain
from scripts.validate_site06_regional_horizon import (
    SITE06_OBSERVER_KM,
    SITE06_UP_ME,
    _make_provider_source_record,
    _load_versioned_5m_profile,
    build_versioned_report,
)
from ui_mission_map import (
    build_site_map_figure,
    build_site_map_geometry,
    load_site_terrain_window,
)
from core.lunar_coordinates import SITE06_NOBILE_RIM_1


VERSIONED_1KM = ROOT / "reports" / "review" / "site06_1km_5m_20m_horizon_v1.json"
VERSIONED_10KM = ROOT / "reports" / "review" / "site06_10km_limited_horizon_v1.json"
PRODUCTION_10KM = ROOT / "reports" / "site06_10km_limited_horizon.json"
CONSISTENCY_REPORT = ROOT / "reports" / "review" / "site06_map_horizon_consistency_v1.json"
RADIUS_M = 10_000.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    high_report, five_m_angles, five_m_counts = _load_versioned_5m_profile(VERSIONED_1KM)
    old_report = json.loads(VERSIONED_10KM.read_text(encoding="utf-8"))
    production = json.loads(PRODUCTION_10KM.read_text(encoding="utf-8-sig"))

    geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, RADIUS_M)
    map_data = load_site_terrain_window(SITE06_NOBILE_RIM_1, RADIUS_M)
    window = map_data.window
    metadata = map_data.metadata

    # Confirm the renderer consumes precisely the cached window object used below.
    figure = build_site_map_figure(SITE06_NOBILE_RIM_1, RADIUS_M, terrain_data=map_data)
    map_z = np.asarray(figure.data[0].z, dtype=np.float64)
    expected_z = np.asarray(window.elevation_m, dtype=np.float64).copy()
    expected_z[np.asarray(window.nodata_mask, dtype=bool) | ~np.isfinite(expected_z)] = np.nan
    if not np.array_equal(map_z, expected_z, equal_nan=True):
        raise RuntimeError("Rendered map elevation values differ from the cached TerrainWindow.")

    rows = (window.row_offset, window.row_offset + window.elevation_m.shape[0])
    cols = (window.column_offset, window.column_offset + window.elevation_m.shape[1])
    window_identity = {
        "source_id": "lola_20m",
        "source_url": map_data.source,
        "rows_half_open": list(rows),
        "columns_half_open": list(cols),
        "shape_rows_columns": list(window.elevation_m.shape),
        "crs_wkt_sha256": _sha256_bytes(metadata.crs.encode("utf-8")),
        "affine": list(metadata.affine),
        "raw_elevation_window_sha256": _sha256_bytes(np.ascontiguousarray(window.elevation_m).tobytes()),
        "nodata_mask_sha256": _sha256_bytes(np.ascontiguousarray(window.nodata_mask, dtype=np.uint8).tobytes()),
        "x_pixel_centers_sha256": _sha256_bytes(np.ascontiguousarray(window.x_centers_m, dtype=np.float64).tobytes()),
        "y_pixel_centers_sha256": _sha256_bytes(np.ascontiguousarray(window.y_centers_m, dtype=np.float64).tobytes()),
        "same_in_process_terrainwindow_object_for_map_and_horizon": True,
    }
    window_identity["window_id"] = (
        f"lola_20m:{rows[0]}:{rows[1]}:{cols[0]}:{cols[1]}:{window_identity['raw_elevation_window_sha256'][:16]}"
    )

    affine = metadata.affine
    x_expected = affine[2] + (np.arange(cols[0], cols[1], dtype=np.float64) + 0.5) * affine[0]
    y_expected = affine[5] + (np.arange(rows[0], rows[1], dtype=np.float64) + 0.5) * affine[4]
    centers_match_affine = bool(
        np.array_equal(window.x_centers_m, x_expected)
        and np.array_equal(window.y_centers_m, y_expected)
    )
    if not centers_match_affine:
        raise RuntimeError("TerrainWindow pixel-center coordinates do not match its source affine.")

    x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
    within_circle, in_coverage = regional_sample_masks(
        x_grid, y_grid, geometry.center_x_m, geometry.center_y_m, RADIUS_M, metadata.bounds_m
    )
    elevation = np.asarray(window.elevation_m)
    missing = np.asarray(window.nodata_mask, dtype=bool) | ~np.isfinite(elevation)
    valid = within_circle & in_coverage & ~missing
    nodata = within_circle & in_coverage & missing
    out_of_coverage = within_circle & ~in_coverage
    status = regional_horizon_status(within_circle, in_coverage, window.nodata_mask, elevation)
    surface = reconstruct_lunar_terrain(window, metadata)
    surface_valid = surface.valid_mask & valid
    positions = surface.positions_km.copy()
    positions[~surface_valid] = np.nan

    # Keep the existing north construction and horizon function unchanged.
    with SpiceGeometryEngine():
        et = spice.str2et("2026-10-05T00:00:00")
        with LunarFrameBridge() as bridge:
            observer_pa = np.asarray(bridge.vector_dem_to_geometry(tuple(SITE06_OBSERVER_KM), et))
            radii_km = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
            ray = observer_pa / np.linalg.norm(observer_pa)
            ellipsoid_point = ray / np.sqrt(float(np.sum((ray / radii_km) ** 2)))
            normal = ellipsoid_point / (radii_km * radii_km)
            normal /= np.linalg.norm(normal)
            latitude = np.degrees(np.arctan2(normal[2], np.hypot(normal[0], normal[1])))
            longitude = np.degrees(np.arctan2(normal[1], normal[0])) % 360.0
            north_me = bridge.vector_geometry_to_dem(
                LocalHorizonFrame.at_site(LunarSite(float(latitude), float(longitude))).north, et
            )

    site_row, site_col = 9699, 19400
    local_observer_index = (site_row - window.row_offset, site_col - window.column_offset)
    observer_index = (
        local_observer_index
        if 0 <= local_observer_index[0] < elevation.shape[0]
        and 0 <= local_observer_index[1] < elevation.shape[1]
        else None
    )
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

    old_20m = np.asarray([
        np.nan if value is None else float(value)
        for value in old_report["horizon"]["profile_angles_deg"]
    ])
    old_20m_source = next(s for s in old_report["sources"] if s["source_id"] == "lola_20m")
    # The old composite may have a 5 m winner. Its pure 20 m profile is also
    # present in the original generator's local production detail report.
    old_20m = np.asarray([
        np.nan if value is None else float(value)
        for value in production["profile_bins_1deg_maximum_obstruction_deg"]
    ])
    common = np.isfinite(old_20m) & np.isfinite(profile.max_obstruction_elevation_deg)
    delta = profile.max_obstruction_elevation_deg[common] - old_20m[common]
    profile_comparison = {
        "comparison_source": str(PRODUCTION_10KM.relative_to(ROOT)),
        "profile_bins_compared": int(common.sum()),
        "maximum_absolute_20m_profile_difference_deg": float(np.max(np.abs(delta))) if delta.size else None,
        "rms_20m_profile_difference_deg": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
        "exactly_equal_finite_bins": int(np.sum(delta == 0.0)),
        "old_report_pixel_arrays_available": False,
        "old_report_pixel_arrays_limitation": "The prior versioned horizon report stores per-window statistics, not its raw samples or NoData mask.",
    }

    map_top = {"rows_half_open": list(rows), "columns_half_open": list(cols)}
    old_rows = old_20m_source["returned_window"]["rows_half_open"]
    old_cols = old_20m_source["returned_window"]["columns_half_open"]
    overlap = {
        "rows_half_open": [max(rows[0], old_rows[0]), min(rows[1], old_rows[1])],
        "columns_half_open": [max(cols[0], old_cols[0]), min(cols[1], old_cols[1])],
    }
    overlap["shape_rows_columns"] = [
        overlap["rows_half_open"][1] - overlap["rows_half_open"][0],
        overlap["columns_half_open"][1] - overlap["columns_half_open"][0],
    ]
    raw_window_counts = {
        "map_window_pixels": int(elevation.size),
        "map_window_nodata_or_nonfinite_pixels": int(missing.sum()),
        "analysis_circle_pixels": int(within_circle.sum()),
        "analysis_circle_valid_pixels_before_observer_exclusion": int(valid.sum()),
        "analysis_circle_nodata_or_nonfinite_pixels": int(nodata.sum()),
        "analysis_circle_out_of_coverage_pixels": int(out_of_coverage.sum()),
        "horizon_samples_after_observer_exclusion": int(profile.sample_count.sum()),
    }

    source_record = _make_provider_source_record(
        SimpleNamespace(source=map_data.source),
        metadata,
        window,
        rows,
        cols,
        {
            "requested_area_pixel_count": int(within_circle.sum()),
            "valid_pixel_count": int(valid.sum()),
            "nodata_pixel_count": int(nodata.sum()),
            "other_invalid_pixel_count": 0,
            "out_of_coverage_pixel_count": int(out_of_coverage.sum()),
        },
    )
    source_record["window_identity"] = window_identity
    source_records = [
        next(s for s in high_report["sources"] if s["source_id"] == "lola_5m"),
        source_record,
    ]
    site_record = dict(old_report["site"])
    site_record["projected_xy_m"] = [geometry.center_x_m, geometry.center_y_m]

    kernels_names = tuple(dict.fromkeys(SpiceGeometryEngine.KERNEL_NAMES + LunarFrameBridge.KERNEL_NAMES))
    software, kernels, unavailable = collect_runtime_provenance(
        ROOT / "scripts" / "validate_site06_regional_horizon.py",
        ROOT / "data" / "spice",
        kernels_names,
    )
    software["map_horizon_consistency_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = build_versioned_report(
        radius_m=RADIUS_M,
        status=status,
        site_record=site_record,
        sources=source_records,
        five_m_angles=five_m_angles,
        five_m_counts=five_m_counts,
        regional_profile=profile,
        software=software,
        kernels=kernels,
        unavailable=unavailable,
    )
    report["map_horizon_window_identity"] = window_identity
    report["map_horizon_window_overlap"] = overlap
    report["map_horizon_20m_profile_comparison"] = profile_comparison
    report["map_horizon_window_counts"] = raw_window_counts
    report["map_horizon_alignment"] = {
        "map_and_horizon_received_same_terrainwindow_object": True,
        "map_rendered_elevations_match_raw_window_after_nodata_masking": True,
        "pixel_center_coordinates_match_affine": centers_match_affine,
        "site_marker_xy_m": [geometry.center_x_m, geometry.center_y_m],
        "site_pixel_row_column": [site_row, site_col],
        "site_pixel_center_xy_m": [
            affine[2] + (site_col + 0.5) * affine[0],
            affine[5] + (site_row + 0.5) * affine[4],
        ],
    }
    validate_horizon_report(report)
    serialize_horizon_report(report, VERSIONED_10KM)

    consistency = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "window_identity": window_identity,
        "map_window": map_top,
        "prior_horizon_20m_window": {
            "rows_half_open": old_rows,
            "columns_half_open": old_cols,
        },
        "common_pixels": overlap,
        "crs_identical_to_prior_horizon_source": metadata.crs == old_20m_source["crs_wkt"],
        "affine_identical_to_prior_horizon_source": list(metadata.affine) == old_20m_source["affine"],
        "source_identity_identical_to_prior_horizon_source": map_data.source == old_20m_source["source_url"],
        "raw_values_and_masks_compared_with_prior_horizon_window": False,
        "prior_profile_comparison": profile_comparison,
        "window_counts": raw_window_counts,
        "site_alignment": report["map_horizon_alignment"],
        "limitations": [
            "The prior horizon report did not preserve raw 20 m pixel arrays or per-pixel masks, so historical pixel-by-pixel equality cannot be tested.",
            "The current map and horizon calculation used one identical in-process TerrainWindow object; its bounded raw values and mask hashes are recorded.",
        ],
    }
    CONSISTENCY_REPORT.write_text(json.dumps(consistency, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "versioned_horizon_report": str(VERSIONED_10KM),
        "consistency_report": str(CONSISTENCY_REPORT),
        "window_id": window_identity["window_id"],
        "common_pixels": overlap,
        "profile_comparison": profile_comparison,
        "window_counts": raw_window_counts,
        "alignment": report["map_horizon_alignment"],
    }, indent=2))


if __name__ == "__main__":
    main()
