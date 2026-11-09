"""Synthetic tests for the Site07 review-only terrain-horizon path."""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np
from pyproj import CRS

from core.horizon_report import validate_horizon_report
from core.terrain_horizon import calculate_terrain_horizon_profile
from core.terrain_provider import MOON_ME_DE421, TerrainMetadata, TerrainWindow
from scripts.review_site07_5km_limited_horizon import (
    analyze_window,
    build_review_report,
    circle_mask,
    select_observer_cell,
)
from scripts.review_site07_5km_row_slab_audit import assemble_terrain_window_from_slabs


ROOT = Path(__file__).resolve().parents[1]
RADIUS_KM = 1737.4
CRS_WKT = CRS.from_proj4(
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs +type=crs"
).to_wkt()
AFFINE = (5.0, 0.0, 22000.0, 0.0, -5.0, -12000.0)


def metadata() -> TerrainMetadata:
    return TerrainMetadata(
        crs=CRS_WKT,
        projection="Polar Stereographic",
        affine=AFFINE,
        width=3200,
        height=3200,
        pixel_size_m=(5.0, 5.0),
        nodata=math.nan,
        vertical_reference="UNVERIFIED",
        source_frame=MOON_ME_DE421,
        bounds_m=(22000.0, -28000.0, 38000.0, -12000.0),
        pixel_registration="PixelIsArea; pixel centers",
        scale=1.0,
        offset=0.0,
        dtype="float32",
        band_unit=None,
    )


def small_window(elevation: np.ndarray | None = None, nodata: np.ndarray | None = None) -> TerrainWindow:
    data = np.zeros((5, 5), dtype=np.float32) if elevation is None else np.asarray(elevation, dtype=np.float32)
    nodata_mask = np.isnan(data) if nodata is None else np.asarray(nodata, dtype=bool)
    return TerrainWindow(
        requested_center_x_m=30000.19238525766,
        requested_center_y_m=-20000.152817410097,
        x_centers_m=22000.0 + (np.arange(1598, 1603) + 0.5) * 5.0,
        y_centers_m=-12000.0 - (np.arange(1598, 1603) + 0.5) * 5.0,
        elevation_m=data,
        nodata_mask=nodata_mask,
        row_offset=1598,
        column_offset=1598,
    )


class Site07ReviewHorizonTest(unittest.TestCase):
    def test_row_slab_assembly_preserves_samples_indices_and_pixel_centers(self) -> None:
        first = np.arange(6, dtype=np.float32).reshape(2, 3)
        second = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
        window = assemble_terrain_window_from_slabs(
            iter(((10, first), (12, second))),
            row_start=10, row_stop=14, column_start=20, column_stop=23,
            center_x_m=102.5, center_y_m=194.5,
            affine=(5.0, 0.0, 0.0, 0.0, -5.0, 250.0),
        )
        np.testing.assert_array_equal(window.elevation_m, np.arange(12, dtype=np.float32).reshape(4, 3))
        self.assertEqual((window.row_offset, window.column_offset), (10, 20))
        np.testing.assert_array_equal(window.x_centers_m, [102.5, 107.5, 112.5])
        np.testing.assert_array_equal(window.y_centers_m, [197.5, 192.5, 187.5, 182.5])

    def test_row_slab_assembly_rejects_gap_or_duplicate(self) -> None:
        with self.assertRaises(ValueError):
            assemble_terrain_window_from_slabs(
                iter(((10, np.zeros((1, 2))), (12, np.zeros((1, 2))))),
                row_start=10, row_stop=12, column_start=20, column_stop=22,
                center_x_m=0.0, center_y_m=0.0, affine=(5.0, 0.0, 0.0, 0.0, -5.0, 0.0),
            )

    def test_circular_mask_excludes_square_corners(self) -> None:
        window = small_window()
        mask = circle_mask(window, window.x_centers_m[2], window.y_centers_m[2], 5.0)
        self.assertEqual(int(mask.sum()), 5)
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[4, 4])

    def test_observer_cell_selection_reports_center_offset(self) -> None:
        window = small_window()
        center = (30000.19238525766, -20000.152817410097)
        cell = select_observer_cell(window, metadata(), center)
        self.assertEqual((cell.row, cell.column), (1600, 1600))
        self.assertEqual((cell.local_row, cell.local_column), (2, 2))
        self.assertAlmostEqual(cell.pixel_center_x_m, 30002.5)
        self.assertAlmostEqual(cell.pixel_center_y_m, -20002.5)
        self.assertAlmostEqual(cell.offset_from_published_center_m, 3.2915577, places=6)

    def test_invalid_observer_normal_stencil_is_rejected(self) -> None:
        missing = np.zeros((5, 5), dtype=bool)
        missing[2, 3] = True
        window = small_window(nodata=missing)
        center = (window.x_centers_m[2], window.y_centers_m[2])
        with self.assertRaisesRegex(ValueError, "normal stencil is invalid"):
            analyze_window(window, metadata(), center, (0.0, 0.0, 1.0), radius_m=100.0)

    def test_known_east_ridge_has_expected_obstruction(self) -> None:
        elevation = np.zeros((5, 5), dtype=np.float32)
        elevation[2, 4] = 100.0
        window = small_window(elevation)
        meta = metadata()
        center = (window.x_centers_m[2], window.y_centers_m[2])
        from core.terrain_surface import calculate_terrain_normals, reconstruct_lunar_terrain

        surface = reconstruct_lunar_terrain(window, meta)
        up = calculate_terrain_normals(surface).normals[2, 2]
        ridge_vector = surface.positions_km[2, 4] - surface.positions_km[2, 2]
        horizontal = ridge_vector - np.dot(ridge_vector, up) * up
        east = horizontal / np.linalg.norm(horizontal)
        north = np.cross(up, east)
        result = analyze_window(window, meta, center, north, radius_m=100.0).horizon_profile
        expected = math.degrees(math.atan2(float(np.dot(ridge_vector, up)), float(np.linalg.norm(horizontal))))
        self.assertEqual(result.sample_count[90], 2)
        self.assertAlmostEqual(result.max_obstruction_elevation_deg[90], expected, places=8)

    def test_azimuth_samples_wrap_across_zero(self) -> None:
        observer = np.array((RADIUS_KM, 0.0, 0.0))
        up = np.array((1.0, 0.0, 0.0))
        north = np.array((0.0, 0.0, 1.0))
        east = np.cross(north, up)

        def point(azimuth_deg: float, distance_km: float, rise_km: float) -> np.ndarray:
            a = math.radians(azimuth_deg)
            return observer + distance_km * (math.cos(a) * north + math.sin(a) * east) + rise_km * up

        profile = calculate_terrain_horizon_profile(
            observer, up,
            np.array((point(359.8, 1.0, 0.1), point(0.2, 2.0, 0.3))),
            observer_frame=MOON_ME_DE421, terrain_frame=MOON_ME_DE421,
            reference_north_km=north,
        )
        self.assertEqual(profile.sample_count[0], 2)
        self.assertAlmostEqual(profile.max_obstruction_elevation_deg[0], math.degrees(math.atan2(0.3, 2.0)))

    def test_report_preserves_provenance_and_unresolved_metadata(self) -> None:
        site_record = json.loads((ROOT / "reports/review/site07_source_record_review_v1.json").read_text(encoding="utf-8"))
        raster_record = json.loads((ROOT / "reports/review/site07_raster_metadata_review_v1.json").read_text(encoding="utf-8"))
        window = small_window()
        meta = metadata()
        center = (30000.19238525766, -20000.152817410097)
        analysis = analyze_window(window, meta, center, (0.0, 0.0, 1.0), radius_m=100.0)
        report = build_review_report(
            window, meta, site_record, raster_record, analysis,
            reference_north_km=np.array((0.0, 0.0, 1.0)),
            north_epoch_utc="2026-10-05T00:00:00Z",
            north_site_latlon_deg={"latitude_deg": -88.8, "longitude_deg": 123.7},
            software={"python": "synthetic-test"},
            spice_kernels=[],
            generated_at_utc="2026-11-03T00:00:00Z",
        )
        validate_horizon_report(report)
        self.assertEqual(report["site"]["geographic"]["latitude_deg"], -88.811)
        self.assertAlmostEqual(report["site"]["observer_cell"]["offset_from_published_center_m"], 3.2915577, places=6)
        self.assertEqual(report["site"]["observer_normal_method"]["name"], "calculate_terrain_normals")
        self.assertEqual(report["horizon"]["radius_m"], 5000.0)
        self.assertEqual(report["horizon"]["coverage_status"], "limited-range")
        self.assertIsNone(report["sources"][0]["elevation_units"])
        self.assertIsNone(report["sources"][0]["vertical_reference"])
        self.assertEqual(report["sources"][0]["source_frame"]["evidence_status"], "code_assigned")
        self.assertIn("meters", report["site"]["elevation_assumption"])
        self.assertTrue(any("not a complete lunar horizon" in item for item in report["limitations"]))
        self.assertEqual(report["horizon"]["comparison_profiles"], [])


if __name__ == "__main__":
    unittest.main()
