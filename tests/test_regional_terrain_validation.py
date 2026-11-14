"""Pure tests for bounded regional raster bounds and coverage masks."""

import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

from scripts.validate_site06_regional_horizon import enclosing_pixel_range, regional_sample_masks
from scripts.validate_site06_regional_horizon import _combined_multiresolution_profile
from core.terrain_provider import MOON_ME_DE421, TerrainMetadata
from core.terrain_region import radius_window_indices, regional_horizon_status
from core.terrain_visibility import SITE06_HORIZON_RANGE_KM, load_site06_horizon_profile


class RegionalTerrainValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = TerrainMetadata(
            crs="synthetic stereographic",
            projection="Polar Stereographic",
            affine=(20.0, 0.0, -304000.0, 0.0, -20.0, 304000.0),
            width=30400,
            height=30400,
            pixel_size_m=(20.0, 20.0),
            nodata=-9999.0,
            vertical_reference="synthetic",
            source_frame=MOON_ME_DE421,
            bounds_m=(-304000.0, -304000.0, 304000.0, 304000.0),
            pixel_registration="Area",
        )

    def test_radius_is_parameterized_for_2_5_and_10_kilometers(self) -> None:
        expected_side = {2.0: 201, 5.0: 501, 10.0: 1001}
        for radius_km, side in expected_side.items():
            with self.subTest(radius_km=radius_km):
                row_start, row_stop, col_start, col_stop = radius_window_indices(
                    84000.01002578877,
                    110000.01079172149,
                    radius_km * 1000.0,
                    self.metadata,
                    max_window_side_px=2048,
                )
                self.assertEqual((row_stop - row_start, col_stop - col_start), (side, side))

    def test_radius_window_rejects_raster_bound_and_provider_limit_overruns(self) -> None:
        with self.assertRaisesRegex(ValueError, "raster bounds"):
            radius_window_indices(84000.0, 110000.0, 200000.0, self.metadata, max_window_side_px=2048)
        with self.assertRaisesRegex(ValueError, "provider limit"):
            radius_window_indices(84000.0, 110000.0, 21000.0, self.metadata, max_window_side_px=2048)

    def test_cell_aligned_window_encloses_site06_five_kilometer_extent(self) -> None:
        center_x, center_y = 84000.01002578877, 110000.01079172149
        left, right = enclosing_pixel_range(center_x - 5000, center_x + 5000, -304000, 20)
        top, bottom = enclosing_pixel_range(center_y + 5000, center_y - 5000, 304000, -20)
        self.assertEqual((top, bottom, left, right), (9449, 9950, 19150, 19651))
        self.assertEqual((bottom - top, right - left), (501, 501))
        self.assertTrue(top <= 9699 < bottom)
        self.assertTrue(left <= 19400 < right)

    def test_circle_and_dem_coverage_masks_count_uncovered_cells(self) -> None:
        x_grid, y_grid = np.meshgrid(np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0]))
        circle, coverage = regional_sample_masks(
            x_grid,
            y_grid,
            center_x_m=0.0,
            center_y_m=0.0,
            radius_m=1.01,
            bounds_m=(-0.5, -2.0, 2.0, 2.0),
        )
        self.assertEqual(int(circle.sum()), 5)
        self.assertEqual(int((circle & coverage).sum()), 4)
        self.assertEqual(int((circle & ~coverage).sum()), 1)

    def test_horizon_status_marks_complete_requested_circle_limited_not_global_complete(self) -> None:
        circle = np.array([[True, True], [True, False]])
        coverage = np.ones((2, 2), dtype=bool)
        nodata = np.zeros((2, 2), dtype=bool)
        values = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(regional_horizon_status(circle, coverage, nodata, values), "limited-range")
        self.assertNotEqual(regional_horizon_status(circle, coverage, nodata, values), "complete")

    def test_horizon_status_marks_nodata_or_uncovered_circle_insufficient(self) -> None:
        circle = np.array([[True, True], [False, False]])
        coverage = np.array([[True, False], [True, True]])
        nodata = np.array([[False, False], [False, False]])
        values = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(regional_horizon_status(circle, coverage, nodata, values), "insufficient-coverage")
        coverage[0, 1] = True
        nodata[0, 0] = True
        self.assertEqual(regional_horizon_status(circle, coverage, nodata, values), "insufficient-coverage")

    def test_saved_one_kilometer_profile_api_remains_backward_compatible(self) -> None:
        profile = load_site06_horizon_profile()
        self.assertEqual(len(profile), 360)
        self.assertEqual(SITE06_HORIZON_RANGE_KM, 1.0)

    def test_multiresolution_union_preserves_stronger_five_meter_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "profile.json"
            high = [1.0] * 360
            high[359] = 9.0
            report_path.write_text(json.dumps({
                "requested_radius_m": 1000.0,
                "source_frame": MOON_ME_DE421,
                "profile_bins_deg": high,
            }), encoding="utf-8")
            regional = np.full(360, 2.0)
            combined = _combined_multiresolution_profile(regional, report_path)
        self.assertEqual(combined[359], 9.0)
        self.assertEqual(combined[0], 2.0)


if __name__ == "__main__":
    unittest.main()
