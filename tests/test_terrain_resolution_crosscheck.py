"""Synthetic regressions for resolution-profile comparison semantics."""

import unittest

import numpy as np

from core.terrain_horizon import TerrainHorizonProfile, calculate_terrain_horizon_profile
from core.terrain_provider import MOON_ME_DE421
from scripts.validate_site06_5m_horizon_crosscheck import compare_profiles


RADIUS = 1737.4
OBSERVER = np.array((RADIUS, 0.0, 0.0))
UP = np.array((1.0, 0.0, 0.0))
NORTH = np.array((0.0, 0.0, 1.0))
EAST = np.array((0.0, 1.0, 0.0))


def profile(values, frame=MOON_ME_DE421):
    return TerrainHorizonProfile(
        azimuth_centers_deg=np.arange(360, dtype=float),
        max_obstruction_elevation_deg=np.asarray(values, dtype=float),
        sample_count=np.ones(360, dtype=np.int64),
        source_frame=frame,
        north_axis=tuple(NORTH),
        east_axis=tuple(EAST),
        up_axis=tuple(UP),
    )


class TerrainResolutionCrosscheckTest(unittest.TestCase):
    def test_profile_comparison_reports_max_difference_rms_and_circular_azimuth_shift(self):
        coarse = np.zeros(360)
        fine = np.zeros(360)
        coarse[359] = 10.0
        fine[0] = 11.0
        result = compare_profiles(profile(fine), profile(coarse))
        self.assertEqual(result["profile_bins_compared"], 360)
        self.assertEqual(result["maximum_absolute_profile_difference_deg"], 11.0)
        self.assertAlmostEqual(result["rms_profile_difference_deg"], np.sqrt((121 + 100) / 360))
        self.assertEqual(result["strongest_azimuth_difference_deg"], 1)

    def test_cardinal_azimuths_follow_north_east_south_west_convention(self):
        points = np.array((OBSERVER + NORTH, OBSERVER + EAST, OBSERVER - NORTH, OBSERVER - EAST))
        result = calculate_terrain_horizon_profile(
            OBSERVER,
            UP,
            points,
            observer_frame=MOON_ME_DE421,
            terrain_frame=MOON_ME_DE421,
            reference_north_km=NORTH,
        )
        for azimuth in (0, 90, 180, 270):
            self.assertEqual(result.sample_count[azimuth], 1)
            self.assertEqual(result.max_obstruction_elevation_deg[azimuth], 0.0)

    def test_nodata_is_ignored_and_empty_bins_remain_nan(self):
        point = OBSERVER + NORTH + 0.1 * UP
        result = calculate_terrain_horizon_profile(
            OBSERVER,
            UP,
            np.array((point, (np.nan, np.nan, np.nan))),
            observer_frame=MOON_ME_DE421,
            terrain_frame=MOON_ME_DE421,
            reference_north_km=NORTH,
            valid_mask=np.array((True, True)),
        )
        self.assertEqual(result.sample_count[0], 1)
        self.assertEqual(result.sample_count.sum(), 1)
        self.assertTrue(np.isnan(result.max_obstruction_elevation_deg[1]))

    def test_frame_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_profiles(profile(np.zeros(360), "MOON_PA_DE440"), profile(np.zeros(360)))

    def test_profile_comparison_reports_nonmonotonic_signed_changes_and_azimuth(self):
        previous = np.zeros(360)
        current = np.zeros(360)
        current[0] = 1.25
        current[359] = -0.5
        result = compare_profiles(profile(current), profile(previous))
        self.assertEqual(result["maximum_absolute_difference_azimuth_deg"], 0.0)
        self.assertEqual(result["maximum_signed_profile_change_deg"], 1.25)
        self.assertEqual(result["minimum_signed_profile_change_deg"], -0.5)
        self.assertEqual(result["profile_bins_compared"], 360)

    def test_nested_radius_profile_can_only_raise_common_bin_maxima(self):
        previous = np.full(360, 0.25)
        current = previous.copy()
        current[42] = 3.0
        result = compare_profiles(profile(current), profile(previous))
        self.assertGreaterEqual(result["minimum_signed_profile_change_deg"], 0.0)
        self.assertEqual(result["maximum_absolute_profile_difference_deg"], 2.75)

    def test_profile_comparison_counts_valid_bins_without_filling_missing_coverage(self):
        previous = np.zeros(360)
        current = np.zeros(360)
        previous[10] = np.nan
        previous[11] = np.nan
        current[11] = 8.0  # New valid bin is reported but not compared to invented data.
        result = compare_profiles(profile(current), profile(previous))
        self.assertEqual(result["first_profile_valid_bins"], 360)
        self.assertEqual(result["second_profile_valid_bins"], 358)
        self.assertEqual(result["first_profile_bins_without_coverage"], 0)
        self.assertEqual(result["second_profile_bins_without_coverage"], 2)
        self.assertEqual(result["profile_bins_compared"], 358)
        self.assertEqual(result["maximum_absolute_profile_difference_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
