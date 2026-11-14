"""Saved-profile terrain-relative clearance tests; no DEM or network access."""

import unittest

import numpy as np

from core.celestial_direction import CelestialDirection
from core.terrain_provider import MOON_ME_DE421
from core.terrain_visibility import (
    azimuth_to_horizon_bin,
    compare_with_terrain_horizon,
    interpolate_horizon_elevation,
    load_site06_horizon_profile,
    site06_terrain_visibility,
)


def direction(azimuth: float, elevation: float, target: str = "SUN") -> CelestialDirection:
    return CelestialDirection(
        target=target,
        utc_timestamp="2026-10-05T00:00:00",
        et_tdb_seconds=1.0,
        frame=MOON_ME_DE421,
        position_vector_moon_me_de421_km=(1.0, 0.0, 0.0),
        unit_vector_moon_me_de421=(1.0, 0.0, 0.0),
        range_km=1.0,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
    )


class TerrainVisibilityTest(unittest.TestCase):
    def test_positive_clearance_is_visible(self) -> None:
        result = compare_with_terrain_horizon(direction(42.0, 10.0), [0.0] * 360)
        self.assertEqual(result.clearance_deg, 10.0)
        self.assertTrue(result.visible)

    def test_negative_clearance_is_obstructed(self) -> None:
        profile = [0.0] * 360
        profile[42] = 10.0
        result = compare_with_terrain_horizon(direction(42.0, 5.0), profile)
        self.assertEqual(result.clearance_deg, -5.0)
        self.assertFalse(result.visible)

    def test_zero_clearance_is_obstructed(self) -> None:
        profile = [0.0] * 360
        profile[42] = 5.0
        result = compare_with_terrain_horizon(direction(42.0, 5.0), profile)
        self.assertEqual(result.clearance_deg, 0.0)
        self.assertFalse(result.visible)

    def test_wrap_azimuths_interpolate_across_359_to_zero(self) -> None:
        profile = [0.0] * 360
        profile[359] = 11.0
        profile[0] = 22.0
        result_359 = compare_with_terrain_horizon(direction(359.8, 15.0), profile)
        result_0 = compare_with_terrain_horizon(direction(0.2, 15.0), profile)
        self.assertAlmostEqual(result_359.terrain_horizon_elevation_deg, 19.8)
        self.assertAlmostEqual(result_0.terrain_horizon_elevation_deg, 17.6)
        self.assertEqual(azimuth_to_horizon_bin(359.8), 359)
        self.assertEqual(azimuth_to_horizon_bin(360.2), 0)

    def test_exact_integer_azimuth_uses_exact_sample(self) -> None:
        profile = [0.0] * 360
        profile[0], profile[1] = 2.0, 100.0
        profile[359] = -30.0
        result = compare_with_terrain_horizon(direction(0.0, 10.0), profile)
        self.assertEqual(result.terrain_horizon_elevation_deg, 2.0)

    def test_midpoint_interpolates_linearly(self) -> None:
        profile = [0.0] * 360
        profile[12], profile[13] = 2.0, 6.0
        self.assertEqual(interpolate_horizon_elevation(12.5, profile), 4.0)

    def test_site06_sun_and_earth_azimuths_interpolate_saved_profile(self) -> None:
        profile = load_site06_horizon_profile()
        sun_azimuth = 217.831350
        earth_azimuth = 324.143875
        sun_expected = profile[217] + (sun_azimuth - 217) * (profile[218] - profile[217])
        earth_expected = profile[324] + (earth_azimuth - 324) * (profile[325] - profile[324])
        self.assertAlmostEqual(interpolate_horizon_elevation(sun_azimuth, profile), sun_expected)
        self.assertAlmostEqual(interpolate_horizon_elevation(earth_azimuth, profile), earth_expected)

    def test_interpolation_does_not_modify_stored_profile(self) -> None:
        profile = [float(index) for index in range(360)]
        original = profile.copy()
        interpolate_horizon_elevation(359.75, profile)
        self.assertEqual(profile, original)

    def test_interpolation_preserves_cardinal_convention(self) -> None:
        self.assertEqual(
            [azimuth_to_horizon_bin(a) for a in (0.0, 90.0, 180.0, 270.0)],
            [0, 90, 180, 270],
        )

    def test_empty_bin_and_wrong_frame_are_rejected(self) -> None:
        profile = list(load_site06_horizon_profile())
        profile[17] = None
        with self.assertRaises(ValueError):
            compare_with_terrain_horizon(direction(17.2, 3.0), profile)
        wrong_frame = direction(20.0, 3.0)
        wrong_frame = CelestialDirection(**{**wrong_frame.__dict__, "frame": "J2000"})
        with self.assertRaises(ValueError):
            compare_with_terrain_horizon(wrong_frame, [0.0] * 360)

    def test_interpolation_rejects_missing_upper_bin_and_missing_wrapped_neighbor(self) -> None:
        profile = [0.0] * 360
        profile[13] = None
        with self.assertRaisesRegex(ValueError, "bin 13"):
            interpolate_horizon_elevation(12.5, profile)

        profile = [0.0] * 360
        profile[0] = None
        with self.assertRaisesRegex(ValueError, "bin 0"):
            interpolate_horizon_elevation(359.5, profile)

    def test_exact_sample_does_not_require_the_adjacent_bin(self) -> None:
        profile = [0.0] * 360
        profile[12] = 7.5
        profile[13] = None
        self.assertEqual(interpolate_horizon_elevation(12.0, profile), 7.5)

    def test_saved_profile_sun_and_earth_clearances_are_finite(self) -> None:
        profile = load_site06_horizon_profile()
        for target in ("SUN", "EARTH"):
            with self.subTest(target=target):
                result = site06_terrain_visibility("2026-10-05T00:00:00", target)
                self.assertTrue(np.isfinite(result.clearance_deg))
                self.assertTrue(np.isfinite(result.terrain_horizon_elevation_deg))
                self.assertEqual(result.horizon_range_km, 1.0)
                self.assertEqual(result.horizon_status, "limited-range")
                self.assertEqual(result.target, target)
                self.assertEqual(result.clearance_deg,
                                 result.celestial_elevation_deg - result.terrain_horizon_elevation_deg)
                self.assertEqual(result.visible, result.clearance_deg > 0.0)


if __name__ == "__main__":
    unittest.main()
