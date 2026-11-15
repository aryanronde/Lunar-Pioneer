"""Tests for lunar-sphere occultation and terrain-limited visibility."""

import unittest

from core.lumen.lumen_optics import TaggedLunarPosition
from core.lumen.lumen_visibility import (
    SiteHorizonProfile,
    evaluate_lumen_visibility,
    segment_occludes_lunar_sphere,
)


R = 1_737.4


def pos(vector, frame="J2000") -> TaggedLunarPosition:
    return TaggedLunarPosition(vector, frame)


def horizon(site, *, bins=None, range_km=10.0, status="limited-range", frame="J2000"):
    return SiteHorizonProfile(
        profile_deg=tuple([0.0] * 360 if bins is None else bins),
        site_position_km=site,
        frame=frame,
        east_axis=(0, 1, 0),
        north_axis=(0, 0, 1),
        up_axis=(1, 0, 0),
        range_km=range_km,
        status=status,
        site_name="synthetic ridge",
    )


class LumenVisibilityTest(unittest.TestCase):
    def test_clear_and_blocked_lunar_sphere_segments(self) -> None:
        self.assertFalse(segment_occludes_lunar_sphere((2_000, 0, 0), (100_000, 0, 0), R))
        self.assertTrue(segment_occludes_lunar_sphere((2_000, 0, 0), (-2_000, 0, 0), R))
        self.assertFalse(segment_occludes_lunar_sphere((2_000, R, 0), (-2_000, R, 0), R))

    def test_lunar_occultation_for_all_three_paths(self) -> None:
        result = evaluate_lumen_visibility(
            pos((100_000, 0, 0)),
            pos((-2_000, 0, 0)),
            pos((R, 0, 0)),
            pos((0, R, 0)),
            lunar_reference_radius_km=R,
        )
        self.assertTrue(result.sun_to_mirror.lunar_occulted)
        self.assertTrue(result.mirror_to_station.lunar_occulted)
        self.assertTrue(result.mirror_to_rover.lunar_occulted)

    def test_geometrically_clear_rays_and_unknown_terrain_without_profile(self) -> None:
        result = evaluate_lumen_visibility(
            pos((100_000, 0, 0)), pos((2_000, 0, 0)), pos((R, 0, 0)), pos((R, 0, 10)),
            lunar_reference_radius_km=R,
        )
        self.assertTrue(result.sun_to_mirror.geometrically_clear)
        self.assertTrue(result.mirror_to_station.geometrically_clear)
        self.assertTrue(result.mirror_to_rover.geometrically_clear)
        self.assertEqual(result.mirror_to_station.terrain_status, "unknown-terrain")
        self.assertIsNone(result.mirror_to_station.visible)

    def test_clear_and_blocked_local_horizon_checks(self) -> None:
        station = pos((R, 0, 0))
        rover = pos((R, 0, 1))
        mirror = pos((R + 2, 2, 1))
        clear_profile = horizon(station.position_km, range_km=10)
        blocked_bins = [60.0] * 360
        blocked_profile = horizon(rover.position_km, bins=blocked_bins, range_km=10)
        result = evaluate_lumen_visibility(
            pos((100_000, 0, 0)), mirror, station, rover,
            lunar_reference_radius_km=R,
            station_horizon=clear_profile,
            rover_horizon=blocked_profile,
        )
        self.assertGreater(result.mirror_to_station.terrain_clearance_deg, 0)
        self.assertTrue(result.mirror_to_station.terrain_visible)
        self.assertLess(result.mirror_to_rover.terrain_clearance_deg, 0)
        self.assertFalse(result.mirror_to_rover.terrain_visible)

    def test_limited_range_never_claims_complete_terrain_visibility(self) -> None:
        station = pos((R, 0, 0))
        mirror = pos((R + 100, 0, 0))
        profile = horizon(station.position_km, range_km=10, status="limited-range")
        result = evaluate_lumen_visibility(
            pos((100_000, 0, 0)), mirror, station, pos((R, 1, 0)),
            lunar_reference_radius_km=R,
            station_horizon=profile,
        )
        ray = result.mirror_to_station
        self.assertEqual(ray.terrain_status, "unknown-beyond-profile-range")
        self.assertIsNone(ray.terrain_visible)
        self.assertIsNone(ray.visible)
        self.assertEqual(ray.terrain_range_km, 10)

    def test_insufficient_coverage_and_complete_profile_status(self) -> None:
        site = pos((R, 0, 0))
        target = pos((R + 2, 2, 0))
        for status, expected in (
            ("insufficient-coverage", None),
            ("complete", True),
        ):
            profile = horizon(site.position_km, status=status, range_km=10)
            result = evaluate_lumen_visibility(
                pos((100_000, 0, 0)), target, site, pos((R, 1, 0)),
                lunar_reference_radius_km=R,
                station_horizon=profile,
            )
            self.assertEqual(result.mirror_to_station.terrain_visible, expected)

    def test_frame_mismatch_and_profile_site_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_lumen_visibility(
                pos((100_000, 0, 0)), pos((2_000, 0, 0), "MOON_PA_DE440"),
                pos((R, 0, 0)), pos((R, 0, 1)), lunar_reference_radius_km=R,
            )
        wrong_profile = horizon((R, 2, 0))
        with self.assertRaises(ValueError):
            evaluate_lumen_visibility(
                pos((100_000, 0, 0)), pos((R + 2, 2, 0)), pos((R, 0, 0)), pos((R, 0, 1)),
                lunar_reference_radius_km=R,
                station_horizon=wrong_profile,
            )

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            segment_occludes_lunar_sphere((0, 0, 0), (0, 0, 0), R)
        with self.assertRaises(ValueError):
            segment_occludes_lunar_sphere((0, 0, 0), (1, 0, 0), 0)
        with self.assertRaises(ValueError):
            horizon((R, 0, 0), range_km=0)
        with self.assertRaises(ValueError):
            horizon((R, 0, 0), status="unknown")
        with self.assertRaises(ValueError):
            horizon((R, 0, 0), bins=[0.0])
        with self.assertRaises(ValueError):
            SiteHorizonProfile(
                tuple([0.0] * 360), (R, 0, 0), "MOON_ME", (0, 1, 0), (0, 0, 1),
                (1, 0, 0), 10, "complete", "bad frame",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
