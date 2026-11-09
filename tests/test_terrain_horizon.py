"""Synthetic tests for sampled terrain horizon profiles."""

import math
import unittest

import numpy as np
import spiceypy as spice

from core.terrain_provider import MOON_ME_DE421
from core.terrain_horizon import calculate_terrain_horizon_profile
from core.lunar_horizon import LocalHorizonFrame, LunarSite
from core.lunar_frame_bridge import LunarFrameBridge
from core.spice_engine import SpiceGeometryEngine


MOON_RADIUS_KM = 1737.4
OBSERVER = np.array((MOON_RADIUS_KM, 0.0, 0.0))
UP = np.array((1.0, 0.0, 0.0))
NORTH = np.array((0.0, 0.0, 1.0))
EAST = np.array((0.0, 1.0, 0.0))


def local_point(azimuth_deg: float, horizontal_range_km: float, up_km: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    horizontal = horizontal_range_km * (
        math.cos(azimuth) * NORTH + math.sin(azimuth) * EAST
    )
    return OBSERVER + horizontal + up_km * UP


def profile(points: np.ndarray, **kwargs):
    return calculate_terrain_horizon_profile(
        OBSERVER,
        UP,
        points,
        observer_frame=MOON_ME_DE421,
        terrain_frame=MOON_ME_DE421,
        reference_north_km=NORTH,
        **kwargs,
    )


class TerrainHorizonTest(unittest.TestCase):
    def test_site06_basis_projects_transformed_v02a_north_onto_terrain_tangent(self) -> None:
        observer = np.array((110.21143659666053, 84.16382863817336, -1737.8192860716285))
        terrain_up = np.array((0.12396688856499274, 0.30047987214444133, -0.9456976562176598))
        epoch = "2026-10-05T00:00:00"
        with SpiceGeometryEngine():
            et = spice.str2et(epoch)
            with LunarFrameBridge() as bridge:
                site_pa = np.asarray(bridge.vector_dem_to_geometry(tuple(observer), et))
                radii = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
                ray = site_pa / np.linalg.norm(site_pa)
                ellipsoid_point = ray / math.sqrt(float(np.sum((ray / radii) ** 2)))
                ellipsoid_normal = ellipsoid_point / (radii * radii)
                ellipsoid_normal /= np.linalg.norm(ellipsoid_normal)
                latitude = math.degrees(math.atan2(
                    ellipsoid_normal[2], math.hypot(ellipsoid_normal[0], ellipsoid_normal[1])
                ))
                longitude = math.degrees(math.atan2(ellipsoid_normal[1], ellipsoid_normal[0])) % 360.0
                reference = LocalHorizonFrame.at_site(LunarSite(latitude, longitude))
                reference_north = np.asarray(bridge.vector_geometry_to_dem(reference.north, et))
                reference_east = np.asarray(bridge.vector_geometry_to_dem(reference.east, et))
                reference_up = np.asarray(bridge.vector_geometry_to_dem(reference.up, et))
        expected_north = reference_north - np.dot(reference_north, terrain_up) * terrain_up
        expected_north /= np.linalg.norm(expected_north)
        expected_east = np.cross(expected_north, terrain_up)
        expected_east /= np.linalg.norm(expected_east)

        result = calculate_terrain_horizon_profile(
            observer,
            terrain_up,
            observer.reshape((1, 3)),
            observer_frame=MOON_ME_DE421,
            terrain_frame=MOON_ME_DE421,
            reference_north_km=reference_north,
            observer_sample_index=0,
        )
        np.testing.assert_allclose(result.up_axis, terrain_up, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(result.north_axis, expected_north, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(result.east_axis, expected_east, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(result.north_axis, result.up_axis), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(result.east_axis, result.up_axis), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(result.north_axis, result.east_axis), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.cross(result.north_axis, result.east_axis), -terrain_up, atol=1e-12)
        np.testing.assert_allclose(np.cross(result.east_axis, result.north_axis), terrain_up, atol=1e-12)
        for actual, expected in zip(
            (result.north_axis, result.east_axis, result.up_axis),
            (expected_north, expected_east, terrain_up),
        ):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        # V0.2A Up is not the measured terrain normal; its angular separation is expected.
        self.assertGreater(math.degrees(math.acos(np.clip(np.dot(result.up_axis, reference_up), -1.0, 1.0))), 0.1)

    def test_cardinal_local_directions_use_required_azimuths(self) -> None:
        points = np.array(
            [OBSERVER + NORTH, OBSERVER + EAST, OBSERVER - NORTH, OBSERVER - EAST]
        )
        result = profile(points)
        for azimuth in (0, 90, 180, 270):
            with self.subTest(azimuth=azimuth):
                self.assertEqual(result.sample_count[azimuth], 1)
                self.assertEqual(result.max_obstruction_elevation_deg[azimuth], 0.0)

    def test_flat_local_terrain_has_zero_tangent_plane_horizon(self) -> None:
        points = np.array(
            [
                local_point(0.0, 0.5, 0.0),
                local_point(90.0, 0.8, 0.0),
                local_point(180.0, 1.0, 0.0),
                local_point(270.0, 1.2, 0.0),
            ]
        )
        result = profile(points)
        for azimuth in (0, 90, 180, 270):
            with self.subTest(azimuth=azimuth):
                self.assertEqual(result.max_obstruction_elevation_deg[azimuth], 0.0)
                self.assertEqual(result.sample_count[azimuth], 1)
        self.assertIn("supplied valid terrain samples only", result.coverage_note)

    def test_single_ridge_appears_at_known_azimuth_and_range(self) -> None:
        range_km = 10.0
        height_km = 0.1
        result = profile(np.array([local_point(90.0, range_km, height_km)]))
        expected = math.degrees(math.atan2(height_km, range_km))
        self.assertAlmostEqual(result.max_obstruction_elevation_deg[90], expected, places=10)
        self.assertEqual(result.sample_count[90], 1)
        self.assertTrue(np.all(np.isnan(result.max_obstruction_elevation_deg[np.arange(360) != 90])))

    def test_farther_ridge_wins_when_its_angular_height_is_greater(self) -> None:
        near = local_point(45.0, 1.0, 0.05)
        far = local_point(45.0, 2.0, 0.2)
        result = profile(np.array([near, far]))
        expected_far = math.degrees(math.atan2(0.2, 2.0))
        expected_near = math.degrees(math.atan2(0.05, 1.0))
        self.assertGreater(expected_far, expected_near)
        self.assertAlmostEqual(result.max_obstruction_elevation_deg[45], expected_far, places=10)
        self.assertEqual(result.sample_count[45], 2)

    def test_lunar_curvature_appears_in_centered_xyz_geometry(self) -> None:
        arc_angle_rad = 0.1
        observer = np.array((MOON_RADIUS_KM, 0.0, 0.0))
        curved_surface_point = np.array(
            (MOON_RADIUS_KM * math.cos(arc_angle_rad), MOON_RADIUS_KM * math.sin(arc_angle_rad), 0.0)
        )
        result = calculate_terrain_horizon_profile(
            observer,
            UP,
            np.array([observer, curved_surface_point]),
            observer_frame=MOON_ME_DE421,
            terrain_frame=MOON_ME_DE421,
                reference_north_km=NORTH,
            observer_sample_index=0,
        )
        # A chord between equal-radius points is below the tangent by half its central angle.
        self.assertAlmostEqual(
            result.max_obstruction_elevation_deg[90], -math.degrees(arc_angle_rad / 2.0), places=11
        )
        self.assertEqual(result.sample_count[90], 1)

    def test_nodata_samples_are_ignored_and_empty_azimuths_are_nan(self) -> None:
        points = np.array(
            [local_point(180.0, 1.0, 0.1), (np.nan, np.nan, np.nan), local_point(0.0, 1.0, 0.2)]
        )
        result = profile(points, valid_mask=np.array([True, True, False]))
        self.assertEqual(result.sample_count[180], 1)
        self.assertAlmostEqual(result.max_obstruction_elevation_deg[180], math.degrees(math.atan2(0.1, 1.0)))
        self.assertEqual(result.sample_count[0], 0)
        self.assertTrue(np.isnan(result.max_obstruction_elevation_deg[0]))

    def test_azimuths_wrap_across_zero_degrees(self) -> None:
        points = np.array(
            [local_point(359.8, 1.0, 0.1), local_point(0.2, 2.0, 0.3)]
        )
        result = profile(points)
        self.assertEqual(result.azimuth_centers_deg[0], 0.0)
        self.assertEqual(result.sample_count[0], 2)
        self.assertAlmostEqual(result.max_obstruction_elevation_deg[0], math.degrees(math.atan2(0.3, 2.0)))

    def test_exact_half_degree_azimuth_ties_assign_to_next_nearest_bin(self) -> None:
        # The documented nearest-center rule rounds an exact 0.5-degree tie
        # clockwise to bin 1; the wraparound tie at 359.5 goes to bin 0.
        points = np.array([local_point(0.5, 1.0, 0.1), local_point(359.5, 1.0, 0.2)])
        result = profile(points)
        self.assertEqual(result.sample_count[0], 1)
        self.assertEqual(result.sample_count[1], 1)
        self.assertEqual(int(result.sample_count.sum()), 2)
        self.assertTrue(np.isnan(result.max_obstruction_elevation_deg[359]))

    def test_observer_pixel_is_excluded_and_frames_must_match_me_de421(self) -> None:
        points = np.array([OBSERVER, local_point(0.0, 1.0, 0.1)])
        result = profile(points, observer_sample_index=0)
        self.assertEqual(result.sample_count.sum(), 1)
        with self.assertRaises(ValueError):
            calculate_terrain_horizon_profile(
                OBSERVER,
                UP,
                points,
                observer_frame="MOON_PA_DE440",
                terrain_frame=MOON_ME_DE421,
                reference_north_km=NORTH,
            )


if __name__ == "__main__":
    unittest.main()
