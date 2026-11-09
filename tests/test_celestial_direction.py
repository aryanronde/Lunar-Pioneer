"""SPICE-only Site06 celestial direction regressions; no DEM access."""

import math
import unittest

import numpy as np
import spiceypy as spice

from core.celestial_direction import (
    SITE06_FRAME_REFERENCE_UTC,
    SITE06_OBSERVER_POSITION_ME_KM,
    SITE06_TERRAIN_UP_ME,
    Site06CelestialDirectionEngine,
)
from core.lunar_frame_bridge import LunarFrameBridge
from core.lunar_horizon import LocalHorizonFrame, LunarSite
from core.spice_engine import SpiceGeometryEngine


UTC = "2026-10-05T00:00:00"


class CelestialDirectionTest(unittest.TestCase):
    def test_sun_and_earth_directions_are_finite_normalized_and_bounded(self) -> None:
        with Site06CelestialDirectionEngine() as engine:
            for target in ("SUN", "EARTH"):
                with self.subTest(target=target):
                    direction = engine.celestial_direction(UTC, target)
                    vector = np.asarray(direction.unit_vector_moon_me_de421)
                    self.assertEqual(direction.target, target)
                    self.assertEqual(direction.utc_timestamp, UTC)
                    self.assertEqual(direction.frame, "MOON_ME_DE421")
                    self.assertTrue(np.all(np.isfinite(vector)))
                    self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=14)
                    self.assertTrue(math.isfinite(direction.azimuth_deg))
                    self.assertTrue(math.isfinite(direction.elevation_deg))
                    self.assertGreaterEqual(direction.azimuth_deg, 0.0)
                    self.assertLess(direction.azimuth_deg, 360.0)
                    self.assertGreaterEqual(direction.elevation_deg, -90.0)
                    self.assertLessEqual(direction.elevation_deg, 90.0)

    def test_changing_utc_changes_sun_direction(self) -> None:
        with Site06CelestialDirectionEngine() as engine:
            first = np.asarray(engine.celestial_direction(UTC, "SUN").unit_vector_moon_me_de421)
            later = np.asarray(
                engine.celestial_direction("2026-10-06T00:00:00", "SUN").unit_vector_moon_me_de421
            )
        self.assertGreater(float(np.linalg.norm(first - later)), 1e-8)

    def test_audited_site06_basis_is_orthonormal_and_uses_terrain_up(self) -> None:
        with Site06CelestialDirectionEngine() as engine:
            north, east, up = (np.asarray(axis) for axis in engine.local_basis)
        np.testing.assert_allclose(up, SITE06_TERRAIN_UP_ME, rtol=0.0, atol=1e-15)
        for axis in (north, east, up):
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=14)
        self.assertAlmostEqual(float(np.dot(north, east)), 0.0, places=14)
        self.assertAlmostEqual(float(np.dot(north, up)), 0.0, places=14)
        self.assertAlmostEqual(float(np.dot(east, up)), 0.0, places=14)
        np.testing.assert_allclose(east, np.cross(north, up), rtol=0.0, atol=1e-15)

    def test_north_matches_projected_v02a_reference(self) -> None:
        with SpiceGeometryEngine() as geometry:
            et = geometry.utc_to_et(SITE06_FRAME_REFERENCE_UTC)
            with LunarFrameBridge() as bridge:
                observer_pa = np.asarray(
                    bridge.vector_dem_to_geometry(SITE06_OBSERVER_POSITION_ME_KM, et)
                )
                radii = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
                ray = observer_pa / np.linalg.norm(observer_pa)
                point = ray / math.sqrt(float(np.sum((ray / radii) ** 2)))
                normal = point / (radii * radii)
                normal /= np.linalg.norm(normal)
                latitude = math.degrees(math.atan2(normal[2], math.hypot(normal[0], normal[1])))
                longitude = math.degrees(math.atan2(normal[1], normal[0])) % 360.0
                v02a = LocalHorizonFrame.at_site(LunarSite(latitude, longitude))
                reference_north_me = np.asarray(
                    bridge.vector_geometry_to_dem(v02a.north, et), dtype=np.float64
                )
        up = np.asarray(SITE06_TERRAIN_UP_ME)
        expected_north = reference_north_me - np.dot(reference_north_me, up) * up
        expected_north /= np.linalg.norm(expected_north)
        with Site06CelestialDirectionEngine() as engine:
            actual_north = np.asarray(engine.local_basis[0])
        np.testing.assert_allclose(actual_north, expected_north, rtol=0.0, atol=1e-12)

    def test_pa_to_me_frame_conversion_preserves_vector_magnitude(self) -> None:
        with SpiceGeometryEngine() as geometry:
            et = geometry.utc_to_et(UTC)
            j2000 = geometry.sun_position_km(et)
            pa = geometry.j2000_to_moon_fixed(j2000, et)
            with LunarFrameBridge() as bridge:
                me = bridge.vector_geometry_to_dem(pa, et)
        self.assertAlmostEqual(float(np.linalg.norm(pa)), float(np.linalg.norm(me)), delta=1e-7)

    def test_target_is_validated(self) -> None:
        with Site06CelestialDirectionEngine() as engine:
            with self.assertRaises(ValueError):
                engine.celestial_direction(UTC, "MOON")


if __name__ == "__main__":
    unittest.main()
