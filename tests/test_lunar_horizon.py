"""Mathematical and SPICE integration checks for V0.2A lunar geometry."""

import math
import unittest

import numpy as np
import spiceypy as spice

from core import LocalHorizonFrame, LunarSite, SpiceGeometryEngine


class LunarHorizonTest(unittest.TestCase):
    utc_timestamp = "2026-10-05T00:00:00"
    site = LunarSite(latitude_deg=-85.99, longitude_deg=357.07)

    def test_site_conversion_is_on_pck_ellipsoid(self) -> None:
        with SpiceGeometryEngine() as engine:
            _dimension, radii = spice.bodvrd("MOON", "RADII", 3)
        point = np.asarray(self.site.surface_point_km(tuple(radii)))
        normalized_radius = np.sum((point / radii) ** 2)
        self.assertAlmostEqual(float(normalized_radius), 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(point)))
        lat_rad = math.radians(self.site.latitude_deg)
        lon_rad = math.radians(self.site.longitude_deg)
        expected_normal = np.array((math.cos(lat_rad) * math.cos(lon_rad),
                                    math.cos(lat_rad) * math.sin(lon_rad),
                                    math.sin(lat_rad)))
        ellipsoid_normal = point / (radii * radii)
        ellipsoid_normal /= np.linalg.norm(ellipsoid_normal)
        self.assertTrue(np.allclose(ellipsoid_normal, expected_normal, atol=1e-14))

    def test_local_axes_are_orthonormal(self) -> None:
        frame = LocalHorizonFrame.at_site(self.site)
        axes = [np.asarray(axis) for axis in (frame.east, frame.north, frame.up)]
        for axis in axes:
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=14)
        for index in range(3):
            for other in range(index + 1, 3):
                self.assertAlmostEqual(float(np.dot(axes[index], axes[other])), 0.0, places=14)

    def test_vector_local_components_preserve_magnitude(self) -> None:
        frame = LocalHorizonFrame.at_site(self.site)
        vector = np.array((1234.0, -5678.0, 9101.0))
        local = np.asarray(frame.components(tuple(vector)))  # East, North, Up
        self.assertAlmostEqual(float(np.linalg.norm(local)), float(np.linalg.norm(vector)), places=10)

    def test_azimuth_elevation_conventions(self) -> None:
        frame = LocalHorizonFrame.at_site(self.site)
        self.assertAlmostEqual(frame.direction_angles(frame.up)[1], 90.0, places=12)
        self.assertAlmostEqual(frame.direction_angles(frame.north)[0], 0.0, places=12)
        self.assertAlmostEqual(frame.direction_angles(frame.east)[0], 90.0, places=12)
        self.assertAlmostEqual(frame.direction_angles(tuple(np.asarray(frame.north) * -1.0))[0], 180.0, places=12)
        self.assertAlmostEqual(frame.direction_angles(tuple(np.asarray(frame.east) * -1.0))[0], 270.0, places=12)
        self.assertAlmostEqual(frame.direction_angles(frame.east)[1], 0.0, places=12)

    def test_spice_moon_fixed_rotation_and_site_angles_are_finite(self) -> None:
        with SpiceGeometryEngine() as engine:
            et = engine.utc_to_et(self.utc_timestamp)
            self.assertEqual(spice.namfrm(engine.MOON_FIXED_FRAME), 31008)
            for inertial_vector in (engine.sun_position_km(et), engine.earth_position_km(et)):
                fixed_vector = engine.j2000_to_moon_fixed(inertial_vector, et)
                self.assertTrue(all(math.isfinite(value) for value in fixed_vector))
                self.assertAlmostEqual(
                    float(np.linalg.norm(fixed_vector)),
                    float(np.linalg.norm(inertial_vector)),
                    delta=1e-7,
                )
                azimuth, elevation = LocalHorizonFrame.at_site(self.site).direction_angles(fixed_vector)
                self.assertTrue(math.isfinite(azimuth))
                self.assertTrue(math.isfinite(elevation))
                self.assertGreaterEqual(azimuth, 0.0)
                self.assertLess(azimuth, 360.0)
                self.assertGreaterEqual(elevation, -90.0)
                self.assertLessEqual(elevation, 90.0)


if __name__ == "__main__":
    unittest.main()
