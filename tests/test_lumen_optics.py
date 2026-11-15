"""Independent analytical checks for LUMEN's ideal geometric optics."""
import math
import unittest

from core.lumen.lumen_optics import (
    AU_KM, MirrorOpticsAssumptions, TaggedLunarPosition, calculate_mirror_optics,
)


def p(xyz):
    return TaggedLunarPosition(tuple(xyz), "J2000")


def calc(sun, mirror, target, **kwargs):
    values = {"mirror_area_m2": 100.0, "reflectivity": 0.9}
    values.update(kwargs)
    return calculate_mirror_optics(p(sun), p(mirror), p(target),
        MirrorOpticsAssumptions(**values),
        geometry_origin="assumed")


class LumenOpticsTest(unittest.TestCase):
    def test_normal_incidence_reference(self):
        r = calc((AU_KM + 2_000, 0, 0), (2_000, 0, 0), (3_000, 0, 0))
        self.assertEqual(r.ideal_mirror_normal_unit, (1.0, 0.0, 0.0))
        self.assertEqual(r.reflected_unit, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(r.incidence_angle_deg, 0)
        self.assertAlmostEqual(r.projected_illuminated_area_m2, 100)

    def test_oblique_reflection_and_bisector_independent_reference(self):
        r = calc((AU_KM + 2_000, 0, 0), (2_000, 0, 0), (2_000, 1_000, 0))
        q = math.sqrt(0.5)
        for actual, expected in zip(r.ideal_mirror_normal_unit, (q, q, 0)):
            self.assertAlmostEqual(actual, expected, places=14)
        for actual, expected in zip(r.reflected_unit, (0, 1, 0)):
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertLess(r.reflection_law_residual, 1e-14)
        self.assertAlmostEqual(sum(x*x for x in r.reflected_unit), 1.0)

    def test_inverse_square_irradiance_and_zero_limit(self):
        m = (0, 0, 0); target = (0, 1_000, 0)
        a = calc((AU_KM, 0, 0), m, target)
        b = calc((2*AU_KM, 0, 0), m, target)
        self.assertAlmostEqual(a.solar_irradiance_at_mirror_w_m2, 1361.0)
        self.assertAlmostEqual(b.solar_irradiance_at_mirror_w_m2 / a.solar_irradiance_at_mirror_w_m2, 0.25)
        zero = calc((AU_KM, 0, 0), m, target, solar_irradiance_w_m2=0)
        self.assertEqual(zero.incident_optical_power_w, 0)
        local_override = calc((2*AU_KM, 0, 0), m, target, solar_irradiance_w_m2=900)
        self.assertEqual(local_override.solar_irradiance_at_mirror_w_m2, 900)

    def test_solar_radius_at_one_au_and_finite_aperture_beam_bound(self):
        r = calc((AU_KM, 0, 0), (0, 0, 0), (0, 100_000, 0), mirror_width_m=10, mirror_height_m=10)
        expected = math.asin(695_700 / AU_KM)
        self.assertAlmostEqual(r.solar_angular_radius_rad, expected, places=14)
        solar_only_diameter_km = 2 * r.mirror_target_distance_km * math.tan(expected)
        self.assertAlmostEqual(r.beam_footprint_diameter_km, solar_only_diameter_km + math.sqrt(200)/1000)
        self.assertAlmostEqual(r.projected_illuminated_area_m2, 100 / math.sqrt(2))

    def test_solar_footprint_component_scales_with_range(self):
        first = calc((AU_KM, 0, 0), (0, 0, 0), (0, 100_000, 0), mirror_area_m2=100)
        second = calc((AU_KM, 0, 0), (0, 0, 0), (0, 200_000, 0), mirror_area_m2=100)
        aperture = math.sqrt(200) / 1000
        d1 = first.beam_footprint_diameter_km - aperture
        d2 = second.beam_footprint_diameter_km - aperture
        self.assertAlmostEqual(d2/d1, 2.0, places=12)

    def test_mirror_dimensions_and_physical_domains(self):
        with self.assertRaises(ValueError):
            MirrorOpticsAssumptions(100, 0.9, mirror_width_m=5, mirror_height_m=5)
        with self.assertRaises(ValueError):
            MirrorOpticsAssumptions(100, 0.9, pointing_error_deg=45)
        with self.assertRaises(ValueError):
            calc((0, 0, 0), (0, 0, 0), (0, 1, 0))

    def test_aperture_orientation_is_projected_into_the_mirror_plane(self):
        r = calc((AU_KM, 0, 0), (0, 0, 0), (0, 100, 0),
            mirror_width_m=20, mirror_height_m=5, mirror_aperture_u_unit=(0, 0, 1))
        self.assertAlmostEqual(sum(a*b for a,b in zip(r.mirror_aperture_u_unit, r.ideal_mirror_normal_unit)), 0.0, places=14)
        self.assertAlmostEqual(sum(v*v for v in r.mirror_aperture_u_unit), 1.0, places=14)
        self.assertAlmostEqual(sum(v*v for v in r.mirror_aperture_v_unit), 1.0, places=14)


if __name__ == "__main__":
    unittest.main()
