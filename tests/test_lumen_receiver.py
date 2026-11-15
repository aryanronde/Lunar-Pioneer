from __future__ import annotations

import math
import unittest
from dataclasses import replace

from core.lumen.energy_storage import BatteryConfig, BatteryState
from core.lumen.lumen_optics import (
    AU_KM, MirrorOpticsAssumptions,
    TaggedLunarPosition,
    calculate_mirror_optics,
)
from core.lumen.lumen_receiver import (
    ReceiverAssumptions,
    calculate_receiver_power,
    integrate_receiver_energy,
    study_receiver_distance_sensitivity,
)


def pos(vector, frame="J2000"):
    return TaggedLunarPosition(vector, frame)


def base_optics(*, distance_km=100.0, reflectivity=0.8, frame="J2000"):
    return calculate_mirror_optics(
        pos((AU_KM + 2_000.0, 0.0, 0.0), frame),
        pos((2_000.0, 0.0, 0.0), frame),
        pos((2_000.0, distance_km, 0.0), frame),
        MirrorOpticsAssumptions(mirror_area_m2=100.0, reflectivity=reflectivity),
        geometry_origin="assumed",
    )


def receiver(*, area=100.0, normal=(0.0, -1.0, 0.0), frame="J2000", pv=0.3, optical=0.9, electrical=0.95, **kwargs):
    return ReceiverAssumptions(area, normal, frame, pv, optical, electrical, **kwargs)


class LumenReceiverTest(unittest.TestCase):
    def test_aligned_receiver_intercepts_projected_area_and_calculates_electrical_power(self):
        result = calculate_receiver_power(base_optics(), receiver(area=4_000_000.0, integration_samples=40))
        self.assertAlmostEqual(result.receiver_incidence_angle_deg, 0.0, places=6)
        self.assertEqual(result.projected_receiver_area_m2, 4_000_000.0)
        self.assertAlmostEqual(result.beam_area_m2, math.pi * result.beam_footprint_diameter_m**2 / 4.0)
        self.assertEqual(result.capture_fraction, 1.0)
        self.assertAlmostEqual(result.intercepted_optical_power_w, result.reflected_optical_power_w * result.capture_fraction)
        self.assertAlmostEqual(result.received_optical_power_w, result.intercepted_optical_power_w * 0.9)
        self.assertAlmostEqual(result.electrical_power_w, result.received_optical_power_w * 0.3 * 0.95)

    def test_receiver_tilted_45_degrees_uses_cosine_projected_area(self):
        tilt = (0.0, -math.sqrt(0.5), math.sqrt(0.5))
        result = calculate_receiver_power(base_optics(), receiver(area=4_000_000.0, normal=tilt))
        self.assertAlmostEqual(result.receiver_incidence_angle_deg, 45.0, places=6)
        self.assertAlmostEqual(result.projected_receiver_area_m2, 4_000_000.0 / math.sqrt(2.0))

    def test_receiver_facing_away_captures_no_power(self):
        result = calculate_receiver_power(base_optics(), receiver(normal=(0.0, 1.0, 0.0)))
        self.assertAlmostEqual(result.receiver_incidence_angle_deg, 180.0)
        self.assertEqual(result.projected_receiver_area_m2, 0.0)
        self.assertEqual(result.capture_fraction, 0.0)
        self.assertEqual(result.intercepted_optical_power_w, 0.0)
        self.assertEqual(result.electrical_power_w, 0.0)

    def test_zero_receiver_area_and_zero_reflected_power(self):
        no_area = calculate_receiver_power(base_optics(), receiver(area=0.0))
        self.assertEqual(no_area.projected_receiver_area_m2, 0.0)
        self.assertEqual(no_area.electrical_power_w, 0.0)
        no_reflection = calculate_receiver_power(base_optics(reflectivity=0.0), receiver())
        self.assertGreater(no_reflection.incident_mirror_power_w, 0.0)
        self.assertEqual(no_reflection.reflected_optical_power_w, 0.0)
        self.assertEqual(no_reflection.intercepted_optical_power_w, 0.0)
        self.assertEqual(no_reflection.electrical_power_w, 0.0)

    def test_invalid_vectors_efficiencies_and_coordinate_frames_rejected(self):
        for normal in ((0, 0), (math.nan, 0, 1), (0, 0, 0), (0, 2, 0)):
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                receiver(normal=normal)
        for changes in (
            {"physical_area_m2": -1.0},
            {"photovoltaic_efficiency": 1.1},
            {"optical_transmission_efficiency": math.inf},
            {"electrical_efficiency": -0.1},
            {"frame": "MOON_ME"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ReceiverAssumptions(
                    physical_area_m2=changes.get("physical_area_m2", 100.0),
                    surface_normal_unit=(0.0, -1.0, 0.0),
                    frame=changes.get("frame", "J2000"),
                    photovoltaic_efficiency=changes.get("photovoltaic_efficiency", 0.3),
                    optical_transmission_efficiency=changes.get("optical_transmission_efficiency", 0.9),
                    electrical_efficiency=changes.get("electrical_efficiency", 0.95),
                )
        with self.assertRaises(ValueError):
            calculate_receiver_power(base_optics(frame="J2000"), receiver(frame="MOON_PA_DE440"))
        with self.assertRaises(ValueError):
            calculate_receiver_power(replace(base_optics(), mirror_to_target_unit=(math.nan, 0, 0)), receiver())

    def test_slant_distance_sensitivity_uses_three_ranges_and_preserves_assumptions(self):
        cases = study_receiver_distance_sensitivity(base_optics(distance_km=100), receiver(area=4_000_000.0))
        self.assertEqual(tuple(case.mirror_to_receiver_distance_km for case in cases), (100.0, 250.0, 500.0))
        aperture = math.hypot(base_optics().mirror_width_m, base_optics().mirror_height_m)
        self.assertAlmostEqual((cases[1].beam_footprint_diameter_m-aperture) / (cases[0].beam_footprint_diameter_m-aperture), 2.5)
        self.assertAlmostEqual((cases[2].beam_footprint_diameter_m-aperture) / (cases[0].beam_footprint_diameter_m-aperture), 5.0)
        self.assertAlmostEqual(cases[1].beam_area_m2 / cases[0].beam_area_m2,
            (cases[1].beam_footprint_diameter_m / cases[0].beam_footprint_diameter_m) ** 2)
        self.assertAlmostEqual(cases[2].beam_area_m2 / cases[0].beam_area_m2,
            (cases[2].beam_footprint_diameter_m / cases[0].beam_footprint_diameter_m) ** 2)
        self.assertAlmostEqual(cases[0].reflected_optical_power_w, cases[2].reflected_optical_power_w)
        self.assertTrue(any("no trajectory" in item for item in cases[0].assumptions))

    def test_very_small_and_large_ranges_remain_finite_and_capture_is_bounded(self):
        cases = study_receiver_distance_sensitivity(base_optics(), receiver(area=1e12), (0.1, 1e9))
        self.assertEqual(cases[0].capture_fraction, 1.0)
        for case in cases:
            for value in (
                case.beam_footprint_diameter_m, case.beam_area_m2,
                case.capture_fraction, case.intercepted_optical_power_w,
                case.electrical_power_w,
            ):
                self.assertTrue(math.isfinite(value))
            self.assertLessEqual(case.capture_fraction, 1.0)
            self.assertLessEqual(case.intercepted_optical_power_w, case.reflected_optical_power_w)

    def test_partial_and_zero_overlap_with_receiver_offset(self):
        centered = calculate_receiver_power(base_optics(), receiver(area=400_000.0, integration_samples=48))
        shifted = calculate_receiver_power(base_optics(), receiver(area=400_000.0, center_offset_u_m=500.0, integration_samples=48))
        far = calculate_receiver_power(base_optics(), receiver(area=400_000.0, center_offset_u_m=2_000.0, integration_samples=48))
        self.assertGreater(centered.capture_fraction, 0.0)
        self.assertLess(centered.capture_fraction, 1.0)
        self.assertLess(shifted.capture_fraction, centered.capture_fraction)
        self.assertEqual(far.capture_fraction, 0.0)

    def test_pointing_error_moves_beam_by_twice_mirror_normal_error(self):
        optics = calculate_mirror_optics(
            pos((AU_KM + 2_000, 0, 0)), pos((2_000, 0, 0)), pos((2_000, 100, 0)),
            MirrorOpticsAssumptions(mirror_area_m2=100, reflectivity=0.8, pointing_error_deg=0.5),
            geometry_origin="assumed",
        )
        self.assertAlmostEqual(optics.pointing_error_displacement_km,
            100 * math.tan(math.radians(1.0)), places=12)
        result = calculate_receiver_power(optics, receiver(area=100, integration_samples=24))
        self.assertEqual(result.capture_fraction, 0.0)

    def test_quadrature_refinement_is_stable_for_partial_overlap(self):
        coarse = calculate_receiver_power(base_optics(), receiver(area=400_000.0, integration_samples=24))
        fine = calculate_receiver_power(base_optics(), receiver(area=400_000.0, integration_samples=48))
        self.assertLess(abs(coarse.capture_fraction - fine.capture_fraction), 0.08)

    def test_energy_conservation_and_battery_storage_are_separate_from_generation(self):
        power = calculate_receiver_power(
            base_optics(),
            receiver(area=100_000.0, optical=0.95, pv=0.3, electrical=0.9),
        )
        generated = integrate_receiver_energy(power, 3_600.0)
        self.assertIsNone(generated.battery_stored_energy_wh)
        self.assertAlmostEqual(generated.electrical_generated_energy_wh, power.electrical_power_w)
        self.assertAlmostEqual(generated.energy_balance_residual_wh, 0.0, delta=1e-8)

        stored = integrate_receiver_energy(
            power,
            3_600.0,
            battery_config=BatteryConfig(1_000.0, 1.0, maximum_charge_power_w=10_000.0),
            battery_state=BatteryState(0.0),
            battery_charge_efficiency=0.9,
        )
        self.assertIsNotNone(stored.battery_charge)
        self.assertLessEqual(stored.battery_stored_energy_wh, stored.electrical_generated_energy_wh)
        self.assertAlmostEqual(stored.energy_balance_residual_wh, 0.0, delta=1e-8)
        self.assertAlmostEqual(
            stored.electrical_generated_energy_wh,
            stored.battery_stored_energy_wh + stored.battery_charging_loss_wh + stored.curtailed_electrical_energy_wh,
            delta=1e-8,
        )

    def test_energy_integration_rejects_partial_battery_inputs(self):
        power = calculate_receiver_power(base_optics(), receiver())
        with self.assertRaises(ValueError):
            integrate_receiver_energy(power, 60.0, battery_config=BatteryConfig(100.0))
        with self.assertRaises(ValueError):
            integrate_receiver_energy(power, 0.0)


if __name__ == "__main__":
    unittest.main()
