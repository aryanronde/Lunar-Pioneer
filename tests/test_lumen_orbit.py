"""Tests for assumed circular lunar orbit geometry."""

import math
import unittest

from core.lumen.lumen_orbit import (
    CircularLunarOrbit,
    ORBIT_FRAME,
    ORBIT_MODEL_LABEL,
)


class FakeSpiceTime:
    def utc_to_et(self, value: str) -> float:
        return {
            "2026-01-01T00:00:00Z": 1000.0,
            "2026-01-01T00:00:00": 1000.0,
            "2026-01-01T01:00:00Z": 4600.0,
        }[value]


def orbit(**changes) -> CircularLunarOrbit:
    values = dict(
        altitude_km=100.0,
        inclination_deg=30.0,
        ascending_node_longitude_deg=45.0,
        initial_orbital_phase_deg=0.0,
        epoch_utc="2026-01-01T00:00:00Z",
    )
    values.update(changes)
    return CircularLunarOrbit(**values)


class LumenOrbitTest(unittest.TestCase):
    def test_radius_and_circular_period(self) -> None:
        model = orbit()
        self.assertAlmostEqual(model.orbital_radius_km, 1_837.4)
        expected = 2 * math.pi * math.sqrt(1_837.4**3 / model.gravitational_parameter_km3_s2)
        self.assertAlmostEqual(model.period_s, expected)
        position = model.position_at_et(1000, 1000)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in position)), 1_837.4)

    def test_phase_and_time_behaviour(self) -> None:
        model = orbit(inclination_deg=0, ascending_node_longitude_deg=0)
        at_epoch = model.position_at_et(0, 0)
        quarter_period = model.position_at_et(model.period_s / 4, 0)
        full_period = model.position_at_et(model.period_s, 0)
        self.assertAlmostEqual(at_epoch[0], model.orbital_radius_km)
        self.assertAlmostEqual(at_epoch[1], 0.0, places=9)
        self.assertAlmostEqual(quarter_period[0], 0.0, places=8)
        self.assertAlmostEqual(quarter_period[1], model.orbital_radius_km)
        for left, right in zip(at_epoch, full_period):
            self.assertAlmostEqual(left, right, places=8)

    def test_inclination_node_and_initial_phase_define_orientation(self) -> None:
        model = orbit(inclination_deg=90, ascending_node_longitude_deg=0, initial_orbital_phase_deg=90)
        position = model.position_at_et(0, 0)
        self.assertAlmostEqual(position[0], 0.0, places=9)
        self.assertAlmostEqual(position[1], 0.0, places=9)
        self.assertAlmostEqual(position[2], model.orbital_radius_km)

    def test_utc_api_uses_spice_time_conversion_but_marks_orbit_assumed(self) -> None:
        record = orbit().position_at_utc("2026-01-01T01:00:00Z", FakeSpiceTime())
        self.assertEqual(record.frame, ORBIT_FRAME)
        self.assertEqual(record.origin, "MOON")
        self.assertEqual(record.epoch_et_s, 1000)
        self.assertEqual(record.et_s, 4600)
        self.assertEqual(record.model, ORBIT_MODEL_LABEL)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in record.position_km)), 1_837.4)

    def test_invalid_orbit_inputs_rejected(self) -> None:
        for changes in (
            {"altitude_km": 0},
            {"altitude_km": -1},
            {"altitude_km": math.inf},
            {"inclination_deg": 181},
            {"inclination_deg": math.nan},
            {"reference_radius_km": 0},
            {"gravitational_parameter_km3_s2": -1},
            {"epoch_utc": "not-a-time"},
            {"epoch_utc": "2026-01-01T00:00:00"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                orbit(**changes)
        with self.assertRaises(ValueError):
            orbit().position_at_et(math.nan, 0)
        with self.assertRaises(TypeError):
            orbit().position_at_utc("2026-01-01T00:00:00Z", object())


if __name__ == "__main__":
    unittest.main()
