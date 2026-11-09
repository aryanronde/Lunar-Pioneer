"""Integration test using the project's real NASA SPICE kernels."""

import math
import unittest

from core import SpiceGeometryEngine


class SpiceGeometryEngineTest(unittest.TestCase):
    def test_sun_and_earth_vectors_relative_to_moon(self) -> None:
        # Fixed, documented UTC instant for a reproducible first geometry check.
        utc_timestamp = "2026-10-05T00:00:00"

        with SpiceGeometryEngine() as engine:
            et = engine.utc_to_et(utc_timestamp)
            sun = engine.sun_position_km(et)
            earth = engine.earth_position_km(et)

        print(f"UTC: {utc_timestamp} (UTC scale)")
        print(f"ET (TDB seconds past J2000): {et:.6f}")
        print(f"Sun relative to Moon, J2000 (km): {sun}")
        print(f"Earth relative to Moon, J2000 (km): {earth}")

        for label, vector in (("Sun", sun), ("Earth", earth)):
            with self.subTest(body=label):
                self.assertEqual(len(vector), 3)
                self.assertTrue(all(math.isfinite(value) for value in vector))


if __name__ == "__main__":
    unittest.main()
