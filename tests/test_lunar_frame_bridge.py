"""SPICE validation for the DE421-ME to DE440-PA lunar frame bridge."""

import math
import unittest

import numpy as np
import spiceypy as spice

from core import SpiceGeometryEngine
from core.lunar_frame_bridge import LunarFrameBridge


class LunarFrameBridgeTest(unittest.TestCase):
    # Fixed project validation instant, explicitly interpreted on the UTC scale.
    UTC_TIMESTAMP = "2026-10-05T00:00:00"
    TEST_VECTORS = ((1.0, 2.0, 3.0), (1737.4, 0.0, 0.0))

    def test_de421_to_de440_frame_transform(self) -> None:
        with SpiceGeometryEngine() as _geometry_kernels:
            et = float(spice.str2et(self.UTC_TIMESTAMP))
            with LunarFrameBridge() as bridge:
                # These are the DE-specific frames, not generic aliases.
                self.assertEqual(spice.namfrm(bridge.DEM_FRAME), 31007)
                self.assertEqual(spice.namfrm(bridge.GEOMETRY_FRAME), 31008)

                matrix = np.asarray(bridge.frame_transform_matrix(et))
                identity = np.eye(3)
                np.testing.assert_allclose(matrix @ matrix.T, identity, rtol=0.0, atol=1e-14)
                np.testing.assert_allclose(matrix.T @ matrix, identity, rtol=0.0, atol=1e-14)

                for original in self.TEST_VECTORS:
                    with self.subTest(vector=original):
                        geometry_vector = bridge.vector_dem_to_geometry(original, et)
                        recovered = bridge.vector_geometry_to_dem(geometry_vector, et)
                        self.assertTrue(all(math.isfinite(value) for value in geometry_vector))
                        np.testing.assert_allclose(recovered, original, rtol=0.0, atol=1e-12)
                        self.assertAlmostEqual(
                            float(np.linalg.norm(geometry_vector)),
                            float(np.linalg.norm(original)),
                            delta=1e-12,
                        )


if __name__ == "__main__":
    unittest.main()
