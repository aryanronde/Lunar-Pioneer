"""Tests for the read-only LOLA metadata and bounded-window layer."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer

from core.terrain_provider import (
    MOON_ME_DE421,
    MOON_REFERENCE_RADIUS_M,
    LOLA20mTerrainProvider,
    TaggedMoonPosition,
    TerrainProvider,
)
from core.lunar_coordinates import SITE06_NOBILE_RIM_1, planetocentric_latlon_to_moon_me_de421


TEST_CRS = CRS.from_proj4("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs +type=crs")


class TerrainProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raster_path = Path(self.temp_dir.name) / "synthetic_lola.tif"
        transform = Affine(20.0, 0.0, -100.0, 0.0, -20.0, 100.0)
        values = np.arange(100, dtype=np.float32).reshape((10, 10))
        values[4, 5] = -9999.0
        with rasterio.open(
            self.raster_path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="float32",
            crs=TEST_CRS,
            transform=transform,
            nodata=-9999.0,
        ) as dataset:
            dataset.write(values, 1)
        self.provider = LOLA20mTerrainProvider(self.raster_path).open()

    def tearDown(self) -> None:
        self.provider.close()
        self.temp_dir.cleanup()

    def test_metadata_and_affine_transform(self) -> None:
        self.assertIsInstance(self.provider, TerrainProvider)
        metadata = self.provider.metadata
        self.assertEqual(metadata.width, 10)
        self.assertEqual(metadata.height, 10)
        self.assertEqual(metadata.pixel_size_m, (20.0, 20.0))
        self.assertEqual(metadata.nodata, -9999.0)
        self.assertEqual(metadata.affine, (20.0, 0.0, -100.0, 0.0, -20.0, 100.0))
        self.assertIn("stere", metadata.crs.lower())
        self.assertEqual(metadata.source_frame, MOON_ME_DE421)
        self.assertEqual(metadata.bounds_m, (-100.0, -100.0, 100.0, 100.0))
        self.assertIn("pixel centers", metadata.pixel_registration.lower())
        self.assertIn("no separate vertical CRS", metadata.vertical_reference)

    def test_known_tagged_moon_me_position_projects_to_synthetic_dem(self) -> None:
        # Build a position from known planetocentric lon/lat, not from the unresolved test-site pair.
        longitude_deg, latitude_deg = 12.0, -85.0
        source_crs = CRS.from_proj4(
            f"+proj=longlat +R={MOON_REFERENCE_RADIUS_M:g} +no_defs +type=crs"
        )
        expected_x, expected_y = Transformer.from_crs(
            source_crs, TEST_CRS, always_xy=True
        ).transform(longitude_deg, latitude_deg)
        radius_km = MOON_REFERENCE_RADIUS_M / 1000.0
        longitude_rad = math.radians(longitude_deg)
        latitude_rad = math.radians(latitude_deg)
        position = TaggedMoonPosition(
            radius_km * math.cos(latitude_rad) * math.cos(longitude_rad),
            radius_km * math.cos(latitude_rad) * math.sin(longitude_rad),
            radius_km * math.sin(latitude_rad),
            MOON_ME_DE421,
        )
        actual_x, actual_y = self.provider.site_position_to_dem_xy(position)
        self.assertAlmostEqual(actual_x, expected_x, places=8)
        self.assertAlmostEqual(actual_y, expected_y, places=8)

    def test_position_frame_must_be_explicit_and_correct(self) -> None:
        with self.assertRaises(ValueError):
            TaggedMoonPosition(1.0, 2.0, 3.0, "MOON_PA_DE440")

    def test_site06_record_and_independent_cartesian_conversion(self) -> None:
        site = SITE06_NOBILE_RIM_1
        self.assertEqual(site.name, "Site06 — Nobile Rim 1")
        self.assertEqual(site.latitude_deg, -85.4381)
        self.assertEqual(site.longitude_deg, 37.36667)
        self.assertEqual(site.latitude_type, "planetocentric")
        self.assertEqual(site.longitude_convention, "east_positive")
        self.assertEqual(site.source_frame, MOON_ME_DE421)
        self.assertEqual(site.source_ephemeris, "DE421")
        self.assertIn("NASA LOLA", site.coordinate_source)

        radius_km = 1737.4
        latitude_rad = math.radians(-85.4381)
        longitude_rad = math.radians(37.36667)
        expected = (
            radius_km * math.cos(latitude_rad) * math.cos(longitude_rad),
            radius_km * math.cos(latitude_rad) * math.sin(longitude_rad),
            radius_km * math.sin(latitude_rad),
        )
        actual = planetocentric_latlon_to_moon_me_de421(-85.4381, 37.36667, radius_km)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_site06_projection_matches_independent_projection(self) -> None:
        # This tests the projection conversion without reading terrain values.
        radius_km = 1737.4
        xyz = planetocentric_latlon_to_moon_me_de421(
            SITE06_NOBILE_RIM_1.latitude_deg,
            SITE06_NOBILE_RIM_1.longitude_deg,
            radius_km,
        )
        position = TaggedMoonPosition(*xyz, frame=SITE06_NOBILE_RIM_1.source_frame)
        projected_x_m, projected_y_m = self.provider.site_position_to_dem_xy(position)

        source_crs = CRS.from_proj4(
            f"+proj=longlat +R={MOON_REFERENCE_RADIUS_M:g} +no_defs +type=crs"
        )
        expected_x_m, expected_y_m = Transformer.from_crs(
            source_crs, TEST_CRS, always_xy=True
        ).transform(SITE06_NOBILE_RIM_1.longitude_deg, SITE06_NOBILE_RIM_1.latitude_deg)
        self.assertAlmostEqual(projected_x_m, expected_x_m, places=7)
        self.assertAlmostEqual(projected_y_m, expected_y_m, places=7)

        # Published LDEM_80S_20MPP_ADJ grid metadata: 20 m pixels and
        # south-polar stereographic bounds of +/-304 km in both axes.
        left, bottom, right, top = (-304000.0, -304000.0, 304000.0, 304000.0)
        self.assertGreaterEqual(projected_x_m, left)
        self.assertLess(projected_x_m, right)
        self.assertGreater(projected_y_m, bottom)
        self.assertLessEqual(projected_y_m, top)
        column = math.floor((projected_x_m - left) / 20.0)
        row = math.floor((top - projected_y_m) / 20.0)
        self.assertEqual((row, column), (9699, 19400))

    def test_pixel_center_conversion_and_bounded_window(self) -> None:
        # DEM point (10, 10) falls in col 5, row 4; a 3x3 window starts at (4, 3).
        result = self.provider.read_window(10.0, 10.0, 3, 3)
        self.assertEqual((result.requested_center_x_m, result.requested_center_y_m), (10.0, 10.0))
        self.assertEqual(result.elevation_m.shape, (3, 3))
        self.assertEqual(result.row_offset, 3)
        self.assertEqual(result.column_offset, 4)
        self.assertEqual(
            (result.row_offset, result.row_offset + result.elevation_m.shape[0]), (3, 6)
        )
        self.assertEqual(
            (result.column_offset, result.column_offset + result.elevation_m.shape[1]), (4, 7)
        )
        np.testing.assert_array_equal(result.x_centers_m, [-10.0, 10.0, 30.0])
        np.testing.assert_array_equal(result.y_centers_m, [30.0, 10.0, -10.0])
        self.assertEqual((result.x_centers_m[1], result.y_centers_m[1]), (10.0, 10.0))
        self.assertEqual(result.elevation_m[1, 1], -9999.0)
        self.assertTrue(result.nodata_mask[1, 1])
        self.assertEqual(int(result.nodata_mask.sum()), 1)
        self.assertFalse(result.nodata_mask[0, 0])

    def test_bounds_and_invalid_window_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            self.provider.read_window(101.0, 0.0, 1, 1)
        with self.assertRaises(ValueError):
            self.provider.read_window(0.0, 0.0, 2049, 1)


if __name__ == "__main__":
    unittest.main()
