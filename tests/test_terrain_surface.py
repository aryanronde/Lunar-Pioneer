"""Tests for spherical LOLA terrain points and finite-difference normals."""

import unittest

import numpy as np
from pyproj import CRS, Transformer

from core.terrain_provider import (
    MOON_ME_DE421,
    MOON_REFERENCE_RADIUS_M,
    TerrainMetadata,
    TerrainWindow,
)
from core.terrain_surface import (
    REFERENCE_RADIUS_KM,
    calculate_terrain_normals,
    reconstruct_lunar_terrain,
)


DEM_CRS = CRS.from_proj4(
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs +type=crs"
)


def make_window(elevation_m: np.ndarray, nodata_mask: np.ndarray | None = None) -> tuple[TerrainWindow, TerrainMetadata]:
    source = CRS.from_proj4(
        f"+proj=longlat +R={MOON_REFERENCE_RADIUS_M:g} +no_defs +type=crs"
    )
    x_center, y_center = Transformer.from_crs(source, DEM_CRS, always_xy=True).transform(37.36667, -85.4381)
    offsets = np.arange(elevation_m.shape[1]) - (elevation_m.shape[1] - 1) / 2.0
    row_offsets = np.arange(elevation_m.shape[0]) - (elevation_m.shape[0] - 1) / 2.0
    x_centers = x_center + offsets * 20.0
    y_centers = y_center - row_offsets * 20.0
    if nodata_mask is None:
        nodata_mask = np.zeros(elevation_m.shape, dtype=bool)
    window = TerrainWindow(
        requested_center_x_m=float(x_center),
        requested_center_y_m=float(y_center),
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        elevation_m=elevation_m.astype(np.float64),
        nodata_mask=nodata_mask,
        row_offset=0,
        column_offset=0,
    )
    metadata = TerrainMetadata(
        crs=DEM_CRS.to_wkt(),
        projection="Polar Stereographic",
        affine=(20.0, 0.0, -304000.0, 0.0, -20.0, 304000.0),
        width=30400,
        height=30400,
        pixel_size_m=(20.0, 20.0),
        nodata=float("nan"),
        vertical_reference="meters relative to 1,737,400 m spherical reference radius",
        source_frame=MOON_ME_DE421,
        bounds_m=(-304000.0, -304000.0, 304000.0, 304000.0),
        pixel_registration="PixelIsArea; pixel centers",
    )
    return window, metadata


class TerrainSurfaceTest(unittest.TestCase):
    def test_flat_spherical_surface_has_outward_radial_normal(self) -> None:
        window, metadata = make_window(np.zeros((5, 5), dtype=np.float64))
        surface = reconstruct_lunar_terrain(window, metadata)
        field = calculate_terrain_normals(surface)
        center = surface.positions_km[2, 2]
        expected_up = center / np.linalg.norm(center)
        actual = field.normals[2, 2]

        self.assertTrue(field.valid_mask[2, 2])
        np.testing.assert_allclose(actual, expected_up, rtol=0.0, atol=1e-7)
        self.assertGreater(float(np.dot(actual, center)), 0.0)
        self.assertEqual(field.source_frame, MOON_ME_DE421)

    def test_normals_are_finite_and_unit_length_for_sloped_surface(self) -> None:
        window, metadata = make_window(np.zeros((5, 5), dtype=np.float64))
        x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
        elevation = 0.12 * (x_grid - window.requested_center_x_m) - 0.08 * (
            y_grid - window.requested_center_y_m
        )
        window = TerrainWindow(
            requested_center_x_m=window.requested_center_x_m,
            requested_center_y_m=window.requested_center_y_m,
            x_centers_m=window.x_centers_m,
            y_centers_m=window.y_centers_m,
            elevation_m=elevation,
            nodata_mask=window.nodata_mask,
            row_offset=window.row_offset,
            column_offset=window.column_offset,
        )
        surface = reconstruct_lunar_terrain(window, metadata)
        field = calculate_terrain_normals(surface)
        valid_normals = field.normals[field.valid_mask]
        self.assertTrue(np.all(np.isfinite(valid_normals)))
        np.testing.assert_allclose(np.linalg.norm(valid_normals, axis=1), 1.0, rtol=0.0, atol=1e-12)

    def test_center_normal_is_orthogonal_to_finite_difference_tangents(self) -> None:
        window, metadata = make_window(np.zeros((5, 5), dtype=np.float64))
        x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
        elevation = 0.12 * (x_grid - window.requested_center_x_m) - 0.08 * (
            y_grid - window.requested_center_y_m
        )
        window = TerrainWindow(
            requested_center_x_m=window.requested_center_x_m,
            requested_center_y_m=window.requested_center_y_m,
            x_centers_m=window.x_centers_m,
            y_centers_m=window.y_centers_m,
            elevation_m=elevation,
            nodata_mask=window.nodata_mask,
            row_offset=window.row_offset,
            column_offset=window.column_offset,
        )
        surface = reconstruct_lunar_terrain(window, metadata)
        normal = calculate_terrain_normals(surface).normals[2, 2]
        tangent_x = surface.positions_km[2, 3] - surface.positions_km[2, 1]
        tangent_y = surface.positions_km[1, 2] - surface.positions_km[3, 2]
        self.assertAlmostEqual(float(np.dot(normal, tangent_x)), 0.0, delta=1e-10)
        self.assertAlmostEqual(float(np.dot(normal, tangent_y)), 0.0, delta=1e-10)

    def test_nodata_neighbor_invalidates_normal_stencil(self) -> None:
        elevations = np.zeros((5, 5), dtype=np.float64)
        nodata = np.zeros((5, 5), dtype=bool)
        nodata[2, 3] = True
        window, metadata = make_window(elevations, nodata)
        window.elevation_m[2, 3] = np.nan
        surface = reconstruct_lunar_terrain(window, metadata)
        field = calculate_terrain_normals(surface)
        self.assertFalse(surface.valid_mask[2, 3])
        self.assertFalse(field.valid_mask[2, 2])
        self.assertTrue(np.all(np.isnan(field.normals[2, 2])))

    def test_surface_rejects_wrong_frame(self) -> None:
        window, metadata = make_window(np.zeros((5, 5), dtype=np.float64))
        wrong_frame_metadata = TerrainMetadata(
            crs=metadata.crs,
            projection=metadata.projection,
            affine=metadata.affine,
            width=metadata.width,
            height=metadata.height,
            pixel_size_m=metadata.pixel_size_m,
            nodata=metadata.nodata,
            vertical_reference=metadata.vertical_reference,
            source_frame="MOON_PA_DE440",
            bounds_m=metadata.bounds_m,
            pixel_registration=metadata.pixel_registration,
        )
        with self.assertRaises(ValueError):
            reconstruct_lunar_terrain(window, wrong_frame_metadata)


if __name__ == "__main__":
    unittest.main()
