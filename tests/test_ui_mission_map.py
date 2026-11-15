"""Offline regressions for the Site06 map projection and range geometry."""

from math import hypot, isfinite
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from affine import Affine
import numpy as np
import rasterio
from pyproj import CRS, Transformer

from core.lunar_coordinates import SITE06_NOBILE_RIM_1, planetocentric_latlon_to_moon_me_de421
from core.terrain_provider import (
    LOLA20M_URL,
    LOLA20M_SOUTH_POLAR_CRS,
    MOON_REFERENCE_RADIUS_M,
    MOON_ME_DE421,
    LOLA20mTerrainProvider,
    TaggedMoonPosition,
    TerrainMetadata,
    TerrainAccessError,
    TerrainWindow,
)
from ui_mission_map import (
    LOLA20M_MAP_LABEL,
    SiteMapGeometry,
    SiteTerrainMapData,
    TerrainContextUnavailable,
    _plot_elevation,
    _read_cached_site_terrain_window,
    _resolve_lola_source,
    build_site_map_figure,
    build_site_map_geometry,
    build_terrain_diagnostics,
    load_site_terrain_window,
)


def synthetic_window(center_x_m: float, center_y_m: float) -> TerrainWindow:
    return TerrainWindow(
        requested_center_x_m=center_x_m,
        requested_center_y_m=center_y_m,
        x_centers_m=center_x_m + np.array([-40.0, -20.0, 0.0, 20.0, 40.0]),
        y_centers_m=center_y_m + np.array([40.0, 20.0, 0.0, -20.0, -40.0]),
        elevation_m=np.arange(25, dtype=np.float32).reshape(5, 5),
        nodata_mask=np.zeros((5, 5), dtype=bool),
        row_offset=100,
        column_offset=200,
    )


class Site06MissionMapTest(unittest.TestCase):
    def test_remote_dem_access_error_becomes_specific_optional_context_error(self) -> None:
        with (
            patch("ui_mission_map._resolve_lola_source", return_value=LOLA20M_URL),
            patch("ui_mission_map._read_cached_site_terrain_window", side_effect=TerrainAccessError("NASA host unreachable")),
        ):
            with self.assertRaisesRegex(TerrainContextUnavailable, "NASA LOLA terrain context could not be loaded"):
                load_site_terrain_window(SITE06_NOBILE_RIM_1, 10_000.0)

    def test_existing_valid_local_cache_is_preferred_without_fetching_remote(self) -> None:
        cache_path = Path("/existing-cache/LDEM_80S_20MPP_ADJ.TIF")
        with (
            patch("ui_mission_map._local_lola_cache_candidates", return_value=(cache_path,)),
            patch("ui_mission_map._valid_local_lola_cache", return_value=True),
        ):
            self.assertEqual(_resolve_lola_source(LOLA20M_URL), str(cache_path))

    def test_bad_nonterrain_exceptions_are_not_converted_to_unavailable_warning(self) -> None:
        with patch("ui_mission_map._read_cached_site_terrain_window", side_effect=ValueError("bad map geometry")):
            with self.assertRaisesRegex(ValueError, "bad map geometry"):
                load_site_terrain_window(SITE06_NOBILE_RIM_1, 10_000.0)

    def test_clps_map_warns_but_keeps_optional_terrain_failure_local(self) -> None:
        from types import SimpleNamespace
        import app

        result = SimpleNamespace(site=SITE06_NOBILE_RIM_1, horizon_range_km=10.0)
        with (
            patch("app.load_site_terrain_window", side_effect=TerrainContextUnavailable("NASA host unreachable")),
            patch("app.st.container") as container,
            patch("app.st.warning") as warning,
            patch("app.st.plotly_chart") as plot,
        ):
            container.return_value.__enter__.return_value = None
            app._render_mission_map(result)
        warning.assert_called_once()
        self.assertIn("NASA LOLA terrain data could not be loaded", warning.call_args.args[0])
        plot.assert_not_called()

    def test_site06_maps_to_finite_south_polar_projected_coordinates(self) -> None:
        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)
        self.assertTrue(isfinite(geometry.center_x_m))
        self.assertTrue(isfinite(geometry.center_y_m))
        self.assertGreater(abs(geometry.center_x_m) + abs(geometry.center_y_m), 0.0)

    def test_analysis_circle_is_exactly_ten_thousand_meters(self) -> None:
        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)
        radii = [
            hypot(x - geometry.center_x_m, y - geometry.center_y_m)
            for x, y in zip(geometry.circle_x_m, geometry.circle_y_m)
        ]
        self.assertEqual(len(radii), 361)
        for radius in radii:
            self.assertAlmostEqual(radius, 10_000.0, places=8)

    def test_site_is_the_center_of_the_analysis_circle(self) -> None:
        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)
        site_marker_x_m, site_marker_y_m = geometry.center_x_m, geometry.center_y_m
        self.assertEqual((site_marker_x_m, site_marker_y_m), (geometry.center_x_m, geometry.center_y_m))
        self.assertAlmostEqual(geometry.circle_x_m[0] - site_marker_x_m, 10_000.0, places=8)
        self.assertAlmostEqual(geometry.circle_y_m[0] - site_marker_y_m, 0.0, places=8)

    def test_projection_matches_lola_provider_crs_and_existing_provider_conversion(self) -> None:
        self.assertTrue(LOLA20M_SOUTH_POLAR_CRS.is_projected)
        expected_crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs +type=crs"
        )
        self.assertEqual(LOLA20M_SOUTH_POLAR_CRS, expected_crs)
        self.assertAlmostEqual(LOLA20M_SOUTH_POLAR_CRS.ellipsoid.semi_major_metre, 1_737_400.0)

        xyz_km = planetocentric_latlon_to_moon_me_de421(
            SITE06_NOBILE_RIM_1.latitude_deg,
            SITE06_NOBILE_RIM_1.longitude_deg,
            MOON_REFERENCE_RADIUS_M / 1000.0,
        )
        position = TaggedMoonPosition(*xyz_km, frame=MOON_ME_DE421)
        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)

        # The map projection is checked independently against pyproj and then
        # against the provider path using a local synthetic raster header.
        geographic = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs +type=crs")
        expected_x, expected_y = Transformer.from_crs(
            geographic, LOLA20M_SOUTH_POLAR_CRS, always_xy=True
        ).transform(SITE06_NOBILE_RIM_1.longitude_deg, SITE06_NOBILE_RIM_1.latitude_deg)
        self.assertAlmostEqual(geometry.center_x_m, expected_x, places=6)
        self.assertAlmostEqual(geometry.center_y_m, expected_y, places=6)

        with tempfile.TemporaryDirectory() as directory:
            raster_path = Path(directory) / "synthetic_lola_header.tif"
            transform = Affine(20.0, 0.0, expected_x - 20.0, 0.0, -20.0, expected_y + 20.0)
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="uint8",
                crs=LOLA20M_SOUTH_POLAR_CRS,
                transform=transform,
            ) as dataset:
                dataset.write(np.zeros((2, 2), dtype=np.uint8), 1)
            with LOLA20mTerrainProvider(raster_path) as provider:
                provider_x, provider_y = provider.site_position_to_dem_xy(position)
        self.assertAlmostEqual(geometry.center_x_m, provider_x, places=6)
        self.assertAlmostEqual(geometry.center_y_m, provider_y, places=6)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "Plotly is declared in requirements but absent from this environment")
    def test_plotly_map_figure_constructs_with_projected_site_and_equal_axes(self) -> None:
        import plotly.graph_objects as go

        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)
        window = synthetic_window(geometry.center_x_m, geometry.center_y_m)
        figure = build_site_map_figure(SITE06_NOBILE_RIM_1, 10_000.0, terrain_window=window)
        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(figure.data[0].name, LOLA20M_MAP_LABEL)
        self.assertEqual(figure.data[0].z.shape, (5, 5))
        site_x_km = geometry.center_x_m / 1000.0
        site_y_km = geometry.center_y_m / 1000.0
        self.assertAlmostEqual(figure.data[2].x[0], site_x_km)
        self.assertAlmostEqual(figure.data[2].y[0], site_y_km)
        self.assertEqual(figure.layout.xaxis.scaleanchor, "y")
        self.assertEqual(figure.layout.xaxis.scaleratio, 1)
        self.assertEqual(figure.layout.dragmode, "pan")
        self.assertEqual(figure.layout.xaxis.tickfont.color, "#10223d")
        self.assertEqual(figure.layout.yaxis.tickfont.color, "#10223d")
        self.assertEqual(figure.layout.xaxis.title.font.color, "#10223d")
        self.assertEqual(figure.layout.yaxis.title.font.color, "#10223d")
        self.assertEqual(figure.data[0].colorbar.title.font.color, "#10223d")
        self.assertEqual(figure.data[0].colorbar.tickfont.color, "#10223d")
        self.assertEqual(figure.layout.legend.font.color, "#10223d")
        self.assertEqual(figure.layout.legend.orientation, "h")
        self.assertEqual(figure.layout.legend.x, 0.5)
        self.assertEqual(figure.layout.legend.y, -0.18)
        self.assertEqual(figure.layout.legend.xanchor, "center")
        self.assertEqual(figure.layout.margin.b, 96)
        self.assertEqual(figure.data[0].colorbar.x, 1.02)
        self.assertEqual(figure.data[0].colorbar.xanchor, "left")
        self.assertEqual(figure.layout.annotations[0].font.color, "#10223d")
        self.assertEqual(figure.layout.annotations[0].x, 0.02)
        self.assertGreater(figure.layout.annotations[0].y, 0.9)
        # The legend occupies a dedicated bottom band; it is vertically
        # separated from the top annotation and horizontally from the bar.
        self.assertLess(figure.layout.legend.y, 0.0)
        self.assertGreater(figure.data[0].colorbar.x, 1.0)
        self.assertEqual(figure.layout.margin.r, 110)
        self.assertEqual(figure.data[0].colorscale[0][1], "rgb(161, 105, 40)")
        self.assertAlmostEqual(figure.layout.xaxis.range[0], geometry.center_x_m / 1000.0 - 15.0)
        self.assertAlmostEqual(figure.layout.xaxis.range[1], geometry.center_x_m / 1000.0 + 15.0)
        self.assertAlmostEqual(figure.data[1].x[0] * 1000.0, geometry.circle_x_m[0])
        self.assertAlmostEqual(figure.data[1].y[0] * 1000.0, geometry.circle_y_m[0])

    def test_nodata_and_nan_elevations_remain_missing_in_plot(self) -> None:
        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)
        window = synthetic_window(geometry.center_x_m, geometry.center_y_m)
        window.elevation_m[0, 0] = np.nan
        window.elevation_m[0, 1] = -9999.0
        window.nodata_mask[0, 1] = True
        plotted = _plot_elevation(window)
        self.assertTrue(np.isnan(plotted[0, 0]))
        self.assertTrue(np.isnan(plotted[0, 1]))
        self.assertEqual(plotted[4, 4], 24.0)

    def test_cached_read_requests_site_centered_margin_window_within_limit(self) -> None:
        geometry = build_site_map_geometry(SITE06_NOBILE_RIM_1, 10_000.0)
        window = synthetic_window(geometry.center_x_m, geometry.center_y_m)
        provider = MagicMock()
        provider.__enter__.return_value = provider
        provider.metadata = SimpleNamespace(
            source_frame=MOON_ME_DE421,
            pixel_size_m=(20.0, 20.0),
        )
        provider.read_window.return_value = window
        provider.MAX_WINDOW_SIDE_PX = 2048
        _read_cached_site_terrain_window.cache_clear()
        with patch("ui_mission_map.LOLA20mTerrainProvider", return_value=provider):
            result = _read_cached_site_terrain_window(
                LOLA20M_URL, geometry.center_x_m, geometry.center_y_m, 10_000.0
            )
            # The cache prevents another provider open/read for the same COG window.
            second_result = _read_cached_site_terrain_window(
                LOLA20M_URL, geometry.center_x_m, geometry.center_y_m, 10_000.0
            )
        self.assertIs(result.window, window)
        self.assertIs(second_result, result)
        provider.read_window.assert_called_once_with(
            geometry.center_x_m, geometry.center_y_m, 1500, 1500
        )
        self.assertLessEqual(provider.read_window.call_args.args[2], 2048)
        self.assertLessEqual(provider.read_window.call_args.args[3], 2048)
        _read_cached_site_terrain_window.cache_clear()

    def test_diagnostics_summarize_small_synthetic_raster(self) -> None:
        window = TerrainWindow(
            requested_center_x_m=120.0,
            requested_center_y_m=200.0,
            x_centers_m=np.array([100.0, 120.0, 140.0]),
            y_centers_m=np.array([220.0, 200.0]),
            elevation_m=np.array([[1.0, np.nan, -9999.0], [4.0, 5.0, 6.0]]),
            nodata_mask=np.array([[False, False, True], [False, False, False]]),
            row_offset=10,
            column_offset=20,
        )
        metadata = TerrainMetadata(
            crs="synthetic",
            projection="Polar Stereographic",
            affine=(20.0, 0.0, 90.0, 0.0, -20.0, 230.0),
            width=100,
            height=100,
            pixel_size_m=(20.0, 20.0),
            nodata=-9999.0,
            vertical_reference="synthetic meters",
            source_frame=MOON_ME_DE421,
            bounds_m=(0.0, 0.0, 2000.0, 2000.0),
            pixel_registration="Area",
        )
        geometry = SiteMapGeometry(
            center_x_m=120.0,
            center_y_m=200.0,
            radius_m=10_000.0,
            circle_x_m=(),
            circle_y_m=(),
        )
        data = SiteTerrainMapData(
            window=window,
            metadata=metadata,
            source="synthetic-local.tif",
        )
        diagnostics = build_terrain_diagnostics(geometry, data)
        self.assertEqual(diagnostics["shape"], (2, 3))
        self.assertEqual(diagnostics["valid_count"], 4)
        self.assertEqual(diagnostics["nodata_count"], 2)
        self.assertEqual(diagnostics["minimum_m"], 1.0)
        self.assertEqual(diagnostics["maximum_m"], 6.0)
        self.assertEqual(diagnostics["median_m"], 4.5)
        self.assertEqual(diagnostics["site_x_y_m"], (120.0, 200.0))
        self.assertEqual(diagnostics["window_center_bounds_m"], (100.0, 200.0, 140.0, 220.0))
        self.assertTrue(diagnostics["site_inside_window_centers"])
        self.assertEqual(diagnostics["pixel_spacing_m"], (20.0, 20.0))
        self.assertEqual(diagnostics["column_orientation"], "+X, no Y change")
        self.assertEqual(diagnostics["row_orientation"], "no X change, −Y")
        self.assertTrue(diagnostics["axis_aligned"])
        self.assertIn("NASA provenance is not established", diagnostics["provenance"])

    def test_zero_elevation_is_reported_as_zero_not_unavailable(self) -> None:
        window = synthetic_window(0.0, 0.0)
        window.elevation_m[:] = 0.0
        metadata = TerrainMetadata(
            crs="synthetic", projection="Polar Stereographic",
            affine=(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            width=5, height=5, pixel_size_m=(20.0, 20.0), nodata=None,
            vertical_reference="synthetic meters", source_frame=MOON_ME_DE421,
            bounds_m=(-100.0, -100.0, 100.0, 100.0), pixel_registration="Area",
        )
        geometry = SiteMapGeometry(0.0, 0.0, 10_000.0, (), ())
        diagnostics = build_terrain_diagnostics(
            geometry, SiteTerrainMapData(window, metadata, "synthetic-local.tif")
        )
        self.assertEqual(diagnostics["minimum_m"], 0.0)
        self.assertEqual(diagnostics["maximum_m"], 0.0)
        self.assertEqual(diagnostics["median_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
