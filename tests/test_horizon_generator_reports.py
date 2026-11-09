"""Offline adapter tests for generator-to-provenance-report integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from core.horizon_report import validate_horizon_report
from core.terrain_horizon import TerrainHorizonProfile
from core.terrain_provider import MOON_ME_DE421, TerrainMetadata, TerrainWindow
from scripts.validate_site06_5m_horizon_crosscheck import (
    SITE_XY_M,
    _make_source_record as make_5m_source_record,
    build_review_report,
)
from scripts.validate_site06_regional_horizon import (
    _make_provider_source_record,
    _union_with_contributors,
    build_versioned_report,
)


def _profile(values, counts=None):
    return TerrainHorizonProfile(
        azimuth_centers_deg=np.arange(360, dtype=np.float64),
        max_obstruction_elevation_deg=np.asarray(values, dtype=np.float64),
        sample_count=np.ones(360, dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64),
        source_frame=MOON_ME_DE421,
        north_axis=(0.0, 0.0, 1.0),
        east_axis=(0.0, 1.0, 0.0),
        up_axis=(1.0, 0.0, 0.0),
    )


class HorizonGeneratorReportIntegrationTest(unittest.TestCase):
    def test_equal_source_horizons_both_receive_composite_attribution(self):
        five_m = np.array([2.0, np.nan, 4.0])
        regional = np.array([2.0, 3.0, 1.0])
        combined, contributors = _union_with_contributors(regional, five_m)
        np.testing.assert_array_equal(combined, np.array([2.0, 3.0, 4.0]))
        self.assertEqual(contributors, [["lola_5m", "lola_20m"], ["lola_20m"], ["lola_5m"]])

    def test_5m_rasterio_window_populates_distinct_report_source_metadata(self):
        values = np.full((400, 400), 100.0, dtype=np.float32)
        transform = from_origin(83000.0, 111000.0, 5.0, 5.0)
        with MemoryFile() as memory_file:
            with memory_file.open(
                driver="GTiff", width=400, height=400, count=1, dtype="float32",
                crs=CRS.from_epsg(3857), transform=transform, nodata=-9999.0,
            ) as dataset:
                dataset.write(values, 1)
                dataset.scales = (1.25,)
                dataset.offsets = (3.0,)
                dataset.update_tags(AREA_OR_POINT="Area")
                window_values = dataset.read(1, window=((199, 201), (199, 201)))
                nodata_mask = window_values == dataset.nodata
                window = SimpleNamespace(elevation_m=window_values, nodata_mask=nodata_mask)
                counts = {
                    "samples_in_1km_circle": 4,
                    "valid_coverage_samples": 4,
                    "nodata_nan_samples": 0,
                    "other_invalid_samples": 0,
                    "out_of_coverage_samples": 0,
                }
                record = make_5m_source_record(
                    "lola_5m", "https://example.invalid/site06_5m.tif", dataset,
                    window, (199, 201, 199, 201), counts,
                )

        p5 = _profile(np.ones(360), np.full(360, 7))
        p20 = _profile(np.full(360, 2.0), np.full(360, 3))
        source20 = dict(record, source_id="lola_20m", source_url="https://example.invalid/lola_20m.tif",
                        product_id="lola_20m.tif", resolution_m=[20.0, 20.0])
        source20["returned_window"] = {"rows_half_open": [199, 201], "columns_half_open": [199, 201]}
        source20["requested_window"] = source20["returned_window"]
        report = build_review_report(p5, p20, [record, source20], {"python": "test"}, [], {})

        validate_horizon_report(report)
        self.assertEqual(report["sources"][0]["resolution_m"], [5.0, 5.0])
        self.assertEqual(report["sources"][0]["scale"], 1.25)
        self.assertEqual(report["sources"][0]["offset"], 3.0)
        self.assertFalse(report["sources"][0]["elevation_units_evidence"]["scale_offset_applied_by_generator"])
        self.assertEqual(report["sources"][0]["returned_window"]["rows_half_open"], [199, 201])
        self.assertEqual(report["horizon"]["comparison_profiles"][0]["profile_id"], "native_20m_1km")

    def test_mocked_provider_window_is_reported_and_composite_contributors_are_explicit(self):
        class Provider:
            source = "https://example.invalid/LDEM_80S_20MPP_ADJ.TIF"

        metadata = TerrainMetadata(
            crs='PROJCRS["synthetic"]', projection="Polar Stereographic",
            affine=(20.0, 0.0, -100.0, 0.0, -20.0, 100.0),
            width=20, height=20, pixel_size_m=(20.0, 20.0), nodata=float("nan"),
            vertical_reference="synthetic", source_frame=MOON_ME_DE421,
            bounds_m=(-100.0, -300.0, 300.0, 100.0), pixel_registration="PixelIsArea",
            scale=1.0, offset=0.0, dtype="float32", band_unit="m",
        )
        elevations = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
        window = TerrainWindow(
            requested_center_x_m=0.0, requested_center_y_m=0.0,
            x_centers_m=np.array([10.0, 30.0]), y_centers_m=np.array([10.0, -10.0]),
            elevation_m=elevations, nodata_mask=np.isnan(elevations),
            row_offset=4, column_offset=5,
        )
        counts = {"requested_area_pixel_count": 4, "valid_pixel_count": 3,
                  "nodata_pixel_count": 1, "other_invalid_pixel_count": 0,
                  "out_of_coverage_pixel_count": 0}
        source20 = _make_provider_source_record(Provider(), metadata, window, (4, 6), (5, 7), counts)
        source5 = dict(source20, source_id="lola_5m", source_url="https://example.invalid/site06_5m.tif",
                       product_id="site06_5m.tif", resolution_m=[5.0, 5.0])
        source5["requested_window"] = {"rows_half_open": [10, 12], "columns_half_open": [11, 13]}
        source5["returned_window"] = source5["requested_window"]

        five_m = np.ones(360)
        regional = np.full(360, 2.0)
        five_m[0] = 3.0
        regional[1] = 2.0
        five_m[2] = regional[2] = 4.0
        report = build_versioned_report(
            radius_m=10000.0, status="limited-range",
            site_record={"id": "Site06", "projected_xy_m": list(SITE_XY_M)},
            sources=[source5, source20], five_m_angles=five_m,
            five_m_counts=np.full(360, 5), regional_profile=_profile(regional, np.full(360, 8)),
            software={"python": "test"}, kernels=[], unavailable={},
        )

        validate_horizon_report(report)
        self.assertEqual(report["sources"][1]["returned_window"]["rows_half_open"], [4, 6])
        self.assertEqual(report["sources"][1]["coverage_counts"]["nodata_pixel_count"], 1)
        self.assertEqual(report["horizon"]["profile_angles_deg"][:3], [3.0, 2.0, 4.0])
        self.assertEqual(report["horizon"]["contributing_source_ids_by_bin"][:3],
                         [["lola_5m"], ["lola_20m"], ["lola_5m", "lola_20m"]])
        self.assertEqual(report["horizon"]["coverage_status"], "limited-range")


if __name__ == "__main__":
    unittest.main()
