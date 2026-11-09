"""Offline tests for versioned terrain-horizon provenance reports."""

from __future__ import annotations

import json
import unittest

import numpy as np

from core.horizon_report import (
    SCHEMA_VERSION,
    make_horizon_report,
    make_raster_source_record,
    serialize_horizon_report,
    summarize_raster_window,
    validate_horizon_report,
    validate_raster_source_record,
)


def _source(source_id: str, resolution: float, *, missing_scale: bool = False):
    unavailable = {"/scale": "The current generator interface does not retain raster scale metadata."} if missing_scale else {}
    return make_raster_source_record(
        source_id=source_id,
        source_url=f"https://example.invalid/{source_id}.tif",
        product_id=source_id,
        source_hash=f"synthetic-hash-{source_id}",
        resolution_m=(resolution, resolution),
        raster_dimensions_rows_columns=(8, 9),
        dtype="float32",
        crs_wkt='PROJCRS["synthetic"]',
        affine=(resolution, 0.0, 100.0, 0.0, -resolution, 200.0),
        pixel_registration="PixelIsArea; pixel centers",
        elevation_units="m",
        scale=None if missing_scale else 1.0,
        offset=0.0,
        nodata_value="NaN",
        nodata_handling="NaN and non-finite samples masked.",
        elevation_units_evidence={"status": "code_assigned", "detail": "Synthetic test assumption."},
        elevation=np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32),
        nodata_mask=np.array([[False, False], [False, False]]),
        coverage_counts={"requested_area_pixel_count": 4, "valid_pixel_count": 3, "nodata_pixel_count": 1, "other_invalid_pixel_count": 0, "out_of_coverage_pixel_count": 0},
        requested_window={"rows_half_open": (2, 4), "columns_half_open": (3, 5)},
        returned_window={"rows_half_open": (2, 4), "columns_half_open": (3, 5)},
        source_frame="MOON_ME_DE421",
        frame_evidence_status="code_assigned",
        frame_evidence={"detail": "Synthetic test label only; not independent product verification."},
        unavailable=unavailable,
    )


def _report(sources=None):
    sources = list(sources or (_source("lola_5m", 5.0), _source("lola_20m", 20.0)))
    return make_horizon_report(
        generated_at_utc="2026-10-30T12:00:00Z",
        site={
            "id": "Site06",
            "geographic": {
                "latitude_deg": -85.4381,
                "longitude_deg": 37.36667,
                "latitude_type": "planetocentric",
                "longitude_convention": "east_positive",
            },
            "projected_xy_m": [84000.0, 110000.0],
            "frame": "MOON_ME_DE421",
        },
        sources=sources,
        radius_m=10000.0,
        coverage_status="limited-range",
        azimuth_bins={
            "count": 4,
            "width_deg": 90.0,
            "centers_deg": [0.0, 90.0, 180.0, 270.0],
            "convention": "0=N, clockwise",
            "assignment": "nearest center",
        },
        algorithm={"name": "sampled max obstruction", "version": "1"},
        profile_angles_deg=[1.0, 2.0, 3.0, 4.0],
        sample_counts_by_bin=[10, 11, 12, 13],
        contributing_source_ids_by_bin=[["lola_5m", "lola_20m"] for _ in range(4)],
        composite_rule="Per-bin maximum across declared sources.",
        software={"python": "synthetic", "numpy": np.__version__},
        spice_kernels=[],
        unavailable={"/spice_kernels": "Synthetic test did not run SPICE."},
        limitations=["Synthetic offline test report."],
    )


class HorizonReportTest(unittest.TestCase):
    def test_summarizes_synthetic_array_and_preserves_nan_as_missing(self):
        summary = summarize_raster_window(
            np.array([[0.0, 2.0], [4.0, np.nan]]),
            np.array([[False, False], [False, False]]),
        )
        self.assertEqual(summary["shape_rows_columns"], [2, 2])
        self.assertEqual(summary["valid_pixel_count"], 3)
        self.assertEqual(summary["nodata_pixel_count"], 1)
        self.assertEqual(summary["minimum_elevation_m"], 0.0)
        self.assertEqual(summary["maximum_elevation_m"], 4.0)
        self.assertEqual(summary["median_elevation_m"], 2.0)

    def test_keeps_5m_and_20m_source_provenance_separate(self):
        report = _report()
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual([item["source_id"] for item in report["sources"]], ["lola_5m", "lola_20m"])
        self.assertEqual([item["resolution_m"] for item in report["sources"]], [[5.0, 5.0], [20.0, 20.0]])
        self.assertNotEqual(report["sources"][0]["source_url"], report["sources"][1]["source_url"])
        validate_horizon_report(report)

    def test_serialization_is_valid_json_and_round_trips(self):
        report = _report()
        encoded = serialize_horizon_report(report)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema_version"], SCHEMA_VERSION)
        self.assertEqual(decoded["horizon"]["coverage_status"], "limited-range")

    def test_null_metadata_requires_an_explanation(self):
        source = _source("lola_20m", 20.0, missing_scale=True)
        source["unavailable"].clear()
        with self.assertRaisesRegex(ValueError, "null value requires"):
            validate_raster_source_record(source)

    def test_code_assigned_frame_is_not_mislabeled_independently_verified(self):
        source = _source("lola_20m", 20.0)
        source["source_frame"]["evidence_status"] = "independently_verified"
        source["source_frame"]["evidence"] = None
        source["unavailable"]["/source_frame/evidence"] = "No independent evidence in this synthetic case."
        with self.assertRaisesRegex(ValueError, "require evidence"):
            validate_raster_source_record(source)

    def test_invalid_window_counts_and_nonfinite_affine_are_rejected(self):
        source = _source("lola_20m", 20.0)
        source["returned_window"]["rows_half_open"] = [4, 2]
        with self.assertRaisesRegex(ValueError, "half-open"):
            validate_raster_source_record(source)

        source = _source("lola_20m", 20.0)
        source["window_statistics"]["valid_pixel_count"] = 5
        with self.assertRaisesRegex(ValueError, "counts must equal"):
            validate_raster_source_record(source)

        source = _source("lola_20m", 20.0)
        source["affine"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "six finite"):
            validate_raster_source_record(source)

    def test_report_rejects_bad_bins_radius_and_unexplained_nulls(self):
        report = _report()
        report["horizon"]["radius_m"] = 0
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_horizon_report(report)

        report = _report()
        report["horizon"]["profile_angles_deg"].pop()
        with self.assertRaisesRegex(ValueError, "match azimuth"):
            validate_horizon_report(report)

        report = _report()
        report["software"]["missing_version"] = None
        with self.assertRaisesRegex(ValueError, "null value requires"):
            validate_horizon_report(report)


if __name__ == "__main__":
    unittest.main()
