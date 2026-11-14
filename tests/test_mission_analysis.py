"""Regression tests for sampled Site06 mission-window analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest
import json
from pathlib import Path

from core.lunar_coordinates import SITE06_NOBILE_RIM_1
from core.terrain_provider import MOON_ME_DE421
from core.mission_analysis import (
    TerrainHorizonConfiguration,
    analyze_mission_window,
    load_site06_10km_horizon_configuration,
)


@dataclass(frozen=True)
class FakeDirection:
    target: str
    azimuth_deg: float
    elevation_deg: float
    frame: str = MOON_ME_DE421


class FakeDirectionEngine:
    def __init__(self, values: dict[tuple[str, str], tuple[float, float]]) -> None:
        self.values = values
        self.calls: list[tuple[str, str]] = []

    def __enter__(self) -> "FakeDirectionEngine":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def celestial_direction(self, utc_timestamp: str, target: str) -> FakeDirection:
        self.calls.append((utc_timestamp, target))
        azimuth, elevation = self.values.get((utc_timestamp, target), (0.0, -1.0))
        return FakeDirection(target, azimuth, elevation)


class MissionAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = TerrainHorizonConfiguration(
            profile_deg=tuple([0.0] * 360),
            horizon_status="limited-range",
            horizon_range_km=10.0,
        )

    def analyze(self, start, end, step, values, profile=None):
        engine = FakeDirectionEngine(values)
        result = analyze_mission_window(
            SITE06_NOBILE_RIM_1,
            start,
            end,
            step,
            profile or self.profile,
            direction_engine_factory=lambda: engine,
        )
        return result, engine

    def test_empty_reversed_interval_returns_no_samples_or_windows(self) -> None:
        result, engine = self.analyze(
            "2026-10-06T00:00:00Z", "2026-10-05T00:00:00Z", 60, {}
        )
        self.assertEqual(result.samples, ())
        self.assertEqual(result.visibility_windows, ())
        self.assertEqual(result.sun_earth_overlap_windows, ())
        self.assertEqual(result.summary_for("SUN").total_visible_minutes, 0)
        self.assertEqual(engine.calls, [])

    def test_single_timestamp_inclusive_interval(self) -> None:
        result, _ = self.analyze(
            "2026-10-05T00:00:00Z",
            "2026-10-05T00:00:00Z",
            60,
            {("2026-10-05T00:00:00", "SUN"): (12.0, 5.0)},
        )
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(result.samples[0].timestamp_utc, "2026-10-05T00:00:00Z")
        self.assertEqual(result.summary_for("SUN").visible_fraction, 1.0)
        self.assertEqual(result.summary_for("SUN").total_visible_minutes, 0)
        self.assertEqual(result.summary_for("SUN").longest_visible_window_minutes, 0)

    def test_no_visible_samples_have_zero_metrics_and_no_windows(self) -> None:
        result, _ = self.analyze(
            "2026-10-05T00:00:00Z", "2026-10-05T02:00:00Z", 60, {}
        )
        for target in ("SUN", "EARTH"):
            summary = result.summary_for(target)
            self.assertEqual(summary.total_visible_minutes, 0)
            self.assertEqual(summary.visible_fraction, 0.0)
            self.assertEqual(summary.number_of_visibility_windows, 0)
            self.assertEqual(summary.longest_visible_window_minutes, 0)
        self.assertEqual(result.visibility_windows, ())

    def test_all_visible_samples_group_and_duration_is_elapsed_time(self) -> None:
        values = {
            (f"2026-10-05T0{hour}:00:00", target): (0.0, 10.0)
            for hour in range(3)
            for target in ("SUN", "EARTH")
        }
        result, _ = self.analyze("2026-10-05T00:00:00Z", "2026-10-05T02:00:00Z", 60, values)
        self.assertEqual(len(result.visibility_windows), 2)
        for target in ("SUN", "EARTH"):
            summary = result.summary_for(target)
            self.assertEqual(summary.visible_fraction, 1.0)
            self.assertEqual(summary.total_visible_minutes, 120)
            self.assertEqual(summary.longest_visible_window_minutes, 120)
            self.assertEqual(summary.number_of_visibility_windows, 1)
        self.assertEqual(len(result.sun_earth_overlap_windows), 1)
        overlap = result.sun_earth_overlap_windows[0]
        self.assertEqual((overlap.start_utc, overlap.end_utc), ("2026-10-05T00:00:00Z", "2026-10-05T02:00:00Z"))
        self.assertEqual(overlap.duration_minutes, 120)
        self.assertEqual(overlap.sample_count, 3)

    def test_visibility_gaps_split_windows_and_summary_metrics(self) -> None:
        values = {
            (f"2026-10-05T0{hour}:00:00", "SUN"): (0.0, 5.0 if hour in (0, 1, 3) else -1.0)
            for hour in range(4)
        }
        result, _ = self.analyze("2026-10-05T00:00:00Z", "2026-10-05T03:00:00Z", 60, values)
        sun_windows = [window for window in result.visibility_windows if window.target == "SUN"]
        self.assertEqual(len(sun_windows), 2)
        self.assertEqual([window.duration_minutes for window in sun_windows], [60, 0])
        self.assertEqual([window.sample_count for window in sun_windows], [2, 1])
        summary = result.summary_for("SUN")
        self.assertEqual(summary.total_visible_minutes, 60)
        self.assertEqual(summary.longest_visible_window_minutes, 60)
        self.assertEqual(summary.number_of_visibility_windows, 2)
        self.assertEqual(summary.visible_fraction, 3 / 4)

    def test_end_timestamp_is_included(self) -> None:
        result, engine = self.analyze(
            "2026-10-05T00:00:00Z", "2026-10-05T02:00:00Z", 60, {}
        )
        self.assertEqual(len({sample.timestamp_utc for sample in result.samples}), 3)
        self.assertEqual(result.samples[-1].timestamp_utc, "2026-10-05T02:00:00Z")
        self.assertIn(("2026-10-05T02:00:00", "EARTH"), engine.calls)

    def test_sun_earth_overlap_is_grouped_only_when_both_are_visible(self) -> None:
        values = {
            (f"2026-10-05T0{hour}:00:00", "SUN"): (0.0, 2.0)
            for hour in range(3)
        }
        values.update({
            ("2026-10-05T00:00:00", "EARTH"): (0.0, -1.0),
            ("2026-10-05T01:00:00", "EARTH"): (0.0, 2.0),
            ("2026-10-05T02:00:00", "EARTH"): (0.0, 2.0),
        })
        result, _ = self.analyze("2026-10-05T00:00:00Z", "2026-10-05T02:00:00Z", 60, values)
        self.assertEqual(len(result.sun_earth_overlap_windows), 1)
        overlap = result.sun_earth_overlap_windows[0]
        self.assertEqual(overlap.start_utc, "2026-10-05T01:00:00Z")
        self.assertEqual(overlap.end_utc, "2026-10-05T02:00:00Z")
        self.assertEqual(overlap.duration_minutes, 60)

    def test_circular_azimuth_interpolation_handles_359_to_zero(self) -> None:
        profile = [0.0] * 360
        profile[359], profile[0] = 10.0, 30.0
        config = TerrainHorizonConfiguration(tuple(profile), "limited-range", 10.0)
        result, _ = self.analyze(
            "2026-10-05T00:00:00Z",
            "2026-10-05T00:00:00Z",
            60,
            {("2026-10-05T00:00:00", "SUN"): (359.75, 40.0)},
            config,
        )
        sun = next(sample for sample in result.samples if sample.target == "SUN")
        self.assertAlmostEqual(sun.terrain_horizon_deg, 25.0)
        self.assertAlmostEqual(sun.clearance_deg, 15.0)
        self.assertEqual(sun.azimuth_deg, 359.75)

    def test_limited_range_metadata_is_preserved_in_result_and_records(self) -> None:
        result, _ = self.analyze(
            "2026-10-05T00:00:00Z", "2026-10-05T00:00:00Z", 60, {}
        )
        self.assertEqual(result.horizon_status, "limited-range")
        self.assertEqual(result.horizon_range_km, 10.0)
        self.assertTrue(all(record.horizon_status == "limited-range" for record in result.samples))
        self.assertTrue(all(record.horizon_range_km == 10.0 for record in result.samples))
        self.assertIn("not exact rise/set", result.sampling_note)

    def test_existing_saved_10km_profile_loads_without_dem_access(self) -> None:
        config = load_site06_10km_horizon_configuration()
        report_path = Path(__file__).resolve().parents[1] / "reports" / "site06_10km_limited_horizon.json"
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        expected = tuple(
            None if value is None else float(value)
            for value in report["profile_bins_1deg_5m_1km_plus_20m_regional_deg"]
        )
        self.assertEqual(config.profile_deg, expected)
        self.assertEqual(len(config.profile_deg), 360)
        self.assertEqual(config.horizon_range_km, 10.0)
        self.assertEqual(config.horizon_status, "limited-range")
        self.assertEqual(config.source_frame, MOON_ME_DE421)
        self.assertEqual(config.profile_name, "Saved Site06 5 m 1 km + 20 m 10 km composite profile")

    def test_real_spice_sun_and_earth_analysis_uses_saved_10km_profile(self) -> None:
        config = load_site06_10km_horizon_configuration()
        result = analyze_mission_window(
            SITE06_NOBILE_RIM_1,
            "2026-10-05T00:00:00Z",
            "2026-10-05T00:00:00Z",
            60,
            config,
        )
        self.assertEqual({sample.target for sample in result.samples}, {"SUN", "EARTH"})
        self.assertEqual(result.horizon_status, "limited-range")
        self.assertEqual(result.horizon_range_km, 10.0)
        for sample in result.samples:
            self.assertTrue(math.isfinite(sample.azimuth_deg))
            self.assertTrue(math.isfinite(sample.celestial_elevation_deg))
            self.assertTrue(math.isfinite(sample.terrain_horizon_deg))
            self.assertTrue(math.isfinite(sample.clearance_deg))
            self.assertEqual(sample.visible, sample.clearance_deg > 0.0)


if __name__ == "__main__":
    unittest.main()
