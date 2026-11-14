"""Discrete visibility window grouping and UTC sampling tests."""

from datetime import timedelta
import unittest
from unittest.mock import patch

from core.terrain_provider import MOON_ME_DE421
import core.visibility_timeseries as visibility_module

from core.visibility_timeseries import (
    VisibilitySample,
    VisibilityTimeSeries,
    generate_timestamps,
    group_both_visible_windows,
    group_visibility_windows,
)


def sample(target: str, hour: int, visible: bool, *, day: int = 5) -> VisibilitySample:
    return VisibilitySample(
        target=target,
        utc_timestamp=f"2026-10-{day:02d}T{hour:02d}:00:00Z",
        azimuth_deg=0.0,
        celestial_elevation_deg=1.0 if visible else -1.0,
        terrain_horizon_elevation_deg=0.0,
        clearance_deg=1.0 if visible else -1.0,
        visible=visible,
        horizon_range_km=1.0,
        horizon_status="limited-range",
    )


def timeseries(samples: tuple[VisibilitySample, ...], windows=()) -> VisibilityTimeSeries:
    return VisibilityTimeSeries(
        site="Site06",
        start_utc="2026-10-05T00:00:00Z",
        end_utc="2026-10-06T00:00:00Z",
        time_step_minutes=60,
        targets=("SUN",),
        samples=samples,
        visibility_windows=tuple(windows),
        both_visible_windows=(),
        horizon_status="limited-range",
        horizon_range_km=1.0,
    )


class VisibilityTimeSeriesTest(unittest.TestCase):
    def test_legacy_site06_timeseries_uses_the_saved_1km_profile(self) -> None:
        profile = tuple([3.25] * 360)

        class FakeDirection:
            def __init__(self, target):
                self.target = target
                self.utc_timestamp = "2026-10-05T00:00:00"
                self.frame = MOON_ME_DE421
                self.azimuth_deg = 10.0
                self.elevation_deg = 10.0

        class FakeEngine:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def celestial_direction(self, utc_timestamp, target):
                return FakeDirection(target)

        with patch.object(visibility_module, "load_site06_horizon_profile", return_value=profile) as loader, \
             patch.object(visibility_module, "Site06CelestialDirectionEngine", FakeEngine), \
             patch.object(visibility_module, "compare_with_terrain_horizon", wraps=visibility_module.compare_with_terrain_horizon) as compare:
            result = visibility_module.run_site06_visibility_timeseries(
                "2026-10-05T00:00:00Z", "2026-10-05T00:00:00Z", 60
            )

        loader.assert_called_once_with()
        self.assertEqual(compare.call_count, 2)
        self.assertTrue(all(call.args[1] is profile for call in compare.call_args_list))
        self.assertEqual(result.horizon_range_km, 1.0)
        self.assertEqual(result.horizon_status, "limited-range")
        self.assertEqual({sample.terrain_horizon_elevation_deg for sample in result.samples}, {3.25})

    def test_timestamp_generation_and_one_hour_stepping(self) -> None:
        times = generate_timestamps(
            "2026-10-05T00:00:00Z", "2026-10-05T03:00:00Z", 60
        )
        self.assertEqual(len(times), 4)
        self.assertEqual([t.hour for t in times], [0, 1, 2, 3])
        self.assertTrue(all(b - a == timedelta(hours=1) for a, b in zip(times, times[1:])))

    def test_validation_interval_has_25_inclusive_hourly_samples(self) -> None:
        times = generate_timestamps("2026-10-05T00:00:00", "2026-10-06T00:00:00", 60)
        self.assertEqual(len(times), 25)
        self.assertEqual(times[0].isoformat(), "2026-10-05T00:00:00+00:00")
        self.assertEqual(times[-1].isoformat(), "2026-10-06T00:00:00+00:00")

    def test_full_inclusive_24_hour_run_has_1440_minute_duration(self) -> None:
        timestamps = generate_timestamps("2026-10-05T00:00:00Z", "2026-10-06T00:00:00Z", 60)
        self.assertEqual(len(timestamps), 25)
        records = tuple(
            VisibilitySample("SUN", timestamp.isoformat().replace("+00:00", "Z"), 0, 1, 0, 1, True, 1, "limited-range")
            for timestamp in timestamps
        )
        windows = group_visibility_windows(records, 60)
        self.assertEqual(windows[0].sampled_duration_minutes, 1440)
        self.assertEqual(timeseries(records, windows).total_visible_sampled_duration_minutes("SUN"), 1440)

    def test_single_sample_visible_window(self) -> None:
        windows = group_visibility_windows([sample("SUN", 4, True)], 60)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].number_of_samples, 1)
        self.assertEqual(windows[0].sampled_duration_minutes, 0)
        self.assertIn("not exact", windows[0].boundary_note)

    def test_two_consecutive_visible_samples_have_60_minute_duration(self) -> None:
        windows = group_visibility_windows(
            [sample("SUN", 0, True), sample("SUN", 1, True)], 60
        )
        self.assertEqual(windows[0].sampled_duration_minutes, 60)

    def test_consecutive_visible_samples_form_one_window(self) -> None:
        windows = group_visibility_windows(
            [sample("SUN", 1, True), sample("SUN", 2, True), sample("SUN", 3, True)], 60
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_utc, "2026-10-05T01:00:00Z")
        self.assertEqual(windows[0].end_utc, "2026-10-05T03:00:00Z")
        self.assertEqual(windows[0].number_of_samples, 3)

    def test_window_includes_requested_end_sample_and_duration_stops_at_end(self) -> None:
        timestamps = generate_timestamps("2026-10-05T23:00:00Z", "2026-10-06T00:00:00Z", 60)
        records = tuple(
            VisibilitySample("SUN", timestamp.isoformat().replace("+00:00", "Z"), 0, 1, 0, 1, True, 1, "limited-range")
            for timestamp in timestamps
        )
        windows = group_visibility_windows(records, 60)
        self.assertEqual(windows[0].start_utc, "2026-10-05T23:00:00Z")
        self.assertEqual(windows[0].end_utc, "2026-10-06T00:00:00Z")
        self.assertEqual(windows[0].sampled_duration_minutes, 60)

    def test_invisible_samples_and_time_gaps_split_windows(self) -> None:
        with_gap = group_visibility_windows(
            [sample("SUN", 1, True), sample("SUN", 2, False), sample("SUN", 3, True)], 60
        )
        self.assertEqual(len(with_gap), 2)
        timestamp_gap = group_visibility_windows(
            [sample("SUN", 1, True), sample("SUN", 3, True)], 60
        )
        self.assertEqual(len(timestamp_gap), 2)

    def test_window_can_cross_midnight_and_date_boundary(self) -> None:
        records = [
            VisibilitySample("SUN", "2026-10-05T23:00:00Z", 0, 1, 0, 1, True, 1, "limited-range"),
            VisibilitySample("SUN", "2026-10-06T00:00:00Z", 0, 1, 0, 1, True, 1, "limited-range"),
            VisibilitySample("SUN", "2026-10-06T01:00:00Z", 0, 1, 0, 1, True, 1, "limited-range"),
        ]
        windows = group_visibility_windows(records, 60)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_utc, "2026-10-05T23:00:00Z")
        self.assertEqual(windows[0].end_utc, "2026-10-06T01:00:00Z")

    def test_targets_are_separated_and_concurrent_visibility_is_grouped(self) -> None:
        records = [
            sample("SUN", 0, True), sample("EARTH", 0, False),
            sample("SUN", 1, True), sample("EARTH", 1, True),
            sample("SUN", 2, False), sample("EARTH", 2, True),
        ]
        windows = group_visibility_windows(records, 60)
        self.assertEqual({window.target for window in windows}, {"SUN", "EARTH"})
        concurrent = group_both_visible_windows(records, 60)
        self.assertEqual(len(concurrent), 1)
        self.assertEqual(concurrent[0].target, "BOTH")
        self.assertEqual(concurrent[0].start_utc, "2026-10-05T01:00:00Z")
        self.assertEqual(concurrent[0].number_of_samples, 1)

    def test_limited_range_status_is_preserved(self) -> None:
        window = group_visibility_windows([sample("EARTH", 4, True)], 60)[0]
        self.assertEqual(window.horizon_status, "limited-range")
        self.assertEqual(window.horizon_range_km, 1.0)

    def test_no_visible_samples_produce_no_windows(self) -> None:
        records = (sample("SUN", 0, False), sample("SUN", 1, False))
        windows = group_visibility_windows(records, 60)
        self.assertEqual(windows, ())
        self.assertEqual(timeseries(records, windows).total_visible_sampled_duration_minutes("SUN"), 0)


if __name__ == "__main__":
    unittest.main()
