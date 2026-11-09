"""Discrete Site06 Sun/Earth visibility sampling against a saved 1 km horizon.

Window boundaries are the first and last visible sample timestamps. They are
sample-based estimates, not exact rise/set transition times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .celestial_direction import Site06CelestialDirectionEngine
from .terrain_visibility import (
    SITE06_HORIZON_RANGE_KM,
    compare_with_terrain_horizon,
    load_site06_horizon_profile,
)


UTC = timezone.utc


@dataclass(frozen=True)
class VisibilitySample:
    target: str
    utc_timestamp: str
    azimuth_deg: float
    celestial_elevation_deg: float
    terrain_horizon_elevation_deg: float
    clearance_deg: float
    visible: bool
    horizon_range_km: float
    horizon_status: str


@dataclass(frozen=True)
class VisibilityWindow:
    """A consecutive run of visible samples; boundaries are not exact events."""

    target: str
    start_utc: str
    end_utc: str
    number_of_samples: int
    horizon_status: str
    horizon_range_km: float
    sampled_duration_minutes: int
    boundary_note: str = "Sample-based boundaries; not exact rise/set transition times."


@dataclass(frozen=True)
class VisibilityTimeSeries:
    site: str
    start_utc: str
    end_utc: str
    time_step_minutes: int
    targets: tuple[str, ...]
    samples: tuple[VisibilitySample, ...]
    visibility_windows: tuple[VisibilityWindow, ...]
    both_visible_windows: tuple[VisibilityWindow, ...]
    horizon_status: str
    horizon_range_km: float

    def visible_sample_count(self, target: str) -> int:
        canonical = target.strip().upper()
        return sum(sample.visible and sample.target == canonical for sample in self.samples)

    def total_visible_sampled_duration_minutes(self, target: str) -> int:
        """Sum elapsed time between visible samples; a singleton has zero duration."""
        canonical = target.strip().upper()
        return sum(
            window.sampled_duration_minutes
            for window in self.visibility_windows
            if window.target == canonical
        )


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_utc(value: datetime, *, spice_compatible: bool = False) -> str:
    plain = value.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
    return plain if spice_compatible else plain + "Z"


def generate_timestamps(start_utc: str, end_utc: str, time_step_minutes: int) -> tuple[datetime, ...]:
    """Generate UTC timestamps from start through end, including an exact end."""
    if not isinstance(time_step_minutes, int) or time_step_minutes <= 0:
        raise ValueError("time_step_minutes must be a positive integer.")
    start = _parse_utc(start_utc)
    end = _parse_utc(end_utc)
    if end < start:
        raise ValueError("end UTC must be at or after start UTC.")
    step = timedelta(minutes=time_step_minutes)
    values: list[datetime] = []
    current = start
    while current <= end:
        values.append(current)
        current += step
    return tuple(values)


def _make_windows(
    samples: Iterable[VisibilitySample],
    time_step_minutes: int,
    *,
    require_both_targets: bool = False,
) -> tuple[VisibilityWindow, ...]:
    samples = tuple(samples)
    step = timedelta(minutes=time_step_minutes)
    visible_by_time: dict[datetime, dict[str, VisibilitySample]] = {}
    targets: set[str] = set()
    for sample in samples:
        targets.add(sample.target)
        if sample.visible:
            visible_by_time.setdefault(_parse_utc(sample.utc_timestamp), {})[sample.target] = sample

    if require_both_targets:
        if not {"SUN", "EARTH"}.issubset(targets):
            return ()
        active: list[VisibilitySample] = []
        windows: list[VisibilityWindow] = []
        previous_time: datetime | None = None
        for timestamp in sorted(visible_by_time):
            records = visible_by_time[timestamp]
            both_visible = "SUN" in records and "EARTH" in records
            if both_visible and (previous_time is None or timestamp - previous_time == step):
                # Store one sample per concurrent timestamp; horizon provenance
                # is identical for both targets at this Site06 run.
                active.append(records["SUN"])
                previous_time = timestamp
            else:
                if active:
                    windows.append(_window_from_run("BOTH", active))
                active = [records["SUN"]] if both_visible else []
                previous_time = timestamp if both_visible else None
        if active:
            windows.append(_window_from_run("BOTH", active))
        return tuple(windows)

    windows = []
    for target in sorted(targets):
        target_samples = sorted(
            (sample for sample in samples if sample.target == target),
            key=lambda sample: _parse_utc(sample.utc_timestamp),
        )
        active = []
        previous_time = None
        for sample in target_samples:
            timestamp = _parse_utc(sample.utc_timestamp)
            if sample.visible and (previous_time is None or timestamp - previous_time == step):
                active.append(sample)
                previous_time = timestamp
            else:
                if active:
                    windows.append(_window_from_run(target, active))
                active = [sample] if sample.visible else []
                previous_time = timestamp if sample.visible else None
        if active:
            windows.append(_window_from_run(target, active))
    return tuple(sorted(windows, key=lambda window: (_parse_utc(window.start_utc), window.target)))


def _window_from_run(target: str, run: list[VisibilitySample]) -> VisibilityWindow:
    first, last = run[0], run[-1]
    # The sample timestamps mark the observed window endpoints. Count only
    # elapsed time between them: n cadence samples span n-1 intervals.
    elapsed = _parse_utc(last.utc_timestamp) - _parse_utc(first.utc_timestamp)
    return VisibilityWindow(
        target=target,
        start_utc=first.utc_timestamp,
        end_utc=last.utc_timestamp,
        number_of_samples=len(run),
        horizon_status=first.horizon_status,
        horizon_range_km=first.horizon_range_km,
        sampled_duration_minutes=int(elapsed.total_seconds() // 60),
    )


def group_visibility_windows(
    samples: Iterable[VisibilitySample], time_step_minutes: int
) -> tuple[VisibilityWindow, ...]:
    """Group visible samples per target, splitting at invisible samples or gaps."""
    if not isinstance(time_step_minutes, int) or time_step_minutes <= 0:
        raise ValueError("time_step_minutes must be a positive integer.")
    return _make_windows(samples, time_step_minutes)


def group_both_visible_windows(
    samples: Iterable[VisibilitySample], time_step_minutes: int
) -> tuple[VisibilityWindow, ...]:
    """Group consecutive timestamps where both Sun and Earth samples are visible."""
    if not isinstance(time_step_minutes, int) or time_step_minutes <= 0:
        raise ValueError("time_step_minutes must be a positive integer.")
    return _make_windows(samples, time_step_minutes, require_both_targets=True)


def run_site06_visibility_timeseries(
    start_utc: str,
    end_utc: str,
    time_step_minutes: int,
    targets: Iterable[str] = ("SUN", "EARTH"),
    site: str = "Site06",
) -> VisibilityTimeSeries:
    """Sample celestial directions and compare them with the saved 1 km horizon."""
    if site.strip().casefold() not in {"site06", "site06 — nobile rim 1"}:
        raise ValueError("This visibility engine currently supports Site06 only.")
    canonical_targets = tuple(dict.fromkeys(str(target).strip().upper() for target in targets))
    if not canonical_targets or any(target not in {"SUN", "EARTH"} for target in canonical_targets):
        raise ValueError("targets must contain SUN and/or EARTH.")
    timestamps = generate_timestamps(start_utc, end_utc, time_step_minutes)
    profile = load_site06_horizon_profile()
    samples: list[VisibilitySample] = []
    with Site06CelestialDirectionEngine() as engine:
        for timestamp in timestamps:
            utc_iso = _format_utc(timestamp)
            spice_utc = _format_utc(timestamp, spice_compatible=True)
            for target in canonical_targets:
                direction = engine.celestial_direction(spice_utc, target)
                result = compare_with_terrain_horizon(direction, profile)
                samples.append(
                    VisibilitySample(
                        target=result.target,
                        utc_timestamp=utc_iso,
                        azimuth_deg=result.azimuth_deg,
                        celestial_elevation_deg=result.celestial_elevation_deg,
                        terrain_horizon_elevation_deg=result.terrain_horizon_elevation_deg,
                        clearance_deg=result.clearance_deg,
                        visible=result.visible,
                        horizon_range_km=result.horizon_range_km,
                        horizon_status=result.horizon_status,
                    )
                )
    sample_tuple = tuple(samples)
    return VisibilityTimeSeries(
        site="Site06",
        start_utc=_format_utc(timestamps[0]),
        end_utc=_format_utc(timestamps[-1]),
        time_step_minutes=time_step_minutes,
        targets=canonical_targets,
        samples=sample_tuple,
        visibility_windows=group_visibility_windows(sample_tuple, time_step_minutes),
        both_visible_windows=group_both_visible_windows(sample_tuple, time_step_minutes),
        horizon_status="limited-range",
        horizon_range_km=SITE06_HORIZON_RANGE_KM,
    )
