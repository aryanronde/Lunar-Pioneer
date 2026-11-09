"""Discrete mission-window analysis using existing SPICE and terrain APIs.

All window boundaries are sample timestamps, not exact rise/set transitions.
Durations measure elapsed time between first and last visible samples, so a
singleton window has zero minutes and n cadence samples span (n - 1) intervals.
This version supports the audited Site06 local frame and reads only a saved
horizon profile; it never accesses the DEM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .celestial_direction import CelestialDirection, Site06CelestialDirectionEngine
from .lunar_coordinates import LunarSiteRecord, SITE06_NOBILE_RIM_1
from .terrain_provider import MOON_ME_DE421
from .terrain_visibility import interpolate_horizon_elevation
from .visibility_timeseries import generate_timestamps


DEFAULT_SITE06_10KM_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "site06_10km_limited_horizon.json"
)
_UTC = timezone.utc


@dataclass(frozen=True)
class TerrainHorizonConfiguration:
    """An explicit saved 360-bin profile and its coverage provenance."""

    profile_deg: tuple[float | None, ...]
    horizon_status: str = "limited-range"
    horizon_range_km: float = 10.0
    source_frame: str = MOON_ME_DE421
    profile_name: str = "Site06 5 m 1 km + 20 m 10 km sampled horizon"

    def __post_init__(self) -> None:
        values = tuple(self.profile_deg)
        if len(values) != 360:
            raise ValueError("Terrain horizon configuration must contain 360 one-degree bins.")
        for index, value in enumerate(values):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Terrain profile bin {index} must be finite or None.")
        if self.source_frame != MOON_ME_DE421:
            raise ValueError(f"Terrain horizon profile must use {MOON_ME_DE421}.")
        if self.horizon_status not in {"limited-range", "insufficient-coverage", "complete"}:
            raise ValueError("horizon_status must be limited-range, insufficient-coverage, or complete.")
        if not math.isfinite(float(self.horizon_range_km)) or self.horizon_range_km <= 0:
            raise ValueError("horizon_range_km must be positive and finite.")
        object.__setattr__(self, "profile_deg", values)


@dataclass(frozen=True)
class MissionSample:
    timestamp_utc: str
    target: str
    azimuth_deg: float
    celestial_elevation_deg: float
    terrain_horizon_deg: float
    clearance_deg: float
    visible: bool
    horizon_status: str
    horizon_range_km: float


@dataclass(frozen=True)
class MissionVisibilityWindow:
    target: str
    start_utc: str
    end_utc: str
    duration_minutes: int
    sample_count: int
    horizon_status: str
    horizon_range_km: float
    boundary_note: str = "Sample-based boundaries; not exact rise/set transition times."


@dataclass(frozen=True)
class TargetMissionSummary:
    target: str
    total_visible_minutes: int
    visible_fraction: float
    longest_visible_window_minutes: int
    number_of_visibility_windows: int
    visible_sample_count: int


@dataclass(frozen=True)
class MissionAnalysisResult:
    site: LunarSiteRecord
    start_utc: str
    end_utc: str
    sampling_interval_minutes: int
    samples: tuple[MissionSample, ...]
    visibility_windows: tuple[MissionVisibilityWindow, ...]
    sun_earth_overlap_windows: tuple[MissionVisibilityWindow, ...]
    target_summaries: tuple[TargetMissionSummary, ...]
    horizon_status: str
    horizon_range_km: float
    sampling_note: str = (
        "Discrete sampled analysis. Window boundaries are sample timestamps, "
        "not exact rise/set transition times."
    )

    def summary_for(self, target: str) -> TargetMissionSummary:
        canonical = target.strip().upper()
        for summary in self.target_summaries:
            if summary.target == canonical:
                return summary
        raise KeyError(f"No mission summary for target {target!r}.")


class _DirectionEngine(Protocol):
    def __enter__(self) -> "_DirectionEngine": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def celestial_direction(self, utc_timestamp: str, target: str) -> CelestialDirection: ...


def load_site06_10km_horizon_configuration(
    report_path: str | Path = DEFAULT_SITE06_10KM_PROFILE_PATH,
) -> TerrainHorizonConfiguration:
    """Load the existing 10 km composite profile without any DEM/network IO."""
    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise FileNotFoundError(f"Saved Site06 10 km horizon report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved Site06 10 km horizon report is invalid JSON: {path}") from exc
    if report.get("source_frame") != MOON_ME_DE421:
        raise ValueError(f"Saved terrain horizon report must use {MOON_ME_DE421}.")
    if float(report.get("requested_radius_m", -1.0)) != 10_000.0:
        raise ValueError("Saved Site06 horizon report must describe the validated 10 km profile.")
    if report.get("horizon_status") != "limited-range":
        raise ValueError("The saved 10 km horizon report must remain explicitly limited-range.")
    bins = report.get("profile_bins_1deg_5m_1km_plus_20m_regional_deg")
    if not isinstance(bins, list) or len(bins) != 360:
        raise ValueError("Saved Site06 10 km report must contain 360 composite profile bins.")
    return TerrainHorizonConfiguration(
        profile_deg=tuple(None if value is None else float(value) for value in bins),
        horizon_status="limited-range",
        horizon_range_km=10.0,
        source_frame=MOON_ME_DE421,
        profile_name="Saved Site06 5 m 1 km + 20 m 10 km composite profile",
    )


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    return parsed.astimezone(_UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(_UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _window(target: str, run: Sequence[MissionSample]) -> MissionVisibilityWindow:
    first, last = run[0], run[-1]
    duration = int((_parse_utc(last.timestamp_utc) - _parse_utc(first.timestamp_utc)).total_seconds() // 60)
    return MissionVisibilityWindow(
        target=target,
        start_utc=first.timestamp_utc,
        end_utc=last.timestamp_utc,
        duration_minutes=duration,
        sample_count=len(run),
        horizon_status=first.horizon_status,
        horizon_range_km=first.horizon_range_km,
    )


def _group_target_windows(
    samples: Sequence[MissionSample], target: str, cadence_minutes: int
) -> tuple[MissionVisibilityWindow, ...]:
    selected = sorted(
        (sample for sample in samples if sample.target == target),
        key=lambda sample: _parse_utc(sample.timestamp_utc),
    )
    cadence = timedelta(minutes=cadence_minutes)
    windows: list[MissionVisibilityWindow] = []
    run: list[MissionSample] = []
    previous: datetime | None = None
    for sample in selected:
        timestamp = _parse_utc(sample.timestamp_utc)
        if sample.visible and (previous is None or timestamp - previous == cadence):
            run.append(sample)
            previous = timestamp
        else:
            if run:
                windows.append(_window(target, run))
            run = [sample] if sample.visible else []
            previous = timestamp if sample.visible else None
    if run:
        windows.append(_window(target, run))
    return tuple(windows)


def _group_overlap_windows(
    samples: Sequence[MissionSample], cadence_minutes: int
) -> tuple[MissionVisibilityWindow, ...]:
    by_time: dict[datetime, dict[str, MissionSample]] = {}
    for sample in samples:
        if sample.visible:
            by_time.setdefault(_parse_utc(sample.timestamp_utc), {})[sample.target] = sample
    cadence = timedelta(minutes=cadence_minutes)
    windows: list[MissionVisibilityWindow] = []
    run: list[MissionSample] = []
    previous: datetime | None = None
    for timestamp in sorted(by_time):
        targets = by_time[timestamp]
        both_visible = "SUN" in targets and "EARTH" in targets
        if both_visible and (previous is None or timestamp - previous == cadence):
            run.append(targets["SUN"])
            previous = timestamp
        else:
            if run:
                windows.append(_window("BOTH", run))
            run = [targets["SUN"]] if both_visible else []
            previous = timestamp if both_visible else None
    if run:
        windows.append(_window("BOTH", run))
    return tuple(windows)


def analyze_mission_window(
    site_definition: LunarSiteRecord,
    start_utc: str,
    end_utc: str,
    sampling_interval_minutes: int,
    terrain_horizon: TerrainHorizonConfiguration,
    *,
    direction_engine_factory: Callable[[], _DirectionEngine] = Site06CelestialDirectionEngine,
) -> MissionAnalysisResult:
    """Analyze sampled Sun/Earth terrain-relative windows for Site06.

    Start and end timestamps are inclusive when aligned with the cadence. A
    reversed interval is treated as empty and returns zero samples; an equal
    start/end interval contains one timestamp. The Site06 celestial engine is
    presently the only supported site implementation because its audited local
    basis is Site06-specific.
    """
    if site_definition != SITE06_NOBILE_RIM_1:
        raise NotImplementedError("Mission analysis currently supports the audited Site06 — Nobile Rim 1 site only.")
    if not isinstance(sampling_interval_minutes, int) or sampling_interval_minutes <= 0:
        raise ValueError("sampling_interval_minutes must be a positive integer.")
    if terrain_horizon.source_frame != MOON_ME_DE421:
        raise ValueError(f"Terrain horizon must be in {MOON_ME_DE421}.")

    start = _parse_utc(start_utc)
    end = _parse_utc(end_utc)
    timestamps = (
        () if end < start
        else generate_timestamps(_utc_text(start), _utc_text(end), sampling_interval_minutes)
    )
    samples: list[MissionSample] = []
    if timestamps:
        with direction_engine_factory() as engine:
            for timestamp in timestamps:
                timestamp_text = _utc_text(timestamp)
                spice_timestamp = timestamp_text[:-1]
                for target in ("SUN", "EARTH"):
                    direction = engine.celestial_direction(spice_timestamp, target)
                    horizon_angle = interpolate_horizon_elevation(
                        direction.azimuth_deg,
                        terrain_horizon.profile_deg,
                    )
                    clearance = float(direction.elevation_deg) - horizon_angle
                    samples.append(
                        MissionSample(
                            timestamp_utc=timestamp_text,
                            target=target,
                            azimuth_deg=float(direction.azimuth_deg) % 360.0,
                            celestial_elevation_deg=float(direction.elevation_deg),
                            terrain_horizon_deg=horizon_angle,
                            clearance_deg=clearance,
                            visible=clearance > 0.0,
                            horizon_status=terrain_horizon.horizon_status,
                            horizon_range_km=float(terrain_horizon.horizon_range_km),
                        )
                    )

    windows = tuple(
        window
        for target in ("SUN", "EARTH")
        for window in _group_target_windows(samples, target, sampling_interval_minutes)
    )
    overlaps = _group_overlap_windows(samples, sampling_interval_minutes)
    timestamp_count = len(timestamps)
    summaries: list[TargetMissionSummary] = []
    for target in ("SUN", "EARTH"):
        target_windows = tuple(window for window in windows if window.target == target)
        visible_count = sum(sample.visible for sample in samples if sample.target == target)
        durations = [window.duration_minutes for window in target_windows]
        summaries.append(
            TargetMissionSummary(
                target=target,
                total_visible_minutes=sum(durations),
                visible_fraction=(visible_count / timestamp_count) if timestamp_count else 0.0,
                longest_visible_window_minutes=max(durations, default=0),
                number_of_visibility_windows=len(target_windows),
                visible_sample_count=visible_count,
            )
        )

    return MissionAnalysisResult(
        site=site_definition,
        start_utc=_utc_text(start),
        end_utc=_utc_text(end),
        sampling_interval_minutes=sampling_interval_minutes,
        samples=tuple(samples),
        visibility_windows=windows,
        sun_earth_overlap_windows=overlaps,
        target_summaries=tuple(summaries),
        horizon_status=terrain_horizon.horizon_status,
        horizon_range_km=float(terrain_horizon.horizon_range_km),
    )

