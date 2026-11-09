"""Presentation-only conversions used by the Streamlit mission dashboard."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from html import escape
from typing import Iterable

from core.mission_analysis import MissionAnalysisResult, MissionVisibilityWindow


UTC = timezone.utc


def build_utc_interval(
    analysis_date: date,
    start_time: time,
    end_time: time,
) -> tuple[str, str]:
    """Build ISO UTC bounds; an end clock time earlier than start means overnight."""
    start = datetime.combine(analysis_date, start_time, tzinfo=UTC)
    end = datetime.combine(analysis_date, end_time, tzinfo=UTC)
    if end_time < start_time:
        end += timedelta(days=1)
    return _iso_utc(start), _iso_utc(end)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def format_duration(minutes: int) -> str:
    """Format a nonnegative elapsed duration for compact metric cards."""
    if minutes < 0:
        raise ValueError("Duration minutes cannot be negative.")
    hours, remaining = divmod(int(minutes), 60)
    if hours and remaining:
        return f"{hours} h {remaining} min"
    if hours:
        return f"{hours} h"
    return f"{remaining} min"


def build_timeline_rows(result: MissionAnalysisResult) -> list[dict[str, object]]:
    """Convert sampled records to categorical tracks without resampling them."""
    rows: list[dict[str, object]] = []
    for sample in result.samples:
        timestamp = datetime.fromisoformat(sample.timestamp_utc.replace("Z", "+00:00"))
        target_label = "Sun" if sample.target == "SUN" else "Earth"
        state = "Visible" if sample.visible else "Blocked"
        rows.append(
            {
                "timestamp_utc": timestamp,
                "track": f"{target_label} · {state.lower()}",
                "target": target_label,
                "visibility": state,
                "azimuth_deg": sample.azimuth_deg,
                "clearance_deg": sample.clearance_deg,
            }
        )
    return rows


def build_timeline_svg(result: MissionAnalysisResult) -> str:
    """Render sampled visible/blocked states as a compact, explicit SVG timeline."""
    rows = build_timeline_rows(result)
    if not rows:
        return ""

    timestamps = [row["timestamp_utc"] for row in rows]
    start = min(timestamps)
    end = max(timestamps)
    span_seconds = (end - start).total_seconds()
    x_left, x_right = 130.0, 960.0

    def x_position(timestamp: datetime) -> float:
        if span_seconds == 0:
            return (x_left + x_right) / 2
        fraction = (timestamp - start).total_seconds() / span_seconds
        return x_left + fraction * (x_right - x_left)

    elements = [
        '<svg viewBox="0 0 1000 190" role="img" aria-label="Sampled Sun and Earth visibility timeline" '
        'style="display:block;width:100%;height:auto;background:#ffffff;border:1px solid #dbe4ef;border-radius:10px">',
        '<style>text{font-family:Arial,sans-serif;fill:#26384f}.axis{stroke:#9aabc0;stroke-width:1}.lane{stroke:#e4eaf1;stroke-width:1}</style>',
        '<text x="18" y="55" font-size="15" font-weight="700">Sun</text>',
        '<text x="18" y="105" font-size="15" font-weight="700">Earth</text>',
        '<line class="lane" x1="130" y1="50" x2="960" y2="50"/>',
        '<line class="lane" x1="130" y1="100" x2="960" y2="100"/>',
        '<line class="axis" x1="130" y1="133" x2="960" y2="133"/>',
    ]
    for row in rows:
        timestamp = row["timestamp_utc"]
        lane_y = 50 if row["target"] == "Sun" else 100
        visible = row["visibility"] == "Visible"
        color = "#16834a" if visible else "#bd3434"
        state = "visible" if visible else "blocked"
        elements.append(
            f'<circle cx="{x_position(timestamp):.2f}" cy="{lane_y}" r="6" fill="{color}" '
            f'stroke="#ffffff" stroke-width="1.5"><title>{escape(str(row["target"]))}: {state} at '
            f'{escape(timestamp.isoformat())}</title></circle>'
        )
    start_label = escape(start.strftime("%Y-%m-%d %H:%M UTC"))
    end_label = escape(end.strftime("%Y-%m-%d %H:%M UTC"))
    elements.extend([
        f'<text x="130" y="157" text-anchor="start" font-size="12">{start_label}</text>',
        f'<text x="960" y="157" text-anchor="end" font-size="12">{end_label}</text>',
        '<circle cx="150" cy="177" r="5" fill="#16834a"/><text x="161" y="181" font-size="12">Visible</text>',
        '<circle cx="245" cy="177" r="5" fill="#bd3434"/><text x="256" y="181" font-size="12">Blocked</text>',
        '</svg>',
    ])
    return "".join(elements)


def build_geometry_rows(result: MissionAnalysisResult) -> list[dict[str, object]]:
    """Return target geometry and local terrain-clearance fields for display."""
    return [
        {
            "UTC timestamp": sample.timestamp_utc,
            "Target": "Sun" if sample.target == "SUN" else "Earth",
            "Azimuth (°)": sample.azimuth_deg,
            "Geometric elevation (°)": sample.celestial_elevation_deg,
            "Terrain horizon elevation (°)": sample.terrain_horizon_deg,
            "Clearance (°)": sample.clearance_deg,
            "Status": "Visible" if sample.visible else "Blocked",
        }
        for sample in result.samples
    ]


def build_window_rows(windows: Iterable[MissionVisibilityWindow]) -> list[dict[str, object]]:
    """Format sampled windows for tables while retaining UTC sample boundaries."""
    return [
        {
            "Start (UTC)": window.start_utc,
            "End (UTC)": window.end_utc,
            "Elapsed duration": format_duration(window.duration_minutes),
            "Samples": window.sample_count,
            "Boundary semantics": "Sample-based; not exact rise/set",
        }
        for window in windows
    ]
