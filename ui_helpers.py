"""Presentation-only conversions used by the Streamlit mission dashboard."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from html import escape
from typing import Iterable

import plotly.graph_objects as go

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


def build_utc_date_range(
    start_date: date,
    end_date: date,
    start_time: time,
    end_time: time,
) -> tuple[str, str]:
    """Build inclusive UTC bounds; same-day reversed clocks retain overnight behavior."""
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")
    start = datetime.combine(start_date, start_time, tzinfo=UTC)
    end = datetime.combine(end_date, end_time, tzinfo=UTC)
    if end_date == start_date and end_time < start_time:
        end += timedelta(days=1)
    if end < start:
        raise ValueError("End of analysis interval must not precede its start.")
    return _iso_utc(start), _iso_utc(end)


def build_daily_summary_rows(
    result: MissionAnalysisResult,
    start_date: date,
    end_date: date,
) -> list[dict[str, object]]:
    """Summarize sampled visibility by UTC date, retaining dates with no samples."""
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")

    days: dict[date, list] = {}
    current = start_date
    while current <= end_date:
        days[current] = []
        current += timedelta(days=1)
    for sample in result.samples:
        timestamp = datetime.fromisoformat(sample.timestamp_utc.replace("Z", "+00:00"))
        sample_day = timestamp.astimezone(UTC).date()
        if sample_day in days:
            days[sample_day].append(sample)

    rows: list[dict[str, object]] = []
    cadence = timedelta(minutes=result.sampling_interval_minutes)
    for day, samples in days.items():
        by_target = {
            target: sorted(
                (sample for sample in samples if sample.target == target and sample.visible),
                key=lambda sample: sample.timestamp_utc,
            )
            for target in ("SUN", "EARTH")
        }

        def elapsed_minutes(visible_samples: list) -> int:
            total = 0
            run: list = []
            previous: datetime | None = None
            for sample in sorted(visible_samples, key=lambda item: item.timestamp_utc):
                timestamp = datetime.fromisoformat(sample.timestamp_utc.replace("Z", "+00:00"))
                if previous is None or timestamp - previous == cadence:
                    run.append(sample)
                else:
                    total += int((previous - datetime.fromisoformat(run[0].timestamp_utc.replace("Z", "+00:00"))).total_seconds() // 60)
                    run = [sample]
                previous = timestamp
            if run:
                last = datetime.fromisoformat(run[-1].timestamp_utc.replace("Z", "+00:00"))
                first = datetime.fromisoformat(run[0].timestamp_utc.replace("Z", "+00:00"))
                total += int((last - first).total_seconds() // 60)
            return total

        timestamp_count = len({sample.timestamp_utc for sample in samples})
        sun_count = sum(sample.target == "SUN" and sample.visible for sample in samples)
        earth_count = sum(sample.target == "EARTH" and sample.visible for sample in samples)
        overlap_samples = [
            timestamp for timestamp in {sample.timestamp_utc for sample in samples}
            if any(sample.target == "SUN" and sample.timestamp_utc == timestamp and sample.visible for sample in samples)
            and any(sample.target == "EARTH" and sample.timestamp_utc == timestamp and sample.visible for sample in samples)
        ]
        overlap_records = [
            next(sample for sample in samples if sample.target == "SUN" and sample.timestamp_utc == timestamp)
            for timestamp in overlap_samples
        ]
        rows.append({
            "UTC date": day.isoformat(),
            "Data availability": "Available" if timestamp_count else "Unavailable — no samples",
            "Sun sampled visibility": sun_count / timestamp_count if timestamp_count else None,
            "Earth sampled visibility": earth_count / timestamp_count if timestamp_count else None,
            "Sun-visible sampled-window duration": elapsed_minutes(by_target["SUN"]) if timestamp_count else None,
            "Earth-visible sampled-window duration": elapsed_minutes(by_target["EARTH"]) if timestamp_count else None,
            "Overlapping Sun/Earth duration": elapsed_minutes(overlap_records) if timestamp_count else None,
            "Sampled timestamps": timestamp_count,
        })
    return rows


def build_site_information_rows(result: MissionAnalysisResult) -> list[dict[str, str]]:
    """Describe the supported site's coordinate, geometry, terrain, and sample provenance."""
    site = result.site
    timestamp_count = len({sample.timestamp_utc for sample in result.samples})
    return [
        {
            "Information": "Analysis support",
            "Availability": "Supported site",
            "Details": "Site06 — Nobile Rim 1 only. Other landing sites are not evaluated or scientifically validated by this pipeline.",
        },
        {
            "Information": "Coordinates",
            "Availability": "Available",
            "Details": (
                f"{abs(site.latitude_deg):.5f}°S, {site.longitude_deg:.5f}°E; "
                f"{site.latitude_type}, east-positive longitude."
            ),
        },
        {
            "Information": "Coordinate provenance",
            "Availability": "Source recorded",
            "Details": f"{site.coordinate_source}; source frame {site.source_frame}, {site.source_ephemeris}.",
        },
        {
            "Information": "Sun/Earth geometry",
            "Availability": "Audited Site06 basis",
            "Details": "NAIF SPICE DE440 geometry transformed to the Site06 local basis in MOON_ME_DE421.",
        },
        {
            "Information": "Terrain horizon used for visibility",
            "Availability": f"{result.horizon_status}; {result.horizon_range_km:g} km range",
            "Details": (
                "Saved reports/site06_10km_limited_horizon.json composite profile "
                "(5 m data within 1 km and 20 m data to 10 km); terrain beyond this radius is not evaluated."
            ),
        },
        {
            "Information": "Sampled observations",
            "Availability": f"Available ({timestamp_count} UTC timestamps)" if timestamp_count else "Unavailable — no samples",
            "Details": f"Sampling interval: {result.sampling_interval_minutes} minutes throughout the selected range.",
        },
        {
            "Information": "LOLA map context",
            "Availability": "Optional; shown separately",
            "Details": "NASA LOLA 20 m DEM is map context only and is not used as a substitute for the saved visibility horizon.",
        },
    ]


def build_daily_visibility_figure(rows: Iterable[dict[str, object]]) -> go.Figure:
    """Build a daily UTC date chart with explicit calendar ticks and missing-value gaps."""
    ordered = sorted(rows, key=lambda row: str(row["UTC date"]))
    utc_dates = [date.fromisoformat(str(row["UTC date"])) for row in ordered]
    figure = go.Figure()
    for label, color, symbol in (
        ("Sun sampled visibility", "#d98b00", "circle"),
        ("Earth sampled visibility", "#2864a5", "diamond"),
    ):
        values = [row[label] for row in ordered]
        figure.add_trace(go.Scatter(
            x=[datetime.combine(day, time.min, tzinfo=UTC) for day in utc_dates],
            y=[None if value is None else float(value) * 100 for value in values],
            mode="lines+markers",
            name=label,
            connectgaps=False,
            line={"color": color, "width": 2},
            marker={"color": color, "size": 9, "symbol": symbol},
        ))
    if utc_dates:
        tickvals = [datetime.combine(day, time.min, tzinfo=UTC) for day in utc_dates]
        lower = tickvals[0] - timedelta(hours=12)
        upper = tickvals[-1] + timedelta(hours=12)
        axis_range = [lower.isoformat(), upper.isoformat()]
    else:
        tickvals = []
        axis_range = None
    figure.update_layout(
        xaxis_title="UTC date",
        yaxis_title="Sampled visibility (%)",
        xaxis={
            "type": "date",
            "range": axis_range,
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": [day.strftime("%d %b") for day in utc_dates],
        },
        yaxis={"range": [0, 100], "ticksuffix": "%", "dtick": 20},
        legend_title_text="Target",
        margin={"t": 20, "b": 20},
    )
    return figure


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
            "Sampled-window elapsed duration": format_duration(window.duration_minutes),
            "Samples": window.sample_count,
            "Boundary semantics": "Sample-based estimate; not exact rise/set; between-sample visibility unverified",
        }
        for window in windows
    ]
