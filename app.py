"""Streamlit prototype for the CLPS Lunar Mission Browser."""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import streamlit as st
from core.lunar_coordinates import SITE06_NOBILE_RIM_1
from core.mission_analysis import (
    MissionAnalysisResult,
    analyze_mission_window,
    load_site06_10km_horizon_configuration,
)
from ui_helpers import (
    build_geometry_rows,
    build_daily_summary_rows,
    build_daily_visibility_figure,
    build_site_information_rows,
    build_timeline_rows,
    build_timeline_svg,
    build_utc_date_range,
    build_window_rows,
    format_duration,
)
from ui_mission_map import (
    TerrainContextUnavailable,
    build_site_map_figure,
    build_site_map_geometry,
    build_terrain_diagnostics,
    load_site_terrain_window,
)


SITE_CHOICES = {"Site06 — Nobile Rim 1": SITE06_NOBILE_RIM_1}


def _render_header() -> None:
    st.markdown(
        """
        <div class="mission-kicker">NASA SPACE APPS CHALLENGE 2026 · CLPS CONCEPT</div>
        <h1 class="mission-title">CLPS Lunar Mission Browser</h1>
        <p class="mission-subtitle">
            Compare sampled Sun and Earth visibility at Site06 — Nobile Rim 1.
            Results use a limited 10 km terrain horizon and do not establish
            landing suitability or continuous visibility between samples.
        </p>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[object, date, date, time, time, int, bool]:
    with st.sidebar:
        st.markdown("## Mission setup")
        site_label = st.selectbox("Landing site", tuple(SITE_CHOICES))
        start_date, end_date = st.date_input(
            "Analysis date range (UTC)", value=(date.today(), date.today())
        )
        start_time = st.time_input("Start time (UTC)", value=time(0, 0), step=900)
        end_time = st.time_input("End time (UTC)", value=time(23, 0), step=900)
        sampling_interval = st.selectbox(
            "Sampling interval",
            options=(15, 30, 60, 120, 360),
            index=2,
            format_func=lambda value: f"{value} minutes",
        )
        st.caption("If the end clock time is earlier than the start, the analysis ends the following day.")
        run = st.button("Run Analysis", type="primary", use_container_width=True)
    return SITE_CHOICES[site_label], start_date, end_date, start_time, end_time, sampling_interval, run


def _render_daily_summary(result: MissionAnalysisResult) -> None:
    start_date = date.fromisoformat(result.start_utc[:10])
    end_date = date.fromisoformat(result.end_utc[:10])
    rows = build_daily_summary_rows(result, start_date, end_date)
    with st.container(border=True):
        st.subheader("Daily visibility comparison")
        st.caption(
            f"Daily percentages use sampled timestamps; daily durations sum first-to-last visible-sample spans per window. "
            f"The {result.sampling_interval_minutes}-minute sampling interval applies across the entire range. "
            "Window boundaries are sample-based estimates, visibility between samples is not independently verified, "
            "and terrain visibility is limited to the existing 10 km horizon profile. Dates with no samples show unavailable percentages."
        )
        display = pd.DataFrame(rows).sort_values("UTC date", kind="stable")
        figure = build_daily_visibility_figure(rows)
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "responsive": True})
        table = display.copy()
        for column in ("Sun sampled visibility", "Earth sampled visibility"):
            table[column] = table[column].map(lambda value: "Unavailable" if pd.isna(value) else f"{value:.1%}")
        for column in ("Sun-visible sampled-window duration", "Earth-visible sampled-window duration", "Overlapping Sun/Earth duration"):
            table[column] = table[column].map(
                lambda value: "Unavailable" if pd.isna(value) else format_duration(value)
            )
        st.dataframe(table, hide_index=True, use_container_width=True)


def _render_site_assessment(result: MissionAnalysisResult) -> None:
    with st.container(border=True):
        st.subheader("Landing-site assessment scope and data")
        st.caption(
            "Site06 is the only supported CLPS analysis location. This identifies the current pipeline's data coverage; "
            "it does not establish that the site is suitable for landing or operations."
        )
        st.dataframe(
            pd.DataFrame(build_site_information_rows(result)),
            hide_index=True,
            use_container_width=True,
        )


def _render_metrics(result: MissionAnalysisResult) -> None:
    sun = result.summary_for("SUN")
    earth = result.summary_for("EARTH")
    overlap_minutes = sum(window.duration_minutes for window in result.sun_earth_overlap_windows)
    has_samples = bool(result.samples)
    sun_visibility = f"{sun.visible_fraction:.1%}" if has_samples else "Unavailable"
    earth_visibility = f"{earth.visible_fraction:.1%}" if has_samples else "Unavailable"
    sun_duration = format_duration(sun.total_visible_minutes) if has_samples else "Unavailable"
    earth_duration = format_duration(earth.total_visible_minutes) if has_samples else "Unavailable"
    overlap_duration = format_duration(overlap_minutes) if has_samples else "Unavailable"

    first, second = st.columns(2)
    first.metric("Sun sampled visibility", sun_visibility, help="Percentage of requested timestamps classified as visible. Visibility between samples is not independently verified.")
    second.metric("Earth sampled visibility", earth_visibility, help="Percentage of requested timestamps classified as visible. Visibility between samples is not independently verified.")
    st.caption(
        f"Visibility percentage is the share of sampled timestamps classified as visible. "
        f"Sampling interval: {result.sampling_interval_minutes} minutes. "
        "Visibility between samples is not independently verified."
    )

    first, second, third = st.columns(3)
    first.metric("Total Sun-visible duration", sun_duration, help="Elapsed time from the first to last visible sample in each window, summed across windows. Boundaries are sample-based estimates.")
    second.metric("Total Earth-visible duration", earth_duration, help="Elapsed time from the first to last visible sample in each window, summed across windows. Boundaries are sample-based estimates.")
    third.metric("Overlapping Sun + Earth", overlap_duration, help="Elapsed time from the first to last jointly visible samples in each window, summed across windows.")

    first, second = st.columns(2)
    first.metric("Longest Sun-visible window", format_duration(sun.longest_visible_window_minutes) if has_samples else "Unavailable")
    second.metric("Longest Earth-visible window", format_duration(earth.longest_visible_window_minutes) if has_samples else "Unavailable")
    st.caption(
        "Visible duration is elapsed time between the first and last visible samples in each window, summed across windows. "
        "Window boundaries are sample-based estimates, not exact rise/set times. Visibility between samples is not independently verified."
    )


def _render_timeline(result: MissionAnalysisResult) -> None:
    with st.container(border=True):
        _render_timeline_card(result)


def _render_timeline_card(result: MissionAnalysisResult) -> None:
    st.subheader("Visibility timeline")
    rows = build_timeline_rows(result)
    if not rows:
        st.info("This interval contains no samples.")
        return
    st.markdown(build_timeline_svg(result), unsafe_allow_html=True)
    st.caption(f"Green markers are sampled visible states; red markers are sampled blocked states. Each marker is one requested sample ({result.sampling_interval_minutes}-minute interval). Visibility between samples is not independently verified.")


def _render_geometry(result: MissionAnalysisResult) -> None:
    with st.container(border=True):
        st.subheader("Sampled celestial and terrain geometry")
        st.caption(
            "Terrain result: 10 km limited-range terrain analysis. "
            "Geometric target elevation is compared with the sampled terrain horizon at the same azimuth; "
            "visibility beyond this radius is not established."
        )
        if result.samples:
            st.dataframe(
                pd.DataFrame(build_geometry_rows(result)),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Unavailable: no timestamps were sampled in this interval.")


def _render_mission_map(result: MissionAnalysisResult) -> None:
    with st.container(border=True):
        st.subheader("Lunar South Polar Mission Map — 20 m terrain context")
        site = result.site
        st.caption(
            f"{site.name} · {abs(site.latitude_deg):.5f}°S, "
            f"{site.longitude_deg:.5f}°E · planetocentric, {site.source_frame}"
        )
        radius_m = result.horizon_range_km * 1000.0
        try:
            terrain_data = load_site_terrain_window(site, radius_m)
        except TerrainContextUnavailable as exc:
            st.warning(
                f"NASA LOLA terrain data could not be loaded. The terrain map is unavailable; "
                f"no substitute terrain is shown. Existing SPICE geometry, mission results, "
                f"tables, and saved limited-range horizon analysis remain available. Details: {exc}"
            )
            return
        figure = build_site_map_figure(site, radius_m, terrain_data=terrain_data)
        st.plotly_chart(
            figure,
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        )
        st.markdown(
            f'<div class="map-legend"><span><b class="site-dot">●</b> {site.name}</span>'
            f'<span><b class="range-ring">○</b> {result.horizon_range_km:g} km limited-range horizon-analysis radius</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Map shading shows the NASA LOLA 20 m DEM as terrain context. "
            "The dashed circle marks the 10 km limited-range analysis radius. "
            "Sun/Earth visibility uses the saved composite 10 km horizon profile, not the map shading. "
            "This does not establish complete lunar visibility beyond that range."
        )
        diagnostics = build_terrain_diagnostics(
            build_site_map_geometry(site, radius_m), terrain_data
        )
        with st.expander("Temporary terrain diagnostics", expanded=False):
            st.caption("Diagnostic panel — reports the same cached window rendered in the map.")
            st.write(f"**Read status:** {diagnostics['status']}")
            st.write(f"**Configured source:** {diagnostics['source']}")
            st.write(f"**Provenance:** {diagnostics['provenance']}")
            st.write(f"**Raster shape (rows × columns):** {diagnostics['shape']}")
            first, second = st.columns(2)
            first.metric("Valid elevation pixels", f"{diagnostics['valid_count']:,}")
            second.metric("NoData / non-finite pixels", f"{diagnostics['nodata_count']:,}")

            def elevation_label(value: object) -> str:
                return "Unavailable (no valid elevations)" if value is None else f"{value:,.3f} m"

            first, second, third = st.columns(3)
            first.metric("Minimum elevation", elevation_label(diagnostics["minimum_m"]))
            second.metric("Maximum elevation", elevation_label(diagnostics["maximum_m"]))
            third.metric("Median elevation", elevation_label(diagnostics["median_m"]))
            st.write(f"**Site06 projected X/Y:** {diagnostics['site_x_y_m'][0]:,.3f} m, {diagnostics['site_x_y_m'][1]:,.3f} m")
            st.write(f"**Window pixel-center bounds (Xmin, Ymin, Xmax, Ymax):** {diagnostics['window_center_bounds_m']}")
            st.write(f"**Source raster bounds (left, bottom, right, top):** {diagnostics['raster_bounds_m']}")
            st.write(f"**Site06 inside window pixel-center bounds:** {diagnostics['site_inside_window_centers']}")
            st.write(f"**Pixel spacing (X, Y):** {diagnostics['pixel_spacing_m']} m")
            st.write(f"**Increasing column direction:** {diagnostics['column_orientation']} ({diagnostics['column_step_m']} m)")
            st.write(f"**Increasing row direction:** {diagnostics['row_orientation']} ({diagnostics['row_step_m']} m)")
            if not diagnostics["axis_aligned"]:
                st.warning("The raster affine includes rotation/shear; the map's rectilinear X/Y grid does not represent those terms.")


def _render_window_table(title: str, windows) -> None:
    st.markdown(f"**{title}**")
    rows = build_window_rows(windows)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No sampled windows in this interval.")


def _render_windows(result: MissionAnalysisResult) -> None:
    with st.container(border=True):
        st.subheader("Visibility windows")
        sun_windows = [window for window in result.visibility_windows if window.target == "SUN"]
        earth_windows = [window for window in result.visibility_windows if window.target == "EARTH"]
        sun_column, earth_column = st.columns(2)
        with sun_column:
            _render_window_table("Sun windows", sun_windows)
        with earth_column:
            _render_window_table("Earth windows", earth_windows)
        _render_window_table("Overlapping Sun/Earth windows", result.sun_earth_overlap_windows)
        st.caption(result.sampling_note)


def _run_analysis(site, start_time_utc: str, end_time_utc: str, interval: int) -> MissionAnalysisResult:
    horizon = load_site06_10km_horizon_configuration()
    return analyze_mission_window(
        site_definition=site,
        start_utc=start_time_utc,
        end_utc=end_time_utc,
        sampling_interval_minutes=interval,
        terrain_horizon=horizon,
    )


def main() -> None:
    st.set_page_config(
        page_title="CLPS Lunar Mission Browser",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    workspace = st.sidebar.radio(
        "Mission workspace",
        options=("CLPS Lunar Mission Browser", "LUMEN Station & Reflector"),
        key="mission_workspace",
    )
    if workspace == "LUMEN Station & Reflector":
        from core.lumen.ui_lumen import render_lumen_dashboard

        render_lumen_dashboard()
        return
    st.markdown(
        """
        <style>
        .stApp { background: #eaf0f7; color: #26384f; }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #eaf0f7; }
        [data-testid="stMain"] .block-container { color: #26384f; }
        [data-testid="stSidebar"] { background: #0b1d39; }
        [data-testid="stSidebar"] * { color: #eef4ff; }
        .mission-kicker { color: #4f83cc; font-size: .78rem; font-weight: 700; letter-spacing: .14em; }
        .mission-title { color: #10284a; margin: .25rem 0 .2rem 0; font-size: 2.35rem; }
        .mission-subtitle { color: #52657d; font-size: 1.05rem; margin: 0 0 1rem 0; max-width: 850px; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: #f6f9fd; border-color: #c8d5e4; border-radius: 12px; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #cbd8e7; border-radius: 10px; padding: 14px 16px; }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #10223d !important; }
        h1, h2, h3, [data-testid="stSubheader"] { color: #10284a !important; }
        [data-testid="stCaptionContainer"], .stMarkdown p { color: #52657d; }
        div[data-testid="stDataFrame"] [role="columnheader"] { background-color: #17283f !important; color: #ffffff !important; }
        div[data-testid="stDataFrame"] [role="gridcell"] { color: #26384f !important; }
        .stButton button[kind="primary"] { background: #e6533b; border-color: #d7442d; color: #ffffff; }
        .terrain-warning { background: #fff3e8; border: 1px solid #e6533b; border-left: 5px solid #e6533b; border-radius: 8px; padding: 13px 16px; color: #482c23; margin: 0 0 10px 0; }
        .terrain-warning strong { color: #7c2d1d; }
        .map-legend { display:flex; flex-wrap:wrap; gap:1.25rem; align-items:center; color:#10223d; font-weight:600; padding:.2rem 0 .35rem; }
        .site-dot { color:#e6533b; font-size:1.1rem; }
        .range-ring { color:#e6533b; font-size:1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _render_header()
    site, start_date, end_date, start_clock, end_clock, interval, run = _render_sidebar()

    st.markdown(
        '<div class="terrain-warning"><strong>⚠ Site06 only · 10 km limited-range terrain analysis</strong><br>'
        'Visibility is evaluated using a sampled terrain profile extending up to 10 km from the landing site. '
        'This does not establish complete lunar visibility beyond the analyzed range.</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Sun and Earth angles use geometric SPICE directions. Sampling interval: {interval} minutes. Visibility percentage is based on sampled timestamps; visible duration spans first-to-last visible samples per window. Window boundaries are sample-based estimates, not exact rise/set times, and visibility between samples is not independently verified.")

    if run:
        try:
            if start_date is None or end_date is None:
                raise ValueError("Select both a start date and an end date.")
            start_utc, end_utc = build_utc_date_range(start_date, end_date, start_clock, end_clock)
            if (date.fromisoformat(end_utc[:10]) - date.fromisoformat(start_utc[:10])).days > 30:
                raise ValueError("Select a maximum of 31 UTC calendar dates to keep analysis and rendering responsive.")
            st.session_state["mission_analysis_result"] = _run_analysis(site, start_utc, end_utc, interval)
            st.session_state["mission_analysis_error"] = None
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
            st.session_state["mission_analysis_error"] = str(exc)
            st.session_state.pop("mission_analysis_result", None)

    error = st.session_state.get("mission_analysis_error")
    if error:
        st.error(f"Analysis could not be completed: {error}")
        if "SPICE kernel" in error or "kernel(s) not found" in error:
            st.info(
                "From the project directory, run `python scripts/setup_spice_kernels.py` "
                "to install required kernels, then `python scripts/setup_spice_kernels.py --check`. "
                "In PowerShell, use `.\\.venv\\Scripts\\python.exe` instead of `python` if the environment is not activated."
            )
    result = st.session_state.get("mission_analysis_result")
    if result is None:
        st.info("Choose a site and UTC interval, then select **Run Analysis** to calculate mission windows.")
        return

    site_title, utc_summary = st.columns((1, 2))
    site_title.markdown(f"### {site.name}")
    utc_summary.caption(
        f"UTC interval used: **{result.start_utc}** through **{result.end_utc}** · "
        f"{len({sample.timestamp_utc for sample in result.samples})} timestamps · "
        f"{result.sampling_interval_minutes}-minute cadence"
    )
    st.caption(f"Site coordinates: {site.latitude_deg:.5f}° latitude, {site.longitude_deg:.5f}° east longitude · {site.source_frame}")
    _render_site_assessment(result)
    with st.container(border=True):
        _render_metrics(result)
    _render_daily_summary(result)
    _render_mission_map(result)
    _render_timeline(result)
    _render_geometry(result)
    _render_windows(result)


if __name__ == "__main__":
    main()
