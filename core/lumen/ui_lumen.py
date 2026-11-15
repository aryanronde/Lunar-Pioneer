"""Streamlit presentation for LUMEN mission results.

The module formats outputs from the existing LUMEN calculation modules. It
does not create orbit, optics, terrain, power, or energy results. Missing
simulation products are rendered as UNKNOWN/unavailable rather than telemetry.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from .station_site import (
    SITE06_NOBILE_RIM_1,
    SITE07_PEAK_NEAR_SHACKLETON_EXPERIMENTAL,
    CriterionStatus,
    SiteComparison,
)


SCENARIO_LABELS = {
    "normal": "Normal exploration and return",
    "rescue": "Reflector-assisted emergency recovery",
    "direct_only_baseline": "Matched direct-only baseline",
}

DESIGN_ASSUMPTIONS = (
    ("Station nominal battery", 20_000, "Wh", "assumed design value"),
    ("Protected station reserve", 5_000, "Wh", "assumed; normal operations must preserve it"),
    ("Rover nominal battery", 1_000, "Wh", "assumed design value"),
    ("Charging efficiency", 0.90, "fraction", "assumed; applies only when configured in the calculation"),
    ("Station average load", 200, "W", "illustrative assumption; use configured simulation value when available"),
)

STAGE12_REPORT_FILES = {
    "normal": "normal.json",
    "rescue": "rescue.json",
    "direct_only_baseline": "direct_only_baseline.json",
}
STAGE12_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "lumen_stage12"


def _scenario_result(simulation_result: Any, scenario_name: str):
    if scenario_name not in SCENARIO_LABELS:
        raise ValueError(f"Unknown LUMEN scenario {scenario_name!r}.")
    if simulation_result is None:
        return None
    try:
        return simulation_result.scenario(scenario_name)
    except (AttributeError, KeyError) as exc:
        raise TypeError("simulation_result must expose scenario(name) from MissionSimulationResult.") from exc


def _duration_hours(step) -> float | None:
    try:
        start = datetime.fromisoformat(step.start_utc.replace("Z", "+00:00"))
        end = datetime.fromisoformat(step.end_utc.replace("Z", "+00:00"))
        seconds = (end - start).total_seconds()
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds / 3600.0 if seconds > 0 else None


def build_lumen_dashboard_model(
    simulation_result: Any = None,
    *,
    scenario_name: str = "normal",
    site_comparison: SiteComparison | None = None,
    mission_config: Any = None,
) -> dict[str, Any]:
    """Build a presentation-only view model from existing LUMEN result objects."""
    selected = _scenario_result(simulation_result, scenario_name)
    rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for step in selected.steps if selected is not None else ():
        row = {
            "UTC": step.sample_utc,
            "Station battery (Wh)": step.station_battery_energy_wh,
            "Rover battery (Wh)": step.rover_battery_energy_wh,
            "Direct generation (Wh)": step.direct_generation_wh,
            "Reflected energy ledger (Wh)": step.reflected_generation_wh,
            "Energy delivered to rover (Wh)": step.rover_energy_delivered_wh,
            "Station load (Wh)": step.station_load_consumed_wh,
            "Rover consumption (Wh)": step.rover_energy_consumed_wh,
            "Conversion losses (Wh)": step.conversion_losses_wh,
            "Transfer losses (Wh)": step.transfer_losses_wh,
            "Rover state": getattr(step.rover_operating_state, "value", step.rover_operating_state),
            "Mirror to rover visibility": _visibility_label(step.mirror_rover_visibility),
            "Mirror to station visibility": _visibility_label(step.mirror_station_visibility),
            "Emergency recovery outcome": _modeled_outcome_label(step.emergency_recovery_success),
        }
        duration_hours = _duration_hours(step)
        row["Direct generation (W)"] = step.direct_generation_wh / duration_hours if duration_hours else None
        row["Reflected energy ledger (W)"] = step.reflected_generation_wh / duration_hours if duration_hours else None
        rows.append(row)
        trajectory.append({
            "UTC": step.sample_utc,
            "X (km)": step.spacecraft_position_km[0],
            "Y (km)": step.spacecraft_position_km[1],
            "Z (km)": step.spacecraft_position_km[2],
            "Frame": step.geometry_frame,
        })

    site_rows = []
    if site_comparison is not None:
        for report in site_comparison.reports:
            site_rows.append({
                "Candidate": f"{report.candidate.identifier} — {report.candidate.name}",
                "Solar illumination": report.solar_illumination.status.value,
                "Reflector visibility": report.reflector_visibility.status.value,
                "Terrain coverage": report.terrain_coverage.status.value,
                "Slope evidence": report.slope_evidence.status.value,
                "Rover access": report.rover_access.status.value,
                "Overall evidence": report.overall_status.value,
            })
    if not site_rows:
        site_rows.append({
            "Candidate": f"{SITE06_NOBILE_RIM_1.identifier} — {SITE06_NOBILE_RIM_1.name} (initial candidate)",
            "Solar illumination": "UNKNOWN",
            "Reflector visibility": "UNKNOWN",
            "Terrain coverage": "UNKNOWN",
            "Slope evidence": "UNKNOWN",
            "Rover access": "UNKNOWN",
            "Overall evidence": "UNKNOWN",
        })

    station_capacity = getattr(getattr(getattr(mission_config, "station_power", None), "battery_config", None), "nominal_capacity_wh", None)
    rover_capacity = getattr(getattr(getattr(mission_config, "rover_config", None), "battery_config", None), "nominal_capacity_wh", None)
    station_reserve = getattr(getattr(getattr(mission_config, "station_power", None), "battery_config", None), "protected_reserve_wh", None)
    safe_return = getattr(getattr(mission_config, "rover_energy_requirements", None), "safe_return_energy_wh", None)
    station_power = getattr(mission_config, "station_power", None)
    rover_config = getattr(mission_config, "rover_config", None)
    configured_inputs = []
    for label, obj, attribute, unit in (
        ("Station average load", station_power, "average_load_w", "W"),
        ("Station direct array area", station_power, "direct_array_area_m2", "m²"),
        ("Station reflected receiver area", station_power, "reflected_receiver_area_m2", "m²"),
        ("Rover charging efficiency", rover_config, "charging_efficiency", "fraction"),
    ):
        value = getattr(obj, attribute, None) if obj is not None else None
        if value is not None:
            configured_inputs.append({"Input": label, "Value": value, "Unit": unit, "Origin": "supplied simulation assumption"})
    rescue = _scenario_result(simulation_result, "rescue")
    last_step = selected.steps[-1] if selected is not None and selected.steps else None
    station_soc = (selected.final_station_energy_wh / station_capacity * 100.0
                   if selected is not None and station_capacity else None)
    rover_soc = (selected.final_rover_energy_wh / rover_capacity * 100.0
                 if selected is not None and rover_capacity else None)
    latest_step = {
        "UTC": last_step.sample_utc,
        "spacecraft_position_km": tuple(last_step.spacecraft_position_km),
        "frame": last_step.geometry_frame,
        "Sun-to-mirror": _visibility_label(last_step.mirror_sun_visibility),
        "Mirror-to-station": _visibility_label(last_step.mirror_station_visibility),
        "Mirror-to-rover": _visibility_label(last_step.mirror_rover_visibility),
    } if last_step is not None else None

    comparison_by_time: dict[str, dict[str, Any]] = {}
    for name in SCENARIO_LABELS:
        scenario_result = _scenario_result(simulation_result, name)
        if scenario_result is None:
            continue
        short = {"normal": "Normal", "rescue": "Rescue", "direct_only_baseline": "Direct-only"}[name]
        for step in scenario_result.steps:
            row = comparison_by_time.setdefault(step.sample_utc, {"UTC": step.sample_utc})
            hours = _duration_hours(step)
            row[f"{short} station energy (Wh)"] = step.station_battery_energy_wh
            row[f"{short} rover energy (Wh)"] = step.rover_battery_energy_wh
            row[f"{short} direct generation (W)"] = step.direct_generation_wh / hours if hours else None
            row[f"{short} reflected energy ledger (W)"] = step.reflected_generation_wh / hours if hours else None
            row[f"{short} rover charging (Wh)"] = step.rover_energy_delivered_wh

    return {
        "scenario_name": scenario_name,
        "scenario_label": SCENARIO_LABELS[scenario_name],
        "available": selected is not None,
        "scenario": selected,
        "normal": _scenario_result(simulation_result, "normal"),
        "rescue": rescue,
        "baseline": _scenario_result(simulation_result, "direct_only_baseline"),
        "comparison": getattr(simulation_result, "comparison", None),
        "simulation_assumptions": tuple(getattr(simulation_result, "assumptions", ())),
        "provenance": tuple(getattr(simulation_result, "provenance", ())),
        "site_rows": site_rows,
        "site_comparison_status": site_comparison.comparison_status.value if site_comparison else "UNKNOWN",
        "site_selected": False,
        "site07_experimental": SITE07_PEAK_NEAR_SHACKLETON_EXPERIMENTAL,
        "station_capacity_wh": station_capacity,
        "rover_capacity_wh": rover_capacity,
        "station_soc_percent": station_soc,
        "rover_soc_percent": rover_soc,
        "station_reserve_wh": station_reserve,
        "safe_return_energy_wh": safe_return,
        "configured_rescue_window_s": getattr(mission_config, "rescue_window_s", None),
        "configured_inputs": configured_inputs,
        "time_series": rows,
        "comparison_time_series": [comparison_by_time[key] for key in sorted(comparison_by_time)],
        "trajectory": trajectory,
        "latest_geometry": latest_step,
        "selected_totals": selected,
        "rescue_success": rescue.rescue_success if rescue is not None else None,
        "rescue_failure_reasons": tuple(rescue.rescue_failure_reasons) if rescue is not None else (),
        "rescue_energy_delivered_wh": rescue.rover_energy_delivered_wh if rescue is not None else None,
        "unsupported_optics_fields": (
            "Mirror normal and reflection residual are not included in MissionSimulationResult.",
            "Beam footprint and intercepted receiver power are not included in MissionSimulationResult.",
            "Rescue deficit and detailed survival deadline are not included in MissionScenarioResult.",
        ),
    }


def _visibility_label(value: Any) -> str:
    if value is True:
        return "VISIBLE"
    if value is False:
        return "BLOCKED"
    return "UNKNOWN"


def _modeled_outcome_label(value: Any) -> str:
    """Label a modeled recovery boolean without conflating it with visibility."""
    if value is True:
        return "MODELED SUCCESS"
    if value is False:
        return "INFEASIBLE"
    return "UNKNOWN"


def _feasibility_label(value: Any) -> str:
    if value is True:
        return "FEASIBLE (MODELED)"
    if value is False:
        return "INFEASIBLE"
    return "UNKNOWN"


def _load_stage12_report(report_name: str, report_dir: Path | None = None) -> dict[str, Any]:
    """Read one fixed Stage 12 report without altering or interpreting its science."""
    if report_name not in STAGE12_REPORT_FILES:
        raise ValueError(f"Unknown Stage 12 report {report_name!r}.")
    path = (report_dir or STAGE12_REPORT_DIR) / STAGE12_REPORT_FILES[report_name]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"scenario_name": report_name, "available": False, "valid_json": False,
                "raw_bytes": None, "payload": None, "error": f"Report unavailable: {exc}"}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"scenario_name": report_name, "available": True, "valid_json": False,
                "raw_bytes": raw, "payload": None, "error": f"Report is not valid UTF-8 JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"scenario_name": report_name, "available": True, "valid_json": False,
                "raw_bytes": raw, "payload": None, "error": "Report JSON must contain an object at the top level."}
    return {"scenario_name": report_name, "available": True, "valid_json": True,
            "raw_bytes": raw, "payload": payload, "error": None}


def _report_value(mapping: Any, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, dict) and key in mapping else None


def _format_report_energy(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "UNKNOWN"
    return f"{float(value):,.6g} Wh"


def _report_audit_label(value: Any) -> str:
    if value == "PASS":
        return "PASS (MODEL AUDIT)"
    if value == "FAIL":
        return "FAIL (MODEL AUDIT)"
    return "UNKNOWN"


def _render_stage12_reports(st, report_dir: Path | None = None) -> None:
    """Render read-only saved reports and exact-byte downloads for fixed files."""
    with st.expander("Read-only Stage 12 demonstration reports", expanded=False):
        st.caption(
            "These saved files are synthetic demonstrations. PASS reports model-audit checks only; "
            "operational lunar mission feasibility remains UNKNOWN."
        )
        selected_name = st.selectbox(
            "Saved scenario report",
            options=tuple(STAGE12_REPORT_FILES),
            format_func=lambda name: SCENARIO_LABELS[name],
            key="lumen_stage12_report_selection",
        )
        selected = _load_stage12_report(selected_name, report_dir)
        if not selected["available"]:
            st.warning(selected["error"])
        elif not selected["valid_json"]:
            st.error(selected["error"])
        else:
            payload = selected["payload"]
            if payload.get("scenario_name") != selected_name:
                st.warning("The report's scenario_name does not match the selected report file.")
            audit = _report_audit_label(payload.get("scenario_audit_status"))
            visibility = payload.get("visibility_status") or "UNKNOWN"
            capture = payload.get("receiver_capture_status") or "UNKNOWN"
            calculated = payload.get("calculated")
            electrical = _report_value(calculated, "electrical_energy_Wh")
            emergency = _report_value(calculated, "emergency_recovery")
            recovery_status = _modeled_outcome_label(_report_value(emergency, "scenario_recovery_success"))
            residual = _format_report_energy(_report_value(calculated, "energy_balance_residual_Wh"))
            operational = payload.get("operational_feasibility") or "UNKNOWN"

            first, second, third = st.columns(3)
            first.metric("Scenario", payload.get("scenario_name") or "UNKNOWN")
            second.metric("Scenario audit", audit)
            third.metric("Operational feasibility", operational)
            first, second, third = st.columns(3)
            first.metric("Visibility", visibility)
            second.metric("Receiver capture", capture)
            third.metric("Emergency recovery", recovery_status)
            direct_energy = _format_report_energy(_report_value(electrical, "direct_generation"))
            reflected_energy = _format_report_energy(_report_value(electrical, "receiver_reflected_generation"))
            first, second, third = st.columns(3)
            first.metric("Direct electrical energy", direct_energy)
            second.metric("Reflected receiver electrical energy", reflected_energy)
            third.metric("Energy-balance residual", residual)
            if operational == "UNKNOWN":
                st.info("Operational feasibility is UNKNOWN; synthetic report results are not mission evidence.")

        st.markdown("**Download original report files**")
        for name, filename in STAGE12_REPORT_FILES.items():
            report = selected if name == selected_name else _load_stage12_report(name, report_dir)
            if report["raw_bytes"] is None:
                st.caption(f"{filename}: unavailable for download.")
                continue
            st.download_button(
                f"Download {filename}",
                data=report["raw_bytes"],
                file_name=filename,
                mime="application/json",
                key=f"lumen_report_download_{name}",
            )


def render_lumen_dashboard(
    simulation_result: Any = None,
    *,
    site_comparison: SiteComparison | None = None,
    mission_config: Any = None,
) -> None:
    """Render the LUMEN workspace from result objects already calculated upstream."""
    import pandas as pd
    import streamlit as st

    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#0b1220; color:#e5edf7; }
        [data-testid="stSidebar"] { background:#101b2b; }
        [data-testid="stSidebar"] * { color:#e5edf7; }
        [data-testid="stMain"] .block-container { max-width:1500px; padding-top:1.5rem; color:#e5edf7; }
        .lumen-kicker { color:#42d6e8; font-size:.78rem; letter-spacing:.18em; font-weight:700; }
        .lumen-title { color:#f1f5fb; font-size:2.5rem; margin:.15rem 0 .25rem; }
        .lumen-subtitle { color:#a9b8ca; max-width:900px; }
        .lumen-warning { background:#2b2518; border:1px solid #765d2c; border-left:4px solid #e5aa42; border-radius:7px; padding:12px 15px; color:#f6dfaa; margin:.5rem 0 1rem; }
        div[data-testid="stMetric"] { background:#131f30; border:1px solid #27384e; border-radius:9px; padding:12px 14px; }
        div[data-testid="stMetric"] * { color:#e5edf7 !important; }
        h1,h2,h3 { color:#edf4fb !important; }
        [data-testid="stCaptionContainer"], .stMarkdown p { color:#a9b8ca; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="lumen-kicker">LUNAR UTILITY THROUGH MIRROR-ENHANCED NETWORKS</div>', unsafe_allow_html=True)
    st.markdown('<div class="lumen-title">LUMEN Mission Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="lumen-subtitle">Evidence-led planning view for a lunar charging station, rover energy management, and a preliminary orbital reflector.</div>', unsafe_allow_html=True)

    simulation_result = simulation_result or st.session_state.get("lumen_mission_simulation_result")
    site_comparison = site_comparison or st.session_state.get("lumen_site_comparison")
    mission_config = mission_config or st.session_state.get("lumen_mission_config")
    scenario_name = st.selectbox(
        "Scenario",
        options=tuple(SCENARIO_LABELS),
        format_func=lambda name: SCENARIO_LABELS[name],
        key="lumen_selected_scenario",
    )
    model = build_lumen_dashboard_model(
        simulation_result, scenario_name=scenario_name,
        site_comparison=site_comparison, mission_config=mission_config,
    )

    st.markdown(
        '<div class="lumen-warning"><b>Model status:</b> the circular orbit is assumed kinematics, not stability-validated. '
        'Limited-range terrain profiles cannot establish complete visibility. A flat mirror redirects sunlight; it does not focus it.</div>',
        unsafe_allow_html=True,
    )
    if not model["available"]:
        st.warning(
            "No LUMEN MissionSimulationResult is available in this session. No telemetry or mission outcome is being shown. "
            "Supply an explicitly configured result through the existing LUMEN simulation API to populate this dashboard."
        )

    _render_stage12_reports(st)

    _render_overview(st, model)
    overview_tab, orbit_tab, station_tab, rover_tab, rescue_tab, compare_tab = st.tabs((
        "Mission overview", "Orbit & reflector", "Charging station", "Rover operations", "Emergency rescue", "Scenario comparison",
    ))
    with overview_tab:
        st.subheader("Architecture and assumptions")
        st.write("Sunlight is converted by elevated station arrays and, when explicitly illuminated and visible, by a redirected orbital beam. Station storage serves the station and rover charging system; rover departure depends on its safe-return energy requirement.")
        st.dataframe(pd.DataFrame(DESIGN_ASSUMPTIONS, columns=("Parameter", "Value", "Unit", "Provenance")), hide_index=True, use_container_width=True)
        if model["configured_inputs"]:
            st.markdown("**Configured simulation assumptions**")
            st.dataframe(pd.DataFrame(model["configured_inputs"]), hide_index=True, use_container_width=True)
        st.caption(f"Selected scenario: {model['scenario_label']} · all timestamps are UTC · calculated energy is Wh and power is W.")
        st.caption("Station battery: 20,000 Wh nominal · 5,000 Wh protected reserve. Rover battery: 1,000 Wh nominal. These are design assumptions, not validated hardware specifications.")
        _render_provenance(st, model)
    with orbit_tab:
        _render_orbit(st, pd, model)
    with station_tab:
        _render_station(st, pd, model)
    with rover_tab:
        _render_rover(st, model)
    with rescue_tab:
        _render_rescue(st, pd, model)
    with compare_tab:
        _render_comparison(st, pd, model)


def _render_overview(st, model):
    """Render the compact mission snapshot used above the detailed tabs."""
    st.subheader("Mission snapshot")
    if not model["available"]:
        st.info("Mission state is UNKNOWN until an existing LUMEN simulation result is supplied; no telemetry has been generated.")
        return
    result = model["scenario"]
    metrics = st.columns(4)
    metrics[0].metric("Station battery", f"{result.final_station_energy_wh:,.1f} Wh")
    metrics[1].metric("Rover battery", f"{result.final_rover_energy_wh:,.1f} Wh")
    metrics[2].metric("Rover energy delivered", f"{result.rover_energy_delivered_wh:,.1f} Wh")
    metrics[3].metric("Rescue result", _modeled_outcome_label(result.rescue_success))
    st.caption(
        f"{model['scenario_label']} · energy-balance residual "
        f"{result.energy_balance_residual_wh:+.6g} Wh · all mission timestamps are UTC."
    )


def _render_provenance(st, model):
    st.subheader("Provenance and assumptions")
    if not model["available"]:
        st.caption("SPICE, DEM, and mission-calculation provenance is unavailable until a result is supplied.")
        return
    for value in model["provenance"]:
        st.write(f"• {value}")
    st.markdown("**Simulation assumptions**")
    for value in model["simulation_assumptions"]:
        st.write(f"• {value}")


def _render_orbit(st, pd, model):
    st.subheader("Assumed orbit and reflector geometry")
    if not model["latest_geometry"]:
        st.info("Spacecraft position and visibility are UNKNOWN until a mission simulation result is supplied.")
        return
    geometry = model["latest_geometry"]
    st.caption(f"Latest sample: {geometry['UTC']} UTC · {geometry['frame']} · assumed circular kinematic orbit")
    a, b, c = st.columns(3)
    a.metric("Sun → mirror", geometry["Sun-to-mirror"])
    b.metric("Mirror → station", geometry["Mirror-to-station"])
    c.metric("Mirror → rover", geometry["Mirror-to-rover"])
    st.write("**Spacecraft position (km):** " + ", ".join(f"{v:,.3f}" for v in geometry["spacecraft_position_km"]))
    if model["trajectory"]:
        st.markdown("**Sampled assumed-orbit positions**")
        st.line_chart(pd.DataFrame(model["trajectory"]).set_index("UTC").drop(columns="Frame"), height=250)
    for item in model["unsupported_optics_fields"][:2]:
        st.warning(item + " Displayed as UNKNOWN; the UI does not recompute optics.")


def _render_station(st, pd, model):
    st.subheader("Candidate charging-station sites")
    st.caption("No automatic site selection or weighted score is used. Site06 is an initial candidate only.")
    st.warning("Site07 (Peak Near Shackleton) remains experimental: DEM source-version consistency is unresolved, so it is not treated as a production candidate.")
    st.dataframe(pd.DataFrame(model["site_rows"]), hide_index=True, use_container_width=True)
    if model["site_comparison_status"] == "UNKNOWN":
        st.warning("Site feasibility is UNKNOWN until compatible terrain, slope, illumination, reflector, and rover-access evidence is supplied.")
    if model["available"]:
        result = model["scenario"]
        columns = st.columns(4)
        columns[0].metric("Station battery", f"{result.final_station_energy_wh:,.1f} Wh")
        columns[1].metric("Direct generation", f"{result.direct_generation_wh:,.1f} Wh")
        columns[2].metric("Reflected energy ledger", f"{result.reflected_generation_wh:,.1f} Wh")
        columns[3].metric("Station load", f"{result.station_load_consumed_wh:,.1f} Wh")
        if model["station_soc_percent"] is not None:
            st.caption(f"Station state of charge: {model['station_soc_percent']:.1f}% of the configured {model['station_capacity_wh']:,.0f} Wh capacity.")
        if model["station_reserve_wh"] is not None:
            st.caption(f"Configured protected reserve: {model['station_reserve_wh']:,.1f} Wh; normal operations preserve this reserve.")
        st.write(f"Conversion losses: **{result.conversion_losses_wh:,.2f} Wh** · curtailed: **{result.curtailed_energy_wh:,.2f} Wh** · transfer losses: **{result.transfer_losses_wh:,.2f} Wh**")
        st.write(f"Energy-balance residual: **{result.energy_balance_residual_wh:+.6g} Wh** · unexplained discrepancy: **{'YES' if result.unexplained_discrepancy else 'no'}**")
        if result.unexplained_discrepancy:
            st.error("The selected scenario reports an unexplained energy-balance discrepancy.")
    st.caption("The simulation result schema does not separate reflected energy delivered to the station from reflected energy delivered to the rover receiver in every scenario. Its reflected-energy ledger is shown without attributing all of it to station generation.")
    if st.checkbox("Load Site06 LOLA terrain context", value=False, key="lumen_site06_terrain_context"):
        try:
            from core.lunar_coordinates import SITE06_NOBILE_RIM_1 as CLPS_SITE06
            from ui_mission_map import build_site_map_figure, build_site_map_geometry, load_site_terrain_window

            radius_m = 10_000.0
            terrain = load_site_terrain_window(CLPS_SITE06, radius_m)
            figure = build_site_map_figure(CLPS_SITE06, radius_m, terrain_data=terrain)
            st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "responsive": True})
            st.caption("LOLA 20 m terrain visualization for Site06 only; it is map context and does not establish site feasibility. The 10 km radius is limited-range.")
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
            st.warning(f"Terrain context unavailable: {exc}")
    else:
        st.metric("Station battery", "UNKNOWN")


def _render_rover(st, model):
    st.subheader("Rover energy state")
    if not model["available"]:
        st.info("Battery state, safe-return margin, and departure feasibility are UNKNOWN without a simulation result and explicit mission energy inputs.")
        return
    result = model["scenario"]
    a, b, c, d = st.columns(4)
    a.metric("Rover battery", f"{result.final_rover_energy_wh:,.1f} Wh")
    b.metric("Consumed", f"{result.rover_energy_consumed_wh:,.1f} Wh")
    c.metric("Charging delivered", f"{result.rover_energy_delivered_wh:,.1f} Wh")
    departure_ok = result.steps[-1].safe_return_feasible if result.steps else None
    d.metric("Safe-return feasibility", _feasibility_label(departure_ok))
    if model["rover_soc_percent"] is not None:
        st.caption(f"Rover state of charge: {model['rover_soc_percent']:.1f}% of the configured {model['rover_capacity_wh']:,.0f} Wh capacity.")
    if model["safe_return_energy_wh"] is not None:
        st.caption(f"Configured safe-return requirement: {model['safe_return_energy_wh']:,.1f} Wh")
    latest = model["time_series"][-1] if model["time_series"] else None
    if latest:
        st.write(f"Latest operating state: **{latest['Rover state']}** · safe-return feasibility: **{_feasibility_label(result.steps[-1].safe_return_feasible)}**")
    st.caption("Rover SOC percentage is shown only when the matching mission configuration is supplied. No departure clearance is inferred beyond the simulation's safe-return feasibility field.")


def _render_rescue(st, pd, model):
    st.subheader("Emergency reflector recovery")
    rescue = model["rescue"]
    if rescue is None:
        st.info("Recovery feasibility is UNKNOWN until a mission simulation result is supplied.")
        return
    st.metric("Emergency recovery", _modeled_outcome_label(rescue.rescue_success))
    energy_deficit, reflected_power = st.columns(2)
    energy_deficit.metric("Rover energy deficit", "UNKNOWN")
    reflected_power.metric("Available reflected receiver power", "UNKNOWN")
    st.caption("The current MissionScenarioResult records delivered rover energy and rescue outcome, but does not expose the instantaneous rescue deficit or received power as separate fields.")
    if model["configured_rescue_window_s"] is not None:
        st.caption(f"Configured survival/recovery window: {model['configured_rescue_window_s']:,.0f} s; this is an input assumption, not measured remaining survival time.")
    if rescue.rescue_success is None:
        st.warning("Geometry or energy evidence did not establish a successful recovery.")
    st.write(f"Energy delivered to rover in rescue scenario: **{rescue.rover_energy_delivered_wh:,.2f} Wh**")
    for reason in rescue.rescue_failure_reasons:
        st.warning(reason)
    for item in model["unsupported_optics_fields"][2:]:
        st.caption(item + " It remains UNKNOWN in the current result schema.")
    if model["time_series"]:
        st.dataframe(pd.DataFrame(model["time_series"])[["UTC", "Mirror to rover visibility", "Energy delivered to rover (Wh)", "Emergency recovery outcome"]], hide_index=True, use_container_width=True)


def _render_comparison(st, pd, model):
    st.subheader("Matched scenario comparison")
    scenarios = (model["normal"], model["rescue"], model["baseline"])
    if any(item is None for item in scenarios):
        st.info("Normal, rescue, and direct-only results are UNKNOWN until all three matched scenarios are supplied.")
        return
    rows = []
    for item in scenarios:
        rows.append({
            "Scenario": SCENARIO_LABELS[item.scenario],
            "Final station energy (Wh)": item.final_station_energy_wh,
            "Final rover energy (Wh)": item.final_rover_energy_wh,
            "Direct generation (Wh)": item.direct_generation_wh,
            "Reflected energy ledger (Wh)": item.reflected_generation_wh,
            "Rover charging delivered (Wh)": item.rover_energy_delivered_wh,
            "Conversion losses (Wh)": item.conversion_losses_wh,
            "Balance residual (Wh)": item.energy_balance_residual_wh,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("**Sampled energy time series**")
    if not model["comparison_time_series"]:
        st.info("The scenario results contain no time-series samples; charts remain unavailable.")
        return
    comparison_series = pd.DataFrame(model["comparison_time_series"]).set_index("UTC")
    energy_columns = [column for column in comparison_series.columns if "energy (Wh)" in column]
    generation_columns = [column for column in comparison_series.columns if "generation (W)" in column or "energy ledger (W)" in column]
    charging_columns = [column for column in comparison_series.columns if "rover charging (Wh)" in column]
    if energy_columns:
        st.line_chart(comparison_series[energy_columns], height=280)
    if generation_columns:
        st.line_chart(comparison_series[generation_columns], height=240)
    if charging_columns:
        st.line_chart(comparison_series[charging_columns], height=220)
    comparison = model["comparison"]
    if comparison is not None:
        st.write(f"Additional reflected generation: **{comparison.additional_reflected_generation_wh:,.2f} Wh** · additional rover delivery: **{comparison.additional_rover_energy_delivered_wh:,.2f} Wh** · additional station stored energy: **{comparison.additional_station_stored_energy_wh:,.2f} Wh**")
