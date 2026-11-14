r"""Reproducible, explicitly synthetic LUMEN demo and JSON report generator.

Run from the repository root with:
    .\.venv\Scripts\python.exe scripts\lumen_stage12_demo.py

The deterministic geometry and horizon are pipeline fixtures, not SPICE or
LOLA evidence. Reported calculations are not operational mission predictions.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lumen.energy_storage import BatteryConfig, BatteryState
from core.lumen.lumen_optics import MirrorOpticsAssumptions, TaggedLunarPosition
from core.lumen.lumen_orbit import CircularLunarOrbit, ORBIT_MODEL_LABEL
from core.lumen.lumen_visibility import SiteHorizonProfile
from core.lumen.mission_simulation import (
    MissionSimulationConfig,
    MissionTimelineSegment,
    RoverSolarReceiverAssumptions,
    StationPowerAssumptions,
    simulate_lumen_mission,
)
from core.lumen.mission_validation import validate_lumen_mission
from core.lumen.rover_energy import MissionEnergyRequirements, RoverConfig


START_UTC = "2026-01-01T00:00:00Z"
MOON_RADIUS_KM = 1_737.4
LUMEN_MODEL_VERSION = "optics-receiver-v2-unified-recovery-v1"


class DemoGeometry:
    """Deterministic placeholder geometry; deliberately not a SPICE adapter."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def utc_to_et(self, timestamp: str) -> float:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            # Mission simulation passes its SPICE timestamp as naive ISO text;
            # the fixture must apply the documented UTC interpretation.
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()

    def sun_position_km(self, et: float) -> tuple[float, float, float]:
        return (149_597_870.7, 0.0, 0.0)

    def j2000_to_moon_fixed(self, vector, et: float) -> tuple[float, float, float]:
        # Identity is an explicitly synthetic frame fixture, not a lunar transform.
        return tuple(float(component) for component in vector)


class DemoBridge:
    """Identity-only frame bridge fixture for same-frame demo coordinates."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def vector_dem_to_geometry(self, vector, et: float) -> tuple[float, float, float]:
        return tuple(float(component) for component in vector)


def _profile(position: TaggedLunarPosition, name: str) -> SiteHorizonProfile:
    return SiteHorizonProfile(
        profile_deg=tuple(0.0 for _ in range(360)),
        site_position_km=position.position_km,
        frame=position.frame,
        east_axis=(0.0, 1.0, 0.0),
        north_axis=(0.0, 0.0, 1.0),
        up_axis=(1.0, 0.0, 0.0),
        range_km=1_000.0,
        status="complete",
        site_name=f"synthetic zero-degree demo profile: {name}",
    )


def make_demo_config(*, rescue_energy_case: bool = False) -> MissionSimulationConfig:
    """Return the fixed assumed-input set used by all demonstration scenarios."""
    station = TaggedLunarPosition((MOON_RADIUS_KM, 0.0, 0.0), "MOON_PA_DE440")
    rover = TaggedLunarPosition((MOON_RADIUS_KM, 10.0, 0.0), "MOON_PA_DE440")
    station_power = StationPowerAssumptions(
        solar_irradiance_1au_w_m2=1_361.0,
        direct_array_area_m2=10.0,
        direct_conversion_efficiency=0.20,
        direct_incidence_angle_deg=0.0,
        reflected_receiver_area_m2=100.0,
        reflected_conversion_efficiency=0.20,
        reflected_incidence_angle_deg=0.0,
        average_load_w=200.0,
        battery_config=BatteryConfig(
            nominal_capacity_wh=20_000.0,
            maximum_depth_of_discharge=0.80,
            protected_reserve_wh=5_000.0,
            maximum_charge_power_w=5_000.0,
            maximum_discharge_power_w=5_000.0,
        ),
        initial_battery_state=BatteryState(10_000.0),
        battery_charge_efficiency=0.90,
        rover_transfer_power_w=20.0,
        rover_transfer_efficiency=0.95,
    )
    rover_config = RoverConfig(
        battery_config=BatteryConfig(
            nominal_capacity_wh=1_000.0,
            maximum_depth_of_discharge=0.80,
            protected_reserve_wh=200.0,
            maximum_charge_power_w=50_000.0,
            maximum_discharge_power_w=1_000.0,
        ),
        charging_efficiency=0.90,
        return_required_margin_wh=20.0,
    )
    energy_terms = (
        "outbound_travel_energy_wh",
        "outbound_communication_navigation_energy_wh",
        "outbound_thermal_energy_wh",
        "return_travel_energy_wh",
        "return_communication_navigation_energy_wh",
        "return_thermal_survival_energy_wh",
        "uncertainty_reserve_energy_wh",
    )
    requirements = MissionEnergyRequirements(
        outbound_travel_energy_wh=10.0,
        outbound_communication_navigation_energy_wh=5.0,
        outbound_thermal_energy_wh=5.0,
        return_travel_energy_wh=50.0,
        return_communication_navigation_energy_wh=10.0,
        return_thermal_survival_energy_wh=20.0,
        uncertainty_reserve_energy_wh=20.0,
        energy_origins=tuple((term, "assumed") for term in energy_terms),
    )
    return MissionSimulationConfig(
        start_utc=START_UTC,
        duration_s=120.0,
        time_step_s=60.0,
        orbit=CircularLunarOrbit(
            altitude_km=100.0,
            inclination_deg=0.0,
            ascending_node_longitude_deg=0.0,
            initial_orbital_phase_deg=0.0,
            epoch_utc=START_UTC,
        ),
        station_position=station,
        rover_position=rover,
        station_horizon=_profile(station, "station"),
        rover_horizon=_profile(rover, "rover"),
        station_power=station_power,
        rover_config=rover_config,
        initial_rover_battery_state=BatteryState(210.0 if rescue_energy_case else 800.0),
        rover_energy_requirements=requirements,
        rover_receiver=RoverSolarReceiverAssumptions(
            receiver_area_m2=100_000.0,
            incidence_angle_deg=0.0,
            conversion_efficiency=0.30,
        ),
        mirror_optics=MirrorOpticsAssumptions(
            mirror_area_m2=1_000_000.0,
            reflectivity=0.90,
            pointing_error_deg=0.0,
            solar_irradiance_w_m2=1_361.0,
        ),
        mission_timeline=(
            MissionTimelineSegment(0.0, 60.0, "exploration", 10.0, "station", 20.0),
            MissionTimelineSegment(60.0, 120.0, "return", 10.0, "station", 20.0),
        ),
        stranded_at_offset_s=60.0,
        stranded_survival_power_w=0.1,
        rescue_window_s=60.0,
        lunar_reference_radius_km=MOON_RADIUS_KM,
        flag_energy_residual_above_wh=1e-5,
    )


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _scenario_json(config: MissionSimulationConfig, simulation, validation, selected_name: str) -> dict:
    scenario = simulation.scenario(selected_name)
    audit = validation.scenario(selected_name)
    if any(check.status.value == "FAIL" for check in audit.checks):
        scenario_validation_status = "FAIL"
    elif any(check.status.value == "UNKNOWN" for check in audit.checks):
        scenario_validation_status = "UNKNOWN"
    else:
        scenario_validation_status = "PASS"
    return {
        "schema_version": 1,
        "model_version": LUMEN_MODEL_VERSION,
        "scenario_name": selected_name,
        "scenario_label": {
            "normal": "Normal exploration and return",
            "rescue": "Reflector-assisted emergency recovery",
            "direct_only_baseline": "Matched direct-only baseline",
        }[selected_name],
        "scenario_audit_status": scenario_validation_status,
        "operational_feasibility": "UNKNOWN",
        "visibility_status": audit.visibility_status,
        "receiver_capture_status": audit.receiver_capture_status,
        "calculated": {
            "optical_energy_Wh": {
                "mirror_incident": audit.mirror_incident_optical_energy_wh,
                "mirror_reflected": audit.mirror_reflected_optical_energy_wh,
                "receiver_intercepted": audit.intercepted_optical_energy_wh,
            },
            "electrical_energy_Wh": {
                "direct_generation": audit.direct_electrical_energy_wh,
                "receiver_reflected_generation": audit.reflected_electrical_energy_wh,
            },
            "battery_change_Wh": {
                "station": audit.station_battery_change_wh,
                "rover": audit.rover_battery_change_wh,
            },
            "emergency_recovery": {
                "energy_deficit": audit.emergency_energy_deficit_wh,
                "energy_delivered_to_battery": audit.emergency_energy_delivered_wh,
                "safe_return_status": audit.safe_return_status.value,
                "scenario_recovery_success": scenario.rescue_success,
            },
            "energy_balance_residual_Wh": audit.energy_balance_residual_wh,
            "ledger_status": audit.ledger_status.value,
            "curtailed_energy_Wh": scenario.curtailed_energy_wh,
            "conversion_losses_Wh": scenario.conversion_losses_wh,
            "transfer_losses_Wh": scenario.transfer_losses_wh,
        },
        "assumed_inputs": list(audit.input_assumptions) + [
            "demo fixture Sun vector: (149597870.7, 0, 0) km in synthetic Moon-centred setup",
            (f"demo mirror irradiance: {config.mirror_optics.solar_irradiance_w_m2:g} W/m^2 explicitly supplied locally at the mirror" if config.mirror_optics.solar_irradiance_w_m2 is not None else "demo mirror irradiance inverse-square scaled from the explicitly named reference input"),
            "demo fixture frame transformations are identity operations; not SPICE-derived",
            "demo fixture terrain horizons are synthetic zero-degree profiles; not LOLA/DEM observations",
            "all station, rover, reflector, orbit and mission timeline values are illustrative assumptions",
        ],
        "provenance": {
            "calculation_engine": "existing core.lumen simulation and validation APIs",
            "model_version": LUMEN_MODEL_VERSION,
            "receiver_capture_model": "core.lumen.lumen_receiver.calculate_receiver_power; finite solar disc and rectangular mirror aperture midpoint integration",
            "emergency_recovery_capture_model": "shared core.lumen.lumen_receiver receiver result; no bounding-circle fallback",
            "time_and_geometry": "deterministic synthetic demo provider; no SPICE kernels queried",
            "orbit": ORBIT_MODEL_LABEL,
            "terrain": "synthetic test horizon only; real site terrain is UNKNOWN",
            "measured_inputs": [],
        },
        "unknown_evidence": [
            "No measured or authoritative lunar-site illumination data is included.",
            "No measured terrain profile is included; synthetic clear horizons exercise code paths only.",
            "No operational orbital perturbation/stability analysis is included.",
            "Hardware performance, reflector pointing, thermal behavior, and mission operations are unvalidated.",
            "Operational mission feasibility remains UNKNOWN even when the synthetic calculation ledger passes.",
        ],
        "checks": [_jsonable(check) for check in audit.checks],
        "known_limitations": list(audit.known_limitations),
        "time_series": [
            {
                "sample_utc": step.sample_utc,
                "mirror_sun_visibility": step.mirror_sun_visibility,
                "mirror_station_visibility": step.mirror_station_visibility,
                "mirror_rover_visibility": step.mirror_rover_visibility,
                "receiver_capture_status": step.receiver_capture_status,
                "direct_generation_Wh": step.direct_generation_wh,
                "reflected_generation_Wh": step.reflected_generation_wh,
                "station_battery_Wh": step.station_battery_energy_wh,
                "rover_battery_Wh": step.rover_battery_energy_wh,
                "emergency_energy_deficit_Wh": step.emergency_energy_deficit_wh,
                "emergency_energy_delivered_Wh": step.emergency_energy_delivered_wh,
            }
            for step in scenario.steps
        ],
    }


def generate_reports(output_dir: Path | None = None) -> tuple[Path, ...]:
    """Run and serialize the three fixed demo scenarios reproducibly."""
    destination = output_dir or ROOT / "reports" / "lumen_stage12"
    destination.mkdir(parents=True, exist_ok=True)
    cases = (
        ("normal", make_demo_config()),
        ("rescue", make_demo_config(rescue_energy_case=True)),
        ("direct_only_baseline", make_demo_config()),
    )
    paths = []
    for name, config in cases:
        simulation = simulate_lumen_mission(
            config,
            geometry_engine_factory=DemoGeometry,
            frame_bridge_factory=DemoBridge,
        )
        validation = validate_lumen_mission(config, simulation)
        payload = _scenario_json(config, simulation, validation, name)
        path = destination / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


if __name__ == "__main__":
    for report_path in generate_reports():
        print(report_path.relative_to(ROOT))
