"""Independent LUMEN mission ledger and scientific-status audit.

This module checks reported simulation results; it does not propagate an orbit,
recalculate optical power, or establish lunar mission feasibility. Energy is Wh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .energy_storage import operational_floor_wh
from .mission_simulation import MissionScenarioResult, MissionSimulationConfig, MissionSimulationResult


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: ValidationStatus
    detail: str


@dataclass(frozen=True)
class ScenarioValidationReport:
    scenario_name: str
    input_assumptions: tuple[str, ...]
    provenance: tuple[str, ...]
    visibility_status: str
    receiver_capture_status: str
    mirror_incident_optical_energy_wh: float
    mirror_reflected_optical_energy_wh: float
    intercepted_optical_energy_wh: float
    direct_electrical_energy_wh: float
    reflected_electrical_energy_wh: float
    station_battery_change_wh: float
    rover_battery_change_wh: float
    energy_balance_residual_wh: float
    emergency_energy_deficit_wh: float | None
    emergency_energy_delivered_wh: float
    safe_return_status: ValidationStatus
    ledger_status: ValidationStatus
    checks: tuple[ValidationCheck, ...]
    known_limitations: tuple[str, ...]


@dataclass(frozen=True)
class MissionValidationReport:
    scenarios: tuple[ScenarioValidationReport, ...]
    overall_status: ValidationStatus
    checks: tuple[ValidationCheck, ...]
    operational_feasibility: ValidationStatus
    limitations: tuple[str, ...]

    def scenario(self, name: str) -> ScenarioValidationReport:
        for report in self.scenarios:
            if report.scenario_name == name:
                return report
        raise KeyError(name)


def validate_lumen_mission(
    config: MissionSimulationConfig,
    result: MissionSimulationResult,
    *,
    tolerance_wh: float = 1e-6,
) -> MissionValidationReport:
    """Audit all simulated scenarios and independently reconstruct energy.

    Battery charge/delivery is deliberately not added to generation: its effect
    is represented by the consecutive battery-state changes. This avoids
    counting stored energy as a second energy source.
    """
    if not isinstance(config, MissionSimulationConfig):
        raise TypeError("config must be MissionSimulationConfig.")
    if not isinstance(result, MissionSimulationResult):
        raise TypeError("result must be MissionSimulationResult.")
    tolerance = float(tolerance_wh)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_wh must be finite and nonnegative.")

    expected_names = ("normal", "rescue", "direct_only_baseline")
    mission_checks: list[ValidationCheck] = []
    if tuple(item.scenario for item in result.scenarios) != expected_names:
        mission_checks.append(_check("scenario-set", ValidationStatus.FAIL, "Expected normal, rescue and direct-only scenarios in order."))
    else:
        mission_checks.append(_check("scenario-set", ValidationStatus.PASS, "All three required scenarios are present."))

    inputs = _input_assumptions(config)
    limitations = (
        "Circular lunar orbit is assumed kinematic and is not stability-validated.",
        "Receiver capture uses a uniform paraxial solar disc, a projected-area-equivalent circular receiver, and midpoint integration over the rectangular mirror aperture.",
        "Diffraction, solar limb darkening, mirror surface slope errors, structural occlusion, and thermal behavior are not modeled.",
        "Terrain horizons are local and limited range; absent or incomplete terrain leaves visibility unknown.",
        "Pointing-offset lateral beam overlap is not modeled; nonzero pointing error must remain unknown.",
        "Successful model energy accounting is not evidence of operational or real lunar mission feasibility.",
    )
    scenario_reports = tuple(
        _validate_scenario(config, item, inputs, limitations, tolerance)
        for item in result.scenarios
    )
    for scenario_report in scenario_reports:
        mission_checks.extend(scenario_report.checks)
    mission_checks.append(_check(
        "required-configuration",
        ValidationStatus.PASS,
        "MissionSimulationConfig validated all required geometry and receiver inputs; absent terrain/pointing evidence is assessed as UNKNOWN per scenario.",
    ))

    if any(check.status is ValidationStatus.FAIL for check in mission_checks):
        overall = ValidationStatus.FAIL
    elif any(check.status is ValidationStatus.UNKNOWN for check in mission_checks):
        overall = ValidationStatus.UNKNOWN
    else:
        overall = ValidationStatus.PASS
    return MissionValidationReport(
        scenarios=scenario_reports,
        overall_status=overall,
        checks=tuple(mission_checks),
        operational_feasibility=ValidationStatus.UNKNOWN,
        limitations=limitations,
    )


def _validate_scenario(config, scenario: MissionScenarioResult, inputs, limitations, tolerance):
    checks: list[ValidationCheck] = []
    if not scenario.steps:
        checks.append(_check("time-series", ValidationStatus.UNKNOWN, "No mission steps are available to audit."))
        residual = math.nan
        ledger_status = ValidationStatus.UNKNOWN
    else:
        residual, ledger_checks = _independent_ledger(config, scenario, tolerance)
        checks.extend(ledger_checks)
        ledger_status = _combine(check.status for check in ledger_checks)

    if scenario.scenario == "direct_only_baseline":
        reflected = abs(scenario.reflected_generation_wh) <= tolerance and all(
            abs(step.reflected_generation_wh) <= tolerance and not step.reflected_illumination_active
            for step in scenario.steps
        )
        checks.append(_check(
            "direct-only-isolation",
            ValidationStatus.PASS if reflected else ValidationStatus.FAIL,
            "No reflector-generated electrical energy is credited." if reflected else "Reflector energy appears in the direct-only baseline.",
        ))

    blocked_generation = False
    for step in scenario.steps:
        target_visible = (
            step.mirror_station_visibility if step.reflector_target == "station"
            else step.mirror_rover_visibility if step.reflector_target == "rover"
            else None
        )
        if (step.mirror_sun_visibility is False or target_visible is False) and step.reflected_generation_wh > tolerance:
            blocked_generation = True
        if step.receiver_capture_status == "ZERO_CAPTURE" and step.reflected_generation_wh > tolerance:
            blocked_generation = True
    checks.append(_check(
        "visibility-and-capture-gating",
        ValidationStatus.FAIL if blocked_generation else ValidationStatus.PASS,
        "No reflected electrical energy is credited on blocked or zero-capture paths." if not blocked_generation else "Reflected generation was credited despite blocked visibility or zero capture.",
    ))

    safe_values = [step.safe_return_feasible for step in scenario.steps]
    safe_status = (
        ValidationStatus.UNKNOWN if not safe_values else
        ValidationStatus.PASS if safe_values[-1] else ValidationStatus.FAIL
    )
    checks.append(_check(
        "safe-return-state",
        safe_status,
        "Modeled safe-return condition holds at the final recorded step." if safe_status is ValidationStatus.PASS else
        "Final recorded battery state is below the modeled safe-return requirement." if safe_status is ValidationStatus.FAIL else
        "Safe-return state is unavailable.",
    ))
    if scenario.scenario != "direct_only_baseline" and (
        _visibility_status(scenario.steps) == "UNKNOWN"
        or _capture_status(scenario.steps) == "UNKNOWN"
    ):
        checks.append(_check(
            "evidence-completeness", ValidationStatus.UNKNOWN,
            "A required reflector visibility or receiver-capture result is unknown; no operational success is inferred.",
        ))
    geometry_complete = all(
        step.geometry_frame in {"MOON_PA_DE440", "MOON_ME_DE421"}
        and len(step.spacecraft_position_km) == 3
        and all(math.isfinite(float(component)) for component in step.spacecraft_position_km)
        for step in scenario.steps
    )
    checks.append(_check(
        "geometry-completeness",
        ValidationStatus.PASS if geometry_complete else ValidationStatus.UNKNOWN,
        "Tagged finite Moon-centred geometry is present at every step." if geometry_complete else "Geometry frame or spacecraft position is missing/invalid; dependent claims remain unknown.",
    ))

    return ScenarioValidationReport(
        scenario_name=scenario.scenario,
        input_assumptions=inputs + scenario.assumptions,
        provenance=scenario.provenance,
        visibility_status=_visibility_status(scenario.steps),
        receiver_capture_status=_capture_status(scenario.steps),
        mirror_incident_optical_energy_wh=sum(step.mirror_incident_optical_energy_wh for step in scenario.steps),
        mirror_reflected_optical_energy_wh=sum(step.mirror_reflected_optical_energy_wh for step in scenario.steps),
        intercepted_optical_energy_wh=scenario.receiver_intercepted_optical_energy_wh,
        direct_electrical_energy_wh=scenario.direct_generation_wh,
        reflected_electrical_energy_wh=scenario.reflected_generation_wh,
        station_battery_change_wh=scenario.final_station_energy_wh - scenario.initial_station_energy_wh,
        rover_battery_change_wh=scenario.final_rover_energy_wh - scenario.initial_rover_energy_wh,
        energy_balance_residual_wh=residual,
        emergency_energy_deficit_wh=next((step.emergency_energy_deficit_wh for step in scenario.steps if step.emergency_energy_deficit_wh is not None), None),
        emergency_energy_delivered_wh=sum(step.emergency_energy_delivered_wh for step in scenario.steps),
        safe_return_status=safe_status,
        ledger_status=ledger_status,
        checks=tuple(checks),
        known_limitations=limitations,
    )


def _independent_ledger(config, scenario, tolerance):
    checks: list[ValidationCheck] = []
    steps = scenario.steps
    aggregate_pairs = (
        ("solar-input-sum", sum(s.energy_balance_input_wh for s in steps), scenario.solar_energy_input_wh),
        ("direct-generation-sum", sum(s.direct_generation_wh for s in steps), scenario.direct_generation_wh),
        ("reflected-generation-sum", sum(s.reflected_generation_wh for s in steps), scenario.reflected_generation_wh),
        ("station-load-sum", sum(s.station_load_consumed_wh for s in steps), scenario.station_load_consumed_wh),
        ("rover-consumption-sum", sum(s.rover_energy_consumed_wh for s in steps), scenario.rover_energy_consumed_wh),
        ("rover-delivery-sum", sum(s.rover_energy_delivered_wh for s in steps), scenario.rover_energy_delivered_wh),
        ("conversion-loss-sum", sum(s.conversion_losses_wh for s in steps), scenario.conversion_losses_wh),
        ("curtailment-sum", sum(s.curtailed_energy_wh for s in steps), scenario.curtailed_energy_wh),
        ("transfer-loss-sum", sum(s.transfer_losses_wh for s in steps), scenario.transfer_losses_wh),
        ("receiver-optical-sum", sum(s.receiver_intercepted_optical_energy_wh for s in steps), scenario.receiver_intercepted_optical_energy_wh),
        ("receiver-electrical-sum", sum(s.receiver_electrical_generation_wh for s in steps), scenario.receiver_electrical_generation_wh),
        ("receiver-stored-sum", sum(s.receiver_battery_stored_energy_wh for s in steps), scenario.receiver_battery_stored_energy_wh),
        ("direct-solar-input-sum", sum(s.direct_solar_input_wh for s in steps), scenario.solar_energy_input_wh - sum(s.mirror_incident_optical_energy_wh for s in steps)),
    )
    for name, calculated, reported in aggregate_pairs:
        ok = math.isfinite(calculated) and math.isfinite(reported) and abs(calculated - reported) <= tolerance
        checks.append(_check(name, ValidationStatus.PASS if ok else ValidationStatus.FAIL,
                             f"Time-step sum {calculated:.12g} Wh; scenario total {reported:.12g} Wh."))

    previous_station = scenario.initial_station_energy_wh
    previous_rover = scenario.initial_rover_energy_wh
    local_residuals = []
    for step in steps:
        storage_delta = (
            step.station_battery_energy_wh - previous_station
            + step.rover_battery_energy_wh - previous_rover
        )
        ledger_delta = (
            step.energy_balance_input_wh
            - step.station_load_consumed_wh
            - step.rover_energy_consumed_wh
            - step.conversion_losses_wh
            - step.curtailed_energy_wh
            - step.transfer_losses_wh
        )
        local_residuals.append(storage_delta - ledger_delta)
        previous_station = step.station_battery_energy_wh
        previous_rover = step.rover_battery_energy_wh
        if not (0.0 <= step.station_battery_energy_wh <= config.station_power.battery_config.nominal_capacity_wh):
            checks.append(_check("station-battery-bounds", ValidationStatus.FAIL, f"Step {step.step_index} station state violates capacity limits."))
        if step.station_battery_energy_wh < operational_floor_wh(config.station_power.battery_config) - tolerance:
            checks.append(_check("station-reserve", ValidationStatus.FAIL, f"Step {step.step_index} crossed the operational reserve."))
        if not (0.0 <= step.rover_battery_energy_wh <= config.rover_config.battery_config.nominal_capacity_wh):
            checks.append(_check("rover-battery-bounds", ValidationStatus.FAIL, f"Step {step.step_index} rover state violates capacity limits."))
        if step.rover_battery_energy_wh < operational_floor_wh(config.rover_config.battery_config) - tolerance:
            checks.append(_check("rover-operational-floor", ValidationStatus.FAIL, f"Step {step.step_index} rover state crossed its configured DoD floor."))
    if not any(check.name == "station-battery-bounds" for check in checks):
        checks.append(_check("station-battery-bounds", ValidationStatus.PASS, "All station states remain within capacity."))
    if not any(check.name == "station-reserve" for check in checks):
        checks.append(_check("station-reserve", ValidationStatus.PASS, "Station states remain above the normal operational floor."))
    if not any(check.name == "rover-battery-bounds" for check in checks):
        checks.append(_check("rover-battery-bounds", ValidationStatus.PASS, "All rover states remain within capacity."))
    if not any(check.name == "rover-operational-floor" for check in checks):
        checks.append(_check("rover-operational-floor", ValidationStatus.PASS, "All rover states remain above their configured DoD floor."))

    residual = (
        scenario.initial_station_energy_wh + scenario.initial_rover_energy_wh + scenario.solar_energy_input_wh
        - scenario.final_station_energy_wh - scenario.final_rover_energy_wh
        - scenario.station_load_consumed_wh - scenario.rover_energy_consumed_wh
        - scenario.conversion_losses_wh - scenario.curtailed_energy_wh - scenario.transfer_losses_wh
    )
    local_ok = all(math.isfinite(value) and abs(value) <= tolerance for value in local_residuals)
    aggregate_ok = math.isfinite(residual) and abs(residual) <= tolerance
    reported_residual_ok = (
        math.isfinite(scenario.energy_balance_residual_wh)
        and abs(residual - scenario.energy_balance_residual_wh) <= tolerance
    )
    optical_ok = all(
        abs(step.energy_balance_input_wh - step.direct_solar_input_wh - step.mirror_incident_optical_energy_wh) <= tolerance
        and abs(step.mirror_incident_optical_energy_wh - step.mirror_reflection_loss_wh - step.mirror_reflected_optical_energy_wh) <= tolerance
        for step in steps
    )
    reflected_output_ok = abs(
        sum(step.receiver_electrical_generation_wh for step in steps) - scenario.reflected_generation_wh
    ) <= tolerance
    checks.append(_check(
        "optical-input-and-reflection-balance",
        ValidationStatus.PASS if optical_ok else ValidationStatus.FAIL,
        "Per-step direct plus mirror input equals modeled input; incident mirror energy equals reflection loss plus reflected energy." if optical_ok else "Optical input or mirror reflection terms do not balance.",
    ))
    checks.append(_check(
        "receiver-to-reflected-electrical-balance",
        ValidationStatus.PASS if reflected_output_ok else ValidationStatus.FAIL,
        "Reflected electrical output is counted once from receiver outputs." if reflected_output_ok else "Receiver electrical output does not match the reflected-generation ledger.",
    ))
    checks.append(_check(
        "reported-residual-consistency",
        ValidationStatus.PASS if reported_residual_ok else ValidationStatus.FAIL,
        f"Independently calculated residual {residual:.12g} Wh; simulation-reported residual {scenario.energy_balance_residual_wh:.12g} Wh.",
    ))
    checks.append(_check(
        "independent-step-energy-balance",
        ValidationStatus.PASS if local_ok and aggregate_ok and reported_residual_ok else ValidationStatus.FAIL,
        f"Maximum per-step residual {max((abs(v) for v in local_residuals), default=math.inf):.12g} Wh; aggregate residual {residual:.12g} Wh. Stored energy is counted only via battery-state change.",
    ))
    return residual, checks


def _visibility_status(steps):
    statuses = []
    for step in steps:
        if step.reflector_target == "station":
            relevant = (step.mirror_sun_visibility, step.mirror_station_visibility)
        elif step.reflector_target == "rover":
            relevant = (step.mirror_sun_visibility, step.mirror_rover_visibility)
        else:
            continue
        if False in relevant:
            statuses.append("BLOCKED")
        elif None in relevant:
            statuses.append("UNKNOWN")
        else:
            statuses.append("CLEAR")
    if not statuses:
        return "NO_REFLECTOR_PATH"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "CLEAR"


def _capture_status(steps):
    values = {step.receiver_capture_status for step in steps if step.reflector_target != "none"}
    if not values or values <= {"UNKNOWN"}:
        return "UNKNOWN"
    if "BLOCKED" in values:
        return "BLOCKED"
    if any(value.startswith("UNKNOWN") for value in values):
        return "UNKNOWN"
    if "VALID" in values and "ZERO_CAPTURE" in values:
        return "MIXED"
    return next(iter(values))


def _input_assumptions(config):
    return (
        f"mission start {config.start_utc}; duration {config.duration_s:g} s; time step {config.time_step_s:g} s",
        f"station direct array {config.station_power.direct_array_area_m2:g} m^2 at {config.station_power.solar_irradiance_1au_w_m2:g} W/m^2 (1 AU), conversion efficiency {config.station_power.direct_conversion_efficiency:g}, incidence {config.station_power.direct_incidence_angle_deg:g} deg",
        f"station load {config.station_power.average_load_w:g} W; charging efficiency {config.station_power.battery_charge_efficiency:g}; protected reserve {config.station_power.battery_config.protected_reserve_wh:g} Wh",
        f"station initial battery: {config.station_power.initial_battery_state.energy_wh:g} Wh; nominal capacity: {config.station_power.battery_config.nominal_capacity_wh:g} Wh",
        f"rover initial battery: {config.initial_rover_battery_state.energy_wh:g} Wh; nominal capacity: {config.rover_config.battery_config.nominal_capacity_wh:g} Wh",
        f"rover charging efficiency {config.rover_config.charging_efficiency:g}; safe-return requirement {config.rover_energy_requirements.safe_return_energy_wh:g} Wh",
        f"receiver areas: station {config.station_power.reflected_receiver_area_m2:g} m^2, rover {config.rover_receiver.receiver_area_m2:g} m^2",
        f"PV conversion efficiencies: station {config.station_power.reflected_conversion_efficiency:g}, rover {config.rover_receiver.conversion_efficiency:g}",
        f"mirror area {config.mirror_optics.mirror_area_m2:g} m^2; reflectivity {config.mirror_optics.reflectivity:g}; pointing error {config.mirror_optics.pointing_error_deg:g} deg",
        (f"mirror irradiance input {config.mirror_optics.solar_irradiance_w_m2:g} W/m^2 is an explicit local-at-mirror value" if config.mirror_optics.solar_irradiance_w_m2 is not None else f"mirror irradiance reference {config.mirror_optics.solar_irradiance_reference_w_m2:g} W/m^2 at {config.mirror_optics.solar_irradiance_reference_distance_km:g} km is inverse-square scaled"),
        f"orbit altitude {config.orbit.altitude_km:g} km; model is assumed kinematic",
    )


def _check(name, status, detail):
    return ValidationCheck(name, status, detail)


def _combine(statuses):
    statuses = tuple(statuses)
    if ValidationStatus.FAIL in statuses:
        return ValidationStatus.FAIL
    if ValidationStatus.UNKNOWN in statuses:
        return ValidationStatus.UNKNOWN
    return ValidationStatus.PASS
