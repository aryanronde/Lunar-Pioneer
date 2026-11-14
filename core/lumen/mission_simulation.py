"""Time-stepped LUMEN mission simulation composed from existing core modules.

All mission inputs are explicit. Geometry is evaluated with SPICE Sun/time
data in J2000 and transformed to ``MOON_PA_DE440`` for lunar-site and orbit
comparisons. Orbit kinematics remain assumed. A terrain profile that is
limited-range or unavailable cannot establish full visibility or successful
rescue. This module provides no operational stability or mission-feasibility
claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import math
from typing import Literal, Protocol

from .emergency_recovery import RecoveryIlluminationInterval, RecoveryRequest, assess_emergency_recovery
from .energy_storage import BatteryConfig, BatteryState, discharge as discharge_battery
from .lumen_optics import MirrorOpticsAssumptions, TaggedLunarPosition, calculate_mirror_optics
from .lumen_orbit import CircularLunarOrbit, ORBIT_MODEL_LABEL
from .lumen_visibility import (
    LumenVisibilityResult,
    SiteHorizonProfile,
    evaluate_lumen_visibility,
    segment_occludes_lunar_sphere,
)
from .rover_energy import (
    MissionEnergyRequirements,
    RoverConfig,
    RoverOperatingState,
    RoverState,
    consume_energy,
)
from ..spice_engine import SpiceGeometryEngine
from ..lunar_frame_bridge import LunarFrameBridge
from .station_power import StationPowerInput, StationPowerResult, calculate_station_power
from .energy_storage import charge as charge_battery, operational_floor_wh
from .lumen_receiver import ReceiverAssumptions, ReceiverPowerResult, calculate_receiver_power
from ..terrain_visibility import interpolate_horizon_elevation


AU_KM = 149_597_870.7
UTC = timezone.utc
ScenarioName = Literal["normal", "rescue", "direct_only_baseline"]
ReflectorTarget = Literal["station", "rover", "none"]


class _GeometryEngine(Protocol):
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, traceback): ...
    def utc_to_et(self, utc_timestamp: str) -> float: ...
    def sun_position_km(self, et: float) -> tuple[float, float, float]: ...
    def j2000_to_moon_fixed(self, vector_j2000, et: float) -> tuple[float, float, float]: ...


class _FrameBridge(Protocol):
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, traceback): ...
    def vector_dem_to_geometry(self, vector_dem, et: float) -> tuple[float, float, float]: ...


def _number(value: float, label: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {bound}.")
    return number


def _utc(value: str, label: str = "UTC timestamp") -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp with UTC offset or Z.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset or Z suffix.")
    return parsed.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class StationPowerAssumptions:
    """Explicit station power/battery assumptions (W, Wh, m², degrees, fractions)."""

    solar_irradiance_1au_w_m2: float
    direct_array_area_m2: float
    direct_conversion_efficiency: float
    direct_incidence_angle_deg: float
    reflected_receiver_area_m2: float
    reflected_conversion_efficiency: float
    reflected_incidence_angle_deg: float
    average_load_w: float
    battery_config: BatteryConfig
    initial_battery_state: BatteryState
    battery_charge_efficiency: float
    rover_transfer_power_w: float
    rover_transfer_efficiency: float

    def __post_init__(self) -> None:
        for name in (
            "solar_irradiance_1au_w_m2", "direct_array_area_m2", "reflected_receiver_area_m2",
            "average_load_w", "rover_transfer_power_w",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in (
            "direct_conversion_efficiency", "reflected_conversion_efficiency",
            "battery_charge_efficiency", "rover_transfer_efficiency",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1].")
            object.__setattr__(self, name, value)
        for name in ("direct_incidence_angle_deg", "reflected_incidence_angle_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 90.0:
                raise ValueError(f"{name} must be finite and in [0, 90] degrees.")
            object.__setattr__(self, name, value)
        if not isinstance(self.battery_config, BatteryConfig) or not isinstance(self.initial_battery_state, BatteryState):
            raise TypeError("Station battery configuration and initial state must use energy_storage dataclasses.")
        if self.initial_battery_state.energy_wh > self.battery_config.nominal_capacity_wh:
            raise ValueError("Initial station battery state exceeds nominal capacity.")
        floor = operational_floor_wh(self.battery_config)
        if self.initial_battery_state.energy_wh < floor:
            raise ValueError("Initial station battery state is below its protected/DoD floor.")
        if self.battery_config.protected_reserve_wh < 5_000.0:
            raise ValueError("Station battery must preserve at least the configured 5,000 Wh emergency reserve.")


@dataclass(frozen=True)
class RoverSolarReceiverAssumptions:
    """Rover receiver area m², panel incidence degrees and efficiency fraction."""

    receiver_area_m2: float
    incidence_angle_deg: float
    conversion_efficiency: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "receiver_area_m2", _number(self.receiver_area_m2, "Rover receiver area"))
        angle = float(self.incidence_angle_deg)
        if not math.isfinite(angle) or not 0.0 <= angle <= 90.0:
            raise ValueError("Rover receiver incidence angle must be in [0, 90] degrees.")
        object.__setattr__(self, "incidence_angle_deg", angle)
        efficiency = float(self.conversion_efficiency)
        if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
            raise ValueError("Rover receiver conversion efficiency must be in (0, 1].")
        object.__setattr__(self, "conversion_efficiency", efficiency)


@dataclass(frozen=True)
class MissionTimelineSegment:
    """Complete timeline segment; power W and offsets seconds from start."""

    start_offset_s: float
    end_offset_s: float
    activity: str
    rover_consumption_power_w: float
    reflector_target: ReflectorTarget
    station_to_rover_transfer_power_w: float

    def __post_init__(self) -> None:
        start = _number(self.start_offset_s, "Segment start")
        end = _number(self.end_offset_s, "Segment end", positive=True)
        if end <= start:
            raise ValueError("Mission segment end must follow its start.")
        object.__setattr__(self, "start_offset_s", start)
        object.__setattr__(self, "end_offset_s", end)
        object.__setattr__(self, "rover_consumption_power_w", _number(self.rover_consumption_power_w, "Rover consumption power"))
        object.__setattr__(
            self,
            "station_to_rover_transfer_power_w",
            _number(self.station_to_rover_transfer_power_w, "Station-to-rover transfer power"),
        )
        if not self.activity.strip():
            raise ValueError("Mission activity must be named explicitly.")
        if self.reflector_target not in ("station", "rover", "none"):
            raise ValueError("reflector_target must be station, rover, or none.")


@dataclass(frozen=True)
class MissionSimulationConfig:
    """Complete, explicit assumptions for one reproducible mission simulation."""

    start_utc: str
    duration_s: float
    time_step_s: float
    orbit: CircularLunarOrbit
    station_position: TaggedLunarPosition
    rover_position: TaggedLunarPosition
    station_horizon: SiteHorizonProfile | None
    rover_horizon: SiteHorizonProfile | None
    station_power: StationPowerAssumptions
    rover_config: RoverConfig
    initial_rover_battery_state: BatteryState
    rover_energy_requirements: MissionEnergyRequirements
    rover_receiver: RoverSolarReceiverAssumptions
    mirror_optics: MirrorOpticsAssumptions
    mission_timeline: tuple[MissionTimelineSegment, ...]
    stranded_at_offset_s: float
    stranded_survival_power_w: float
    rescue_window_s: float
    lunar_reference_radius_km: float
    flag_energy_residual_above_wh: float = 1e-6

    def __post_init__(self) -> None:
        _utc(self.start_utc, "start_utc")
        for name in ("duration_s", "time_step_s", "rescue_window_s", "lunar_reference_radius_km"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        for name in ("stranded_at_offset_s", "stranded_survival_power_w", "flag_energy_residual_above_wh"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.stranded_at_offset_s >= self.duration_s:
            raise ValueError("stranded_at_offset_s must lie within the mission duration.")
        if self.stranded_at_offset_s + self.rescue_window_s > self.duration_s + 1e-9:
            raise ValueError("The rescue window must end within the simulated mission duration.")
        if self.flag_energy_residual_above_wh < 0:
            raise ValueError("Energy residual threshold must be nonnegative.")
        for name, value in (
            ("orbit", self.orbit), ("station_position", self.station_position),
            ("rover_position", self.rover_position), ("station_power", self.station_power),
            ("rover_config", self.rover_config), ("initial_rover_battery_state", self.initial_rover_battery_state),
            ("rover_energy_requirements", self.rover_energy_requirements),
            ("rover_receiver", self.rover_receiver), ("mirror_optics", self.mirror_optics),
        ):
            if value is None:
                raise ValueError(f"{name} is required; missing inputs are not inferred.")
        if self.station_position.frame not in ("MOON_PA_DE440", "MOON_ME_DE421"):
            raise ValueError("Station coordinates must use MOON_PA_DE440 or MOON_ME_DE421.")
        if self.rover_position.frame not in ("MOON_PA_DE440", "MOON_ME_DE421"):
            raise ValueError("Rover coordinates must use MOON_PA_DE440 or MOON_ME_DE421.")
        if self.initial_rover_battery_state.energy_wh > self.rover_config.battery_config.nominal_capacity_wh:
            raise ValueError("Initial rover battery state exceeds nominal capacity.")
        floor = operational_floor_wh(self.rover_config.battery_config)
        if self.initial_rover_battery_state.energy_wh < floor:
            raise ValueError("Initial rover battery state is below its configured DoD floor.")
        timeline = tuple(self.mission_timeline)
        if not timeline or any(not isinstance(item, MissionTimelineSegment) for item in timeline):
            raise ValueError("mission_timeline must contain MissionTimelineSegment values.")
        ordered = tuple(sorted(timeline, key=lambda item: item.start_offset_s))
        if abs(ordered[0].start_offset_s) > 1e-9 or abs(ordered[-1].end_offset_s - self.duration_s) > 1e-9:
            raise ValueError("Mission timeline must cover exactly [0, duration_s].")
        for left, right in zip(ordered, ordered[1:]):
            if abs(left.end_offset_s - right.start_offset_s) > 1e-9:
                raise ValueError("Mission timeline segments must be contiguous without overlap or gaps.")
        object.__setattr__(self, "mission_timeline", ordered)
        for profile, site in ((self.station_horizon, self.station_position), (self.rover_horizon, self.rover_position)):
            if profile is not None and profile.frame != site.frame:
                raise ValueError("Site horizon profile and site position must share an explicit frame.")
            if profile is not None and _norm(tuple(profile.site_position_km[i] - site.position_km[i] for i in range(3))) > 1e-6:
                raise ValueError("Each site horizon profile must belong to its configured site coordinate.")
        if any(segment.reflector_target == "rover" for segment in ordered):
            raise ValueError("The mission timeline targets the station or none; rover targeting is reserved for the rescue scenario.")
        if any(
            segment.station_to_rover_transfer_power_w > self.station_power.rover_transfer_power_w
            for segment in ordered
        ):
            raise ValueError("Timeline station-to-rover transfer power exceeds its configured limit.")


@dataclass(frozen=True)
class MissionStepResult:
    """One scenario/time-step result; all energy values are Wh, power values W."""

    scenario: ScenarioName
    step_index: int
    start_utc: str
    end_utc: str
    sample_utc: str
    spacecraft_position_km: tuple[float, float, float]
    geometry_frame: str
    reflector_target: str
    direct_station_illumination: bool | None
    mirror_sun_visibility: bool | None
    mirror_station_visibility: bool | None
    mirror_rover_visibility: bool | None
    reflected_illumination_active: bool
    direct_generation_wh: float
    reflected_generation_wh: float
    station_battery_energy_wh: float
    rover_battery_energy_wh: float
    rover_operating_state: RoverOperatingState
    station_load_consumed_wh: float
    rover_energy_consumed_wh: float
    rover_energy_delivered_wh: float
    station_to_rover_transfer_energy_wh: float
    conversion_losses_wh: float
    curtailed_energy_wh: float
    transfer_losses_wh: float
    safe_return_feasible: bool
    emergency_recovery_attempted: bool
    emergency_recovery_success: bool | None
    emergency_failure_reasons: tuple[str, ...]
    energy_balance_input_wh: float
    provenance: tuple[str, ...]
    receiver_capture_status: str = "UNKNOWN"
    receiver_intercepted_optical_energy_wh: float = 0.0
    receiver_electrical_generation_wh: float = 0.0
    receiver_battery_stored_energy_wh: float = 0.0
    direct_solar_input_wh: float = 0.0
    mirror_incident_optical_energy_wh: float = 0.0
    mirror_reflected_optical_energy_wh: float = 0.0
    mirror_reflection_loss_wh: float = 0.0
    emergency_energy_deficit_wh: float | None = None
    emergency_energy_delivered_wh: float = 0.0


@dataclass(frozen=True)
class MissionScenarioResult:
    """Scenario time series, final states and complete energy ledger."""

    scenario: ScenarioName
    steps: tuple[MissionStepResult, ...]
    initial_station_energy_wh: float
    final_station_energy_wh: float
    initial_rover_energy_wh: float
    final_rover_energy_wh: float
    direct_generation_wh: float
    reflected_generation_wh: float
    solar_energy_input_wh: float
    station_load_consumed_wh: float
    rover_energy_consumed_wh: float
    rover_energy_delivered_wh: float
    conversion_losses_wh: float
    curtailed_energy_wh: float
    transfer_losses_wh: float
    energy_balance_residual_wh: float
    unexplained_discrepancy: bool
    rescue_success: bool | None
    rescue_failure_reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    provenance: tuple[str, ...]
    receiver_intercepted_optical_energy_wh: float = 0.0
    receiver_electrical_generation_wh: float = 0.0
    receiver_battery_stored_energy_wh: float = 0.0


@dataclass(frozen=True)
class ScenarioComparison:
    """Matched direct-only versus reflected scenario energy comparison."""

    baseline_scenario: str
    reflected_scenario: str
    baseline_station_final_energy_wh: float
    reflected_station_final_energy_wh: float
    baseline_rover_final_energy_wh: float
    reflected_rover_final_energy_wh: float
    additional_rover_energy_delivered_wh: float
    additional_station_stored_energy_wh: float
    additional_reflected_generation_wh: float


@dataclass(frozen=True)
class MissionSimulationResult:
    """Three reproducible mission scenarios and matched comparison."""

    scenarios: tuple[MissionScenarioResult, ...]
    comparison: ScenarioComparison
    assumptions: tuple[str, ...]
    provenance: tuple[str, ...]

    def scenario(self, name: ScenarioName) -> MissionScenarioResult:
        for result in self.scenarios:
            if result.scenario == name:
                return result
        raise KeyError(name)


def simulate_lumen_mission(
    config: MissionSimulationConfig,
    *,
    geometry_engine_factory=SpiceGeometryEngine,
    frame_bridge_factory=LunarFrameBridge,
) -> MissionSimulationResult:
    """Run normal, stranded-rescue and matched direct-only scenarios.

    SPICE supplies UTC→ET, Sun geometry, and J2000→PA frame rotation. The
    orbit model supplies assumed circular motion only. Missing terrain data
    yields unknown visibility and disables energy credited to that path.
    """
    if not isinstance(config, MissionSimulationConfig):
        raise TypeError("config must be MissionSimulationConfig.")
    with geometry_engine_factory() as geometry, frame_bridge_factory() as bridge:
        epoch_et = float(geometry.utc_to_et(config.orbit.epoch_utc))
        results = tuple(
            _simulate_scenario(config, name, geometry, bridge, epoch_et)
            for name in ("normal", "rescue", "direct_only_baseline")
        )
    normal = results[0]
    baseline = results[2]
    comparison = ScenarioComparison(
        baseline_scenario="direct_only_baseline",
        reflected_scenario="normal",
        baseline_station_final_energy_wh=baseline.final_station_energy_wh,
        reflected_station_final_energy_wh=normal.final_station_energy_wh,
        baseline_rover_final_energy_wh=baseline.final_rover_energy_wh,
        reflected_rover_final_energy_wh=normal.final_rover_energy_wh,
        additional_rover_energy_delivered_wh=(
            normal.rover_energy_delivered_wh - baseline.rover_energy_delivered_wh
        ),
        additional_station_stored_energy_wh=(
            normal.final_station_energy_wh - baseline.final_station_energy_wh
        ),
        additional_reflected_generation_wh=normal.reflected_generation_wh - baseline.reflected_generation_wh,
    )
    return MissionSimulationResult(
        scenarios=results,
        comparison=comparison,
        assumptions=(
            "circular lunar orbit is assumed kinematic and not stability-validated",
            "mission events, rover loads, station loads, hardware areas and efficiencies are supplied assumptions",
            "flat reflector redirects but does not focus sunlight",
            "reflector illumination is enabled only for an explicit target and clear Sun/mirror/target geometry",
            "receiver interception uses lumen_receiver's centered uniform circular footprint approximation",
            "nonzero mirror pointing error is reported unknown for capture because lateral overlap is not modeled",
            "receiver electrical output and battery-stored energy are ledgered as separate transitions",
            "limited-range terrain profiles do not establish visibility beyond their stated range",
            "step geometry and power are treated as constant over each integration interval",
        ),
        provenance=(
            "Sun position, UTC-to-ET and J2000-to-MOON_PA_DE440 rotation from SPICE kernels",
            "MOON_ME_DE421 site vectors and local axes rotated through LunarFrameBridge",
            f"orbit source: {ORBIT_MODEL_LABEL}",
            "station power, rover energy, battery, receiver interception and emergency recovery from core/lumen modules",
        ),
    )


def _simulate_scenario(config, scenario, geometry, bridge, epoch_et) -> MissionScenarioResult:
    station_energy = config.station_power.initial_battery_state
    rover_energy = config.initial_rover_battery_state
    initial_station = station_energy.energy_wh
    initial_rover = rover_energy.energy_wh
    steps: list[MissionStepResult] = []
    totals = {
        "direct_generation_wh": 0.0,
        "reflected_generation_wh": 0.0,
        "solar_energy_input_wh": 0.0,
        "station_load_consumed_wh": 0.0,
        "rover_energy_consumed_wh": 0.0,
        "rover_energy_delivered_wh": 0.0,
        "conversion_losses_wh": 0.0,
        "curtailed_energy_wh": 0.0,
        "transfer_losses_wh": 0.0,
        "receiver_intercepted_optical_energy_wh": 0.0,
        "receiver_electrical_generation_wh": 0.0,
        "receiver_battery_stored_energy_wh": 0.0,
    }
    rescue_success: bool | None = None
    rescue_reasons: list[str] = []
    elapsed = 0.0
    step_index = 0
    while elapsed < config.duration_s - 1e-9:
        dt = min(config.time_step_s, config.duration_s - elapsed)
        step_start = _utc(config.start_utc) + timedelta(seconds=elapsed)
        step_end = step_start + timedelta(seconds=dt)
        sample = step_start + timedelta(seconds=dt / 2.0)
        sample_text = _utc_text(sample)
        spice_timestamp = sample.strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0").rstrip(".")
        et = float(geometry.utc_to_et(spice_timestamp))
        sun_j2000 = geometry.sun_position_km(et)
        sun_pa = geometry.j2000_to_moon_fixed(sun_j2000, et)
        orbit_j2000 = config.orbit.position_at_et(et, epoch_et)
        mirror_pa = geometry.j2000_to_moon_fixed(orbit_j2000, et)
        station_pa, station_horizon_pa = _site_to_pa(
            config.station_position, config.station_horizon, et, bridge
        )
        rover_pa, rover_horizon_pa = _site_to_pa(
            config.rover_position, config.rover_horizon, et, bridge
        )
        sun = TaggedLunarPosition(sun_pa, "MOON_PA_DE440")
        mirror = TaggedLunarPosition(mirror_pa, "MOON_PA_DE440")
        station = TaggedLunarPosition(station_pa, "MOON_PA_DE440")
        rover = TaggedLunarPosition(rover_pa, "MOON_PA_DE440")
        visibility = evaluate_lumen_visibility(
            sun, mirror, station, rover,
            lunar_reference_radius_km=config.lunar_reference_radius_km,
            station_horizon=station_horizon_pa,
            rover_horizon=rover_horizon_pa,
        )
        direct_visible, direct_status = _direct_site_visibility(
            sun, station, station_horizon_pa, config.lunar_reference_radius_km
        )
        segment = _mission_segment_at(config.mission_timeline, elapsed + dt / 2.0)
        rescue_deadline_dt = _utc(config.start_utc) + timedelta(
            seconds=config.stranded_at_offset_s + config.rescue_window_s
        )
        stranded = scenario == "rescue" and elapsed >= config.stranded_at_offset_s - 1e-9
        if scenario == "rescue" and rescue_success:
            stranded = False
        reflector_target = "none" if scenario == "direct_only_baseline" else segment.reflector_target
        if stranded:
            reflector_target = "rover"

        selected_optics = None
        target_visibility = None
        if reflector_target == "station":
            target_pos = station
            target_visibility = visibility.mirror_to_station
        elif reflector_target == "rover":
            target_pos = rover
            target_visibility = visibility.mirror_to_rover
        else:
            target_pos = None
        if target_pos is not None:
            solar_irradiance_at_mirror = (
                config.station_power.solar_irradiance_1au_w_m2
                * (AU_KM / _norm(sun_j2000)) ** 2
            )
            selected_assumptions = replace(
                config.mirror_optics,
                solar_irradiance_w_m2=solar_irradiance_at_mirror,
            )
            selected_optics = calculate_mirror_optics(
                sun, mirror, target_pos, selected_assumptions, geometry_origin="assumed"
            )

        source_sun_mirror_visible = visibility.sun_to_mirror.visible is True
        mirror_available = (
            scenario != "direct_only_baseline"
            and reflector_target in ("station", "rover")
            and source_sun_mirror_visible
            and target_visibility is not None
            and target_visibility.visible is True
        )
        irradiance_1au = config.station_power.solar_irradiance_1au_w_m2
        sun_range_km = _norm(sun_j2000)
        station_direct_irradiance = (
            irradiance_1au * (AU_KM / sun_range_km) ** 2
            if direct_visible is True else 0.0
        )
        reflected_irradiance = reflected_area_eff = 0.0
        reflected_incidence_deg = 0.0
        mirror_incident_source_wh = 0.0
        mirror_reflected_energy_wh = 0.0
        mirror_optical_loss_wh = mirror_spill_wh = 0.0
        station_receiver_power: ReceiverPowerResult | None = None
        rover_receiver_power: ReceiverPowerResult | None = None
        if mirror_available and selected_optics is not None:
            mirror_incident_source_wh = selected_optics.incident_optical_power_w * dt / 3600.0
            mirror_reflected_wh = selected_optics.reflected_optical_power_w * dt / 3600.0
            mirror_reflected_energy_wh = mirror_reflected_wh
            mirror_optical_loss_wh = mirror_incident_source_wh - mirror_reflected_wh
            if selected_optics.pointing_error_deg <= 1e-12:
                if reflector_target == "station":
                    station_receiver_power = calculate_receiver_power(
                        selected_optics,
                        _receiver_assumptions(
                            selected_optics,
                            config.station_power.reflected_receiver_area_m2,
                            config.station_power.reflected_incidence_angle_deg,
                            config.station_power.reflected_conversion_efficiency,
                        ),
                    )
                elif reflector_target == "rover":
                    rover_receiver_power = calculate_receiver_power(
                        selected_optics,
                        _receiver_assumptions(
                            selected_optics,
                            config.rover_receiver.receiver_area_m2,
                            config.rover_receiver.incidence_angle_deg,
                            config.rover_receiver.conversion_efficiency,
                        ),
                    )
            if reflector_target == "station":
                if station_receiver_power is None:
                    mirror_spill_wh = mirror_reflected_wh
                else:
                    reflected_area_eff = config.station_power.reflected_receiver_area_m2
                    reflected_irradiance = (
                        station_receiver_power.electrical_power_w / reflected_area_eff
                        if reflected_area_eff > 0.0 else 0.0
                    )
                    mirror_spill_wh = station_receiver_power.beam_spill_power_w * dt / 3600.0
                    mirror_spill_wh += (
                        selected_optics.reflected_optical_power_w
                        - station_receiver_power.intercepted_optical_power_w
                        - station_receiver_power.beam_spill_power_w
                    ) * dt / 3600.0
            elif reflector_target == "rover" and rover_receiver_power is None:
                # Pointing error is known, but its lateral overlap is outside
                # this receiver model; do not credit uncertain rover power.
                mirror_spill_wh = mirror_reflected_wh

        station_power_input = StationPowerInput(
            duration_s=dt,
            direct_irradiance_w_m2=station_direct_irradiance,
            direct_array_area_m2=config.station_power.direct_array_area_m2,
            direct_incidence_angle_deg=(
                config.station_power.direct_incidence_angle_deg if direct_visible is True else 0.0
            ),
            direct_conversion_efficiency=config.station_power.direct_conversion_efficiency,
            reflected_irradiance_w_m2=reflected_irradiance,
            reflected_receiver_area_m2=reflected_area_eff,
            reflected_incidence_angle_deg=reflected_incidence_deg,
            reflected_conversion_efficiency=1.0,
            reflected_illumination_available=(
                station_receiver_power is not None
                and station_receiver_power.capture_fraction > 0.0
                and station_receiver_power.electrical_power_w > 0.0
            ),
            station_load_w=config.station_power.average_load_w,
            battery_charge_efficiency=config.station_power.battery_charge_efficiency,
            battery_config=config.station_power.battery_config,
            battery_state=station_energy,
            direct_irradiance_origin="calculated",
            direct_incidence_origin="assumed",
            reflected_irradiance_origin="calculated",
            reflected_incidence_origin="calculated",
        )
        station_power = calculate_station_power(station_power_input)
        station_energy = (
            station_power.battery_charge.resulting_state
            if station_power.battery_charge is not None else station_energy
        )
        unserved_load = station_power.load_unserved_energy_wh
        load_discharge = discharge_battery(
            station_energy,
            config.station_power.battery_config,
            unserved_load,
            "normal",
            duration_s=dt,
        )
        station_energy = load_discharge.resulting_state
        station_load_served = station_power.station_load_energy_wh - load_discharge.unmet_energy_wh

        step_losses = (
            station_power.direct_conversion_loss_wh
            + station_power.reflected_conversion_loss_wh
            + station_power.battery_charge_loss_wh
            + mirror_optical_loss_wh
            + mirror_spill_wh
        )
        if station_receiver_power is not None:
            step_losses += (
                station_receiver_power.optical_transmission_loss_w
                + station_receiver_power.photovoltaic_loss_w
                + station_receiver_power.electrical_conversion_loss_w
            ) * dt / 3600.0
        step_curtailed = station_power.curtailed_energy_wh
        step_transfer_loss = 0.0
        rover_delivered = 0.0
        rover_receiver_stored_wh = 0.0
        station_transfer = 0.0

        transfer_power = (
            segment.station_to_rover_transfer_power_w
            if not stranded else 0.0
        )
        if transfer_power > 0:
            requested_transfer = transfer_power * dt / 3600.0
            transfer_result = discharge_battery(
                station_energy,
                config.station_power.battery_config,
                requested_transfer,
                "normal",
                duration_s=dt,
            )
            station_energy = transfer_result.resulting_state
            station_transfer = transfer_result.accepted_energy_wh
            rover_input = station_transfer * config.station_power.rover_transfer_efficiency
            step_transfer_loss = station_transfer - rover_input
            rover_charge = charge_battery(
                rover_energy,
                config.rover_config.battery_config,
                rover_input,
                config.rover_config.charging_efficiency,
                duration_s=dt,
            )
            rover_energy = rover_charge.resulting_state
            rover_delivered += rover_charge.stored_energy_wh
            step_losses += rover_charge.loss_energy_wh
            step_curtailed += rover_charge.curtailed_input_energy_wh

        rover_power = config.stranded_survival_power_w if stranded else segment.rover_consumption_power_w
        rover_consumption_duration_s = dt
        if stranded:
            rover_consumption_duration_s = max(
                0.0, min(dt, (rescue_deadline_dt - step_start).total_seconds())
            )
        requested_consumption = rover_power * rover_consumption_duration_s / 3600.0
        consumption_result = consume_energy(
            RoverState(rover_energy), config.rover_config, requested_consumption,
            config.rover_energy_requirements, duration_s=max(rover_consumption_duration_s, 0.0),
        )
        rover_energy = BatteryState(consumption_result.battery_energy_wh)
        rover_consumed = consumption_result.accepted_energy_wh

        recovery_attempted = False
        recovery = None
        recovery_success: bool | None = None
        recovery_reasons: tuple[str, ...] = ()
        recovery_deficit_wh: float | None = None
        recovery_delivered_wh = 0.0
        rover_operating_state = consumption_result.operating_state
        if (
            stranded
            and rover_operating_state == RoverOperatingState.EMERGENCY
            and step_start < rescue_deadline_dt
        ):
            recovery_attempted = True
            recovery_visibility = visibility
            interval = RecoveryIlluminationInterval(
                start_utc=_utc_text(step_start),
                end_utc=_utc_text(step_end),
                illumination_available=(
                    scenario == "rescue"
                    and reflector_target == "rover"
                    and source_sun_mirror_visible
                    and rover_receiver_power is not None
                    and rover_receiver_power.capture_fraction > 0.0
                    and rover_receiver_power.electrical_power_w > 0.0
                ),
                visibility=recovery_visibility,
                optics=selected_optics or _zero_optics_for_frame("MOON_PA_DE440"),
                receiver_area_m2=config.rover_receiver.receiver_area_m2,
                receiver_incidence_angle_deg=config.rover_receiver.incidence_angle_deg,
                receiver_conversion_efficiency=config.rover_receiver.conversion_efficiency,
                receiver_power=rover_receiver_power,
            )
            rescue_deadline = rescue_deadline_dt
            recovery = assess_emergency_recovery(
                RecoveryRequest(
                    current_battery_energy_wh=rover_energy.energy_wh,
                    safe_return_energy_wh=config.rover_energy_requirements.safe_return_energy_wh,
                    survival_start_utc=_utc_text(step_start),
                    survival_deadline_utc=_utc_text(max(step_end, rescue_deadline)),
                    rover_config=config.rover_config,
                    intervals=(interval,),
                )
            )
            # The recovery calculator clips its interval to the survival
            # deadline, but the survival start is this interval's beginning.
            rover_energy = BatteryState(recovery.final_battery_energy_wh)
            rover_delivered += recovery.delivered_battery_energy_wh
            recovery_deficit_wh = recovery.energy_deficit_wh
            recovery_delivered_wh = recovery.delivered_battery_energy_wh
            step_losses += recovery.receiver_conversion_loss_wh + recovery.battery_charging_loss_wh
            step_losses += recovery.optical_spill_energy_wh
            step_losses += recovery.receiver_optical_transmission_loss_wh
            step_curtailed += recovery.charge_power_or_capacity_curtailed_wh
            # Mirror input and reflectivity loss are accounted before the
            # receiver's optical/electrical conversion terms.
            recovery_success = recovery.feasible
            recovery_reasons = recovery.failure_reasons
            rescue_success = recovery_success if recovery_success else rescue_success
            if not recovery_success:
                rescue_reasons.extend(recovery.failure_reasons)
            rover_operating_state = (
                RoverOperatingState.RETURN_REQUIRED if recovery.feasible
                else RoverOperatingState.EMERGENCY
            )
        elif (
            mirror_available
            and reflector_target == "rover"
            and rover_receiver_power is not None
            and rover_receiver_power.capture_fraction > 0.0
            and rover_receiver_power.electrical_power_w > 0.0
        ):
            # Outside the emergency assessor window, pass the receiver's
            # electrical output to the rover battery exactly once.
            receiver_input_wh = rover_receiver_power.electrical_power_w * dt / 3600.0
            rover_charge = charge_battery(
                rover_energy,
                config.rover_config.battery_config,
                receiver_input_wh,
                config.rover_config.charging_efficiency,
                duration_s=dt,
            )
            rover_energy = rover_charge.resulting_state
            rover_delivered += rover_charge.stored_energy_wh
            rover_receiver_stored_wh += rover_charge.stored_energy_wh
            step_losses += rover_charge.loss_energy_wh
            step_curtailed += rover_charge.curtailed_input_energy_wh
        elif stranded and step_start >= rescue_deadline_dt and rescue_success is not True:
            rescue_success = False
            if not rescue_reasons:
                rescue_reasons.append("Emergency recovery deadline expired before safe-return energy was restored.")

        if mirror_available and reflector_target == "rover" and selected_optics is not None:
            if recovery_attempted and recovery is not None:
                if recovery.receiver_intercepted_optical_energy_wh <= 0.0 and mirror_spill_wh <= 1e-12:
                    step_losses += selected_optics.reflected_optical_power_w * dt / 3600.0
                # Otherwise the recovery result accounts receiver spill and losses.
            elif rover_receiver_power is not None:
                spill = rover_receiver_power.beam_spill_power_w * dt / 3600.0
                spill += (
                    selected_optics.reflected_optical_power_w
                    - rover_receiver_power.intercepted_optical_power_w
                    - rover_receiver_power.beam_spill_power_w
                ) * dt / 3600.0
                step_losses += spill
                step_losses += (
                    rover_receiver_power.optical_transmission_loss_w
                    + rover_receiver_power.photovoltaic_loss_w
                    + rover_receiver_power.electrical_conversion_loss_w
                ) * dt / 3600.0
            else:
                step_losses += selected_optics.reflected_optical_power_w * dt / 3600.0

        safe_floor = operational_floor_wh(config.rover_config.battery_config)
        safe_margin = rover_energy.energy_wh - safe_floor - config.rover_energy_requirements.safe_return_energy_wh
        safe_return_feasible = safe_margin >= -1e-9
        direct_incident_wh = (
            station_direct_irradiance
            * config.station_power.direct_array_area_m2
            * math.cos(math.radians(config.station_power.direct_incidence_angle_deg))
            * dt / 3600.0
        )
        solar_input = direct_incident_wh + mirror_incident_source_wh
        step_input = solar_input
        totals["direct_generation_wh"] += station_power.direct_generation_wh
        totals["reflected_generation_wh"] += station_power.reflected_generation_wh
        if rover_receiver_power is not None and reflector_target == "rover" and not recovery_attempted:
            totals["reflected_generation_wh"] += rover_receiver_power.electrical_power_w * dt / 3600.0
        if recovery_attempted:
            totals["reflected_generation_wh"] += recovery.receiver_electrical_energy_wh
        totals["solar_energy_input_wh"] += solar_input
        totals["station_load_consumed_wh"] += station_load_served
        totals["rover_energy_consumed_wh"] += rover_consumed
        totals["rover_energy_delivered_wh"] += rover_delivered
        totals["conversion_losses_wh"] += step_losses
        totals["curtailed_energy_wh"] += step_curtailed
        totals["transfer_losses_wh"] += step_transfer_loss

        receiver_intercepted_wh = 0.0
        receiver_electrical_wh = 0.0
        receiver_stored_wh = rover_receiver_stored_wh
        if station_receiver_power is not None and reflector_target == "station":
            receiver_intercepted_wh += station_receiver_power.intercepted_optical_power_w * dt / 3600.0
            receiver_electrical_wh += station_receiver_power.electrical_power_w * dt / 3600.0
        if rover_receiver_power is not None and reflector_target == "rover" and not recovery_attempted:
            receiver_intercepted_wh += rover_receiver_power.intercepted_optical_power_w * dt / 3600.0
            receiver_electrical_wh += rover_receiver_power.electrical_power_w * dt / 3600.0
        if recovery_attempted and recovery is not None:
            receiver_intercepted_wh += recovery.receiver_intercepted_optical_energy_wh
            receiver_electrical_wh += recovery.receiver_electrical_energy_wh
            receiver_stored_wh += recovery.delivered_battery_energy_wh
        totals["receiver_intercepted_optical_energy_wh"] += receiver_intercepted_wh
        totals["receiver_electrical_generation_wh"] += receiver_electrical_wh
        totals["receiver_battery_stored_energy_wh"] += receiver_stored_wh

        steps.append(MissionStepResult(
            scenario=scenario,
            step_index=step_index,
            start_utc=_utc_text(step_start),
            end_utc=_utc_text(step_end),
            sample_utc=sample_text,
            spacecraft_position_km=mirror_pa,
            geometry_frame="MOON_PA_DE440",
            reflector_target=reflector_target,
            direct_station_illumination=direct_visible,
            mirror_sun_visibility=visibility.sun_to_mirror.visible,
            mirror_station_visibility=visibility.mirror_to_station.visible,
            mirror_rover_visibility=visibility.mirror_to_rover.visible,
            reflected_illumination_active=mirror_available,
            direct_generation_wh=station_power.direct_generation_wh,
            reflected_generation_wh=(
                station_power.reflected_generation_wh
                + (
                    rover_receiver_power.electrical_power_w * dt / 3600.0
                    if rover_receiver_power is not None and reflector_target == "rover" and not recovery_attempted
                    else 0.0
                )
                + (recovery.receiver_electrical_energy_wh if recovery_attempted else 0.0)
            ),
            station_battery_energy_wh=station_energy.energy_wh,
            rover_battery_energy_wh=rover_energy.energy_wh,
            rover_operating_state=rover_operating_state,
            station_load_consumed_wh=station_load_served,
            rover_energy_consumed_wh=rover_consumed,
            rover_energy_delivered_wh=rover_delivered,
            station_to_rover_transfer_energy_wh=station_transfer,
            conversion_losses_wh=step_losses,
            curtailed_energy_wh=step_curtailed,
            transfer_losses_wh=step_transfer_loss,
            safe_return_feasible=safe_return_feasible,
            emergency_recovery_attempted=recovery_attempted,
            emergency_recovery_success=recovery_success,
            emergency_failure_reasons=recovery_reasons,
            energy_balance_input_wh=step_input,
            provenance=(
                "SPICE Sun vector and PA frame at sample UTC",
                ORBIT_MODEL_LABEL,
                f"direct station terrain status: {direct_status}",
                "station/rover loads and hardware parameters are supplied assumptions",
            ),
            receiver_capture_status=_capture_status(
                mirror_available,
                selected_optics,
                station_receiver_power or rover_receiver_power,
                target_visibility,
            ),
            receiver_intercepted_optical_energy_wh=receiver_intercepted_wh,
            receiver_electrical_generation_wh=receiver_electrical_wh,
            receiver_battery_stored_energy_wh=receiver_stored_wh,
            direct_solar_input_wh=direct_incident_wh,
            mirror_incident_optical_energy_wh=mirror_incident_source_wh,
            mirror_reflected_optical_energy_wh=mirror_reflected_energy_wh,
            mirror_reflection_loss_wh=mirror_optical_loss_wh,
            emergency_energy_deficit_wh=recovery_deficit_wh,
            emergency_energy_delivered_wh=recovery_delivered_wh,
        ))
        elapsed += dt
        step_index += 1

    final_station = station_energy.energy_wh
    final_rover = rover_energy.energy_wh
    initial_total = initial_station + initial_rover
    final_total = final_station + final_rover
    consumed = totals["station_load_consumed_wh"] + totals["rover_energy_consumed_wh"]
    residual = (
        initial_total + totals["solar_energy_input_wh"]
        - final_total - consumed - totals["conversion_losses_wh"]
        - totals["curtailed_energy_wh"] - totals["transfer_losses_wh"]
    )
    if scenario == "rescue" and rescue_success is None and rescue_reasons:
        rescue_success = False
    if scenario != "rescue":
        rescue_success = None
    assumptions = (
        "identical initial station/rover battery states across scenarios",
        "normal and direct-only scenarios share the same timeline; baseline disables reflector assistance",
        "rescue scenario applies stranded survival load after configured stranding offset",
        "station normal discharge uses energy_storage reserve-protected discharge",
        "site coordinates are fixed in a lunar body-fixed frame during the run",
    )
    provenance = (
        "SPICE-derived Sun and UTC/ET geometry; assumed orbit kinematics",
        "station_power for direct/reflected station electrical generation and station load",
        "energy_storage for all station/rover battery transitions",
        "rover_energy for consumption and safe-return state logic",
        "lumen_visibility and emergency_recovery for mirror links and recovery assessment",
    )
    return MissionScenarioResult(
        scenario=scenario,
        steps=tuple(steps),
        initial_station_energy_wh=initial_station,
        final_station_energy_wh=final_station,
        initial_rover_energy_wh=initial_rover,
        final_rover_energy_wh=final_rover,
        direct_generation_wh=totals["direct_generation_wh"],
        reflected_generation_wh=totals["reflected_generation_wh"],
        solar_energy_input_wh=totals["solar_energy_input_wh"],
        station_load_consumed_wh=totals["station_load_consumed_wh"],
        rover_energy_consumed_wh=totals["rover_energy_consumed_wh"],
        rover_energy_delivered_wh=totals["rover_energy_delivered_wh"],
        conversion_losses_wh=totals["conversion_losses_wh"],
        curtailed_energy_wh=totals["curtailed_energy_wh"],
        transfer_losses_wh=totals["transfer_losses_wh"],
        energy_balance_residual_wh=residual,
        unexplained_discrepancy=abs(residual) > config.flag_energy_residual_above_wh,
        rescue_success=rescue_success,
        rescue_failure_reasons=tuple(dict.fromkeys(rescue_reasons)),
        assumptions=assumptions,
        provenance=provenance,
        receiver_intercepted_optical_energy_wh=totals["receiver_intercepted_optical_energy_wh"],
        receiver_electrical_generation_wh=totals["receiver_electrical_generation_wh"],
        receiver_battery_stored_energy_wh=totals["receiver_battery_stored_energy_wh"],
    )


def _receiver_assumptions(optics, area_m2, incidence_angle_deg, efficiency):
    """Encode the supplied scalar incidence in the optics frame.

    The current mission interface supplies incidence but no panel azimuth.
    Azimuth is irrelevant to this model's projected-area calculation; the
    deterministic tangent below only constructs a unit normal with that angle.
    """
    toward_mirror = tuple(-value for value in optics.mirror_to_target_unit)
    axis = (1.0, 0.0, 0.0) if abs(toward_mirror[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(axis[i] * toward_mirror[i] for i in range(3))
    tangent = tuple(axis[i] - projection * toward_mirror[i] for i in range(3))
    tangent_norm = _norm(tangent)
    tangent = tuple(value / tangent_norm for value in tangent)
    angle = math.radians(incidence_angle_deg)
    normal = tuple(
        math.cos(angle) * toward_mirror[i] + math.sin(angle) * tangent[i]
        for i in range(3)
    )
    return ReceiverAssumptions(
        physical_area_m2=area_m2,
        surface_normal_unit=normal,
        frame=optics.frame,
        photovoltaic_efficiency=efficiency,
    )


def _capture_status(mirror_available, optics, receiver_power, target_visibility):
    if target_visibility is not None and target_visibility.visible is False:
        return "BLOCKED"
    if target_visibility is None or target_visibility.visible is None or not mirror_available:
        return "UNKNOWN"
    if optics is None or optics.pointing_error_deg > 1e-12 or receiver_power is None:
        return "UNKNOWN"
    return "VALID" if receiver_power.capture_fraction > 0.0 else "ZERO_CAPTURE"


def _mission_segment_at(timeline, offset_s):
    for item in timeline:
        if item.start_offset_s <= offset_s < item.end_offset_s:
            return item
    return timeline[-1]


def _site_to_pa(position, profile, et, bridge):
    if position.frame == "MOON_PA_DE440":
        site_pa = position.position_km
    else:
        site_pa = bridge.vector_dem_to_geometry(position.position_km, et)
    if profile is None:
        return site_pa, None
    if profile.frame == "MOON_PA_DE440":
        return site_pa, profile
    transformed = SiteHorizonProfile(
        profile_deg=profile.profile_deg,
        site_position_km=bridge.vector_dem_to_geometry(profile.site_position_km, et),
        frame="MOON_PA_DE440",
        east_axis=bridge.vector_dem_to_geometry(profile.east_axis, et),
        north_axis=bridge.vector_dem_to_geometry(profile.north_axis, et),
        up_axis=bridge.vector_dem_to_geometry(profile.up_axis, et),
        range_km=profile.range_km,
        status=profile.status,
        site_name=profile.site_name,
    )
    return site_pa, transformed


def _direct_site_visibility(sun, site, profile, lunar_radius_km):
    if segment_occludes_lunar_sphere(site.position_km, sun.position_km, lunar_radius_km):
        return False, "lunar-occultation"
    if profile is None:
        return None, "unknown-terrain-no-profile"
    ray = tuple(sun.position_km[i] - site.position_km[i] for i in range(3))
    east = sum(ray[i] * profile.east_axis[i] for i in range(3))
    north = sum(ray[i] * profile.north_axis[i] for i in range(3))
    up = sum(ray[i] * profile.up_axis[i] for i in range(3))
    azimuth = math.degrees(math.atan2(east, north)) % 360.0
    elevation = math.degrees(math.atan2(up, math.hypot(east, north)))
    try:
        horizon = interpolate_horizon_elevation(azimuth, profile.profile_deg)
    except ValueError:
        return None, "unknown-terrain-profile-bin"
    clearance = elevation - horizon
    if profile.status == "insufficient-coverage":
        return None, "insufficient-terrain-coverage"
    if profile.status == "limited-range" and _norm(ray) > profile.range_km:
        return None, "unknown-beyond-limited-terrain-range"
    return clearance > 0.0, "complete-profile-clearance" if profile.status == "complete" else "limited-range-clearance"


def _norm(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _zero_optics_for_frame(frame):
    """Construct a zero-power record for an unavailable orbit link.

    Emergency recovery still rejects this interval from visibility/illumination
    flags. The record preserves the correct frame without inventing power.
    """
    from .lumen_optics import MirrorOpticsResult

    return MirrorOpticsResult(
        frame=frame,
        sun_to_mirror_unit=(0.0, 0.0, 0.0),
        mirror_to_target_unit=(0.0, 0.0, 0.0),
        ideal_mirror_normal_unit=(0.0, 0.0, 0.0),
        reflected_unit=(0.0, 0.0, 0.0),
        reflection_law_residual=0.0,
        sun_mirror_distance_km=0.0,
        mirror_target_distance_km=0.0,
        solar_angular_radius_rad=0.0,
        beam_footprint_diameter_km=0.0,
        pointing_error_deg=0.0,
        pointing_error_displacement_km=0.0,
        incidence_angle_deg=0.0,
        incident_optical_power_w=0.0,
        reflected_optical_power_w=0.0,
        geometry_origin="assumed",
        assumed_parameters=("no valid illumination geometry; zero power assigned",),
    )
