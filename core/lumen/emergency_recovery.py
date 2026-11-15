"""Estimate whether reflected sunlight can restore rover safe-return energy.

This is a first-order optical/energy estimate, not a rescue guarantee. It
requires explicit illumination windows, visibility results, receiver
assumptions, and a survival deadline. A flat mirror redirects a diverging
solar beam; receiver power is reduced by finite-Sun footprint, pointing offset,
panel orientation, panel conversion efficiency and battery charging losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

from .energy_storage import BatteryState, operational_floor_wh, charge
from .lumen_optics import MirrorOpticsResult
from .lumen_visibility import LumenVisibilityResult
from .rover_energy import RoverConfig
from .lumen_receiver import ReceiverPowerResult
from .lumen_receiver import ReceiverAssumptions, calculate_receiver_power


_UTC = timezone.utc


def _nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return number


def _efficiency(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1].")
    return number


def _angle(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 90.0:
        raise ValueError(f"{label} must be finite and in [0, 90] degrees.")
    return number


def _utc(value: str, label: str) -> datetime:
    try:
        text = value.strip()
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp with UTC offset or Z.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset or Z suffix.")
    return parsed.astimezone(_UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(_UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RecoveryIlluminationInterval:
    """One interval with explicit geometry and receiver assumptions.

    Times are UTC ISO strings. Receiver area is m²; panel incidence is
    degrees; panel conversion efficiency is a fraction. ``illumination_available``
    must be supplied from an explicit temporal illumination assessment; it is
    never inferred from a clear line of sight alone.
    """

    start_utc: str
    end_utc: str
    illumination_available: bool
    visibility: LumenVisibilityResult
    optics: MirrorOpticsResult
    receiver_area_m2: float
    receiver_incidence_angle_deg: float
    receiver_conversion_efficiency: float
    receiver_power: ReceiverPowerResult | None = None

    def __post_init__(self) -> None:
        start = _utc(self.start_utc, "start_utc")
        end = _utc(self.end_utc, "end_utc")
        if end <= start:
            raise ValueError("Recovery interval end_utc must be after start_utc.")
        if not isinstance(self.illumination_available, bool):
            raise ValueError("illumination_available must be bool.")
        if not isinstance(self.visibility, LumenVisibilityResult):
            raise TypeError("visibility must be a LumenVisibilityResult.")
        if not isinstance(self.optics, MirrorOpticsResult):
            raise TypeError("optics must be a MirrorOpticsResult.")
        if self.visibility.frame != self.optics.frame:
            raise ValueError("Visibility and optics geometry must use the same coordinate frame.")
        object.__setattr__(self, "receiver_area_m2", _nonnegative(self.receiver_area_m2, "Receiver area"))
        object.__setattr__(
            self, "receiver_incidence_angle_deg", _angle(self.receiver_incidence_angle_deg, "Receiver incidence angle")
        )
        object.__setattr__(
            self,
            "receiver_conversion_efficiency",
            _efficiency(self.receiver_conversion_efficiency, "Receiver conversion efficiency"),
        )
        if self.receiver_power is not None:
            if not isinstance(self.receiver_power, ReceiverPowerResult):
                raise TypeError("receiver_power must be ReceiverPowerResult when supplied.")
            if self.receiver_power.frame != self.optics.frame:
                raise ValueError("Receiver power and optics geometry must use the same coordinate frame.")
            if not math.isclose(self.receiver_power.receiver_physical_area_m2, self.receiver_area_m2, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Receiver power result area does not match receiver_area_m2.")
            if not math.isclose(self.receiver_power.reflected_optical_power_w, self.optics.reflected_optical_power_w, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Receiver power result does not match the interval optics result.")


@dataclass(frozen=True)
class RecoveryRequest:
    """Rover energy state and survival window; battery/safe-return energies in Wh."""

    current_battery_energy_wh: float
    safe_return_energy_wh: float
    survival_start_utc: str
    survival_deadline_utc: str
    rover_config: RoverConfig
    intervals: tuple[RecoveryIlluminationInterval, ...]

    def __post_init__(self) -> None:
        energy = _nonnegative(self.current_battery_energy_wh, "Current battery energy")
        safe = _nonnegative(self.safe_return_energy_wh, "Safe-return energy")
        start = _utc(self.survival_start_utc, "survival_start_utc")
        deadline = _utc(self.survival_deadline_utc, "survival_deadline_utc")
        if deadline <= start:
            raise ValueError("Survival deadline must be after survival start.")
        if not isinstance(self.rover_config, RoverConfig):
            raise TypeError("rover_config must be RoverConfig.")
        if energy > self.rover_config.battery_config.nominal_capacity_wh:
            raise ValueError("Current rover battery energy exceeds nominal capacity.")
        floor = operational_floor_wh(self.rover_config.battery_config)
        if energy < floor:
            raise ValueError("Current rover energy is below the configured battery DoD floor.")
        intervals = tuple(self.intervals)
        if any(not isinstance(interval, RecoveryIlluminationInterval) for interval in intervals):
            raise TypeError("intervals must contain RecoveryIlluminationInterval values.")
        ordered = sorted(intervals, key=lambda item: _utc(item.start_utc, "interval start"))
        for previous, following in zip(ordered, ordered[1:]):
            if _utc(following.start_utc, "interval start") < _utc(previous.end_utc, "interval end"):
                raise ValueError("Recovery illumination intervals must not overlap.")
        object.__setattr__(self, "current_battery_energy_wh", energy)
        object.__setattr__(self, "safe_return_energy_wh", safe)
        object.__setattr__(self, "intervals", ordered)


@dataclass(frozen=True)
class FeasibleChargingInterval:
    """Clipped interval that contributed battery charging."""

    start_utc: str
    end_utc: str
    duration_s: float
    receiver_optical_power_w: float
    receiver_electrical_power_w: float
    receiver_optical_energy_wh: float
    receiver_electrical_energy_wh: float
    battery_stored_energy_wh: float
    battery_loss_energy_wh: float


@dataclass(frozen=True)
class EmergencyRecoveryResult:
    """Recovery energy, time, loss and feasibility results (energy in Wh)."""

    feasible: bool
    current_battery_energy_wh: float
    current_usable_energy_wh: float
    safe_return_energy_wh: float
    energy_deficit_wh: float
    delivered_battery_energy_wh: float
    final_battery_energy_wh: float
    final_usable_energy_wh: float
    final_safe_return_margin_wh: float
    receiver_optical_energy_wh: float
    receiver_electrical_energy_wh: float
    peak_receiver_optical_power_w: float
    peak_receiver_electrical_power_w: float
    optical_spill_energy_wh: float
    receiver_conversion_loss_wh: float
    battery_charging_loss_wh: float
    charge_power_or_capacity_curtailed_wh: float
    earliest_feasible_start_utc: str | None
    latest_feasible_end_utc: str | None
    feasible_intervals: tuple[FeasibleChargingInterval, ...]
    failure_reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    provenance: tuple[str, ...]
    receiver_intercepted_optical_energy_wh: float = 0.0
    receiver_optical_transmission_loss_wh: float = 0.0


def assess_emergency_recovery(request: RecoveryRequest) -> EmergencyRecoveryResult:
    """Assess reflected-light charging within the explicit survival window.

    Every eligible interval uses ``lumen_receiver.calculate_receiver_power``:
    a caller-supplied result is reused after consistency checks, otherwise a
    receiver assumption is constructed from this interval's area, incidence,
    and conversion efficiency. The shared battery ``charge`` function then
    applies rover charging efficiency and power/capacity limits. Recovery
    succeeds only if resulting usable battery energy meets the safe-return
    requirement; geometric visibility by itself never constitutes success.
    """
    if not isinstance(request, RecoveryRequest):
        raise TypeError("request must be RecoveryRequest.")
    config = request.rover_config
    battery_config = config.battery_config
    floor = operational_floor_wh(battery_config)
    current = request.current_battery_energy_wh
    current_usable = max(0.0, current - floor)
    deficit = max(0.0, request.safe_return_energy_wh - current_usable)
    battery_state = BatteryState(current)
    start_window = _utc(request.survival_start_utc, "survival_start_utc")
    end_window = _utc(request.survival_deadline_utc, "survival_deadline_utc")

    optical_intercepted_wh = optical_received_wh = electrical_generated_wh = stored_wh = 0.0
    optical_transmission_loss_wh = 0.0
    peak_optical_power_w = peak_electrical_power_w = 0.0
    optical_spill_wh = panel_loss_wh = battery_loss_wh = curtailed_wh = 0.0
    contributing: list[FeasibleChargingInterval] = []
    failure_reasons: list[str] = []
    eligible_duration_s = 0.0
    blocked_seen = False
    unavailable_seen = False
    unknown_seen = False

    for interval in request.intervals:
        interval_start = max(start_window, _utc(interval.start_utc, "interval start"))
        interval_end = min(end_window, _utc(interval.end_utc, "interval end"))
        if interval_end <= interval_start:
            continue
        duration_s = (interval_end - interval_start).total_seconds()
        visibility_clear = (
            interval.visibility.sun_to_mirror.visible is True
            and interval.visibility.mirror_to_rover.visible is True
        )
        if not visibility_clear:
            known_block = (
                not interval.visibility.sun_to_mirror.geometrically_clear
                or not interval.visibility.mirror_to_rover.geometrically_clear
                or interval.visibility.mirror_to_rover.terrain_visible is False
            )
            if known_block:
                blocked_seen = True
            else:
                unknown_seen = True
            continue
        if not interval.illumination_available:
            unavailable_seen = True
            continue

        eligible_duration_s += duration_s
        receiver_result = interval.receiver_power or _calculate_receiver_power(interval)
        received_power_w = receiver_result.received_optical_power_w
        intercepted_power_w = receiver_result.intercepted_optical_power_w
        spill_power_w = receiver_result.beam_spill_power_w
        transmission_loss_power_w = receiver_result.optical_transmission_loss_w
        electrical_power_w = receiver_result.electrical_power_w
        receiver_conversion_loss_power_w = (
            receiver_result.photovoltaic_loss_w + receiver_result.electrical_conversion_loss_w
        )
        peak_optical_power_w = max(peak_optical_power_w, received_power_w)
        peak_electrical_power_w = max(peak_electrical_power_w, electrical_power_w)
        receiver_optical_wh = received_power_w * duration_s / 3600.0
        receiver_electrical_wh = electrical_power_w * duration_s / 3600.0
        optical_spill_wh += spill_power_w * duration_s / 3600.0
        optical_intercepted_wh += intercepted_power_w * duration_s / 3600.0
        optical_transmission_loss_wh += transmission_loss_power_w * duration_s / 3600.0
        panel_loss_wh += receiver_conversion_loss_power_w * duration_s / 3600.0
        optical_received_wh += receiver_optical_wh
        electrical_generated_wh += receiver_electrical_wh

        charged = charge(
            battery_state,
            battery_config,
            receiver_electrical_wh,
            config.charging_efficiency,
            duration_s=duration_s,
        )
        battery_state = charged.resulting_state
        stored_wh += charged.stored_energy_wh
        battery_loss_wh += charged.loss_energy_wh
        curtailed_wh += charged.curtailed_input_energy_wh
        if charged.stored_energy_wh > 0.0:
            contributing.append(
                FeasibleChargingInterval(
                    start_utc=_utc_text(interval_start),
                    end_utc=_utc_text(interval_end),
                    duration_s=duration_s,
                    receiver_optical_power_w=received_power_w,
                    receiver_electrical_power_w=electrical_power_w,
                    receiver_optical_energy_wh=receiver_optical_wh,
                    receiver_electrical_energy_wh=receiver_electrical_wh,
                    battery_stored_energy_wh=charged.stored_energy_wh,
                    battery_loss_energy_wh=charged.loss_energy_wh,
                )
            )

    final_battery = battery_state.energy_wh
    final_usable = max(0.0, final_battery - floor)
    final_margin = final_usable - request.safe_return_energy_wh
    feasible = final_margin >= -1e-9

    if blocked_seen:
        failure_reasons.append("One or more candidate intervals have blocked Sun-to-mirror or mirror-to-rover visibility.")
    if unknown_seen:
        failure_reasons.append("One or more candidate intervals have unknown terrain or visibility status.")
    if unavailable_seen:
        failure_reasons.append("Reflected illumination is unavailable in one or more otherwise visible intervals.")
    if eligible_duration_s <= 0.0 and deficit > 0:
        failure_reasons.append("No eligible reflected-illumination interval overlaps the survival window.")
    if eligible_duration_s > 0.0 and deficit > 0:
        total_available_s = (end_window - start_window).total_seconds()
        if eligible_duration_s < total_available_s:
            failure_reasons.append("Eligible illumination time is shorter than the full survival window.")
    if final_margin < -1e-9:
        failure_reasons.append("Delivered battery energy is insufficient to restore safe-return feasibility before the deadline.")
    if not failure_reasons and deficit <= 0.0:
        failure_reasons.append("Rover already meets the safe-return energy requirement; orbital recovery is not required.")
    if feasible:
        failure_reasons = []

    earliest = contributing[0].start_utc if contributing else None
    latest = contributing[-1].end_utc if contributing else None
    return EmergencyRecoveryResult(
        feasible=feasible,
        current_battery_energy_wh=current,
        current_usable_energy_wh=current_usable,
        safe_return_energy_wh=request.safe_return_energy_wh,
        energy_deficit_wh=deficit,
        delivered_battery_energy_wh=stored_wh,
        final_battery_energy_wh=final_battery,
        final_usable_energy_wh=final_usable,
        final_safe_return_margin_wh=final_margin,
        receiver_optical_energy_wh=optical_received_wh,
        receiver_electrical_energy_wh=electrical_generated_wh,
        peak_receiver_optical_power_w=peak_optical_power_w,
        peak_receiver_electrical_power_w=peak_electrical_power_w,
        optical_spill_energy_wh=optical_spill_wh,
        receiver_conversion_loss_wh=panel_loss_wh,
        battery_charging_loss_wh=battery_loss_wh,
        charge_power_or_capacity_curtailed_wh=curtailed_wh,
        earliest_feasible_start_utc=earliest,
        latest_feasible_end_utc=latest,
        feasible_intervals=tuple(contributing),
        failure_reasons=tuple(failure_reasons),
        assumptions=(
            "flat mirror redirects but does not focus sunlight",
            "receiver capture and conversion use core.lumen.lumen_receiver",
            "receiver is modeled as a projected-area-equivalent circular aperture by the shared receiver calculation",
            "receiver incidence and PV efficiency are supplied interval assumptions",
            "illumination availability is explicitly supplied per interval",
        ),
        provenance=(
            "mirror reflected optical power and finite-Sun footprint from lumen_optics.MirrorOpticsResult",
            "ray visibility from lumen_visibility.LumenVisibilityResult",
            "battery charge efficiency, capacity and power limits from RoverConfig/BatteryConfig",
            "time-window clipping and energy integration calculated in emergency_recovery",
        ),
        receiver_intercepted_optical_energy_wh=optical_intercepted_wh,
        receiver_optical_transmission_loss_wh=optical_transmission_loss_wh,
    )


def _calculate_receiver_power(interval: RecoveryIlluminationInterval) -> ReceiverPowerResult:
    """Build a shared receiver input from legacy interval-level assumptions."""
    toward_mirror = tuple(-value for value in interval.optics.mirror_to_target_unit)
    axis = (1.0, 0.0, 0.0) if abs(toward_mirror[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(axis[i] * toward_mirror[i] for i in range(3))
    tangent = tuple(axis[i] - projection * toward_mirror[i] for i in range(3))
    tangent_norm = math.sqrt(sum(value * value for value in tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("Cannot construct receiver orientation from the supplied beam direction.")
    tangent = tuple(value / tangent_norm for value in tangent)
    angle = math.radians(interval.receiver_incidence_angle_deg)
    normal = tuple(
        math.cos(angle) * toward_mirror[i] + math.sin(angle) * tangent[i]
        for i in range(3)
    )
    assumptions = ReceiverAssumptions(
        physical_area_m2=interval.receiver_area_m2,
        surface_normal_unit=normal,
        frame=interval.optics.frame,
        photovoltaic_efficiency=interval.receiver_conversion_efficiency,
    )
    return calculate_receiver_power(interval.optics, assumptions)
