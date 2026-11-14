"""First-order interception and electrical output for a lunar PV receiver.

Geometry and the finite-Sun footprint are taken from ``MirrorOpticsResult``;
this module does not recalculate mirror reflection or solar angular size. The
uniform circular beam is an approximation, not validated optical propagation.
Receiver electrical energy is reported separately from optional battery-stored
energy so callers can credit each stage only once in a mission ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .energy_storage import BatteryConfig, BatteryState, ChargeResult, charge
from .lumen_optics import MirrorOpticsResult, PositionFrame


_FRAMES = {"J2000", "MOON_PA_DE440", "MOON_ME_DE421"}
SENSITIVITY_SLANT_DISTANCES_KM = (100.0, 250.0, 500.0)


def _finite_nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return number


def _fraction(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1].")
    return number


def _unit_vector(values: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} must have three components.")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} components must be finite.")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} must be a unit vector.")
    return tuple(value / norm for value in vector)


def _normalise_vector(values, label):
    length = math.sqrt(sum(float(v) * float(v) for v in values))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"{label} must be nonzero and numerically stable.")
    return tuple(float(v) / length for v in values)


@dataclass(frozen=True)
class ReceiverAssumptions:
    """Receiver properties; area is m², normal is unitless, efficiencies are fractions."""

    physical_area_m2: float
    surface_normal_unit: tuple[float, float, float]
    frame: PositionFrame
    photovoltaic_efficiency: float
    optical_transmission_efficiency: float = 1.0
    electrical_efficiency: float = 1.0
    center_offset_u_m: float = 0.0
    center_offset_v_m: float = 0.0
    integration_samples: int = 96

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_area_m2", _finite_nonnegative(self.physical_area_m2, "Receiver physical area"))
        object.__setattr__(self, "surface_normal_unit", _unit_vector(self.surface_normal_unit, "Receiver surface normal"))
        if self.frame not in _FRAMES:
            raise ValueError("Receiver frame must be J2000, MOON_PA_DE440, or MOON_ME_DE421.")
        object.__setattr__(self, "photovoltaic_efficiency", _fraction(self.photovoltaic_efficiency, "Photovoltaic efficiency"))
        object.__setattr__(self, "optical_transmission_efficiency", _fraction(self.optical_transmission_efficiency, "Optical transmission efficiency"))
        object.__setattr__(self, "electrical_efficiency", _fraction(self.electrical_efficiency, "Electrical efficiency"))
        for name in ("center_offset_u_m", "center_offset_v_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite metres.")
            object.__setattr__(self, name, value)
        if isinstance(self.integration_samples, bool) or not isinstance(self.integration_samples, int) or self.integration_samples < 4:
            raise ValueError("integration_samples must be an integer of at least 4.")


@dataclass(frozen=True)
class ReceiverPowerResult:
    """Power terms for one receiver case; powers W, areas m², distance km."""

    frame: str
    mirror_to_receiver_distance_km: float
    beam_footprint_diameter_m: float
    beam_area_m2: float
    receiver_physical_area_m2: float
    receiver_incidence_angle_deg: float
    projected_receiver_area_m2: float
    capture_fraction: float
    incident_mirror_power_w: float
    mirror_reflection_loss_w: float
    reflected_optical_power_w: float
    intercepted_optical_power_w: float
    beam_spill_power_w: float
    optical_transmission_loss_w: float
    received_optical_power_w: float
    photovoltaic_loss_w: float
    electrical_conversion_loss_w: float
    electrical_power_w: float
    assumptions: tuple[str, ...]
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.frame not in _FRAMES:
            raise ValueError("Receiver power result has an unsupported frame.")
        for name in (
            "mirror_to_receiver_distance_km", "beam_footprint_diameter_m", "beam_area_m2",
            "receiver_physical_area_m2", "projected_receiver_area_m2",
            "incident_mirror_power_w", "reflected_optical_power_w", "intercepted_optical_power_w",
            "mirror_reflection_loss_w", "beam_spill_power_w", "optical_transmission_loss_w", "received_optical_power_w",
            "photovoltaic_loss_w", "electrical_conversion_loss_w", "electrical_power_w",
        ):
            value = _finite_nonnegative(getattr(self, name), name.replace("_", " "))
            if name in {"mirror_to_receiver_distance_km", "beam_footprint_diameter_m", "beam_area_m2"} and value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        incidence = float(self.receiver_incidence_angle_deg)
        if not math.isfinite(incidence) or not 0.0 <= incidence <= 180.0:
            raise ValueError("Receiver incidence angle must be finite and in [0, 180] degrees.")
        capture = _fraction(self.capture_fraction, "Capture fraction")
        if self.reflected_optical_power_w > self.incident_mirror_power_w + max(1e-9, self.incident_mirror_power_w * 1e-12):
            raise ValueError("Reflected optical power cannot exceed incident mirror power.")
        if self.projected_receiver_area_m2 > self.receiver_physical_area_m2 + 1e-9:
            raise ValueError("Projected receiver area cannot exceed physical receiver area.")
        if self.intercepted_optical_power_w > self.reflected_optical_power_w + 1e-9:
            raise ValueError("Intercepted optical power cannot exceed reflected optical power.")
        if self.electrical_power_w > self.received_optical_power_w + 1e-9:
            raise ValueError("Electrical power cannot exceed received optical power.")
        if not math.isclose(self.beam_area_m2, math.pi * self.beam_footprint_diameter_m**2 / 4.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Beam area must match the circular footprint diameter.")
        if not 0.0 <= capture <= 1.0:
            raise ValueError("Capture fraction must be bounded by [0, 1].")
        for total, parts, label in (
            (self.incident_mirror_power_w, (self.mirror_reflection_loss_w, self.reflected_optical_power_w), "mirror input"),
            (self.reflected_optical_power_w, (self.intercepted_optical_power_w, self.beam_spill_power_w), "reflected beam"),
            (self.intercepted_optical_power_w, (self.optical_transmission_loss_w, self.received_optical_power_w), "receiver optics"),
            (self.received_optical_power_w, (self.photovoltaic_loss_w, self.electrical_conversion_loss_w, self.electrical_power_w), "receiver electrical output"),
        ):
            if not math.isclose(total, sum(parts), rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"{label} power terms do not conserve energy.")


@dataclass(frozen=True)
class ReceiverEnergyResult:
    """Interval energy balance; energies Wh. Battery fields are absent if not supplied."""

    duration_s: float
    incident_mirror_energy_wh: float
    mirror_reflection_loss_wh: float
    reflected_optical_energy_wh: float
    intercepted_optical_energy_wh: float
    beam_spill_energy_wh: float
    optical_transmission_loss_wh: float
    received_optical_energy_wh: float
    photovoltaic_loss_wh: float
    electrical_conversion_loss_wh: float
    electrical_generated_energy_wh: float
    battery_stored_energy_wh: float | None
    battery_charging_loss_wh: float
    curtailed_electrical_energy_wh: float
    battery_charge: ChargeResult | None
    energy_balance_residual_wh: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite_positive(self.duration_s, "Duration"))
        for name in (
            "incident_mirror_energy_wh", "mirror_reflection_loss_wh", "reflected_optical_energy_wh",
            "intercepted_optical_energy_wh", "beam_spill_energy_wh", "optical_transmission_loss_wh",
            "received_optical_energy_wh", "photovoltaic_loss_wh", "electrical_conversion_loss_wh",
            "electrical_generated_energy_wh", "battery_charging_loss_wh", "curtailed_electrical_energy_wh",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name.replace("_", " ")))
        if self.battery_stored_energy_wh is not None:
            object.__setattr__(self, "battery_stored_energy_wh", _finite_nonnegative(self.battery_stored_energy_wh, "Battery stored energy"))
        residual = float(self.energy_balance_residual_wh)
        if not math.isfinite(residual):
            raise ValueError("Energy balance residual must be finite Wh.")
        object.__setattr__(self, "energy_balance_residual_wh", residual)


def calculate_receiver_power(
    optics: MirrorOpticsResult,
    receiver: ReceiverAssumptions,
) -> ReceiverPowerResult:
    """Estimate optical interception and PV output from existing optics geometry.

    The receiver normal points out of its front surface. Incoming light at the
    receiver travels opposite ``optics.mirror_to_target_unit``. A back-facing
    receiver therefore has zero projected area and captures zero power.
    """
    if not isinstance(optics, MirrorOpticsResult):
        raise TypeError("optics must be MirrorOpticsResult from calculate_mirror_optics().")
    if not isinstance(receiver, ReceiverAssumptions):
        raise TypeError("receiver must be ReceiverAssumptions.")
    if receiver.frame != optics.frame:
        raise ValueError("Receiver normal and optics vectors must use the same coordinate frame.")
    distance_km = _finite_positive(optics.mirror_target_distance_km, "Mirror-to-receiver distance")
    diameter_m = _finite_positive(optics.beam_footprint_diameter_km * 1_000.0, "Beam footprint diameter")
    incident = _finite_nonnegative(optics.incident_optical_power_w, "Incident mirror power")
    reflected = _finite_nonnegative(optics.reflected_optical_power_w, "Reflected optical power")
    if reflected > incident + max(1e-9, incident * 1e-12):
        raise ValueError("Reflected optical power cannot exceed incident mirror power.")
    direction = _pointed_direction(optics)
    toward_mirror = tuple(-value for value in direction)
    dot = sum(receiver.surface_normal_unit[i] * toward_mirror[i] for i in range(3))
    incidence = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    projected_area = receiver.physical_area_m2 * max(0.0, dot)
    beam_area = math.pi * diameter_m * diameter_m / 4.0
    if not math.isfinite(beam_area) or beam_area <= 0.0:
        raise ValueError("Beam footprint area is not representable as a finite positive value.")
    capture = _overlap_capture(
        optics, receiver,
        _unit_vector(optics.mirror_to_target_unit, "Mirror-to-receiver target direction"),
    )
    intercepted = reflected * capture
    spilled = max(0.0, reflected - intercepted)
    received = intercepted * receiver.optical_transmission_efficiency
    optical_loss = intercepted - received
    pv_output = received * receiver.photovoltaic_efficiency
    pv_loss = received - pv_output
    electrical = pv_output * receiver.electrical_efficiency
    electrical_loss = pv_output - electrical
    values = (incidence, projected_area, capture, intercepted, spilled, optical_loss, received, pv_loss, electrical_loss, electrical)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Receiver calculation produced a non-finite or negative result.")
    return ReceiverPowerResult(
        frame=optics.frame,
        mirror_to_receiver_distance_km=distance_km,
        beam_footprint_diameter_m=diameter_m,
        beam_area_m2=beam_area,
        receiver_physical_area_m2=receiver.physical_area_m2,
        receiver_incidence_angle_deg=incidence,
        projected_receiver_area_m2=projected_area,
        capture_fraction=capture,
        incident_mirror_power_w=incident,
        mirror_reflection_loss_w=incident - reflected,
        reflected_optical_power_w=reflected,
        intercepted_optical_power_w=intercepted,
        beam_spill_power_w=spilled,
        optical_transmission_loss_w=optical_loss,
        received_optical_power_w=received,
        photovoltaic_loss_w=pv_loss,
        electrical_conversion_loss_w=electrical_loss,
        electrical_power_w=electrical,
        assumptions=(
            "deterministic midpoint quadrature over the rectangular mirror aperture with analytic uniform-disc/receiver overlap",
            "receiver is a circular planar aperture with configured in-plane center offsets",
            "beam footprint diameter is a conservative circular bounding diameter, not a uniform illuminated area",
            "uniform specular reflectivity; no diffraction, mirror slope error, structural occlusion, or thermal effects",
            f"ray quadrature resolution {receiver.integration_samples} by {receiver.integration_samples}",
            "receiver efficiency values are supplied assumptions",
        ),
        provenance=("beam diameter, mirror-target distance, reflected direction and mirror powers from lumen_optics.MirrorOpticsResult",),
    )


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _add_scaled(a, b, scale):
    return tuple(a[i] + b[i] * scale for i in range(3))


def _pointed_direction(optics):
    nominal = _unit_vector(optics.mirror_to_target_unit, "Mirror-to-receiver propagation direction")
    error = math.radians(optics.pointing_error_deg)
    if error == 0:
        return nominal
    normal = _unit_vector(optics.ideal_mirror_normal_unit, "Mirror normal")
    axis = _cross(normal, nominal)
    length = math.sqrt(sum(v*v for v in axis))
    if length <= 1e-12:
        return nominal
    axis = tuple(v/length for v in axis)
    c, s = math.cos(error), math.sin(error)
    cross_an = _cross(axis, normal)
    normal = tuple(normal[i]*c + cross_an[i]*s + axis[i]*sum(axis[j]*normal[j] for j in range(3))*(1-c) for i in range(3))
    incoming = tuple(-v for v in _unit_vector(optics.sun_to_mirror_unit, "Mirror-to-Sun direction"))
    nd = sum(incoming[i]*normal[i] for i in range(3))
    return _normalise_vector(tuple(incoming[i]-2*nd*normal[i] for i in range(3)), "Perturbed reflected direction")


def _overlap_capture(optics, receiver, central_direction):
    """Midpoint-integrate analytic finite-Sun circle overlap over aperture.

    The source disk is uniform and paraxial; its footprint is circular. The
    receiver is represented by a projected-area-equivalent circle. The
    rectangular mirror aperture is sampled n by n; each sample's overlap is
    evaluated analytically. Diffraction, limb darkening, mirror slope errors,
    and the exact tilted receiver ellipse are not modeled.
    """
    n = receiver.integration_samples
    nominal_direction = _unit_vector(central_direction, "Mirror-to-receiver direction")
    direction = _pointed_direction(optics)
    normal = _unit_vector(optics.ideal_mirror_normal_unit, "Mirror normal")
    error = math.radians(optics.pointing_error_deg)
    aperture_u = getattr(optics, "mirror_aperture_u_unit", (0.0, 0.0, 0.0))
    if error:
        axis = _cross(normal, nominal_direction)
        axis_norm = math.sqrt(sum(v*v for v in axis))
        if axis_norm > 1e-12:
            axis = tuple(v/axis_norm for v in axis)
            c, s = math.cos(error), math.sin(error)
            cross_n = _cross(axis, normal)
            normal = tuple(normal[i]*c + cross_n[i]*s + axis[i]*sum(axis[j]*normal[j] for j in range(3))*(1-c) for i in range(3))
            cross_u = _cross(axis, aperture_u)
            aperture_u = tuple(aperture_u[i]*c + cross_u[i]*s + axis[i]*sum(axis[j]*aperture_u[j] for j in range(3))*(1-c) for i in range(3))
    rn = receiver.surface_normal_unit
    cos_incidence = dot_front(rn, direction)
    if receiver.physical_area_m2 == 0 or cos_incidence <= 0:
        return 0.0
    if math.sqrt(sum(v*v for v in aperture_u)) <= 1e-12:
        aperture_ref = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
        aperture_u = _normalise_vector(_cross(normal, aperture_ref), "Mirror aperture axis")
    mu = _normalise_vector(aperture_u, "Mirror aperture axis")
    mv = _normalise_vector(_cross(normal, mu), "Mirror aperture axis v")
    ru0 = _cross(rn, nominal_direction)
    if math.sqrt(sum(v*v for v in ru0)) <= 1e-12:
        ru0 = _cross(rn, (0, 0, 1) if abs(rn[2]) < 0.9 else (0, 1, 0))
    ru = _normalise_vector(ru0, "Receiver aperture axis")
    rv = _cross(rn, ru)
    receiver_center = tuple(nominal_direction[i]*optics.mirror_target_distance_km*1000 + ru[i]*receiver.center_offset_u_m + rv[i]*receiver.center_offset_v_m for i in range(3))
    receiver_radius = math.sqrt(receiver.physical_area_m2*cos_incidence/math.pi)
    half_w, half_h = optics.mirror_width_m/2, optics.mirror_height_m/2
    aperture_weight = 1.0/(n*n)
    total = 0.0
    for iy in range(n):
        y = -half_h + (iy+0.5)*optics.mirror_height_m/n
        for ix in range(n):
            x = -half_w + (ix+0.5)*optics.mirror_width_m/n
            origin = tuple(mu[i]*x + mv[i]*y for i in range(3))
            denom = sum(rn[i]*direction[i] for i in range(3))
            if abs(denom) <= 1e-12:
                continue
            distance = sum(rn[i]*(receiver_center[i]-origin[i]) for i in range(3))/denom
            if distance <= 0:
                continue
            intercept = _add_scaled(origin, direction, distance)
            delta = tuple(intercept[i]-receiver_center[i] for i in range(3))
            axial = sum(delta[i]*direction[i] for i in range(3))
            transverse = tuple(delta[i]-axial*direction[i] for i in range(3))
            separation = math.sqrt(sum(v*v for v in transverse))
            beam_radius = distance*math.tan(optics.solar_angular_radius_rad)
            if beam_radius > 0:
                total += aperture_weight*_circle_overlap_area(beam_radius, receiver_radius, separation)/(math.pi*beam_radius**2)
    total = min(1.0, max(0.0, total))
    return 1.0 if math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12) else total


def _circle_overlap_area(r1, r2, separation):
    if r1 <= 0 or r2 <= 0 or separation >= r1+r2:
        return 0.0
    if separation <= abs(r1-r2):
        return math.pi*min(r1, r2)**2
    a1 = math.acos(max(-1.0, min(1.0, (separation**2+r1**2-r2**2)/(2*separation*r1))))
    a2 = math.acos(max(-1.0, min(1.0, (separation**2+r2**2-r1**2)/(2*separation*r2))))
    radicand = max(0.0, (-separation+r1+r2)*(separation+r1-r2)*(separation-r1+r2)*(separation+r1+r2))
    return r1*r1*a1 + r2*r2*a2 - 0.5*math.sqrt(radicand)


def dot_front(normal, direction):
    return -sum(normal[i] * direction[i] for i in range(3))


def study_receiver_distance_sensitivity(
    optics: MirrorOpticsResult,
    receiver: ReceiverAssumptions,
    slant_distances_km: tuple[float, ...] = SENSITIVITY_SLANT_DISTANCES_KM,
) -> tuple[ReceiverPowerResult, ...]:
    """Recompute finite-Sun footprint and overlap over explicit slant ranges.

    This is a range sensitivity calculation only. It does not create mirror
    positions, orbital altitudes, trajectories, or new reflection solutions.
    Mirror power, receiver properties, and the source optics angular radius
    remain fixed; the solar-disc component scales with distance and aperture
    contribution remains fixed.
    """
    if not isinstance(optics, MirrorOpticsResult):
        raise TypeError("optics must be MirrorOpticsResult.")
    ranges = tuple(_finite_positive(value, "Mirror-to-receiver slant distance") for value in slant_distances_km)
    if not ranges:
        raise ValueError("At least one mirror-to-receiver slant distance is required.")
    reference_distance = _finite_positive(optics.mirror_target_distance_km, "Reference optics distance")
    _finite_positive(optics.beam_footprint_diameter_km * 1_000.0, "Reference beam diameter")
    results = []
    for distance in ranges:
        aperture_diagonal_km = math.hypot(optics.mirror_width_m, optics.mirror_height_m) / 1000.0
        diameter_km = 2.0 * distance * math.tan(optics.solar_angular_radius_rad) + aperture_diagonal_km
        diameter_m = _finite_positive(diameter_km * 1_000.0, "Scaled beam diameter")
        beam_area_m2 = math.pi * diameter_m * diameter_m / 4.0
        if not math.isfinite(beam_area_m2) or beam_area_m2 <= 0.0:
            raise ValueError("Scaled beam area is not representable as a finite positive value.")
        # Reuse the authoritative optics record while substituting only the
        # requested slant range and its proportionally scaled footprint.
        power = _calculate_with_range(optics, receiver, distance, diameter_km)
        results.append(replace(
            power,
            assumptions=power.assumptions + (
                "distance case is a slant-range sensitivity; footprint scaled from the supplied optics result; no trajectory or spacecraft position is implied",
            ),
        ))
    return tuple(results)


def _calculate_with_range(optics, receiver, distance_km, diameter_km):
    return calculate_receiver_power(
        replace(optics, mirror_target_distance_km=distance_km, beam_footprint_diameter_km=diameter_km),
        receiver,
    )


def integrate_receiver_energy(
    power: ReceiverPowerResult,
    duration_s: float,
    *,
    battery_config: BatteryConfig | None = None,
    battery_state: BatteryState | None = None,
    battery_charge_efficiency: float | None = None,
) -> ReceiverEnergyResult:
    """Integrate each receiver power stage once; optionally store PV output.

    Without a battery, electrical generated energy is the ledger output and
    ``battery_stored_energy_wh`` remains ``None``. With a battery, generated
    energy is passed once to the shared storage ``charge`` API; stored energy,
    charging loss, and curtailment are reported separately.
    """
    if not isinstance(power, ReceiverPowerResult):
        raise TypeError("power must be ReceiverPowerResult.")
    duration = _finite_nonnegative(duration_s, "Integration duration")
    if duration <= 0.0:
        raise ValueError("Integration duration must be positive seconds.")
    battery_values = (battery_config is not None, battery_state is not None, battery_charge_efficiency is not None)
    if any(battery_values) and not all(battery_values):
        raise ValueError("Battery configuration, state, and charge efficiency must be supplied together.")
    if battery_config is not None and not isinstance(battery_config, BatteryConfig):
        raise TypeError("battery_config must be BatteryConfig.")
    if battery_state is not None and not isinstance(battery_state, BatteryState):
        raise TypeError("battery_state must be BatteryState.")
    efficiency = _fraction(battery_charge_efficiency, "Battery charge efficiency") if battery_charge_efficiency is not None else None
    hours = duration / 3_600.0
    incident = power.incident_mirror_power_w * hours
    reflected = power.reflected_optical_power_w * hours
    intercepted = power.intercepted_optical_power_w * hours
    spilled = power.beam_spill_power_w * hours
    optical_loss = power.optical_transmission_loss_w * hours
    received = power.received_optical_power_w * hours
    pv_loss = power.photovoltaic_loss_w * hours
    electrical_loss = power.electrical_conversion_loss_w * hours
    electrical = power.electrical_power_w * hours
    reflection_loss = power.mirror_reflection_loss_w * hours
    battery_charge = None
    stored = None
    battery_loss = curtailed = 0.0
    if battery_config is not None and battery_state is not None and efficiency is not None:
        battery_charge = charge(
            battery_state,
            battery_config,
            electrical,
            efficiency,
            duration_s=duration,
        )
        stored = battery_charge.stored_energy_wh
        battery_loss = battery_charge.loss_energy_wh
        curtailed = battery_charge.curtailed_input_energy_wh
    final_energy = electrical if stored is None else stored
    residual = incident - (
        reflection_loss + spilled + optical_loss + pv_loss + electrical_loss
        + battery_loss + curtailed + final_energy
    )
    return ReceiverEnergyResult(
        duration_s=duration,
        incident_mirror_energy_wh=incident,
        mirror_reflection_loss_wh=reflection_loss,
        reflected_optical_energy_wh=reflected,
        intercepted_optical_energy_wh=intercepted,
        beam_spill_energy_wh=spilled,
        optical_transmission_loss_wh=optical_loss,
        received_optical_energy_wh=received,
        photovoltaic_loss_wh=pv_loss,
        electrical_conversion_loss_wh=electrical_loss,
        electrical_generated_energy_wh=electrical,
        battery_stored_energy_wh=stored,
        battery_charging_loss_wh=battery_loss,
        curtailed_electrical_energy_wh=curtailed,
        battery_charge=battery_charge,
        energy_balance_residual_wh=residual,
    )


def _finite_positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive.")
    return number
