"""Compare saved Site06 horizon profiles at nested limited terrain radii."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.terrain_horizon import TerrainHorizonProfile
from core.terrain_provider import MOON_ME_DE421
from scripts.validate_site06_5m_horizon_crosscheck import compare_profiles


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROFILE_1KM = REPORTS / "site06_1km_5m_20m_horizon_crosscheck.json"


def _as_profile(values: list[float | None]) -> TerrainHorizonProfile:
    if len(values) != 360:
        raise ValueError("Site06 profiles must have exactly 360 one-degree bins.")
    angles = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    return TerrainHorizonProfile(
        azimuth_centers_deg=np.arange(360, dtype=np.float64),
        max_obstruction_elevation_deg=angles,
        sample_count=np.isfinite(angles).astype(np.int64),
        source_frame=MOON_ME_DE421,
        north_axis=(0.0, 0.0, 1.0),
        east_axis=(0.0, 1.0, 0.0),
        up_axis=(1.0, 0.0, 0.0),
    )


def _union_near_5m(regional_20m: list[float | None], near_5m: list[float | None]) -> list[float | None]:
    regional = np.asarray([np.nan if value is None else float(value) for value in regional_20m])
    fine = np.asarray([np.nan if value is None else float(value) for value in near_5m])
    combined = np.fmax(regional, fine)
    return [float(value) if np.isfinite(value) else None for value in combined]


def load_site06_range_profiles() -> tuple[dict[str, object], ...]:
    """Load the common 5 m inner + 20 m regional profiles; no network access."""
    one = json.loads(PROFILE_1KM.read_text(encoding="utf-8-sig"))
    if float(one.get("requested_radius_m", -1.0)) != 1000.0:
        raise ValueError("The existing one-kilometer profile does not match its expected radius.")
    if one.get("source_frame") != MOON_ME_DE421:
        raise ValueError("The one-kilometer profile is not in MOON_ME_DE421.")
    near_5m = one.get("profile_bins_deg")
    one_20m = one.get("20m_profile_bins_deg")
    if not isinstance(near_5m, list) or not isinstance(one_20m, list):
        raise ValueError("The one-kilometer report lacks 5 m or 20 m profile bins.")

    result: list[dict[str, object]] = []
    for radius_km in (1, 2, 5, 10):
        if radius_km == 1:
            report = one
            regional = one_20m
            raw_valid_samples = one["regional_20m_window"]["valid_samples"]
        else:
            path = REPORTS / f"site06_{radius_km}km_limited_horizon.json"
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            if abs(float(report.get("requested_radius_m", -1.0)) - radius_km * 1000.0) > 1e-8:
                raise ValueError(f"Saved {radius_km} km report has an unexpected requested radius.")
            if report.get("source_frame") != MOON_ME_DE421:
                raise ValueError(f"Saved {radius_km} km profile is not in MOON_ME_DE421.")
            if report.get("horizon_status", "limited-range") != "limited-range":
                raise ValueError(f"Saved {radius_km} km region is not a fully covered limited-range profile.")
            regional = report.get("profile_bins_1deg_maximum_obstruction_deg")
            raw_valid_samples = report.get("horizon_sample_count")
        if not isinstance(regional, list) or len(regional) != 360:
            raise ValueError(f"The {radius_km} km report lacks 360 regional profile bins.")
        profile_values = _union_near_5m(regional, near_5m)
        profile = _as_profile(profile_values)
        angles = profile.max_obstruction_elevation_deg
        valid = np.flatnonzero(np.isfinite(angles))
        maximum_index = int(valid[np.argmax(angles[valid])])
        entry: dict[str, object] = {
            "radius_km": radius_km,
            "profile": profile,
            "profile_values": profile_values,
            "max_obstruction_deg": float(angles[maximum_index]),
            "max_azimuth_deg": float(profile.azimuth_centers_deg[maximum_index]),
            "min_obstruction_deg": float(np.min(angles[valid])),
            "valid_bins": int(valid.size),
            "bins_without_coverage": int(360 - valid.size),
            "valid_20m_horizon_samples_observer_excluded": int(raw_valid_samples),
            "status": "limited-range",
        }
        result.append(entry)

    for previous, current in zip(result, result[1:]):
        comparison = compare_profiles(current["profile"], previous["profile"])
        current["comparison_to_previous_radius"] = comparison
        # With the same native 20 m data source, observer, frame, and bins, a
        # nested radius's max-over-samples profile cannot decrease in a common bin.
        if float(comparison["minimum_signed_profile_change_deg"]) < -1e-8:
            current["nested_profile_monotonic"] = False
        else:
            current["nested_profile_monotonic"] = True
    return tuple(result)


def markdown_table(entries: tuple[dict[str, object], ...]) -> str:
    rows = ["| Radius | Max horizon | Max azimuth | Min horizon | Valid bins | Max Δ | RMS Δ | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for entry in entries:
        comparison = entry.get("comparison_to_previous_radius")
        if comparison is None:
            max_delta = rms_delta = "—"
        else:
            max_delta = f"{comparison['maximum_absolute_profile_difference_deg']:.6f}°"
            rms_delta = f"{comparison['rms_profile_difference_deg']:.6f}°"
        rows.append(
            f"| {entry['radius_km']} km | {entry['max_obstruction_deg']:.6f}° | "
            f"{entry['max_azimuth_deg']:.0f}° | {entry['min_obstruction_deg']:.6f}° | "
            f"{entry['valid_bins']}/360 | {max_delta} | {rms_delta} | {entry['status']} |"
        )
    return "\n".join(rows)


def main() -> None:
    entries = load_site06_range_profiles()
    print(markdown_table(entries))
    print("\nAnnulus valid 20 m horizon samples (observer pixel excluded):")
    previous_count = 0
    for entry in entries:
        count = int(entry["valid_20m_horizon_samples_observer_excluded"])
        print(f"  through {entry['radius_km']} km: {count - previous_count}")
        previous_count = count
    print("\nAll profiles are limited-range; no visibility or convergence claim is made.")


if __name__ == "__main__":
    main()
