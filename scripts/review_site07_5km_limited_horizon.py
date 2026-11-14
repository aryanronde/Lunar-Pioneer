"""Review-only Site07 native 5 m, 5 km limited-range horizon profile.

The default invocation is offline. A new bounded NASA raster read can only be
started with --read-dem; this task does not invoke that option.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import spiceypy as spice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.horizon_report import (  # noqa: E402
    collect_runtime_provenance,
    make_horizon_report,
    make_raster_source_record,
    serialize_horizon_report,
)
from core.lunar_coordinates import planetocentric_latlon_to_moon_me_de421  # noqa: E402
from core.lunar_frame_bridge import LunarFrameBridge  # noqa: E402
from core.lunar_horizon import LocalHorizonFrame, LunarSite  # noqa: E402
from core.spice_engine import SpiceGeometryEngine  # noqa: E402
from core.terrain_horizon import TerrainHorizonProfile, calculate_terrain_horizon_profile  # noqa: E402
from core.terrain_provider import (  # noqa: E402
    MOON_ME_DE421,
    MOON_REFERENCE_RADIUS_M,
    TerrainMetadata,
    TerrainWindow,
)
from core.terrain_region import regional_horizon_status  # noqa: E402
from core.terrain_surface import calculate_terrain_normals, reconstruct_lunar_terrain  # noqa: E402


SITE_RECORD_PATH = ROOT / "reports" / "review" / "site07_source_record_review_v1.json"
RASTER_REVIEW_PATH = ROOT / "reports" / "review" / "site07_raster_metadata_review_v1.json"
OUTPUT_PATH = ROOT / "reports" / "review_site07_5km_limited_horizon_v1.json"
RADIUS_M = 5000.0
REFERENCE_RADIUS_KM = MOON_REFERENCE_RADIUS_M / 1000.0
PROFILE_SOURCE_ID = "site07_lola_5m"
ELEVATION_ASSUMPTION = (
    "Existing reconstruct_lunar_terrain interpretation: source elevation values are meters and are added to the "
    "1,737,400 m spherical reference radius. The TIFF does not expose band units or a vertical reference; "
    "this is an explicit calculation assumption, not raster-metadata verification."
)


@dataclass(frozen=True)
class ObserverCell:
    row: int
    column: int
    local_row: int
    local_column: int
    pixel_center_x_m: float
    pixel_center_y_m: float
    offset_from_published_center_m: float


@dataclass(frozen=True)
class WindowAnalysis:
    observer_cell: ObserverCell
    observer_position_km: np.ndarray
    observer_up: np.ndarray
    in_circle_mask: np.ndarray
    valid_circle_mask: np.ndarray
    coverage_status: str
    horizon_profile: TerrainHorizonProfile
    surface: Any
    normal_field: Any


def circle_mask(window: TerrainWindow, center_x_m: float, center_y_m: float, radius_m: float) -> np.ndarray:
    """Select native pixel centers within the requested projected circle."""
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Circle radius must be positive and finite.")
    x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
    return (x_grid - center_x_m) ** 2 + (y_grid - center_y_m) ** 2 <= radius_m**2


def select_observer_cell(
    window: TerrainWindow,
    metadata: TerrainMetadata,
    published_center_xy_m: tuple[float, float],
) -> ObserverCell:
    """Find the raster cell containing the published center and its center offset."""
    a, b, c, d, e, f = metadata.affine
    determinant = a * e - b * d
    if not math.isfinite(determinant) or abs(determinant) <= 1e-15:
        raise ValueError("Raster affine is singular; observer pixel cannot be selected.")
    x, y = map(float, published_center_xy_m)
    dx, dy = x - c, y - f
    column_f = (e * dx - b * dy) / determinant
    row_f = (-d * dx + a * dy) / determinant
    column, row = math.floor(column_f), math.floor(row_f)
    local_row, local_column = row - window.row_offset, column - window.column_offset
    if not (0 <= local_row < window.elevation_m.shape[0] and 0 <= local_column < window.elevation_m.shape[1]):
        raise ValueError("Published Site07 center's containing pixel is absent from the returned TerrainWindow.")
    px = c + (column + 0.5) * a + (row + 0.5) * b
    py = f + (column + 0.5) * d + (row + 0.5) * e
    return ObserverCell(row, column, local_row, local_column, px, py, math.hypot(px - x, py - y))


def analyze_window(
    window: TerrainWindow,
    metadata: TerrainMetadata,
    published_center_xy_m: tuple[float, float],
    reference_north_km: tuple[float, float, float] | np.ndarray,
    *,
    radius_m: float = RADIUS_M,
) -> WindowAnalysis:
    """Reconstruct this bounded window and calculate its sample-only profile."""
    if metadata.source_frame != MOON_ME_DE421:
        raise ValueError(f"Site07 raster frame input must be tagged {MOON_ME_DE421}.")
    cell = select_observer_cell(window, metadata, published_center_xy_m)
    surface = reconstruct_lunar_terrain(window, metadata, REFERENCE_RADIUS_KM)
    normal_field = calculate_terrain_normals(surface)
    index = (cell.local_row, cell.local_column)
    if not normal_field.valid_mask[index] or not np.all(np.isfinite(normal_field.normals[index])):
        raise ValueError("Site07 observer normal stencil is invalid; refusing to calculate a horizon.")
    observer = np.asarray(surface.positions_km[index], dtype=np.float64)
    observer_up = np.asarray(normal_field.normals[index], dtype=np.float64)
    in_circle = circle_mask(window, *published_center_xy_m, radius_m)
    x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
    left, bottom, right, top = metadata.bounds_m
    in_coverage = (x_grid >= left) & (x_grid < right) & (y_grid > bottom) & (y_grid <= top)
    valid = surface.valid_mask & in_circle & in_coverage & ~window.nodata_mask & np.isfinite(window.elevation_m)
    status = regional_horizon_status(in_circle, in_coverage, window.nodata_mask, window.elevation_m)
    profile = calculate_terrain_horizon_profile(
        observer,
        observer_up,
        surface.positions_km,
        observer_frame=MOON_ME_DE421,
        terrain_frame=surface.source_frame,
        reference_north_km=reference_north_km,
        valid_mask=valid,
        observer_sample_index=index,
        azimuth_bin_count=360,
    )
    return WindowAnalysis(cell, observer, observer_up, in_circle, valid, status, profile, surface, normal_field)


def _site07_north_in_dem_frame(observer_position_km: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Reuse the established Site06 V0.2A North and SPICE bridge procedure."""
    with SpiceGeometryEngine():
        epoch = "2026-10-05T00:00:00"
        et = spice.str2et(epoch)
        with LunarFrameBridge() as bridge:
            observer_pa = np.asarray(bridge.vector_dem_to_geometry(tuple(observer_position_km), et))
            radii = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
            ray = observer_pa / np.linalg.norm(observer_pa)
            ellipsoid_point = ray / math.sqrt(float(np.sum((ray / radii) ** 2)))
            pck_normal = ellipsoid_point / (radii * radii)
            pck_normal /= np.linalg.norm(pck_normal)
            latitude = math.degrees(math.atan2(pck_normal[2], math.hypot(pck_normal[0], pck_normal[1])))
            longitude = math.degrees(math.atan2(pck_normal[1], pck_normal[0])) % 360.0
            local_frame = LocalHorizonFrame.at_site(LunarSite(latitude, longitude))
            north_me = np.asarray(bridge.vector_geometry_to_dem(local_frame.north, et), dtype=np.float64)
    return north_me, epoch, {"latitude_deg": latitude, "longitude_deg": longitude}


def _load_review_inputs() -> tuple[dict[str, Any], dict[str, Any], TerrainMetadata]:
    site = json.loads(SITE_RECORD_PATH.read_text(encoding="utf-8"))
    raster = json.loads(RASTER_REVIEW_PATH.read_text(encoding="utf-8"))
    rm = raster["raster_metadata"]
    metadata = TerrainMetadata(
        crs=rm["crs_wkt"],
        projection="Polar Stereographic",
        affine=tuple(float(value) for value in rm["affine_gdal_order"]),
        width=int(rm["raster_dimensions_rows_columns"][1]),
        height=int(rm["raster_dimensions_rows_columns"][0]),
        pixel_size_m=tuple(float(value) for value in rm["resolution_m"]),
        nodata=math.nan,
        vertical_reference="UNVERIFIED: raster metadata did not encode a vertical reference.",
        source_frame=MOON_ME_DE421,
        bounds_m=(
            float(rm["bounds_m"]["left"]), float(rm["bounds_m"]["bottom"]),
            float(rm["bounds_m"]["right"]), float(rm["bounds_m"]["top"]),
        ),
        pixel_registration=rm["pixel_registration"],
        scale=float(rm["scales_reported_by_rasterio"][0]),
        offset=float(rm["offsets_reported_by_rasterio"][0]),
        dtype=rm["dtype"],
        band_unit=rm["band_units"],
        pixel_registration_observed=True,
    )
    return site, raster, metadata


def build_review_report(
    window: TerrainWindow,
    metadata: TerrainMetadata,
    site_record: Mapping[str, Any],
    raster_record: Mapping[str, Any],
    analysis: WindowAnalysis,
    *,
    reference_north_km: np.ndarray,
    north_epoch_utc: str,
    north_site_latlon_deg: Mapping[str, float],
    software: Mapping[str, Any],
    spice_kernels: list[dict[str, Any]],
    unavailable: Mapping[str, str] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build schema-1.0 report, preserving unresolved raster provenance."""
    coords = site_record["coordinates"]
    product = raster_record["product"]
    raster_meta = raster_record["raster_metadata"]
    center = raster_record["coordinates"]["projected_center_m"]
    center_xy = (float(center["x"]), float(center["y"]))
    circle = analysis.in_circle_mask
    finite = np.isfinite(window.elevation_m)
    missing = window.nodata_mask | ~finite
    valid_circle = circle & analysis.valid_circle_mask
    nodata_circle = circle & window.nodata_mask
    other_invalid_circle = circle & ~valid_circle & ~nodata_circle
    coverage_counts = {
        "requested_area_pixel_count": int(circle.sum()),
        "valid_pixel_count": int(valid_circle.sum()),
        "nodata_pixel_count": int(nodata_circle.sum()),
        "other_invalid_pixel_count": int(other_invalid_circle.sum()),
        "out_of_coverage_pixel_count": 0,
    }
    window_info = {
        "rows_half_open": [window.row_offset, window.row_offset + window.elevation_m.shape[0]],
        "columns_half_open": [window.column_offset, window.column_offset + window.elevation_m.shape[1]],
    }
    source_unavailable = {
        "/source_hash": "A cryptographic hash of the full remote raster was not computed; only bounded TIFF byte ranges were read.",
        "/elevation_units": "No explicit band elevation unit was exposed in the inspected raster metadata.",
        "/vertical_reference": "No vertical reference was encoded in the inspected raster metadata.",
        "/source_frame/evidence": "The raster contains no named source-frame tag; MOON_ME_DE421 comes from Product 78 documentation and application assignment.",
        "/scale_offset_encoding": "Scale 1 and offset 0 are driver-reported defaults; explicit GeoTIFF scale/offset tags were not observed.",
    }
    source = make_raster_source_record(
        source_id=PROFILE_SOURCE_ID,
        source_url=product["url"],
        product_id="NASA PGDA Product 78: Site07_final_adj_5mpp_surf.tif",
        source_hash=None,
        resolution_m=metadata.pixel_size_m,
        raster_dimensions_rows_columns=(metadata.height, metadata.width),
        dtype=metadata.dtype,
        crs_wkt=metadata.crs,
        affine=metadata.affine,
        pixel_registration=metadata.pixel_registration,
        elevation_units=None,
        scale=metadata.scale,
        offset=metadata.offset,
        nodata_value="NaN",
        nodata_handling="NaN is source NoData; all non-finite elevations and samples outside the 5 km circle are excluded from horizon calculation; no interpolation or fill.",
        elevation_units_evidence={
            "status": "unverified",
            "calculation_assumption": ELEVATION_ASSUMPTION,
            "scale_offset_applied": False,
        },
        elevation=window.elevation_m,
        nodata_mask=window.nodata_mask,
        coverage_counts=coverage_counts,
        requested_window=window_info,
        returned_window=window_info,
        source_frame=MOON_ME_DE421,
        frame_evidence_status="code_assigned",
        frame_evidence={
            "references": coords["frame_evidence"]["references"],
            "detail": "MOON_ME/DE421 is supported by NASA Product 78 documentation; it is not a named frame tag in the TIFF.",
        },
        unavailable=source_unavailable,
    )
    source["vertical_reference"] = None
    source["scale_offset_tags_observed"] = False
    source["raster_metadata_reference"] = str(RASTER_REVIEW_PATH.relative_to(ROOT))
    source["coverage_region"] = "Pixel centers within 5,000 m of the published Site07 projected center."

    cell = analysis.observer_cell
    x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
    stencil = [
        [cell.row, cell.column], [cell.row - 1, cell.column], [cell.row + 1, cell.column],
        [cell.row, cell.column - 1], [cell.row, cell.column + 1],
    ]
    source_frames = {
        "value": MOON_ME_DE421,
        "evidence_status": "code_assigned",
        "evidence": {
            "references": coords["frame_evidence"]["references"],
            "detail": "The site coordinates and Product 78 documentation support ME/DE421; the raster itself has no named frame tag.",
        },
    }
    site = {
        "id": "Site07",
        "name": "Peak Near Shackleton",
        "geographic": {
            "latitude_deg": float(coords["latitude_deg"]),
            "longitude_deg": float(coords["longitude_deg"]),
            "latitude_type": coords["latitude_type"],
            "longitude_convention": coords["longitude_convention"],
            "longitude_range_deg": coords["longitude_range_deg"],
            "source_references": [item["document"] if "document" in item else item.get("product_name") for item in site_record["sources"]],
            "convention_evidence": "LOLA RDR coordinate documentation; the Lunar Surface Data Book table supplies the values but does not repeat the conventions.",
        },
        "projected_center_xy_m": list(center_xy),
        "observer_xyz_km": analysis.observer_position_km.tolist(),
        "observer_frame": source_frames,
        "observer_cell": {
            "row": cell.row, "column": cell.column,
            "pixel_center_xy_m": [cell.pixel_center_x_m, cell.pixel_center_y_m],
            "published_center_xy_m": list(center_xy),
            "offset_from_published_center_m": cell.offset_from_published_center_m,
            "selection": "Raster cell containing the published projected Site07 center, selected by inverse affine and floor registration.",
        },
        "observer_up": analysis.observer_up.tolist(),
        "observer_normal_method": {
            "name": "calculate_terrain_normals",
            "method": "Centered 3D finite differences; outward orientation checked against Moon-centered position.",
            "stencil_global_row_column": stencil,
            "stencil_required_valid": True,
        },
        "reference_north": {
            "method": "Existing V0.2A LocalHorizonFrame North method bridged MOON_PA_DE440 to MOON_ME_DE421; projected against measured terrain Up by the existing horizon routine.",
            "epoch_utc": north_epoch_utc,
            "site_ellipsoid_latlon_deg": dict(north_site_latlon_deg),
            "vector_km": np.asarray(reference_north_km, dtype=float).tolist(),
        },
        "frame": source_frames,
        "elevation_assumption": ELEVATION_ASSUMPTION,
    }
    angles = analysis.horizon_profile.max_obstruction_elevation_deg
    counts = analysis.horizon_profile.sample_count
    contributors = [[PROFILE_SOURCE_ID] if int(counts[i]) else [] for i in range(360)]
    runtime = dict(software)
    runtime["terrain_reconstruction_sha256"] = _sha256(ROOT / "core" / "terrain_surface.py")
    runtime["terrain_horizon_sha256"] = _sha256(ROOT / "core" / "terrain_horizon.py")
    runtime["site07_reader_sha256"] = _sha256(ROOT / "scripts" / "review_site07_5km_row_slab_audit.py")
    runtime["frame_bridge_sha256"] = _sha256(ROOT / "core" / "lunar_frame_bridge.py")
    report_unavailable = dict(unavailable or {})
    report_unavailable["/horizon/algorithm/version"] = "The existing scientific horizon routine has no declared semantic version; its exact source hash is recorded."
    report_unavailable["/horizon/composite_rule"] = "This profile has one dedicated 5 m source and is not a multi-source composite."
    return make_horizon_report(
        generated_at_utc=generated_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        site=site,
        sources=[source],
        radius_m=RADIUS_M,
        coverage_status=analysis.coverage_status,
        azimuth_bins={
            "count": 360, "width_deg": 1.0, "centers_deg": list(range(360)),
            "convention": "0=N, 90=E, 180=S, 270=W; clockwise",
            "assignment": "nearest 1 degree center with circular wrap; exact half-bin ties map to the increasing-azimuth bin",
        },
        algorithm={
            "name": "calculate_terrain_horizon_profile", "version": None,
            "implementation_sha256": runtime["terrain_horizon_sha256"],
            "terrain_reconstruction": "Existing reconstruct_lunar_terrain; elevations interpreted under the explicit unverified meter/reference-sphere assumption.",
            "geometry": "Exact Moon-centered Cartesian chord Q-P; lunar curvature included; no flat elevation-over-horizontal-range approximation.",
            "lunar_reference_radius_km": REFERENCE_RADIUS_KM,
            "refraction_model": "None applied.",
            "sample_scope": "Only valid pixel centers inside the 5 km projected circle; observer raster sample excluded.",
        },
        profile_angles_deg=[float(value) if np.isfinite(value) else None for value in angles],
        sample_counts_by_bin=[int(value) for value in counts],
        contributing_source_ids_by_bin=contributors,
        composite_rule=None,
        comparison_profiles=[],
        software=runtime,
        spice_kernels=spice_kernels,
        limitations=[
            "5 km limited-range terrain analysis; it is not a complete lunar horizon and makes no complete lunar visibility claim.",
            ELEVATION_ASSUMPTION,
            "MOON_ME/DE421 is documented for Product 78 but is not encoded as a named raster frame tag.",
            "The profile uses only the dedicated Site07 5 m raster; shared 20 m DEM coverage and suitability were not assessed.",
        ],
        unavailable=report_unavailable,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_review_report() -> None:
    """Make one bounded Site07 5 km read and save a review-only report."""
    from scripts.review_site07_5km_row_slab_audit import read_site07_5km_terrain_window

    site_record, raster_record, metadata = _load_review_inputs()
    window, transfer = read_site07_5km_terrain_window()
    center = raster_record["coordinates"]["projected_center_m"]
    center_xy = (float(center["x"]), float(center["y"]))
    provisional_surface = reconstruct_lunar_terrain(window, metadata, REFERENCE_RADIUS_KM)
    cell = select_observer_cell(window, metadata, center_xy)
    observer = provisional_surface.positions_km[cell.local_row, cell.local_column].copy()
    del provisional_surface
    reference_north, epoch, north_latlon = _site07_north_in_dem_frame(observer)
    analysis = analyze_window(window, metadata, center_xy, reference_north)
    kernel_names = tuple(dict.fromkeys(SpiceGeometryEngine.KERNEL_NAMES + LunarFrameBridge.KERNEL_NAMES))
    software, kernels, unavailable = collect_runtime_provenance(
        Path(__file__), ROOT / "data" / "spice", kernel_names
    )
    software["bounded_response_body_bytes"] = transfer["response_body_bytes_received"]
    software["bounded_response_body_cap_bytes"] = transfer["aggregate_response_body_cap_bytes"]
    software["bounded_range_request_count"] = len(transfer["request_ranges"])
    software["bounded_range_etag_values_observed"] = transfer["etag_values_observed"]
    report = build_review_report(
        window, metadata, site_record, raster_record, analysis,
        reference_north_km=reference_north,
        north_epoch_utc=epoch,
        north_site_latlon_deg=north_latlon,
        software=software,
        spice_kernels=kernels,
        unavailable=unavailable,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialize_horizon_report(report, OUTPUT_PATH)
    print(f"Review-only Site07 report written: {OUTPUT_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-dem", action="store_true", help="Perform a bounded remote Site07 read and generate the review report.")
    args = parser.parse_args(argv)
    if not args.read_dem:
        parser.error("No raster read performed. Review inputs first; --read-dem is required for generation.")
    generate_review_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
