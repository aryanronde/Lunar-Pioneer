"""Compare native-resolution 5 m and 20 m Site06 horizons over 1 km."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import rasterio
import spiceypy as spice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lunar_frame_bridge import LunarFrameBridge
from core.lunar_coordinates import SITE06_NOBILE_RIM_1
from core.lunar_horizon import LocalHorizonFrame, LunarSite
from core.horizon_report import collect_runtime_provenance, make_horizon_report, make_raster_source_record, serialize_horizon_report
from core.spice_engine import SpiceGeometryEngine
from core.terrain_horizon import calculate_terrain_horizon_profile
from core.terrain_horizon import TerrainHorizonProfile
from core.terrain_provider import LOLA20M_URL, MOON_ME_DE421, TerrainMetadata, TerrainWindow
from core.terrain_surface import reconstruct_lunar_terrain


SITE06_5M_URL = (
    "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site06/"
    "Site06_final_adj_5mpp_surf.tif"
)
RADIUS_M = 1000.0
OBSERVER_KM = np.array((110.21143659666053, 84.16382863817336, -1737.8192860716285))
UP_ME = np.array((0.12396688856499274, 0.30047987214444133, -0.9456976562176598))
SITE_XY_M = (84000.01002578877, 110000.01079172149)
GEOMETRY_UTC = "2026-10-05T00:00:00"
REVIEW_REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "review" / "site06_1km_5m_20m_horizon_v1.json"


def enclosing_window(dataset: rasterio.io.DatasetReader, center_x: float, center_y: float, radius: float):
    """Smallest cell-aligned window enclosing the square around a radius."""
    t = dataset.transform
    col_start = math.floor((center_x - radius - t.c) / t.a)
    col_stop = math.ceil((center_x + radius - t.c) / t.a)
    row_start = math.floor((center_y + radius - t.f) / t.e)
    row_stop = math.ceil((center_y - radius - t.f) / t.e)
    col_start, row_start = max(0, col_start), max(0, row_start)
    col_stop, row_stop = min(dataset.width, col_stop), min(dataset.height, row_stop)
    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError("Requested Site06 window does not overlap this raster.")
    return row_start, row_stop, col_start, col_stop


def read_native_window(dataset: rasterio.io.DatasetReader, center_x: float, center_y: float):
    row_start, row_stop, col_start, col_stop = enclosing_window(dataset, center_x, center_y, RADIUS_M)
    array = dataset.read(
        1,
        window=((row_start, row_stop), (col_start, col_stop)),
        masked=False,
    )
    rows = np.arange(row_start, row_stop)
    cols = np.arange(col_start, col_stop)
    t = dataset.transform
    x_centers = t.c + (cols + 0.5) * t.a
    y_centers = t.f + (rows + 0.5) * t.e
    nodata = dataset.nodata
    nodata_mask = ~np.isfinite(array) if nodata is None or math.isnan(nodata) else array == nodata
    window = TerrainWindow(
        requested_center_x_m=center_x,
        requested_center_y_m=center_y,
        x_centers_m=x_centers.astype(np.float64),
        y_centers_m=y_centers.astype(np.float64),
        elevation_m=array,
        nodata_mask=nodata_mask,
        row_offset=row_start,
        column_offset=col_start,
    )
    return window, (row_start, row_stop, col_start, col_stop)


def metadata_from_dataset(dataset: rasterio.io.DatasetReader) -> TerrainMetadata:
    t = dataset.transform
    return TerrainMetadata(
        crs=dataset.crs.to_wkt(),
        projection="Polar Stereographic" if dataset.crs.is_projected else "Geographic",
        affine=(t.a, t.b, t.c, t.d, t.e, t.f),
        width=dataset.width,
        height=dataset.height,
        pixel_size_m=(abs(dataset.res[0]), abs(dataset.res[1])),
        nodata=dataset.nodata,
        vertical_reference="LOLA adjusted elevations in meters relative to the 1,737,400 m lunar reference sphere.",
        source_frame=MOON_ME_DE421,
        bounds_m=(dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top),
        pixel_registration=f"GeoTIFF PixelIs{dataset.tags().get('AREA_OR_POINT', 'Area')}; pixel-center coordinates.",
    )


def make_profile(
    window: TerrainWindow,
    dataset: rasterio.io.DatasetReader,
    north_me: tuple[float, ...],
):
    metadata = metadata_from_dataset(dataset)
    surface = reconstruct_lunar_terrain(window, metadata)
    x_grid, y_grid = np.meshgrid(window.x_centers_m, window.y_centers_m)
    inside = (x_grid - SITE_XY_M[0]) ** 2 + (y_grid - SITE_XY_M[1]) ** 2 <= RADIUS_M**2
    in_coverage = (
        (x_grid >= metadata.bounds_m[0]) & (x_grid < metadata.bounds_m[2])
        & (y_grid > metadata.bounds_m[1]) & (y_grid <= metadata.bounds_m[3])
    )
    nodata = window.nodata_mask | ~np.isfinite(window.elevation_m)
    valid = inside & in_coverage & ~nodata & surface.valid_mask
    valid_coverage = valid.copy()
    observer_row, observer_col = dataset.index(*SITE_XY_M)
    local_observer_index = (observer_row - window.row_offset, observer_col - window.column_offset)
    if 0 <= local_observer_index[0] < valid.shape[0] and 0 <= local_observer_index[1] < valid.shape[1]:
        valid[local_observer_index] = False
    points = surface.positions_km.copy()
    points[~valid] = np.nan
    profile = calculate_terrain_horizon_profile(
        OBSERVER_KM,
        UP_ME,
        points,
        observer_frame=MOON_ME_DE421,
        terrain_frame=MOON_ME_DE421,
        reference_north_km=north_me,
        valid_mask=valid,
    )
    return profile, {
        "window_shape_rows_columns": list(window.elevation_m.shape),
        "window_bounds_rows_columns_half_open": [
            window.row_offset,
            window.row_offset + window.elevation_m.shape[0],
            window.column_offset,
            window.column_offset + window.elevation_m.shape[1],
        ],
        "samples_in_1km_circle": int(inside.sum()),
        "valid_samples": int(valid.sum()),
        "valid_coverage_samples": int(valid_coverage.sum()),
        "nodata_nan_samples": int((inside & in_coverage & nodata).sum()),
        "other_invalid_samples": int((inside & in_coverage & ~nodata & ~surface.valid_mask).sum()),
        "out_of_coverage_samples": int((inside & ~in_coverage).sum()),
        "native_pixel_size_m": list(metadata.pixel_size_m),
        "block_shapes": dataset.block_shapes,
        "is_tiled": dataset.is_tiled,
        "raster_bounds_m": metadata.bounds_m,
        "observer_pixel_row_column": [int(observer_row), int(observer_col)],
        "observer_pixel_excluded": True,
    }


def _make_source_record(source_id, source_url, dataset, window, bounds, counts):
    nodata = dataset.nodata
    nodata_value = "NaN" if nodata is not None and math.isnan(float(nodata)) else nodata
    units = (dataset.units[0] or None) if dataset.units else None
    scales = dataset.scales
    offsets = dataset.offsets
    unavailable = {
        "/source_hash": "Remote source content was not hashed; a complete-file hash would require a full download.",
    }
    pixel_tag = dataset.tags().get("AREA_OR_POINT")
    if pixel_tag is None:
        unavailable["/pixel_registration"] = "The opened raster has no AREA_OR_POINT tag; pixel coordinates are still calculated at affine pixel centers."
    if units is None:
        unavailable["/elevation_units"] = "The raster band does not expose a unit tag; generator geometry interprets heights as meters."
        unavailable["/elevation_units_evidence/value"] = unavailable["/elevation_units"]
    if nodata_value is None:
        unavailable["/nodata_value"] = "The opened raster exposes no NoData value; non-finite samples are still masked."
    if not scales:
        unavailable["/scale"] = "The opened raster exposes no band scale metadata."
    if not offsets:
        unavailable["/offset"] = "The opened raster exposes no band offset metadata."
    frame_evidence = {
        "status": "code_assigned",
        "detail": "The generator assigns MOON_ME_DE421; this raster metadata does not independently identify its body-fixed frame.",
    }
    row0, row1, col0, col1 = bounds
    return make_raster_source_record(
        source_id=source_id,
        source_url=source_url,
        product_id=source_url.split("?", 1)[0].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1],
        source_hash=None,
        resolution_m=(abs(dataset.res[0]), abs(dataset.res[1])),
        raster_dimensions_rows_columns=(dataset.height, dataset.width),
        dtype=str(dataset.dtypes[0]) if dataset.dtypes else None,
        crs_wkt=dataset.crs.to_wkt() if dataset.crs else None,
        affine=(dataset.transform.a, dataset.transform.b, dataset.transform.c,
                dataset.transform.d, dataset.transform.e, dataset.transform.f),
        pixel_registration=(f"GeoTIFF PixelIs{pixel_tag}; coordinate arrays use pixel centers." if pixel_tag else None),
        elevation_units=units,
        scale=float(scales[0]) if scales else None,
        offset=float(offsets[0]) if offsets else None,
        nodata_value=nodata_value,
        nodata_handling="Mask non-finite samples and exact equality to the raster NoData value; no fill/interpolation.",
        elevation_units_evidence={"status": "raster_metadata" if units else "unverified", "value": units,
                                  "calculation_assumption": "Existing reconstruction treats raw samples as meters.",
                                  "scale_offset_applied_by_generator": False},
        elevation=window.elevation_m,
        nodata_mask=window.nodata_mask,
        coverage_counts={
            "requested_area_pixel_count": counts["samples_in_1km_circle"],
            "valid_pixel_count": counts["valid_coverage_samples"],
            "nodata_pixel_count": counts["nodata_nan_samples"],
            "other_invalid_pixel_count": counts["other_invalid_samples"],
            "out_of_coverage_pixel_count": counts["out_of_coverage_samples"],
        },
        requested_window={"rows_half_open": (row0, row1), "columns_half_open": (col0, col1)},
        returned_window={"rows_half_open": (row0, row1), "columns_half_open": (col0, col1)},
        source_frame=MOON_ME_DE421,
        frame_evidence_status="code_assigned",
        frame_evidence=frame_evidence,
        unavailable=unavailable,
    )


def build_review_report(p5, p20, source_records, software, kernels, unavailable):
    """Build a versioned 1 km report; the primary profile remains native 5 m."""
    angles5 = [float(x) if np.isfinite(x) else None for x in p5.max_obstruction_elevation_deg]
    angles20 = [float(x) if np.isfinite(x) else None for x in p20.max_obstruction_elevation_deg]
    contributors = [["lola_5m"] if angle is not None else [] for angle in angles5]
    unavailable = dict(unavailable)
    unavailable["/horizon/algorithm/version"] = "The existing horizon implementation has no semantic version; its source hash is recorded in software metadata."
    software = dict(software)
    software["horizon_algorithm_sha256"] = _sha256(Path(__file__).resolve().parents[1] / "core" / "terrain_horizon.py")
    return make_horizon_report(
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        site={
            "id": SITE06_NOBILE_RIM_1.name,
            "geographic": {"latitude_deg": SITE06_NOBILE_RIM_1.latitude_deg,
                           "longitude_deg": SITE06_NOBILE_RIM_1.longitude_deg,
                           "latitude_type": SITE06_NOBILE_RIM_1.latitude_type,
                           "longitude_convention": SITE06_NOBILE_RIM_1.longitude_convention,
                           "coordinate_source": SITE06_NOBILE_RIM_1.coordinate_source},
            "projected_xy_m": list(SITE_XY_M),
            "projected_xy_method": "Fixed projected coordinates carried by this cross-check script; the 5 m source CRS is recorded independently below.",
            "observer_xyz_km": OBSERVER_KM.tolist(),
            "observer_up": UP_ME.tolist(),
            "frame": {"value": MOON_ME_DE421, "evidence_status": "code_assigned",
                      "evidence": {"kind": "application_coordinate_record", "coordinate_source": SITE06_NOBILE_RIM_1.coordinate_source,
                                   "detail": "Site and observer are tagged in code; DEM source frame is separately marked code_assigned per source."}},
            "north_source": "V0.2A North transformed MOON_PA_DE440 to MOON_ME_DE421 using local SPICE kernels.",
        },
        sources=source_records,
        radius_m=RADIUS_M,
        coverage_status="limited-range",
        azimuth_bins={"count": 360, "width_deg": 1.0, "centers_deg": list(range(360)),
                      "convention": "0=N, 90=E, 180=S, 270=W; clockwise",
                      "assignment": "nearest 1 degree center with circular wrap"},
        algorithm={"name": "calculate_terrain_horizon_profile", "version": None,
                   "implementation_sha256": _sha256(Path(__file__).resolve().parents[1] / "core" / "terrain_horizon.py")},
        profile_angles_deg=angles5,
        sample_counts_by_bin=[int(x) for x in p5.sample_count],
        contributing_source_ids_by_bin=contributors,
        composite_rule="Native 5 m profile is primary; 20 m profile is retained as a same-radius resolution comparison, not unioned.",
        comparison_profiles=[{"profile_id": "native_20m_1km", "source_ids": ["lola_20m"],
                              "radius_m": RADIUS_M, "profile_angles_deg": angles20,
                              "sample_counts_by_bin": [int(x) for x in p20.sample_count]}],
        unavailable=unavailable,
        software=software,
        spice_kernels=kernels,
        limitations=["1 km limited-range horizon; not a complete lunar horizon.",
                     "Raster frame and absent band units may rely on code/product assumptions; see evidence fields."],
    )


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_profiles(
    profile_5m: TerrainHorizonProfile,
    profile_20m: TerrainHorizonProfile,
) -> dict[str, float | int]:
    """Compare common finite 1-degree bins without filling unsupported bins.

    The signed change is first profile minus second profile. For range
    sensitivity, pass the larger-radius profile first and the previous-radius
    profile second. This helper reports profile deltas; it does not identify
    which terrain sample caused a change.
    """
    if profile_5m.source_frame != profile_20m.source_frame:
        raise ValueError("Cannot compare horizon profiles expressed in different reference frames.")
    if not np.array_equal(profile_5m.azimuth_centers_deg, profile_20m.azimuth_centers_deg):
        raise ValueError("Horizon profiles must use identical azimuth-bin centers.")
    a = profile_5m.max_obstruction_elevation_deg
    b = profile_20m.max_obstruction_elevation_deg
    valid_a = np.isfinite(a)
    valid_b = np.isfinite(b)
    common = valid_a & valid_b
    if not np.any(common):
        raise ValueError("The profiles have no common finite azimuth bins to compare.")
    difference = a[common] - b[common]
    common_indices = np.flatnonzero(common)
    max_absolute_index = int(common_indices[int(np.argmax(np.abs(difference)))])
    max5 = int(np.nanargmax(a))
    max20 = int(np.nanargmax(b))
    return {
        "profile_bins_compared": int(common.sum()),
        "first_profile_valid_bins": int(valid_a.sum()),
        "second_profile_valid_bins": int(valid_b.sum()),
        "first_profile_bins_without_coverage": int((~valid_a).sum()),
        "second_profile_bins_without_coverage": int((~valid_b).sum()),
        "maximum_absolute_profile_difference_deg": float(np.max(np.abs(difference))),
        "rms_profile_difference_deg": float(np.sqrt(np.mean(difference**2))),
        "maximum_signed_profile_change_deg": float(np.max(difference)),
        "minimum_signed_profile_change_deg": float(np.min(difference)),
        "maximum_absolute_difference_azimuth_deg": float(profile_5m.azimuth_centers_deg[max_absolute_index]),
        "strongest_azimuth_difference_deg": int(abs((max5 - max20 + 180) % 360 - 180)),
    }


def main() -> None:
    # Open metadata first. The 5 m file is untiled scanline data, so the 1 km
    # request spans about 400 full-width 4,000-pixel rows (~6.4 MB raw).
    # Obtain the exact V0.2A Site06 North in PA/DE440 and bridge it to ME/DE421.
    # This uses local SPICE kernels only, before any terrain values are read.
    with SpiceGeometryEngine():
        et = spice.str2et(GEOMETRY_UTC)
        with LunarFrameBridge() as bridge:
            observer_pa = np.asarray(bridge.vector_dem_to_geometry(tuple(OBSERVER_KM), et))
            radii = np.asarray(spice.bodvrd("MOON", "RADII", 3)[1], dtype=np.float64)
            ray = observer_pa / np.linalg.norm(observer_pa)
            ellipsoid_point = ray / math.sqrt(float(np.sum((ray / radii) ** 2)))
            normal = ellipsoid_point / (radii * radii)
            normal /= np.linalg.norm(normal)
            lat = math.degrees(math.atan2(normal[2], math.hypot(normal[0], normal[1])))
            lon = math.degrees(math.atan2(normal[1], normal[0])) % 360.0
            north_me = bridge.vector_geometry_to_dem(LocalHorizonFrame.at_site(LunarSite(lat, lon)).north, et)

    with rasterio.Env(CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF"):
        source_records = []
        with rasterio.open("/vsicurl/" + SITE06_5M_URL, sharing=False) as ds5:
            meta5 = metadata_from_dataset(ds5)
            info5 = {
                "url": SITE06_5M_URL,
                "dimensions": [ds5.width, ds5.height],
                "dtype": ds5.dtypes,
                "crs": ds5.crs.to_wkt(),
                "transform": tuple(ds5.transform),
                "bounds_m": tuple(ds5.bounds),
                "pixel_size_m": tuple(ds5.res),
                "nodata": str(ds5.nodata),
                "block_shapes": ds5.block_shapes,
                "is_tiled": ds5.is_tiled,
                "overviews": ds5.overviews(1),
                "image_structure": ds5.tags(ns="IMAGE_STRUCTURE"),
                "file_size_bytes": 64025603,
            }
            if ds5.width != 4000 or ds5.height != 4000 or ds5.block_shapes[0][0] != 1:
                raise RuntimeError("The inspected 5 m raster layout changed; stop before reading a window.")
            row_span = math.ceil(2 * RADIUS_M / ds5.res[1])
            estimated_raw_range_bytes = row_span * ds5.width * np.dtype(ds5.dtypes[0]).itemsize
            if estimated_raw_range_bytes > 10_000_000:
                raise RuntimeError(
                    f"Estimated 5 m scanline transfer {estimated_raw_range_bytes:,} bytes exceeds 10 MB; stopped before raster window read."
                )
            win5, bounds5 = read_native_window(ds5, *SITE_XY_M)
            p5, counts5 = make_profile(win5, ds5, north_me)
            source_records.append(_make_source_record("lola_5m", SITE06_5M_URL, ds5, win5, bounds5, counts5))

        with rasterio.open("/vsicurl/" + LOLA20M_URL, sharing=False) as ds20:
            win20, bounds20 = read_native_window(ds20, *SITE_XY_M)
            info20 = {
                "dimensions": [ds20.width, ds20.height],
                "block_shapes": ds20.block_shapes,
                "is_tiled": ds20.is_tiled,
                "pixel_size_m": tuple(ds20.res),
            }
            p20, counts20 = make_profile(win20, ds20, north_me)
            source_records.append(_make_source_record("lola_20m", LOLA20M_URL, ds20, win20, bounds20, counts20))

    comparison = compare_profiles(p5, p20)
    max5 = int(np.nanargmax(p5.max_obstruction_elevation_deg))
    max20 = int(np.nanargmax(p20.max_obstruction_elevation_deg))
    report = {
        "classification": "1 km Site06 resolution cross-check; limited-range horizon, not a complete lunar horizon",
        "site06_5m": info5,
        "regional_20m": info20,
        "requested_radius_m": RADIUS_M,
        "site_projected_xy_m": SITE_XY_M,
        "site06_5m_window": {**counts5, "row_column_bounds_inclusive_exclusive": list(bounds5)},
        "regional_20m_window": {**counts20, "row_column_bounds_inclusive_exclusive": list(bounds20)},
        "profile_bins_deg": [float(x) if np.isfinite(x) else None for x in p5.max_obstruction_elevation_deg],
        "20m_profile_bins_deg": [float(x) if np.isfinite(x) else None for x in p20.max_obstruction_elevation_deg],
        "5m_max_obstruction_deg": float(p5.max_obstruction_elevation_deg[max5]),
        "5m_max_obstruction_azimuth_deg": int(max5),
        "20m_max_obstruction_deg": float(p20.max_obstruction_elevation_deg[max20]),
        "20m_max_obstruction_azimuth_deg": int(max20),
        **comparison,
        "source_frame": MOON_ME_DE421,
        "observer_frame": MOON_ME_DE421,
        "azimuth_convention": "0=N, 90=E, 180=S, 270=W; bins are 1 degree",
        "observer_up": UP_ME.tolist(),
        "north_source": "V0.2A LocalHorizonFrame North transformed MOON_PA_DE440 -> MOON_ME_DE421 then projected against Site06 terrain Up",
        "east_definition": "East = North x Up",
        "transfer_bytes": "Populate from HTTP Content-Range response headers captured for this run.",
        "outside_requested_bounded_regions_read": True,
        "storage_block_overfetch_note": (
            "HTTP byte ranges include complete 5 m scanlines and 20 m COG tiles, "
            "so some pixels outside each requested rectangular RasterIO window "
            "were transferred. No full raster or additional spatial window was read."
        ),
    }
    spice_names = tuple(dict.fromkeys(SpiceGeometryEngine.KERNEL_NAMES + LunarFrameBridge.KERNEL_NAMES))
    software, kernels, unavailable = collect_runtime_provenance(
        Path(__file__), Path(__file__).resolve().parents[1] / "data" / "spice", spice_names
    )
    versioned = build_review_report(p5, p20, source_records, software, kernels, unavailable)
    serialize_horizon_report(versioned, REVIEW_REPORT_PATH)
    print(json.dumps({key: value for key, value in report.items() if "profile_bins" not in key}, indent=2))
    print(f"Versioned review report: {REVIEW_REPORT_PATH}")


if __name__ == "__main__":
    main()
