"""Read-only, windowed access to lunar terrain rasters.

This module deliberately stops at raw elevation windows. It does not calculate
terrain normals, horizons, or visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window


LOLA20M_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_20MPP_ADJ.TIF"
MOON_ME_DE421 = "MOON_ME_DE421"
MOON_REFERENCE_RADIUS_M = 1_737_400.0
_MOON_SPHERE_GEOGRAPHIC_CRS = CRS.from_proj4(
    f"+proj=longlat +R={MOON_REFERENCE_RADIUS_M:g} +no_defs +type=crs"
)
LOLA20M_SOUTH_POLAR_CRS = CRS.from_proj4(
    f"+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R={MOON_REFERENCE_RADIUS_M:g} +units=m +no_defs +type=crs"
)


def project_moon_me_de421_position(
    position: TaggedMoonPosition,
    destination_crs: CRS | str,
) -> tuple[float, float]:
    """Project a tagged ME/DE421 position using a supplied lunar map CRS.

    This shared coordinate operation lets presentation maps use the validated
    projection without opening or reading the remote raster. The provider's
    normal path still supplies the opened product's own CRS.
    """
    if position.frame != MOON_ME_DE421:
        raise ValueError(f"Position frame must be {MOON_ME_DE421}; got {position.frame!r}.")
    longitude_deg = degrees(atan2(position.y_km, position.x_km))
    latitude_deg = degrees(atan2(position.z_km, hypot(position.x_km, position.y_km)))
    transformer = Transformer.from_crs(
        _MOON_SPHERE_GEOGRAPHIC_CRS,
        CRS.from_user_input(destination_crs),
        always_xy=True,
    )
    x_m, y_m = transformer.transform(longitude_deg, latitude_deg, errcheck=True)
    if not isfinite(x_m) or not isfinite(y_m):
        raise ValueError("The tagged MOON_ME_DE421 position did not project to finite DEM coordinates.")
    return float(x_m), float(y_m)


@dataclass(frozen=True)
class TaggedMoonPosition:
    """Origin-centered Moon position in km, with its body-fixed frame explicit."""

    x_km: float
    y_km: float
    z_km: float
    frame: str

    def __post_init__(self) -> None:
        if self.frame != MOON_ME_DE421:
            raise ValueError(f"Expected an explicitly tagged {MOON_ME_DE421} position; got {self.frame!r}.")
        if not all(isfinite(float(value)) for value in (self.x_km, self.y_km, self.z_km)):
            raise ValueError("Moon position components must be finite.")
        if hypot(self.x_km, self.y_km) == 0.0 and self.z_km == 0.0:
            raise ValueError("The Moon-center position has no longitude or latitude to project.")


@dataclass(frozen=True)
class TerrainMetadata:
    """CRS, grid, and provenance properties read from the opened raster."""

    crs: str
    projection: str
    affine: tuple[float, float, float, float, float, float]
    width: int
    height: int
    pixel_size_m: tuple[float, float]
    nodata: float | None
    vertical_reference: str
    source_frame: str
    bounds_m: tuple[float, float, float, float]
    pixel_registration: str
    scale: float | None = None
    offset: float | None = None
    dtype: str | None = None
    band_unit: str | None = None
    pixel_registration_observed: bool = False


@dataclass(frozen=True)
class TerrainWindow:
    """Unmodified raster samples and their native pixel-center coordinates."""

    requested_center_x_m: float
    requested_center_y_m: float
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    elevation_m: np.ndarray
    nodata_mask: np.ndarray
    row_offset: int
    column_offset: int
    pixel_center_semantics: str = "Coordinates are affine-transformed pixel centers; elevations are raw source samples."


class TerrainAccessError(RuntimeError):
    """A raster could not be opened or a bounded terrain window could not be read."""


@runtime_checkable
class TerrainProvider(Protocol):
    """Interface for local or remote read-only DEM metadata and window access."""

    @property
    def metadata(self) -> TerrainMetadata: ...

    def site_position_to_dem_xy(self, position: TaggedMoonPosition) -> tuple[float, float]: ...

    def read_window(
        self, center_x_m: float, center_y_m: float, width_px: int, height_px: int
    ) -> TerrainWindow: ...


class LOLA20mTerrainProvider:
    """Windowed reader for the NASA 80S 20 m LOLA adjusted DEM COG.

    Input site positions are explicitly MOON_ME_DE421 Cartesian coordinates
    in km. Their planetocentric latitude and east-positive longitude are
    projected on the product's 1,737,400 m Moon sphere. This does not resolve
    the frame/convention of any untagged site latitude/longitude pair.

    Raster windows are centered on the pixel containing the supplied DEM X/Y.
    The upper-left outer raster edge is not a pixel center; sample coordinates
    returned in TerrainWindow use pixel centers. Even-sized windows place the
    containing pixel at index floor(size/2) on each axis.
    """

    MAX_WINDOW_SIDE_PX = 2048

    def __init__(self, source: str | Path = LOLA20M_URL) -> None:
        self.source = str(source)
        self._dataset: rasterio.io.DatasetReader | None = None

    def open(self) -> "LOLA20mTerrainProvider":
        if self._dataset is not None:
            return self
        source = self.source
        if source.lower().startswith("https://"):
            # GDAL's virtual filesystem issues byte-range requests against the COG.
            source = "/vsicurl/" + source
        try:
            self._dataset = rasterio.open(source, sharing=False)
        except (rasterio.errors.RasterioError, OSError) as exc:
            raise TerrainAccessError(f"Could not open LOLA DEM {self.source!r}: {exc}") from exc
        if self._dataset.crs is None:
            self.close()
            raise ValueError("The LOLA raster has no CRS; refusing to project site coordinates.")
        return self

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self) -> "LOLA20mTerrainProvider":
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def metadata(self) -> TerrainMetadata:
        dataset = self._require_open()
        transform = dataset.transform
        scale_x, scale_y = dataset.res
        projection = "Polar Stereographic" if dataset.crs.is_projected else "Geographic"
        return TerrainMetadata(
            crs=dataset.crs.to_wkt(),
            projection=projection,
            affine=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
            width=dataset.width,
            height=dataset.height,
            pixel_size_m=(abs(float(scale_x)), abs(float(scale_y))),
            nodata=dataset.nodata,
            vertical_reference=(
                "LOLA adjusted surface elevation in meters relative to the product's "
                "1,737,400 m Moon reference sphere; no separate vertical CRS is encoded in this GeoTIFF."
            ),
            source_frame=MOON_ME_DE421,
            bounds_m=(dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top),
            pixel_registration=(
                f"GeoTIFF PixelIs{dataset.tags().get('AREA_OR_POINT', 'Area')}; "
                "returned coordinate arrays identify pixel centers."
            ),
            scale=float(dataset.scales[0]) if dataset.scales else None,
            offset=float(dataset.offsets[0]) if dataset.offsets else None,
            dtype=str(dataset.dtypes[0]) if dataset.dtypes else None,
            band_unit=((dataset.units[0] or None) if dataset.units else None),
            pixel_registration_observed="AREA_OR_POINT" in dataset.tags(),
        )

    def site_position_to_dem_xy(self, position: TaggedMoonPosition) -> tuple[float, float]:
        """Project a tagged ME/DE421 Cartesian vector (km) into raster X/Y (m).

        Latitude is planetocentric; longitude is east-positive from the ME
        reference meridian. pyproj applies the raster's own CRS/projection.
        """
        dataset = self._require_open()
        return project_moon_me_de421_position(position, dataset.crs)

    def read_window(
        self, center_x_m: float, center_y_m: float, width_px: int, height_px: int
    ) -> TerrainWindow:
        """Read a bounded pixel window centered on the pixel containing DEM X/Y.

        Samples and elevations remain in source coordinates and meters. Reads
        are clipped at raster edges, and oversized requests are rejected.
        """
        dataset = self._require_open()
        if not all(isfinite(float(value)) for value in (center_x_m, center_y_m)):
            raise ValueError("DEM X/Y coordinates must be finite.")
        if not isinstance(width_px, int) or not isinstance(height_px, int) or width_px < 1 or height_px < 1:
            raise ValueError("Window width_px and height_px must be positive integers.")
        if width_px > self.MAX_WINDOW_SIDE_PX or height_px > self.MAX_WINDOW_SIDE_PX:
            raise ValueError(f"Window dimensions may not exceed {self.MAX_WINDOW_SIDE_PX} pixels per side.")
        bounds = dataset.bounds
        if not (bounds.left <= center_x_m < bounds.right and bounds.bottom < center_y_m <= bounds.top):
            raise ValueError("Requested DEM X/Y center lies outside raster bounds.")

        row, column = dataset.index(center_x_m, center_y_m)
        row_start = max(0, row - height_px // 2)
        column_start = max(0, column - width_px // 2)
        row_end = min(dataset.height, row_start + height_px)
        column_end = min(dataset.width, column_start + width_px)
        # Shift the window back from an edge so its requested size is retained when possible.
        row_start = max(0, row_end - height_px)
        column_start = max(0, column_end - width_px)
        window = Window(column_start, row_start, column_end - column_start, row_end - row_start)
        try:
            elevation = dataset.read(1, window=window, masked=False)
        except (rasterio.errors.RasterioError, OSError) as exc:
            raise TerrainAccessError(f"Could not read bounded LOLA DEM window: {exc}") from exc

        rows = np.arange(row_start, row_end)
        columns = np.arange(column_start, column_end)
        x_centers = dataset.transform.c + (columns + 0.5) * dataset.transform.a
        y_centers = dataset.transform.f + (rows + 0.5) * dataset.transform.e
        nodata = dataset.nodata
        if nodata is None:
            nodata_mask = np.zeros(elevation.shape, dtype=bool)
        elif np.isnan(nodata):
            nodata_mask = np.isnan(elevation)
        else:
            nodata_mask = elevation == nodata
        return TerrainWindow(
            requested_center_x_m=float(center_x_m),
            requested_center_y_m=float(center_y_m),
            x_centers_m=np.asarray(x_centers, dtype=np.float64),
            y_centers_m=np.asarray(y_centers, dtype=np.float64),
            elevation_m=elevation,
            nodata_mask=nodata_mask,
            row_offset=row_start,
            column_offset=column_start,
        )

    def _require_open(self) -> rasterio.io.DatasetReader:
        if self._dataset is None:
            raise RuntimeError("Open the LOLA20mTerrainProvider before reading metadata or terrain windows.")
        return self._dataset
