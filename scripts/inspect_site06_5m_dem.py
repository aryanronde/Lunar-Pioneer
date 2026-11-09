"""Inspect only metadata for the NASA Site06 5 m LOLA GeoTIFF."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SITE06_5M_URL = (
    "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site06/"
    "Site06_final_adj_5mpp_surf.tif"
)


def main() -> None:
    with rasterio.Env(CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF"):
        with rasterio.open("/vsicurl/" + SITE06_5M_URL, sharing=False) as dataset:
            report = {
                "source": SITE06_5M_URL,
                "driver": dataset.driver,
                "width": dataset.width,
                "height": dataset.height,
                "count": dataset.count,
                "dtype": dataset.dtypes,
                "crs": dataset.crs.to_wkt() if dataset.crs else None,
                "transform": tuple(dataset.transform),
                "bounds": tuple(dataset.bounds),
                "resolution": tuple(dataset.res),
                "nodata": dataset.nodata,
                "block_shapes": dataset.block_shapes,
                "is_tiled": dataset.is_tiled,
                "overviews": dataset.overviews(1),
                "image_structure": dataset.tags(ns="IMAGE_STRUCTURE"),
                "dataset_tags": dataset.tags(),
                "files": dataset.files,
            }
            print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
