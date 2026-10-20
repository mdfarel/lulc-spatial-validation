"""ESA WorldCover 2021 labels for spatially independent LULC training."""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
CLASS_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
CLASS_NAMES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare/sparse vegetation",
    70: "snow and ice",
    80: "permanent water",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
CODE_TO_INDEX = {code: index for index, code in enumerate(CLASS_CODES)}
INDEX_TO_CODE = {index: code for code, index in CODE_TO_INDEX.items()}
_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE="TRUE")


def _tile_name(south: int, west: int) -> str:
    ns = f"{'N' if south >= 0 else 'S'}{abs(south):02d}"
    ew = f"{'E' if west >= 0 else 'W'}{abs(west):03d}"
    return f"{ns}{ew}"


def tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """WorldCover 3° tile names intersecting a WGS84 bbox."""
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("invalid WorldCover bbox")
    names: list[str] = []
    latitude = math.floor(south / 3) * 3
    while latitude < north:
        longitude = math.floor(west / 3) * 3
        while longitude < east:
            names.append(_tile_name(latitude, longitude))
            longitude += 3
        latitude += 3
    return names


def _href(tile: str) -> str:
    return f"{BASE}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def read_labels(bbox: tuple[float, float, float, float], shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour WorldCover labels on the caller's exact WGS84 grid."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("WorldCover output shape must be positive")
    destination = np.zeros(shape, dtype=np.uint8)
    destination_transform = from_bounds(*bbox, width=width, height=height)
    first = True
    for tile in tile_names(bbox):
        try:
            with rasterio.Env(**_ENV):
                with rasterio.open(_href(tile)) as source:
                    reproject(
                        source=rasterio.band(source, 1),
                        destination=destination,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=destination_transform,
                        dst_crs="EPSG:4326",
                        dst_nodata=0,
                        resampling=Resampling.nearest,
                        init_dest_nodata=first,
                    )
        except rasterio.errors.RasterioIOError as exc:
            raise RuntimeError(f"WorldCover tile unavailable: {tile}") from exc
        first = False
    return destination


def to_index(codes: np.ndarray) -> np.ndarray:
    """Convert official class codes to dense model indices; unknown is -1."""
    codes = np.asarray(codes)
    indices = np.full(codes.shape, -1, dtype=np.int16)
    for code, index in CODE_TO_INDEX.items():
        indices[codes == code] = index
    return indices
