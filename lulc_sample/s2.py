"""Sentinel-2 L2A reads for the land-cover sample.

A subset of the Sentinel-2 module in GEOAI's Enviro engine: the STAC search,
the windowed COG reads and the land-cover feature stack. Uses the Element84
Earth Search STAC API and the AWS `sentinel-cogs` open bucket, both public,
no credentials.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD = 60.0

# SCL classes to mask: no-data, saturated/defective, cast/cloud shadow,
# unclassified, water, cloud med/high, cirrus, and snow. Since processing
# baseline 04.00, class 2 is CAST_SHADOW rather than a generic dark-area class;
# retaining it creates exactly the false temporal changes this mask prevents.
SCL_INVALID = (0, 1, 2, 3, 6, 7, 8, 9, 10, 11)
# Surface-water analysis must retain SCL class 6 (water), while still masking
# unusable, cloud-shadow, cloud, cirrus, snow, and unclassified observations.


def scene_cloud(item: dict[str, Any]) -> float:
    value = item.get("properties", {}).get("eo:cloud_cover")
    return 100.0 if value is None else float(value)



async def stac_search(
    bbox: tuple[float, float, float, float],
    datetime_range: str,
    limit: int = 40,
    *,
    collection: str = COLLECTION,
) -> list[dict[str, Any]]:
    """Search Earth Search with bounded pages and retry transient failures."""
    page_size = min(limit, 200)
    body: dict[str, Any] = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": datetime_range,
        "limit": page_size,
        "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    features: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        while len(features) < limit:
            resp: httpx.Response | None = None
            for attempt in range(4):
                try:
                    resp = await client.post(STAC_URL, json=body)
                    if resp.status_code != 429 and resp.status_code < 500:
                        break
                except httpx.RequestError:
                    if attempt == 3:
                        raise
                if attempt < 3:
                    await asyncio.sleep(0.5 * (2 ** attempt))
            if resp is None:
                raise RuntimeError("Earth Search did not return a response")
            resp.raise_for_status()
            payload = resp.json()
            page = payload.get("features", [])
            features.extend(page[:limit - len(features)])
            next_link = next((link for link in payload.get("links", []) if link.get("rel") == "next"), None)
            if not page or not next_link or len(features) >= limit:
                break
            body = next_link.get("body") or {**body, "next": next_link.get("href")}
            body["limit"] = page_size
    return features




def _public_raster_href(href: str) -> str:
    """Prefer fixed unsigned-S3 config, with public HTTPS as a fallback."""
    if not href.startswith("s3://"):
        return href
    if os.environ.get("AWS_NO_SIGN_REQUEST", "").upper() == "YES":
        return href
    bucket_and_key = href[5:].split("/", 1)
    if len(bucket_and_key) != 2:
        return href
    bucket, key = bucket_and_key
    return f"https://{bucket}.s3.amazonaws.com/{key}"


# Earth Search's sentinel-2-l2a COGs live under this path and were written with
# the BOA offset already removed, although their raster:bands still advertise it.
HARMONIZED_COG_PATH = "sentinel-s2-l2a-cogs"


def _boa_offset(asset: dict) -> float:
    """Native-DN BOA_ADD_OFFSET still present in a reflectance asset's pixels.

    Baseline 04.00+ (25 Jan 2022 onward) stores L2A reflectance as
    DN = rho * 10000 + BOA_ADD_OFFSET (offset -1000 in practice) so
    atmospherically-corrected values can go slightly negative instead of
    clipping at 0. STAC exposes it per asset via raster:bands scale/offset
    (offset in reflectance-ratio units, e.g. -0.1); dividing by scale converts
    it to native DN so it adds directly onto the raw pixel array.

    The COGs this module reads are the exception: Element 84 removed the
    offset when converting them, but left -0.1 in raster:bands. Checked on
    2026-09-15 against clear open water, whose true NIR/SWIR1 reflectance is
    near zero: in 15 Jakarta Bay items from 2021-2026 its median DN was 1-485,
    and whole-tile blue-band 1st percentiles in Riau, Central Kalimantan,
    Merauke and the Netherlands were 54-245. Pixels still carrying the offset
    sit near 1000 or above, as the same Jakarta scenes do in the
    sentinel-2-c1-l2a collection (1195-1311). Applying the advertised offset
    here pushed water, shadow and dense canopy below zero, so the `> 0` gates
    dropped them and every normalized-difference index came out inflated.
    Older items (pre-04.00) and non-reflectance assets carry no offset field,
    so this returns 0.0 for them too.
    """
    if HARMONIZED_COG_PATH in asset.get("href", ""):
        return 0.0
    bands = asset.get("raster:bands") or [{}]
    offset = bands[0].get("offset") or 0.0
    scale = bands[0].get("scale") or 0.0001
    return offset / scale


def _read_window(href: str, bbox4326: tuple[float, float, float, float]) -> tuple[np.ndarray, list[float], Any]:
    href = _public_raster_href(href)
    for attempt in range(3):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(href) as ds:
                    wb = transform_bounds("EPSG:4326", ds.crs, *bbox4326)
                    win = from_bounds(*wb, transform=ds.transform).round_offsets().round_lengths()
                    # boundless: an AOI overlapping a tile edge would otherwise
                    # silently return a window smaller than `win`, while the
                    # bounds below are computed from the full (unclipped) win —
                    # the mismatch misregisters every pixel past the edge.
                    data = ds.read(1, window=win, boundless=True, fill_value=0)
                    # actual bounds of the rounded window, back in EPSG:4326
                    actual = rasterio.windows.bounds(win, ds.transform)
                    b4326 = transform_bounds(ds.crs, "EPSG:4326", *actual)
                    return data, list(b4326), ds.crs
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(0.4 * (2 ** attempt))
    raise RuntimeError("unreachable raster read retry state")


def _read_window_resampled(
    href: str,
    bbox4326: tuple[float, float, float, float],
    shape: tuple[int, int],
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    href = _public_raster_href(href)
    for attempt in range(3):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(href) as ds:
                    wb = transform_bounds("EPSG:4326", ds.crs, *bbox4326)
                    win = from_bounds(*wb, transform=ds.transform).round_offsets().round_lengths()
                    return ds.read(
                        1, window=win, out_shape=shape, resampling=resampling,
                        boundless=True, fill_value=0,
                    )
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(0.4 * (2 ** attempt))
    raise RuntimeError("unreachable raster read retry state")



def read_scene_lulc(item: dict, bbox4326: tuple[float, float, float, float]) -> dict:
    """Read one scene on the 20 m SWIR grid as a land-cover feature stack.

    Training and inference must see identical features in identical order, so
    both call this. LULC_FEATURES is that contract — reorder it and every stored
    model becomes silently wrong, which is why the model artifact records it.

    Reflectance is kept alongside the indices: indices alone cannot separate
    bright bare soil from built surfaces, and absolute brightness is what does it.
    """
    assets = item["assets"]
    swir1, bounds, _ = _read_window(assets["swir16"]["href"], bbox4326)
    height, width = swir1.shape
    swir1 = swir1.astype(np.float32) + _boa_offset(assets["swir16"])
    swir2 = _read_window_resampled(assets["swir22"]["href"], bbox4326, (height, width)).astype(np.float32) + _boa_offset(assets["swir22"])
    red = _read_window_resampled(assets["red"]["href"], bbox4326, (height, width)).astype(np.float32) + _boa_offset(assets["red"])
    green = _read_window_resampled(assets["green"]["href"], bbox4326, (height, width)).astype(np.float32) + _boa_offset(assets["green"])
    blue = _read_window_resampled(assets["blue"]["href"], bbox4326, (height, width)).astype(np.float32) + _boa_offset(assets["blue"])
    nir = _read_window_resampled(assets["nir"]["href"], bbox4326, (height, width)).astype(np.float32) + _boa_offset(assets["nir"])
    scl = _read_window_resampled(assets["scl"]["href"], bbox4326, (height, width))

    valid = (
        (~np.isin(scl, SCL_INVALID))
        & (blue > 0) & (green > 0) & (red > 0) & (nir > 0) & (swir1 > 0) & (swir2 > 0)
    )

    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full((height, width), np.nan, dtype=np.float32)
        denominator = a + b
        ok = valid & (denominator != 0)
        out[ok] = (a[ok] - b[ok]) / denominator[ok]
        return out

    bsi = np.full((height, width), np.nan, dtype=np.float32)
    bsi_denominator = swir1 + red + nir + blue
    ok = valid & (bsi_denominator != 0)
    bsi[ok] = ((swir1[ok] + red[ok]) - (nir[ok] + blue[ok])) / bsi_denominator[ok]

    return {
        "blue": blue, "green": green, "red": red,
        "nir": nir, "swir1": swir1, "swir2": swir2,
        "ndvi": ratio(nir, red),
        "ndwi": ratio(green, nir),
        "mndwi": ratio(green, swir1),
        "ndbi": ratio(swir1, nir),
        "bsi": bsi,
        "ndmi": ratio(nir, swir1),
        "valid": valid,
        "bounds": bounds,
        "scene_id": item["id"],
        "datetime": item["properties"]["datetime"],
        "cloud_cover": item["properties"].get("eo:cloud_cover"),
        "tile": item["properties"].get("grid:code"),
    }

