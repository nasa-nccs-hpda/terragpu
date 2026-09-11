"""WorldView ARD NDVI/NDWI from local STAC, analytic COG and quality masks."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from .engine import array_module


def _asset_path(item_path, asset):
    href = asset['href']
    path = (item_path.parent / href).resolve()
    if '://' in href or path.parent != item_path.parent.resolve():
        raise ValueError('Assets must be local files beside the STAC item')
    return path


def process_worldview(item, destination, *, backend='numpy', tile_size=512):
    """Process archived Maxar WorldView ARD; raw DN/IMD products are unsupported.

    Band positions come from ms_analytic eo:bands. Surface reflectance is DN/10000
    for this product. Require clear cloud-mask class 1 and saturation class 0,
    preserving raster validity masks. QA is nearest-neighbor aligned to the
    analytic grid. Output NDVI and green-NIR NDWI contain NaN for invalid pixels.
    """
    if backend not in {'numpy', 'cupy'}:
        raise ValueError('backend must be numpy or cupy')
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    item, destination = Path(item), Path(destination)
    metadata = json.loads(item.read_text())
    props = metadata['properties']
    if props.get('platform') not in {'WV02', 'WV03', 'WV04'} or 'ard_metadata_version' not in props:
        raise ValueError('Expected WorldView ARD STAC metadata')
    analytic = metadata['assets']['ms_analytic']
    bands = [b.get('common_name') for b in analytic['eo:bands']]
    if any(bands.count(b) != 1 for b in ('red', 'green', 'nir08')):
        raise ValueError('STAC must identify exactly one red, green and nir08 band')
    positions = [bands.index(b)+1 for b in ('red', 'green', 'nir08')]
    if destination.suffix.lower() not in {'.tif', '.tiff'}:
        raise ValueError('Output must be a GeoTIFF')
    if destination.exists():
        raise FileExistsError(destination)
    xp = array_module(backend)
    start = time.perf_counter()
    temporary = None
    valid_pixels = tiles = 0
    try:
        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open(_asset_path(item, analytic)))
            if src.count != len(bands) or src.crs is None or src.dtypes != ('uint16',)*src.count:
                raise ValueError('Analytic raster must match STAC bands and ARD storage type')
            if (list(src.shape) != analytic['proj:shape']
                    or not np.allclose(tuple(src.transform), analytic['proj:transform'], rtol=0, atol=1e-8)
                    or src.crs.to_epsg() != props['proj:epsg']):
                raise ValueError('STAC and analytic raster grids differ')
            masks = []
            for key in ('cloud-mask-raster', 'ms-saturation-mask-raster'):
                qa = stack.enter_context(rasterio.open(_asset_path(item, metadata['assets'][key])))
                if qa.count != 1 or qa.crs is None:
                    raise ValueError('Expected single-band georeferenced quality mask')
                masks.append(stack.enter_context(WarpedVRT(qa, crs=src.crs, transform=src.transform,
                    width=src.width, height=src.height, resampling=Resampling.nearest, add_alpha=True)))
            profile = dict(driver='GTiff', width=src.width, height=src.height, count=2,
                           dtype='float32', nodata=np.nan, crs=src.crs, transform=src.transform,
                           tiled=True, blockxsize=256, blockysize=256, compress='LZW', BIGTIFF='IF_SAFER')
            fd, temporary = tempfile.mkstemp(prefix='.terragpu-worldview-', suffix='.tif', dir=destination.parent)
            os.close(fd)
            with rasterio.open(temporary, 'w', **profile) as dst:
                dst.descriptions = ('ndvi', 'ndwi')
                dst.update_tags(source_item=metadata['id'], platform=props['platform'],
                                reflectance_scale='0.0001', qa_policy='cloud class 1; saturation class 0; nearest resampling')
                for row in range(0, src.height, tile_size):
                    for col in range(0, src.width, tile_size):
                        window = Window(col, row, min(tile_size, src.width-col), min(tile_size, src.height-row))
                        host = src.read(positions, window=window, masked=True, out_dtype='float32').filled(np.nan)
                        cloud, sat = [q.read(1, window=window, masked=True) for q in masks]
                        invalid = (np.ma.getmaskarray(cloud) | np.ma.getmaskarray(sat)
                                   | (cloud.data != 1) | (sat.data != 0))
                        r, g, n = xp.asarray(host) * xp.float32(.0001)
                        outputs = []
                        for a, b in ((n, r), (g, n)):
                            denominator = a+b
                            valid = ~xp.asarray(invalid) & xp.isfinite(a) & xp.isfinite(b) & (denominator != 0)
                            result = xp.full(a.shape, xp.nan, dtype=xp.float32)
                            xp.divide(a-b, denominator, out=result, where=valid)
                            outputs.append(result)
                        result = xp.stack(outputs)
                        values = xp.asnumpy(result) if backend == 'cupy' else result
                        valid_pixels += int(np.all(np.isfinite(values), axis=0).sum())
                        dst.write(values, window=window)
                        tiles += 1
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return dict(backend=backend, tile_size=tile_size, tiles=tiles, valid_pixels=valid_pixels,
                total_seconds=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('item', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--backend', choices=('numpy', 'cupy'), default='numpy')
    parser.add_argument('--tile-size', type=int, default=512)
    print(json.dumps(process_worldview(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
