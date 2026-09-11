"""Windowed NDVI from one native HLS V2 L30 or S30 granule.

Product conventions: https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import rasterio
from rasterio.windows import Window

from .engine import array_module


def process_hls_ndvi(source, destination, *, backend='numpy', tile_size=512):
    """Process separate red/NIR/Fmask COGs without assembling a full scene.

    Reject fill, cloud, adjacent cloud/shadow, shadow and snow (Fmask bits 1–4).
    Water and all aerosol levels are retained. Reflectance uses the HLS V2
    scale 0.0001 and fill -9999; zero denominators become NaN. No clipping is
    applied. The source directory must contain exactly one V2 granule.
    """
    if backend not in {'numpy', 'cupy'}:
        raise ValueError('backend must be numpy or cupy')
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    source, destination = Path(source), Path(destination)
    prefixes = set()
    for path in source.glob('HLS.*.tif'):
        match = re.fullmatch(r'(HLS\.(L30|S30)\.T\w{5}\.\d{7}T\d{6}\.v2\.0)\.[^.]+\.tif', path.name)
        if match:
            prefixes.add((match[1], match[2]))
    if len(prefixes) != 1:
        raise ValueError('Source must contain exactly one native HLS V2 granule')
    prefix, product = prefixes.pop()
    band_names = ('B04', 'B05' if product == 'L30' else 'B8A', 'Fmask')
    paths = [source / f'{prefix}.{band}.tif' for band in band_names]
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
            red, nir, qa = [stack.enter_context(rasterio.open(path)) for path in paths]
            for dataset, dtype in zip((red, nir, qa), ('int16', 'int16', 'uint8')):
                if (dataset.count != 1 or dataset.dtypes != (dtype,)
                        or dataset.shape != red.shape or dataset.crs != red.crs
                        or dataset.transform != red.transform or dataset.crs is None):
                    raise ValueError('HLS bands must have matching geospatial grids and native types')
            profile = dict(driver='GTiff', width=red.width, height=red.height,
                           count=1, dtype='float32', nodata=np.nan, crs=red.crs,
                           transform=red.transform, tiled=True, blockxsize=256,
                           blockysize=256, compress='LZW', BIGTIFF='IF_SAFER')
            fd, temporary = tempfile.mkstemp(prefix='.terragpu-hls-', suffix='.tif', dir=destination.parent)
            os.close(fd)
            with rasterio.open(temporary, 'w', **profile) as dst:
                dst.descriptions = ('ndvi',)
                dst.update_tags(source_granule=prefix, reflectance_scale='0.0001',
                                qa_policy='reject fill and Fmask bits 1,2,3,4; retain water and aerosol levels')
                for row in range(0, red.height, tile_size):
                    for col in range(0, red.width, tile_size):
                        window = Window(col, row, min(tile_size, red.width-col), min(tile_size, red.height-row))
                        r, n, q = [s.read(1, window=window, masked=True) for s in (red, nir, qa)]
                        invalid = (np.ma.getmaskarray(r) | np.ma.getmaskarray(n)
                                   | np.ma.getmaskarray(q) | (r.data == -9999)
                                   | (n.data == -9999) | (q.data == 255) | ((q.data & 30) != 0))
                        r_device = xp.asarray(r.data, dtype=xp.float32) * xp.float32(0.0001)
                        n_device = xp.asarray(n.data, dtype=xp.float32) * xp.float32(0.0001)
                        denominator = n_device + r_device
                        valid = ~xp.asarray(invalid) & (denominator != 0)
                        result = xp.full(r.shape, xp.nan, dtype=xp.float32)
                        xp.divide(n_device-r_device, denominator, out=result, where=valid)
                        output = xp.asnumpy(result) if backend == 'cupy' else result
                        valid_pixels += int(np.count_nonzero(np.isfinite(output)))
                        dst.write(output, 1, window=window)
                        tiles += 1
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return dict(granule=prefix, backend=backend, tile_size=tile_size, tiles=tiles,
                valid_pixels=valid_pixels, total_seconds=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--backend', choices=('numpy', 'cupy'), default='numpy')
    parser.add_argument('--tile-size', type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(process_hls_ndvi(args.source, args.destination,
                                     backend=args.backend, tile_size=args.tile_size), indent=2))


if __name__ == '__main__':
    main()
