"""Windowed focal processing with explicit halos and NaN-aware boundaries."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import rasterio
from rasterio.windows import Window

from .engine import array_module
from .workloads import focal_mean


def process_focal_mean(source, destination, *, band=1, size=15, tile_size=512,
                       backend='numpy'):
    """Mean of finite neighbors, including partial support at raster boundaries.

    Read a size//2 halo around each output tile, filter, then write only its core.
    Nodata holes are filled when at least one neighbor is valid; an all-invalid
    window remains NaN. Stored band scale/offset are applied before filtering.
    No additional product QA or calibration is inferred. Application tile memory
    scales with (tile_size + size - 1)^2; GDAL/allocator caches also consume memory.
    """
    if backend not in {'numpy', 'cupy'}:
        raise ValueError('backend must be numpy or cupy')
    if not isinstance(size, int) or size < 1 or size % 2 != 1:
        raise ValueError('size must be a positive odd integer')
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    if not isinstance(band, int) or band < 1:
        raise ValueError('band must be a positive integer')
    destination = Path(destination)
    if destination.suffix.lower() not in {'.tif', '.tiff'}:
        raise ValueError('Output must be a GeoTIFF')
    if destination.exists():
        raise FileExistsError(destination)
    xp = array_module(backend)
    halo = size//2
    timing = dict(read_seconds=0., compute_seconds=0., write_seconds=0., tiles=0)
    started = time.perf_counter()
    temporary = None
    try:
        with rasterio.open(source) as src:
            if band > src.count or src.crs is None:
                raise ValueError('Expected an existing band in a georeferenced raster')
            profile = dict(driver='GTiff', width=src.width, height=src.height,
                           count=1, dtype='float32', nodata=np.nan, crs=src.crs,
                           transform=src.transform, tiled=True, blockxsize=256,
                           blockysize=256, compress='LZW', BIGTIFF='IF_SAFER')
            fd, temporary = tempfile.mkstemp(prefix='.terragpu-focal-', suffix='.tif', dir=destination.parent)
            os.close(fd)
            with rasterio.open(temporary, 'w', **profile) as dst:
                dst.descriptions = ('focal_mean',)
                dst.update_tags(source_band=band, window_size=size,
                                boundary_policy='finite neighbors only; no reflection; holes filled with finite support',
                                source_scale=src.scales[band-1], source_offset=src.offsets[band-1])
                if src.units[band-1]:
                    dst.set_band_unit(1, src.units[band-1])
                for row in range(0, src.height, tile_size):
                    for col in range(0, src.width, tile_size):
                        height, width = min(tile_size, src.height-row), min(tile_size, src.width-col)
                        top, left = max(0, row-halo), max(0, col-halo)
                        bottom, right = min(src.height, row+height+halo), min(src.width, col+width+halo)
                        start = time.perf_counter()
                        host = src.read(band, window=Window(left, top, right-left, bottom-top),
                                        masked=True, out_dtype='float32').filled(np.nan)
                        host = host * np.float32(src.scales[band-1]) + np.float32(src.offsets[band-1])
                        timing['read_seconds'] += time.perf_counter()-start
                        start = time.perf_counter()
                        result = focal_mean(xp.asarray(host), size, xp=xp)
                        core = result[row-top:row-top+height, col-left:col-left+width]
                        output = xp.asnumpy(core) if backend == 'cupy' else core
                        timing['compute_seconds'] += time.perf_counter()-start
                        start = time.perf_counter()
                        dst.write(output, 1, window=Window(col, row, width, height))
                        timing['write_seconds'] += time.perf_counter()-start
                        timing['tiles'] += 1
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    timing.update(total_seconds=time.perf_counter()-started, backend=backend,
                  tile_size=tile_size, window_size=size, halo=halo)
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--band', type=int, default=1)
    parser.add_argument('--size', type=int, default=15)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--backend', choices=['numpy', 'cupy'], default='numpy')
    print(json.dumps(process_focal_mean(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
