"""Single-device windowed spectral processing without a task scheduler."""
from pathlib import Path
import os
import tempfile
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr

from .engine import array_module
from .indices.wv_indices import get_indices


def process_indices(source, destination, *, bands, indices=('ndvi',),
                    backend='numpy', tile_size=1024, apply_scale=False):
    """Read → compute → write one spatial tile at a time on CPU or one GPU.

    Input nodata masks become NaN. If apply_scale is True, apply each source
    band's stored scale/offset. Product-specific QA flags remain the caller's
    responsibility. Output contains only requested float32 indices. A temporary
    sibling GeoTIFF is atomically renamed after success. Existing outputs are
    rejected. Memory use scales with tile size and band count, not scene size
    (GDAL's block cache and source block layout also affect resident memory).
    """
    if backend not in {'numpy', 'cupy'}:
        raise ValueError("streaming backend must be numpy or cupy")
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    names = [n.lower() for n in indices]
    if not names or len(set(names)) != len(names):
        raise ValueError('indices must contain unique index names')
    functions = [get_indices(name) for name in names]
    bands = [b.lower() for b in bands]
    destination = Path(destination)
    if destination.suffix.lower() not in {'.tif', '.tiff'}:
        raise ValueError('Output must be a GeoTIFF')
    if Path(source).resolve() == destination.resolve() or destination.exists():
        raise FileExistsError(destination)
    xp = array_module(backend)
    timing = dict(read_seconds=0.0, compute_seconds=0.0, write_seconds=0.0, tiles=0)
    total_start = time.perf_counter()
    temporary = None
    try:
        with rasterio.open(source) as src:
            if len(bands) != src.count or len(set(bands)) != len(bands):
                raise ValueError('bands must contain one unique name per input band')
            # Validate required bands before creating output or reading real tiles.
            fixture = xr.DataArray(np.ones((src.count, 1, 1), dtype='float32'),
                                   dims=('band', 'y', 'x'), attrs={'band_names': bands})
            for fn in functions:
                fn(fixture)
            profile = dict(driver='GTiff', width=src.width, height=src.height,
                           count=len(names), dtype='float32', nodata=np.nan,
                           crs=src.crs, transform=src.transform, tiled=True,
                           blockxsize=256, blockysize=256, compress='LZW', BIGTIFF='IF_SAFER')
            fd, temporary = tempfile.mkstemp(suffix='.tif', prefix='.terragpu-', dir=destination.parent)
            os.close(fd)
            with rasterio.open(temporary, 'w', **profile) as dst:
                dst.descriptions = tuple(names)
                for row in range(0, src.height, tile_size):
                    for col in range(0, src.width, tile_size):
                        window = Window(col, row, min(tile_size, src.width-col), min(tile_size, src.height-row))
                        start = time.perf_counter()
                        host = src.read(window=window, masked=True, out_dtype='float32').filled(np.nan)
                        if apply_scale:
                            host *= np.asarray(src.scales, dtype='float32')[:, None, None]
                            host += np.asarray(src.offsets, dtype='float32')[:, None, None]
                        timing['read_seconds'] += time.perf_counter() - start
                        start = time.perf_counter()
                        data = xp.asarray(host)
                        tile = xr.DataArray(data, dims=('band', 'y', 'x'), attrs={'band_names': bands})
                        result = xp.concatenate([fn(tile).data for fn in functions], axis=0)
                        output = xp.asnumpy(result) if backend == 'cupy' else result
                        # asnumpy is blocking; compute time includes both device transfers.
                        timing['compute_seconds'] += time.perf_counter() - start
                        start = time.perf_counter()
                        dst.write(output, window=window)
                        timing['write_seconds'] += time.perf_counter() - start
                        timing['tiles'] += 1
                        del data, tile, result, output, host
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    timing['total_seconds'] = time.perf_counter() - total_start
    timing['backend'] = backend
    timing['tile_size'] = tile_size
    return timing
