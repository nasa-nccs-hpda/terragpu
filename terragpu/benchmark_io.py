"""Compare windowed and Dask NDVI including GeoTIFF read/compute/write."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin

from . import io
from .indices.wv_indices import ndvi
from .streaming import process_indices


def run(size=2048, tile_size=1024, repeat=3, device='numpy'):
    if min(size, tile_size, repeat) < 1:
        raise ValueError('size, tile_size and repeat must be positive')
    io._require_dask()
    records = {'streaming': [], 'dask': []}
    with tempfile.TemporaryDirectory(prefix='terragpu-benchmark-') as directory:
        root = Path(directory)
        source = root / 'source.tif'
        values = np.random.default_rng(42).uniform(.01, 1, (2, size, size)).astype('float32')
        expected = (values[1].astype('float64') - values[0]) / (values[1].astype('float64') + values[0])
        with rasterio.open(source, 'w', driver='GTiff', count=2, width=size, height=size,
                           dtype='float32', tiled=True, blockxsize=256, blockysize=256,
                           transform=from_origin(500000, 4000000, 30, 30), crs='EPSG:32618') as dst:
            dst.write(values)
        del values
        # Alternate ordering to reduce systematic first-reader bias. Cache is not flushed.
        for iteration in range(repeat):
            for mode in (('streaming', 'dask') if iteration % 2 == 0 else ('dask', 'streaming')):
                target = root / f'{mode}-{iteration}.tif'
                start = time.perf_counter()
                if mode == 'streaming':
                    process_indices(source, target, bands=['red', 'nir1'], backend=device, tile_size=tile_size)
                else:
                    import dask
                    backend = 'dask-cupy' if device == 'cupy' else 'dask'
                    with dask.config.set(scheduler='single-threaded'):
                        with io.imread(source, bands=['red', 'nir1'], backend=backend,
                                       chunks={'band': -1, 'x': tile_size, 'y': tile_size}) as raster:
                            # Match streaming output tiling/compression for a fair I/O comparison.
                            output = ndvi(raster)
                            io.to_tif(output, target, tiled=True, blockxsize=256, blockysize=256)
                seconds = time.perf_counter() - start
                with rasterio.open(target) as dst:
                    actual = dst.read(1)
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                records[mode].append(seconds)
                target.unlink()
    return {'workload': 'synthetic_geotiff_ndvi', 'device': device, 'size': size,
            'tile_size': tile_size, 'repeat': repeat, 'samples_seconds': records,
            'median_seconds': {k: float(np.median(v)) for k, v in records.items()},
            'correctness_passed': True, 'dask_scheduler': 'single-threaded',
            'scope': 'read, graph/array construction, computation, transfers, compressed write and close; validation excluded; filesystem cache uncontrolled'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--device', choices=['numpy', 'cupy'], default='numpy')
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args())
    output = args.pop('output')
    result = json.dumps(run(**args), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result + '\n')
    print(result)


if __name__ == '__main__':
    main()
