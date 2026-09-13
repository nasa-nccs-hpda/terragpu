"""Download a small public raster and exercise geospatial CPU/GPU processing."""
import argparse
from pathlib import Path
import json
import numpy as np
import rasterio

from .datasets import fetch_sample, SAMPLE
from .streaming import process_indices


def run(cache_dir='data/examples', output='data/example-si.tif', backend='numpy'):
    source = fetch_sample(cache_dir)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result = process_indices(source, output, bands=SAMPLE['bands'], indices=['si'],
                             backend=backend, tile_size=256)
    with rasterio.open(source) as src, rasterio.open(output) as dst:
        rgb = src.read(masked=True).astype('float64')
        expected = np.cbrt(rgb[0] * rgb[1] * rgb[2])
        actual = dst.read(1, masked=True)
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
        valid = ~np.ma.getmaskarray(expected)
        np.testing.assert_allclose(actual.data[valid], expected.data[valid], rtol=1e-5, atol=1e-5)
        assert dst.crs == src.crs and dst.transform == src.transform
    return {**result, 'output': str(output), 'correctness_passed': True,
            'note': 'RGB geometric-mean arithmetic demo on display bytes; not NDVI or calibrated reflectance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='data/examples')
    parser.add_argument('--output', default='data/example-si.tif')
    parser.add_argument('--backend', choices=['numpy', 'cupy'], default='numpy')
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
