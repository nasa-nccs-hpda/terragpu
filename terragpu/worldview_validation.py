"""Independent float64 ARD ratio check; align QA with reproject, not WarpedVRT."""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def validate(item, output):
    item = Path(item)
    metadata = json.loads(item.read_text())
    assets = metadata['assets']
    bands = [v['common_name'] for v in assets['ms_analytic']['eo:bands']]
    count, error = 0, 0.
    with rasterio.open(item.parent / assets['ms_analytic']['href']) as src, rasterio.open(output) as dst:
        assert (src.shape, src.crs, src.transform) == (dst.shape, dst.crs, dst.transform)
        assert dst.descriptions == ('ndvi', 'ndwi')
        valid_qa = np.ones(src.shape, dtype=bool)
        for key, clear in [('cloud-mask-raster', 1), ('ms-saturation-mask-raster', 0)]:
            with rasterio.open(item.parent / assets[key]['href']) as qa:
                aligned = np.full(src.shape, 255, dtype='uint8')
                reproject(qa.read(1, masked=True).filled(255), aligned,
                          src_transform=qa.transform, src_crs=qa.crs, src_nodata=255,
                          dst_transform=src.transform, dst_crs=src.crs, dst_nodata=255,
                          resampling=Resampling.nearest)
                valid_qa &= aligned == clear
        for _, window in dst.block_windows(1):
            r, g, n = src.read([bands.index(b)+1 for b in ('red', 'green', 'nir08')],
                               window=window, masked=True, out_dtype='float64').filled(np.nan)
            refs = []
            for a, b in ((n, r), (g, n)):
                valid = valid_qa[window.toslices()] & np.isfinite(a) & np.isfinite(b) & ((a+b) != 0)
                ref = np.full(a.shape, np.nan)
                np.divide(a-b, a+b, where=valid, out=ref)
                refs.append(ref)
            expected = np.stack(refs)
            actual = dst.read(window=window)
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-7, equal_nan=True)
            count += int(np.all(np.isfinite(actual), axis=0).sum())
            valid = np.isfinite(expected)
            if valid.any():
                error = max(error, float(np.max(np.abs(actual[valid]-expected[valid]))))
        assert count > 0
        return dict(valid_pixels=count, max_absolute_error=error, shape=list(src.shape),
                    masks_and_georeferencing_match=True)

