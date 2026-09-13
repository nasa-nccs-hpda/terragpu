"""Independent windowed float64 HLS ratio, QA and georeferencing validation."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio


def validate(source, output):
    source = Path(source)
    red_path, = source.glob('HLS.*.B04.tif')
    nir_band = 'B05' if '.L30.' in red_path.name else 'B8A'
    prefix = red_path.name.removesuffix('B04.tif')
    count, error = 0, 0.
    with ExitStack() as stack:
        red, nir, qa, dst = [stack.enter_context(rasterio.open(path)) for path in
                            (red_path, source/(prefix+nir_band+'.tif'),
                             source/(prefix+'Fmask.tif'), output)]
        assert (red.shape, red.crs, red.transform) == (dst.shape, dst.crs, dst.transform)
        for _, window in dst.block_windows(1):
            r, n, q = [ds.read(1, window=window, masked=True) for ds in (red, nir, qa)]
            valid = ~(np.ma.getmaskarray(r) | np.ma.getmaskarray(n) | np.ma.getmaskarray(q))
            valid &= (r.data != -9999) & (n.data != -9999) & (q.data != 255)
            for bit in (1, 2, 3, 4):
                valid &= ((q.data >> bit) & 1) == 0
            r, n = r.data.astype('float64'), n.data.astype('float64')
            valid &= (r+n) != 0
            expected = np.full(r.shape, np.nan)
            np.divide(n-r, n+r, out=expected, where=valid)
            actual = dst.read(1, window=window)
            np.testing.assert_equal(np.isfinite(actual), valid)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)
            count += int(valid.sum())
            if valid.any():
                error = max(error, float(np.max(np.abs(actual[valid]-expected[valid]))))
        assert count > 0, 'No valid pixels'
        return dict(valid_pixels=count, max_absolute_error=error, shape=list(red.shape),
                    masks_and_georeferencing_match=True, rtol=1e-5, atol=1e-6)
