import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip('scipy')
from terragpu.focal import process_focal_mean
from terragpu.workloads import focal_mean


def scene(path):
    a = np.arange(17*19, dtype='int16').reshape(17, 19)
    a[3:14, 4:15] = -9999
    with rasterio.open(path, 'w', driver='GTiff', width=19, height=17, count=1,
                       dtype='int16', nodata=-9999, crs='EPSG:32618',
                       transform=from_origin(500000, 4000000, 30, 30)) as dst:
        dst.write(a, 1)
        dst.scales = (.01,)
        dst.offsets = (1.,)
    data = a.astype('float64')*.01+1
    data[a == -9999] = np.nan
    return data


@pytest.mark.parametrize('tile_size', [2, 7, 64])
def test_halo_edges_holes_and_scale(tmp_path, tile_size):
    src, dst = tmp_path/'src.tif', tmp_path/'dst.tif'
    data = scene(src)
    expected = np.full(data.shape, np.nan)
    for y in range(17):
        for x in range(19):
            patch = data[max(0, y-2):y+3, max(0, x-2):x+3]
            if np.isfinite(patch).any():
                expected[y, x] = np.nanmean(patch)
    process_focal_mean(src, dst, size=5, tile_size=tile_size)
    with rasterio.open(src) as source, rasterio.open(dst) as output:
        np.testing.assert_allclose(output.read(1), expected, rtol=1e-6, atol=2e-6, equal_nan=True)
        assert output.transform == source.transform and output.crs == source.crs
        assert output.scales == (1.,) and output.offsets == (0.,)
    with pytest.raises(FileExistsError):
        process_focal_mean(src, dst)


def test_mixed_image_all_invalid_neighborhood_stays_nan():
    a = np.ones((37, 41), dtype='float32')
    a[4:33, 4:37] = np.nan
    result = focal_mean(a, 15)
    assert np.isnan(result[11:26, 11:30]).all()


@pytest.mark.gpu
def test_focal_streaming_gpu(tmp_path):
    pytest.importorskip('cupy')
    src = tmp_path/'src.tif'
    scene(src)
    for backend in ('numpy', 'cupy'):
        process_focal_mean(src, tmp_path/f'{backend}.tif', size=5, tile_size=2, backend=backend)
    with rasterio.open(tmp_path/'numpy.tif') as cpu, rasterio.open(tmp_path/'cupy.tif') as gpu:
        np.testing.assert_allclose(cpu.read(), gpu.read(), rtol=1e-6, atol=2e-6, equal_nan=True)
