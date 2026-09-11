import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terragpu.hls import process_hls_ndvi


def make_scene(directory, product='L30', mismatch=False):
    prefix = f'HLS.{product}.T18SUJ.2024158T154508.v2.0'
    red = np.full((3, 11), 1000, dtype='int16')
    nir = np.full_like(red, 3000)
    red[1, 0] = -9999
    nir[1, 1] = -9999
    red[1, 2] = -3000  # zero denominator
    red[1, 3] = -1000  # valid negative reflectance: NDVI is not clipped
    qa = np.zeros(red.shape, dtype='uint8')
    qa[0] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 255]
    for band, values in [('B04', red), ('B05' if product == 'L30' else 'B8A', nir), ('Fmask', qa)]:
        with rasterio.open(directory / f'{prefix}.{band}.tif', 'w', driver='GTiff',
                           width=11, height=3, count=1, dtype=values.dtype,
                           crs='EPSG:32618', transform=from_origin(500030 if mismatch and band == 'Fmask' else 500000, 4000000, 30, 30)) as dst:
            dst.write(values, 1)
    return prefix


@pytest.mark.parametrize('product', ['L30', 'S30'])
def test_native_hls_masks_and_band_mapping(tmp_path, product):
    prefix = make_scene(tmp_path, product)
    target = tmp_path / 'ndvi.tif'
    report = process_hls_ndvi(tmp_path, target, tile_size=2)
    expected = np.full((3, 11), .5, dtype='float32')
    expected[0, [2, 3, 4, 5, 10]] = np.nan
    expected[1, :3] = np.nan
    expected[1, 3] = 2
    assert report['valid_pixels'] == 25
    assert report['tiles'] == 12
    with rasterio.open(target) as dst:
        np.testing.assert_allclose(dst.read(1), expected, atol=1e-6)
        assert dst.tags()['source_granule'] == prefix
        assert dst.descriptions == ('ndvi',)
        assert dst.crs.to_epsg() == 32618
    with pytest.raises(FileExistsError):
        process_hls_ndvi(tmp_path, target)


def test_hls_rejects_misaligned_grid(tmp_path):
    make_scene(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match='grids'):
        process_hls_ndvi(tmp_path, tmp_path / 'out.tif')
    assert not (tmp_path / 'out.tif').exists()
    assert not list(tmp_path.glob('.terragpu-hls-*'))


def test_hls_rejects_multiple_granules(tmp_path):
    make_scene(tmp_path, 'L30')
    make_scene(tmp_path, 'S30')
    with pytest.raises(ValueError, match='exactly one'):
        process_hls_ndvi(tmp_path, tmp_path / 'out.tif')


@pytest.mark.gpu
def test_hls_gpu_parity(tmp_path):
    pytest.importorskip('cupy')
    make_scene(tmp_path)
    process_hls_ndvi(tmp_path, tmp_path / 'cpu.tif', tile_size=2)
    process_hls_ndvi(tmp_path, tmp_path / 'gpu.tif', backend='cupy', tile_size=2)
    with rasterio.open(tmp_path / 'cpu.tif') as cpu, rasterio.open(tmp_path / 'gpu.tif') as gpu:
        np.testing.assert_allclose(cpu.read(), gpu.read(), rtol=1e-6, equal_nan=True)
