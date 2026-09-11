import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from terragpu import Raster
from terragpu.streaming import process_indices


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / 'scene.tif'
    values = np.full((2, 17, 19), 1000, dtype='uint16')
    values[1] = 3000
    values[:, 0, 0] = 65535
    with rasterio.open(path, 'w', driver='GTiff', count=2, width=19, height=17,
                       dtype='uint16', nodata=65535, crs='EPSG:32618',
                       transform=from_origin(500000, 4000000, 30, 30)) as dst:
        dst.write(values)
        dst.scales = (0.001, 0.001)
        dst.offsets = (1, 1)
    return path


@pytest.mark.parametrize('scale,expected', [(False, .5), (True, 1/3)])
def test_streaming_edges_masks_and_scaling(scene, tmp_path, scale, expected):
    output = tmp_path / 'indices.tif'
    result = process_indices(scene, output, bands=['red', 'nir1'], tile_size=8, apply_scale=scale)
    assert result['tiles'] == 9
    with rasterio.open(output) as dst, rasterio.open(scene) as src:
        data = dst.read(masked=True)
        assert data.mask[0, 0, 0]
        np.testing.assert_allclose(data.compressed(), expected, rtol=1e-6)
        assert dst.crs == src.crs and dst.transform == src.transform
        assert dst.descriptions == ('ndvi',)
    assert not list(tmp_path.glob('.terragpu-*'))
    with pytest.raises(FileExistsError):
        process_indices(scene, output, bands=['red', 'nir1'])


def test_raster_handle(scene):
    with Raster(scene, bands=['red', 'nir1']) as raster:
        assert raster.bands == ('red', 'nir1')
        np.testing.assert_allclose(raster.index('ndvi')[0, 1:, :], .5)
        assert raster.add_indices(['ndvi']).sizes['band'] == 3
        assert raster.data.sizes['band'] == 2


def test_invalid_streaming_leaves_no_output(scene, tmp_path):
    target = tmp_path / 'bad.tif'
    with pytest.raises(ValueError, match='nir1'):
        process_indices(scene, target, bands=['red', 'blue'])
    assert not target.exists()


@pytest.mark.gpu
def test_streaming_gpu(scene, tmp_path):
    pytest.importorskip('cupy')
    target = tmp_path / 'gpu.tif'
    process_indices(scene, target, bands=['red', 'nir1'], backend='cupy', tile_size=8)
    with rasterio.open(target) as dst:
        np.testing.assert_allclose(dst.read(masked=True).compressed(), .5)
