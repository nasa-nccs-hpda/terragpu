import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terragpu.worldview import process_worldview


def scene(root):
    transform = from_origin(0, 6, 1, 1)
    # Deliberately shuffled bands: never assume native order.
    data = np.broadcast_to(np.array([3000, 1000, 2000, 500], dtype='uint16')[:, None, None], (4, 6, 6)).copy()
    data[:, 5, 5] = 0
    cloud = np.array([[1, 2, 3], [0, 1, 1], [1, 1, 1]], dtype='uint8')
    sat = np.zeros((6, 6), dtype='uint8'); sat[3, 3] = 1
    for name, values, tr in [('ms.tif', data, transform), ('cloud.tif', cloud[None], from_origin(0, 6, 2, 2)), ('sat.tif', sat[None], transform)]:
        with rasterio.open(root/name, 'w', driver='GTiff', count=len(values), width=values.shape[2],
                           height=values.shape[1], dtype=values.dtype, crs='EPSG:32637', transform=tr) as dst:
            dst.write(values)
            if name == 'ms.tif':
                mask = np.full((6, 6), 255, dtype='uint8'); mask[4, 4] = 0;dst.write_mask(mask)
    meta = {'id':'test', 'properties':{'platform':'WV03', 'ard_metadata_version':'0.0.1', 'proj:epsg':32637},
            'assets':{'ms_analytic':{'href':'ms.tif', 'proj:shape':[6, 6], 'proj:transform':list(transform),
                                    'eo:bands':[{'common_name':b} for b in ('nir08','red','green','blue')]},
                      'cloud-mask-raster':{'href':'cloud.tif'}, 'ms-saturation-mask-raster':{'href':'sat.tif'}}}
    path = root/'item.json';path.write_text(json.dumps(meta));return path


def test_worldview_metadata_qa_and_ratios(tmp_path):
    item = scene(tmp_path);dst = tmp_path/'out.tif'
    result = process_worldview(item, dst, tile_size=2)
    assert result['valid_pixels'] == 21
    with rasterio.open(dst) as ds:
        values = ds.read()
        assert np.isnan(values[:, 0, 2]).all()  # coarse cloud expansion
        assert np.isnan(values[:, 3, 3]).all()  # saturation
        assert np.isnan(values[:, 4, 4]).all()  # internal validity mask
        assert np.isnan(values[:, 5, 5]).all()  # zero denominator
        np.testing.assert_allclose(values[:, 0, 0], [.5, -.2], atol=1e-7)
    from scripts.validate_worldview import validate
    assert validate(item, dst)['valid_pixels'] == 21
    with pytest.raises(FileExistsError):process_worldview(item, dst)


def test_worldview_rejects_wrong_grid(tmp_path):
    item = scene(tmp_path);meta=json.loads(item.read_text())
    meta['assets']['ms_analytic']['proj:shape']=[10,10];item.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='grids'):process_worldview(item,tmp_path/'out.tif')
    assert not (tmp_path/'out.tif').exists()


@pytest.mark.gpu
def test_worldview_gpu(tmp_path):
    pytest.importorskip('cupy');item=scene(tmp_path)
    process_worldview(item,tmp_path/'cpu.tif');process_worldview(item,tmp_path/'gpu.tif',backend='cupy')
    with rasterio.open(tmp_path/'cpu.tif') as cpu, rasterio.open(tmp_path/'gpu.tif') as gpu:
        np.testing.assert_allclose(cpu.read(),gpu.read(),atol=2e-7,equal_nan=True)
