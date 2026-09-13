import json
import numpy as np
import pytest
import rasterio

pytest.importorskip('scipy')
pytest.importorskip('psutil')
from terragpu.scaling_benchmark import repeat_scene,scaling,plot_scaling
from test_gpu_io import scene


def test_physical_repetition_preserves_values_masks_and_labels(tmp_path):
    source=tmp_path/'source.tif';values=scene(source)
    with rasterio.open(source,'r+') as src:
        src.scales=(2.,);src.offsets=(3.,)
        values=src.read(1,masked=True,out_dtype='float32').filled(np.nan)
    destination=repeat_scene(source,tmp_path/'expanded.tif',3,tile=4)
    with rasterio.open(destination) as dst,rasterio.open(source) as src:
        np.testing.assert_allclose(dst.read(1),np.tile(values*2+3,(3,3)),equal_nan=True)
        assert dst.transform==src.transform and dst.crs==src.crs
        assert dst.tags()['terragpu_synthetic_extent']=='true'
        assert dst.scales==(1.,) and dst.offsets==(0.,)
    # BigTIFF magic, independent of GDAL's profile representation.
    assert destination.read_bytes()[:4] in (b'II+\x00',b'MM\x00+')
    with pytest.raises(FileExistsError):repeat_scene(source,destination,2)


def test_small_complete_scaling_sweep(tmp_path,monkeypatch):
    source=tmp_path/'source.tif';scene(source)
    monkeypatch.setattr('terragpu.scaling_benchmark.available_cpus',lambda:2)
    result=scaling(tmp_path/'results',source=source,work_root=tmp_path/'work',scales=[1,2],
                   workers=[1],tiles=[4,8],query_counts=[1],backends=['numpy'],repeat=1,warmup=0,allow_dirty=True)
    assert result['status']=='complete'
    assert [r['pixel_multiplier'] for r in result['runs']]==[1,4]
    first=json.loads((tmp_path/'results/scale-1.json').read_text())
    second=json.loads((tmp_path/'results/scale-2.json').read_text())
    assert second['source_shape'][1:]==[v*2 for v in first['source_shape'][1:]]
    assert all(r['correctness_passed'] for r in second['records'])
    assert (tmp_path/'results/scaling.csv').read_text().count('\n')==5
    assert not list((tmp_path/'work').iterdir())
    pytest.importorskip('matplotlib')
    plot_scaling(tmp_path/'results',allow_dirty=True)
    assert (tmp_path/'results/scaling.png').stat().st_size>1000
    assert len(json.loads((tmp_path/'results/best-tested.json').read_text()))==2


def test_scaling_disk_guard_cleans_scratch(tmp_path,monkeypatch):
    from types import SimpleNamespace
    source=tmp_path/'source.tif';scene(source)
    monkeypatch.setattr('terragpu.scaling_benchmark.shutil.disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match='free scratch'):
        scaling(tmp_path/'results',source=source,work_root=tmp_path/'work',scales=[1],workers=[1])
    assert not list((tmp_path/'work').iterdir())
    assert json.loads((tmp_path/'results/scaling.json').read_text())['status']=='running'
