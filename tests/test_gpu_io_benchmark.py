import json
import numpy as np
import pytest
pytest.importorskip('scipy')
from terragpu.benchmark_gpu_io import features,reference,run,process,validate
from terragpu.gpu_io import pack_raster,RasterCache
from test_gpu_io import scene


def test_multiscale_against_manual_neighbors():
    tile=np.arange(2*9*11,dtype='float32').reshape(2,9,11)/100
    tile[:,2:5,3:7]=np.nan
    actual=features(tile,[3,5],np)
    expected=[]
    for band in tile:
        for size in [3,5]:
            mean=np.full(band.shape,np.nan);var=mean.copy();h=size//2
            for y in range(band.shape[0]):
                for x in range(band.shape[1]):
                    patch=band[max(0,y-h):y+h+1,max(0,x-h):x+h+1].astype('float64')
                    if np.isfinite(patch).any():mean[y,x]=np.nanmean(patch);var[y,x]=np.nanvar(patch)
            expected.extend([mean,var])
    np.testing.assert_allclose(actual,np.stack(expected),rtol=2e-5,atol=2e-6,equal_nan=True)


def test_full_cpu_experiment_and_cleanup(tmp_path):
    src=tmp_path/'source.tif';scene(src)
    result=run(tmp_path/'result.json',work_root=tmp_path/'work',source=src,modes=['numpy'],
               tiles=[8,32],sizes=[3,5],repeat=1,warmup=0,allow_dirty=True)
    assert result['status']=='complete' and not result['gds_verified']
    assert len(result['records'])==2
    assert all(r['correctness_passed'] and len(r['phase_samples'])==1 for r in result['records'])
    assert not list((tmp_path/'work').iterdir())
    assert not (tmp_path/'result.partial.json').exists()
    assert (tmp_path/'result.csv').exists()
    assert all(r['geotiff_export_seconds']>0 for r in result['preparation'])


@pytest.mark.gpu
def test_gpu_chain_and_io_matches_independent_reference(tmp_path):
    pytest.importorskip('cupy')
    src=tmp_path/'source.tif';scene(src);pack_raster(src,tmp_path/'cache',tile_size=8,halo=2)
    cache=RasterCache(tmp_path/'cache')
    modes=['cupy']
    try:import kvikio
    except ImportError:pass
    else:modes.append('kvikio-compat')
    for mode in modes:
        process(cache,tmp_path/mode,[3,5],mode)
        assert validate(cache,tmp_path/mode,[3,5])['finite_values']>0


def test_reject_insufficient_halo_before_writing(tmp_path):
    src=tmp_path/'source.tif';scene(src)
    pack_raster(src,tmp_path/'cache',tile_size=8,halo=1)
    with pytest.raises(ValueError,match='halo'):
        process(RasterCache(tmp_path/'cache'),tmp_path/'output',[5],'numpy')
    assert not (tmp_path/'output').exists()


def test_optional_cufile_failure_is_explicit(tmp_path,monkeypatch):
    import terragpu.benchmark_gpu_io as bench
    original=bench.probe
    def probe(mode,work):
        if mode=='kvikio-cufile':raise RuntimeError('unavailable')
        return original(mode,work)
    monkeypatch.setattr(bench,'probe',probe)
    src=tmp_path/'source.tif';scene(src)
    result=run(tmp_path/'result.json',work_root=tmp_path/'work',source=src,
               modes=['numpy','kvikio-cufile'],tiles=[32],sizes=[3],repeat=1,warmup=0,allow_dirty=True)
    assert result['status']=='complete_available_modes'
    assert result['skipped_modes']['kvikio-cufile']['error_type']=='RuntimeError'
    assert not result['gds_verified']
    with pytest.raises(RuntimeError,match='unavailable'):
        run(tmp_path/'required.json',work_root=tmp_path/'work',source=src,
            modes=['kvikio-cufile'],tiles=[32],sizes=[3],repeat=1,warmup=0,allow_dirty=True,require_cufile=True)
