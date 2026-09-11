import subprocess
import sys
import numpy as np
import pytest
import xarray as xr
import dask.array as da
from dask.callbacks import Callback
from terragpu import engine, io
from terragpu.indices import wv_indices as indices, hls_indices


def raster(values=(2, 4, 8, 16, 32, 64), dtype='uint16'):
    return xr.DataArray(np.array(values, dtype=dtype)[:, None, None], dims=('band', 'y', 'x'),
                        coords={'band': range(1, len(values) + 1)},
                        attrs={'band_names': ['blue', 'green', 'red', 'nir1', 'rededge', 'nir2'][:len(values)]})


def test_light_import():
    code = """
import importlib.abc
import sys
class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'cupy', 'xarray', 'rioxarray', 'numpy'}:
            raise AssertionError('Unexpected eager import: ' + fullname)
sys.meta_path.insert(0, RejectNumericalImports())
import terragpu
assert not {'torch', 'cupy', 'xarray', 'rioxarray', 'numpy'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_backend_and_cpu_cluster():
    assert engine.array_module('numpy') is np
    with pytest.raises(ValueError):
        engine.array_module('bogus')
    with pytest.raises(ValueError):
        engine.configure_dask(device='bogus')
    client = engine.configure_dask(device='cpu', n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None)
    cluster = client.cluster
    try:
        assert client.submit(sum, [1, 2]).result() == 3
    finally:
        client.close()
        cluster.close()


@pytest.mark.parametrize('name, expected', [('cs1', 48/14), ('cs2', 7.5), ('dvi', 8), ('dwi', -12), ('fdi', 30), ('ndvi', 1/3), ('gndvi', .6), ('ndwi', -.6), ('si', 4), ('sr', 2)])
def test_formulas(name, expected):
    np.testing.assert_allclose(indices.get_indices(name)(raster()), expected, rtol=1e-6)


def test_integer_overflow_zero_nodata():
    source = raster((2, 4, 60000, 1000))
    np.testing.assert_allclose(indices.ndvi(source), -59000/61000, rtol=1e-6)
    source = raster((2, 4, 0, 0))
    assert np.isnan(indices.ndvi(source)).all()
    source.attrs['_FillValue'] = 4
    assert np.isnan(indices.ndwi(source)).all()


def test_lazy_and_metadata():
    source = raster().chunk({'band': 2})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = indices.add_indices(source, ['NDVI', 'FDI'])
        hls = hls_indices.add_indices(source.to_dataset(name='band_data').assign_attrs(source.attrs), ['ndvi'])
    assert not tasks
    assert isinstance(result.data, da.Array)
    assert source.attrs['band_names'] == ['blue', 'green', 'red', 'nir1', 'rededge', 'nir2']
    assert result.attrs['band_names'][-2:] == ['ndvi', 'fdi']
    np.testing.assert_allclose(hls.band_data[-1].compute(), 1/3)
    assert indices.fdi(raster((2, 4, 8, 16))).item() == 6
    with pytest.raises(ValueError):
        indices.add_indices(source, ['ndvi', 'ndvi'])


def test_tiff_roundtrip(tmp_path):
    from rasterio.transform import from_origin
    source = raster().rio.write_crs('EPSG:32618').rio.write_transform(from_origin(500000, 4000000, 30, 30))
    path = tmp_path / 'input.TIF'
    io.imsave(source, path)
    with io.imread(path, bands=source.attrs['band_names'], backend='dask', chunks={'band': -1, 'x': 1, 'y': 1}) as loaded:
        assert isinstance(loaded.data, da.Array)
        result = indices.ndvi(loaded)
        before = result.data
        io.imsave(result, tmp_path / 'ndvi.tif')
        assert result.data is before
        with io.imread(tmp_path / 'ndvi.tif', backend='numpy') as saved:
            np.testing.assert_allclose(saved, 1/3, rtol=1e-6)
            assert saved.rio.crs == source.rio.crs
            assert saved.rio.transform() == source.rio.transform()
    with pytest.raises(ValueError):
        io.imread(path, bands=['red'])
    with pytest.raises(ValueError):
        io.imread(path, backend='bogus')


@pytest.mark.gpu
def test_gpu_parity():
    pytest.importorskip('cupy')
    cp = engine.array_module('cupy')
    source = raster()
    gpu = source.copy(data=cp.asarray(source.data))
    for name in indices.indices_registry:
        np.testing.assert_allclose(cp.asnumpy(indices.get_indices(name)(gpu).data), indices.get_indices(name)(source), rtol=1e-6)
    lazy = gpu.chunk({'band': -1, 'x': 1, 'y': 1})
    result = indices.ndvi(lazy).compute()
    assert isinstance(result.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(result.data), 1/3, rtol=1e-6)


def test_missing_gpu_is_explicit(monkeypatch):
    real_import = engine.import_module
    def without_cupy(name):
        if name == 'cupy':
            raise ImportError('test: unavailable')
        return real_import(name)
    monkeypatch.setattr(engine, 'import_module', without_cupy)
    assert engine.array_module('auto') is np
    with pytest.raises(RuntimeError, match='CUDA'):
        engine.array_module('cupy')


def test_masked_multichunk_output(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    values = np.full((2, 5, 7), 1000, dtype='uint16')
    values[1] = 3000
    values[:, 0, 0] = 65535
    path = tmp_path / 'masked.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=5, width=7, count=2,
                       dtype='uint16', nodata=65535, crs='EPSG:32618',
                       transform=from_origin(500000, 4000000, 30, 30)) as dst:
        dst.write(values)
    with io.imread(path, bands=['red', 'nir1'], backend='dask', chunks={'band': -1, 'y': 2, 'x': 3}) as source:
        result = indices.ndvi(source)
        target = tmp_path / 'output.tif'
        io.imsave(result, target)
        with io.imread(target, backend='numpy') as output:
            assert np.isnan(output[0, 0, 0])
            np.testing.assert_allclose(output[0, 1:, :], .5)
