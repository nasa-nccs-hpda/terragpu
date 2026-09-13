import numpy as np
import pytest

nc = pytest.importorskip('netCDF4')
from terragpu.pace import DEFAULT_FLAGS, PaceSwath, process_pace


def scene(path):
    with nc.Dataset(path, 'w') as ds:
        ds.platform, ds.instrument, ds.processing_version = 'PACE', 'OCI', '3.2'
        dims = ('number_of_lines', 'pixels_per_line')
        for name, size in zip(dims + ('wavelength',), (3, 5, 4)):
            ds.createDimension(name, size)
        sensor = ds.createGroup('sensor_band_parameters')
        wave = sensor.createVariable('wavelength_3d', 'f4', ('wavelength',))
        wave.units = 'nm'
        wave[:] = [390, 400, 500, 700]
        geo = ds.createGroup('geophysical_data')
        rrs = geo.createVariable('Rrs', 'i2', dims + ('wavelength',), fill_value=-32767)
        rrs.setncatts(dict(scale_factor=np.float32(.0001), add_offset=np.float32(.01),
                          valid_min=np.int16(-1000), valid_max=np.int16(1000), units='sr^-1'))
        rrs.set_auto_maskandscale(False)
        raw = np.broadcast_to(np.array([1, 0, 100, 400], dtype='int16'), (3, 5, 4)).copy()
        raw[0, 1, 2] = -32767
        raw[0, 2, 2] = 1001
        raw[1, 1, :] = -200  # valid negative reflectance remains valid
        rrs[:] = raw
        flags = geo.createVariable('l2_flags', 'i4', dims)
        flags.flag_meanings = ' '.join(DEFAULT_FLAGS) + ' HIGHBIT'
        flags.flag_masks = np.array([2**i for i in range(len(DEFAULT_FLAGS))] + [-2147483648], dtype='int32')
        flags[:] = 0
        flags[0, 3] = 1
        flags[1, 2] = -2147483648
        nav = ds.createGroup('navigation_data')
        for name, value in [('latitude', 35.), ('longitude', -75.)]:
            var = nav.createVariable(name, 'f4', dims, fill_value=-999.)
            var[:] = value
        nav.variables['latitude'][0, 4] = -999.


def test_packed_swath_reduction(tmp_path):
    src, dst = tmp_path / 'source.nc', tmp_path / 'out.nc'
    scene(src)
    stats = process_pace(src, dst, tile_size=2)
    # Uneven interval trapezoids: (.01+.02)/2*100 + (.02+.05)/2*200.
    expected = np.full((3, 5), 8.5/300)
    expected[0, 1:] = np.nan
    expected[1, 1] = -.01
    assert stats['tiles'] == 6 and stats['valid_pixels'] == 11
    assert stats['wavelength_count'] == 3
    with nc.Dataset(dst) as ds:
        np.testing.assert_allclose(ds['mean_Rrs'][:].filled(np.nan), expected, atol=1e-8)
        assert ds['latitude'][0, 4] is np.ma.masked
        assert ds['longitude'][1, 1] == -75
        assert ds.wavelength_min_nm == 400 and ds.wavelength_max_nm == 700
    with PaceSwath(src) as swath:
        assert swath.flag_mask(['HIGHBIT']) == 2147483648
    from scripts.validate_pace import validate
    assert validate(src, dst)['valid_pixels'] == 11
    with pytest.raises(FileExistsError):
        process_pace(src, dst)


@pytest.mark.parametrize('kwargs', [dict(wavelength_range=(800, 900)),
                                  dict(reject_flags=('UNKNOWN',)), dict(tile_size=0)])
def test_invalid_request_no_output(tmp_path, kwargs):
    src, dst = tmp_path / 'source.nc', tmp_path / 'out.nc'
    scene(src)
    with pytest.raises(ValueError):
        process_pace(src, dst, **kwargs)
    assert not dst.exists()
    assert not list(tmp_path.glob('.terragpu-pace-*'))


def test_bad_wavelength_coordinate(tmp_path):
    src = tmp_path / 'source.nc'
    scene(src)
    with nc.Dataset(src, 'a') as ds:
        ds.groups['sensor_band_parameters']['wavelength_3d'][:] = [400, 500, 450, 700]
    with pytest.raises(ValueError, match='spectral coordinate'):
        PaceSwath(src)


def test_failed_processing_cleans_staging(tmp_path, monkeypatch):
    src, dst = tmp_path / 'source.nc', tmp_path / 'out.nc'
    scene(src)
    def failed_sum(*args, **kwargs):
        raise RuntimeError('simulated compute failure')
    monkeypatch.setattr(np, 'sum', failed_sum)
    with pytest.raises(RuntimeError, match='simulated'):
        process_pace(src, dst)
    assert not dst.exists()
    assert not list(tmp_path.glob('.terragpu-pace-*'))


@pytest.mark.gpu
def test_pace_gpu_parity(tmp_path):
    pytest.importorskip('cupy')
    src = tmp_path / 'source.nc'
    scene(src)
    process_pace(src, tmp_path / 'cpu.nc', tile_size=2)
    process_pace(src, tmp_path / 'gpu.nc', backend='cupy', tile_size=2)
    with nc.Dataset(tmp_path / 'cpu.nc') as cpu, nc.Dataset(tmp_path / 'gpu.nc') as gpu:
        np.testing.assert_allclose(cpu['mean_Rrs'][:].filled(np.nan), gpu['mean_Rrs'][:].filled(np.nan),
                                   rtol=1e-6, atol=1e-8, equal_nan=True)
