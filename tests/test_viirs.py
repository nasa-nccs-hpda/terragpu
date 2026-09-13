import numpy as np
import pytest

nc = pytest.importorskip('netCDF4')
from terragpu.viirs import ViirsSwath, process_viirs
from terragpu.pace import DEFAULT_FLAGS


def scene(path):
    with nc.Dataset(path, 'w') as ds:
        ds.instrument='VIIRS';ds.platform='JPSS-2';ds.processing_version='R2025'
        dims=('number_of_lines','pixels_per_line')
        for name in dims:ds.createDimension(name,3)
        geo=ds.createGroup('geophysical_data')
        for wave,offset in [(667,.03),(411,.01),(556,.02)]:
            v=geo.createVariable(f'Rrs_{wave}','i2',dims,fill_value=-32767)
            v.setncatts(dict(scale_factor=np.float32(.0001),add_offset=np.float32(offset),
                            valid_min=np.int16(-1000),valid_max=np.int16(1000),units='sr^-1'))
            v.set_auto_maskandscale(False);v[:]=0
        geo['Rrs_556'][0,0]=-32767
        q=geo.createVariable('l2_flags','i4',dims);q.flag_meanings=' '.join(DEFAULT_FLAGS)
        q.flag_masks=np.array([2**i for i in range(len(DEFAULT_FLAGS))],dtype='int32');q[:]=0;q[0,1]=1
        nav=ds.createGroup('navigation_data')
        for name,value in [('latitude',35.),('longitude',-75.)]:
            v=nav.createVariable(name,'f4',dims,fill_value=-999.);v[:]=value


def test_viirs_separate_packing_and_flags(tmp_path):
    path=tmp_path/'source.nc';out=tmp_path/'out.nc';scene(path)
    with ViirsSwath(path) as swath:
        np.testing.assert_equal(swath.wavelengths,[411,556,667])
    report=process_viirs(path,out,tile_size=2)
    assert report['valid_pixels']==7
    expected=(.015*145+.025*111)/256
    with nc.Dataset(out) as ds:
        np.testing.assert_allclose(ds['mean_Rrs'][:].compressed(),expected,atol=1e-8)
    from scripts.validate_pace import validate
    assert validate(path,out)['valid_pixels']==7


def test_viirs_invalid_units(tmp_path):
    path=tmp_path/'source.nc';scene(path)
    with nc.Dataset(path,'a') as ds:ds.groups['geophysical_data']['Rrs_411'].units='DN'
    with pytest.raises(ValueError,match='sr'):ViirsSwath(path)


@pytest.mark.gpu
def test_viirs_gpu(tmp_path):
    pytest.importorskip('cupy');path=tmp_path/'source.nc';scene(path)
    process_viirs(path,tmp_path/'cpu.nc');process_viirs(path,tmp_path/'gpu.nc',backend='cupy')
    with nc.Dataset(tmp_path/'cpu.nc') as cpu,nc.Dataset(tmp_path/'gpu.nc') as gpu:
        np.testing.assert_allclose(cpu['mean_Rrs'][:].filled(np.nan),gpu['mean_Rrs'][:].filled(np.nan),
                                   atol=1e-8,equal_nan=True)
