from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terragpu.gpu_io import RasterCache, CacheWriter, pack_raster


def scene(path):
    values=np.arange(17*19,dtype='int16').reshape(17,19)
    values[3:12,4:15]=-9999
    with rasterio.open(path,'w',driver='GTiff',height=17,width=19,count=1,dtype='int16',nodata=-9999,
                       crs='EPSG:32618',transform=from_origin(500000,4000000,30,30)) as dst:
        dst.write(values,1);dst.scales=(.01,);dst.offsets=(1.,);dst.descriptions=('reflectance',)
    expected=values.astype('float32')*.01+1
    expected[values==-9999]=np.nan
    return expected


@pytest.mark.parametrize('tile,halo',[(2,3),(8,2),(32,0)])
def test_pack_halos_metadata_and_export(tmp_path,tile,halo):
    src=tmp_path/'src.tif';expected=scene(src)
    stats=pack_raster(src,tmp_path/'cache',tile,halo)
    cache=RasterCache(tmp_path/'cache',verify=True)
    for row,col in cache.tiles():
        data=cache.read_tile(row,col)
        wanted=np.full_like(data,np.nan)
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                yy,xx=row+y-halo,col+x-halo
                if 0<=yy<17 and 0<=xx<19:wanted[0,y,x]=expected[yy,xx]
        np.testing.assert_allclose(data,wanted,rtol=1e-6,equal_nan=True)
        assert cache.filename(row,col).stat().st_size%4096==0
    dst=tmp_path/'export.tif';cache.to_geotiff(dst)
    with rasterio.open(dst) as out,rasterio.open(src) as original:
        np.testing.assert_allclose(out.read(1),expected,rtol=1e-6,equal_nan=True)
        assert out.crs==original.crs and out.transform==original.transform
        assert out.descriptions==('reflectance',)
    assert stats['physical_bytes']>=stats['logical_bytes']
    with pytest.raises(FileExistsError):pack_raster(src,tmp_path/'cache',tile,halo)
    with pytest.raises(FileExistsError):cache.to_geotiff(dst)


def test_corruption_and_incomplete_publication(tmp_path):
    src=tmp_path/'src.tif';scene(src);pack_raster(src,tmp_path/'cache',8,2)
    cache=RasterCache(tmp_path/'cache',verify=True)
    with pytest.raises(ValueError,match='incomplete'):
        with CacheWriter(tmp_path/'incomplete',cache.meta):pass
    assert not (tmp_path/'incomplete').exists()
    assert not list(tmp_path.glob('.raster-cache-*'))
    file=cache.filename(0,0);file.write_bytes(b'\0'*file.stat().st_size)
    with pytest.raises(ValueError,match='checksum'):RasterCache(cache.path,verify=True)
    file.write_bytes(b'short')
    with pytest.raises(ValueError):RasterCache(cache.path)


@pytest.mark.gpu
def test_kvikio_compat_device_roundtrip(tmp_path):
    cp=pytest.importorskip('cupy');pytest.importorskip('kvikio')
    from terragpu.benchmark_gpu_io import probe
    result=probe('kvikio-compat',tmp_path)
    assert result['gds_verified'] is False
    assert not list(tmp_path.iterdir())
