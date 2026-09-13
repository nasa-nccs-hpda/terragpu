"""Experimental uncompressed, halo-tiled raster cache for NumPy/CuPy/KvikIO.

This is a versioned TerraGPU working format, not GeoTIFF or Zarr. Metadata stays
on the CPU. KvikIO payload I/O can use device buffers; cuFile is not proof of GDS.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import rasterio
from rasterio.windows import Window

from .datasets import sha256, _write_json
from .engine import array_module

MODES = ('numpy', 'cupy', 'kvikio-compat', 'kvikio-cufile')


def module(mode):
    if mode not in MODES:
        raise ValueError('Unknown I/O mode')
    return array_module('numpy' if mode == 'numpy' else 'cupy')


def synchronize(mode):
    if mode != 'numpy':
        module(mode).cuda.get_current_stream().synchronize()


@contextmanager
def io_mode(mode):
    module(mode)
    if mode.startswith('kvikio'):
        try:
            import kvikio.defaults
        except ImportError as exc:
            raise ImportError('Install the gpu-io extra on a CUDA Linux node') from exc
        with kvikio.defaults.set('compat_mode', 'ON' if mode == 'kvikio-compat' else 'OFF'):
            yield
    else:
        yield


def io_evidence(mode):
    result = dict(mode=mode, gds_verified=False)
    if mode.startswith('kvikio'):
        import kvikio
        result['kvikio_version'] = kvikio.__version__
        if mode == 'kvikio-cufile':
            try:
                from kvikio import cufile_driver
                for key in ('is_gds_available', 'allow_compat_mode', 'major_version', 'minor_version'):
                    result[key] = cufile_driver.get(key)
            except Exception as error:
                result['driver_probe_error_type'] = type(error).__name__
        result['note'] = 'cuFile-requested I/O may still use internal compatibility paths; mount-specific GDS telemetry is required.'
    return result


def _shape(meta):
    return (meta['count'], meta['tile_size']+2*meta['halo'], meta['tile_size']+2*meta['halo'])


def _elements(meta):
    # Each file is padded to a multiple of 4096 bytes. Padding is not raster data.
    return math.ceil(math.prod(_shape(meta))*4/4096)*1024


def _read(path, meta, mode):
    count = _elements(meta)
    if path.stat().st_size != count*4:
        raise ValueError('Truncated or oversized raster chunk')
    if mode in ('numpy', 'cupy'):
        data = np.fromfile(path, dtype='<f4', count=count)
        data = module(mode).asarray(data)
    else:
        import kvikio
        data = module(mode).empty(count, dtype='float32')
        synchronize(mode)
        with kvikio.CuFile(str(path), 'r') as handle:
            if handle.read(data) != data.nbytes:
                raise OSError('Short KvikIO read')
    return data[:math.prod(_shape(meta))].reshape(_shape(meta))


def _write(path, meta, array, mode):
    xp = module(mode)
    if array.shape != _shape(meta):
        raise ValueError('Tile shape does not match cache layout')
    data = xp.zeros(_elements(meta), dtype='float32')
    data[:array.size] = array.ravel()
    synchronize(mode)
    if mode in ('numpy', 'cupy'):
        host = xp.asnumpy(data) if mode == 'cupy' else data
        host.astype('<f4', copy=False).tofile(path)
    else:
        import kvikio
        with kvikio.CuFile(str(path), 'w') as handle:
            if handle.write(data) != data.nbytes:
                raise OSError('Short KvikIO write')


class RasterCache:
    """Read halo tiles; each returned array has band,y,x axes and float32 NaNs."""
    def __init__(self, path, verify=False):
        self.path = Path(path)
        self.meta = json.loads((self.path/'metadata.json').read_text())
        m = self.meta
        if m.get('schema') != 'terragpu-raster-cache-v1' or m.get('dtype') != '<f4':
            raise ValueError('Unsupported raster cache')
        for key in ('height','width','count','tile_size'):
            if type(m.get(key)) is not int or m[key] < 1:
                raise ValueError('Invalid raster cache dimensions')
        if type(m.get('halo')) is not int or m['halo'] < 0:
            raise ValueError('Invalid halo')
        if len(m['transform']) != 6 or len(m['bands']) != m['count'] or m.get('nodata') != 'NaN':
            raise ValueError('Invalid raster metadata')
        rasterio.crs.CRS.from_wkt(m['crs_wkt'])
        for row, col in self.tiles():
            file = self.filename(row,col)
            if file.is_symlink() or file.stat().st_size != _elements(m)*4:
                raise ValueError('Invalid raster chunk')
            if verify and sha256(file) != m.get('sha256',{}).get(file.name):
                raise ValueError('Cache checksum mismatch or missing checksum')

    def tiles(self):
        m=self.meta
        for row in range(0,m['height'],m['tile_size']):
            for col in range(0,m['width'],m['tile_size']):
                yield row,col

    def filename(self,row,col):
        m=self.meta
        if not (0 <= row < m['height'] and 0 <= col < m['width']) or row%m['tile_size'] or col%m['tile_size']:
            raise ValueError('Invalid tile origin')
        return self.path/f'{row}-{col}.bin'

    def read_tile(self,row,col,mode='numpy'):
        with io_mode(mode):
            return _read(self.filename(row,col),self.meta,mode)

    def core_shape(self,row,col):
        m=self.meta
        return min(m['tile_size'],m['height']-row),min(m['tile_size'],m['width']-col)

    def to_geotiff(self,destination):
        """Explicit CPU export; its cost is separate from cache-only benchmarks."""
        destination=Path(destination)
        if destination.exists():raise FileExistsError(destination)
        m=self.meta
        fd,temp=tempfile.mkstemp(suffix='.tif',dir=destination.parent);os.close(fd)
        try:
            with rasterio.open(temp,'w',driver='GTiff',count=m['count'],height=m['height'],width=m['width'],
                               dtype='float32',nodata=np.nan,crs=m['crs_wkt'],transform=rasterio.Affine(*m['transform']),
                               tiled=True,blockxsize=256,blockysize=256,compress='LZW') as dst:
                dst.descriptions=tuple(m['bands'])
                for row,col in self.tiles():
                    h,w=self.core_shape(row,col);halo=m['halo']
                    dst.write(self.read_tile(row,col)[:,halo:halo+h,halo:halo+w],window=Window(col,row,w,h))
            os.replace(temp,destination)
        finally:
            if os.path.exists(temp):os.unlink(temp)


class CacheWriter:
    """Atomic cache creation; partial chunks disappear on any processing failure."""
    def __init__(self,path,metadata,mode='numpy'):
        self.path=Path(path);self.meta=dict(metadata);self.mode=mode;self.written=set()
        self.meta.pop('sha256',None)

    def __enter__(self):
        if self.path.exists():raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temp=Path(tempfile.mkdtemp(prefix='.raster-cache-',dir=self.path.parent))
        return self

    def write_tile(self,row,col,data):
        m=self.meta
        if (row,col) in self.written or not (0<=row<m['height'] and 0<=col<m['width']) or row%m['tile_size'] or col%m['tile_size']:
            raise ValueError('Duplicate or invalid tile origin')
        with io_mode(self.mode):
            _write(self.temp/f'{row}-{col}.bin',m,data,self.mode)
        self.written.add((row,col))

    def __exit__(self,kind,value,trace):
        try:
            if kind is None:
                m=self.meta
                if len(self.written) != math.ceil(m['height']/m['tile_size'])*math.ceil(m['width']/m['tile_size']):
                    raise ValueError('Cannot publish incomplete raster cache')
                _write_json(self.temp/'metadata.json',m)
                if self.path.exists():raise FileExistsError(self.path)
                self.temp.rename(self.path)
        finally:
            if self.temp.exists():shutil.rmtree(self.temp)


def pack_raster(source,destination,tile_size=512,halo=15):
    """Decode GeoTIFF once on CPU, preserving georeferencing and scale/offset.

    Halos are duplicated in storage to allow one contiguous read per compute
    tile. NaNs pad outside the raster. This storage overhead is reported.
    """
    if type(tile_size) is not int or tile_size<1 or type(halo) is not int or halo<0:
        raise ValueError('Invalid tile or halo size')
    started=time.perf_counter()
    with rasterio.open(source) as src:
        if src.crs is None:raise ValueError('Georeferenced source required')
        meta=dict(schema='terragpu-raster-cache-v1',dtype='<f4',nodata='NaN',height=src.height,width=src.width,
                  count=src.count,tile_size=tile_size,halo=halo,crs_wkt=src.crs.to_wkt(),
                  transform=list(src.transform)[:6],bands=[v or f'band{i+1}' for i,v in enumerate(src.descriptions)],
                  source_sha256=sha256(source),source_scales=list(src.scales),source_offsets=list(src.offsets),sha256={})
        hashes={}
        with CacheWriter(destination,meta) as writer:
            for row in range(0,src.height,tile_size):
                for col in range(0,src.width,tile_size):
                    top,left=max(0,row-halo),max(0,col-halo)
                    bottom,right=min(src.height,row+tile_size+halo),min(src.width,col+tile_size+halo)
                    values=src.read(window=Window(left,top,right-left,bottom-top),masked=True,out_dtype='float32').filled(np.nan)
                    values=values*np.asarray(src.scales,dtype='float32')[:,None,None]+np.asarray(src.offsets,dtype='float32')[:,None,None]
                    tile=np.full(_shape(meta),np.nan,dtype='float32')
                    y,x=top-row+halo,left-col+halo
                    tile[:,y:y+values.shape[1],x:x+values.shape[2]]=values
                    writer.write_tile(row,col,tile)
                    file=writer.temp/f'{row}-{col}.bin';hashes[file.name]=sha256(file)
            writer.meta['sha256']=hashes
    physical=sum(p.stat().st_size for p in Path(destination).glob('*.bin'))
    return dict(conversion_seconds=time.perf_counter()-started,logical_bytes=src.count*src.height*src.width*4,
                physical_bytes=physical,metadata_sha256=sha256(Path(destination)/'metadata.json'))
