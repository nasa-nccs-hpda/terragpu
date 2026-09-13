"""GeoTIFF I/O with explicit CPU/GPU residency and lazy chunked reads."""
from pathlib import Path
import numpy as np
import rioxarray as rxr
from terragpu import engine

CHUNKS = {'band': -1, 'x': 2048, 'y': 2048}


def _xarray_to_cupy_(data_array):
    cp = engine.array_module('cupy')
    if _is_dask(data_array):
        return data_array.map_blocks(cp.asarray, meta=cp.empty((0,) * data_array.ndim, dtype=data_array.dtype))
    return cp.asarray(data_array)


def _xarray_to_numpy_(data_array):
    meta = data_array._meta if _is_dask(data_array) else data_array
    if isinstance(meta, np.ndarray):
        return data_array
    import cupy as cp
    if _is_dask(data_array):
        return data_array.map_blocks(cp.asnumpy, meta=np.empty((0,) * data_array.ndim, dtype=data_array.dtype))
    return cp.asnumpy(data_array)


def imread(filename, bands=None, backend='numpy', *, chunks=None):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {'.tif', '.tiff'}:
        raise ValueError(f'Unsupported raster format: {path.suffix}')
    return read_tif(path, bands, backend, chunks=chunks)


def read_tif(filename, bands=None, backend='numpy', *, chunks=None):
    """Read a raster. dask/numpy are CPU; cupy/dask-cupy explicitly require GPU.

    Masked source nodata becomes NaN. Data are not radiometrically scaled.
    """
    if backend.startswith('dask'):
        _require_dask()
    if backend not in {'dask', 'numpy', 'cupy', 'dask-cupy'}:
        raise ValueError('backend must be dask, numpy, cupy, or dask-cupy')
    if 'cupy' in backend:
        engine.array_module('cupy')
    raster = rxr.open_rasterio(filename, chunks=(chunks or CHUNKS) if backend.startswith('dask') else None, masked=True)
    if bands is not None:
        if len(bands) != raster.sizes['band'] or len(set(b.lower() for b in bands)) != len(bands):
            raster.close()
            raise ValueError('bands must contain one unique name per raster band')
        raster.attrs['band_names'] = [b.lower() for b in bands]
    if 'cupy' in backend:
        raster.data = _xarray_to_cupy_(raster.data)
    return raster


def imsave(data, filename, compress='LZW', crs=None):
    return to_tif(data, filename, compress, crs)


def to_tif(raster, filename, compress='LZW', crs=None, **creation_options):
    """Write without mutating the source; Dask arrays are written in chunks."""
    if Path(filename).suffix.lower() not in {'.tif', '.tiff'}:
        raise ValueError('Only .tif and .tiff output is supported')
    output = raster.copy(deep=False)
    output.data = _xarray_to_numpy_(raster.data)
    # Derived indices must not inherit integer storage encoding from source.
    output.encoding = {k: v for k, v in output.encoding.items() if k not in {'dtype', 'scale_factor', 'add_offset', 'rasterio_dtype'}}
    if crs is not None:
        output = output.rio.write_crs(crs)
    options = dict(creation_options)
    if _is_dask(output.data):
        from dask.utils import SerializableLock
        options['lock'] = SerializableLock()
    output.rio.to_raster(filename, BIGTIFF='IF_SAFER', compress=compress, **options)


def _unsupported(*args, **kwargs):
    raise NotImplementedError('This format is not implemented; use GeoTIFF')


read_hdf = read_shp = to_hdf = to_shp = to_cog = to_zarr = _unsupported


def _is_dask(data):
    return hasattr(data, "__dask_graph__")


def _require_dask():
    try:
        import dask.array
    except ImportError as exc:
        raise ImportError("Install terragpu[parallel] for Dask raster backends") from exc
