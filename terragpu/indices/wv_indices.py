"""Lazy spectral indices for band/y/x DataArrays on NumPy, CuPy, or Dask.

Arithmetic uses at least float32 to avoid integer overflow. Invalid pixels and
zero denominators propagate as NaN. No function computes or transfers arrays.
"""
import numpy as np
import xarray as xr


def _band(raster, name):
    names = [b.lower() for b in raster.attrs.get('band_names', [])]
    if len(names) != raster.sizes.get('band', 0) or len(set(names)) != len(names):
        raise ValueError('band_names must contain one unique name per band')
    if name not in names:
        raise ValueError(f'{name} not in raster bands {names}')
    band = raster.isel(band=names.index(name), drop=True)
    band = band.astype(np.result_type(band.dtype, np.float32))
    nodata = raster.attrs.get('_FillValue')
    if nodata is not None:
        band = band.where(band != nodata)
    return band


def _ratio(a, b):
    return a / b.where(b != 0)


def _finish(index):
    return index.expand_dims(band=[1])


def cs1(raster):
    return _finish(_ratio(3 * _band(raster, 'nir1'), sum(_band(raster, b) for b in ('blue', 'green', 'red'))))


def cs2(raster):
    return _finish(sum(_band(raster, b) for b in ('blue', 'green', 'red', 'nir1')) / 4)


def dvi(raster):
    return _finish(_band(raster, 'nir1') - _band(raster, 'red'))


def dwi(raster):
    return _finish(_band(raster, 'green') - _band(raster, 'nir1'))


def fdi(raster):
    names = [b.lower() for b in raster.attrs.get('band_names', [])]
    nir, red = ('nir2', 'rededge') if all(b in names for b in ('nir2', 'rededge')) else ('nir1', 'red')
    return _finish(_band(raster, nir) - (_band(raster, red) + _band(raster, 'blue')))


def _normalized(raster, first, second):
    a, b = _band(raster, first), _band(raster, second)
    return _finish(_ratio(a - b, a + b))


def ndvi(raster):
    """(NIR1 - Red) / (NIR1 + Red)."""
    return _normalized(raster, 'nir1', 'red')


def gndvi(raster):
    return _normalized(raster, 'nir1', 'green')


def ndwi(raster):
    return _normalized(raster, 'green', 'nir1')


def si(raster):
    """Cube root of Blue * Green * Red (legacy documented definition)."""
    return _finish(np.cbrt(_band(raster, 'blue') * _band(raster, 'green') * _band(raster, 'red')))


def sr(raster):
    return _finish(_ratio(_band(raster, 'nir1'), _band(raster, 'red')))


indices_registry = {f.__name__: f for f in (cs1, cs2, dvi, dwi, fdi, gndvi, ndvi, ndwi, si, sr)}
__all__ = [*indices_registry, 'get_indices', 'add_indices']


def get_indices(index_key):
    try:
        return indices_registry[index_key.lower()]
    except KeyError:
        raise ValueError(f'Invalid indices mapping: {index_key}.') from None


def add_indices(raster, indices):
    """Append indices in one concat, preserving source metadata without mutation.

    Output band coordinates are numbered from 1; band_names stores semantics.
    """
    names = [name.lower() for name in indices]
    original = list(raster.attrs.get('band_names', []))
    if len(set(names + [n.lower() for n in original])) != len(names) + len(original):
        raise ValueError('Requested indices duplicate existing or requested band names')
    additions = [get_indices(name)(raster) for name in names]
    result = xr.concat([raster, *additions], dim='band') if additions else raster.copy(deep=False)
    result = result.assign_coords(band=np.arange(1, result.sizes['band'] + 1))
    result.attrs = {**raster.attrs, 'band_names': original + names}
    return result
