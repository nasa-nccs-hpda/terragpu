import os
import logging
import pathlib
import xarray as xr
import rioxarray as rxr
import dask.array as da
import numpy as np

from terragpu import engine

xp = engine.array_module()
xf = engine.df_module()

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}

# References
# https://geoexamples.com/other/2019/02/08/cog-tutorial.html/

# -------------------------------------------------------------------------------
# Backend Methods
# -------------------------------------------------------------------------------

def _xarray_to_cupy_(data_array):
    return data_array.map_blocks(xp.asarray)


def _xarray_to_numpy_(data_array):
    return data_array.map_blocks(xp.asnumpy)

# -------------------------------------------------------------------------------
# Read Methods
# -------------------------------------------------------------------------------

def imread(filename: str = None, bands: list = None, backend: str = 'dask'):
    """
    Read imagery based on suffix
    """
    assert os.path.isfile(filename) 
    suffix = pathlib.Path(filename).suffix

    # choose which file to read from here
    engines = {
        '.tif': read_tif,
        '.tiff': read_tif,
        '.hdf': read_hdf,
        '.shp': read_shp,
    }
    return engines[suffix](filename, bands, backend)


def read_tif(filename: str, bands: list = None, backend: str = 'dask'):
    """
    Read TIF Imagery to GPU.
    Next Release: cucim support for built-in GPU read.
    """
    raster = rxr.open_rasterio(filename, chunks=CHUNKS)
    if xp.__name__ == 'cupy':
        raster.data = _xarray_to_cupy_(raster.data)
    if bands is not None:
        raster.attrs['band_names'] = [b.lower() for b in bands]
    return raster


def read_hdf(filename: str, bands: list = None, backend: str = 'dask'):
    # rioxarray or cupy
    raise NotImplementedError


def read_shp(filename: str, bands: list = None, backend: str = 'dask'):
    # cuspatial or geopandas
    raise NotImplementedError

# -------------------------------------------------------------------------------
# Output Methods
# -------------------------------------------------------------------------------

def imsave(data, filename: str, compress: str = 'LZW', crs: str = None):
    """
    Save imagery based on format
    """
    suffix = pathlib.Path(filename).suffix
    
    # choose which file to save from here
    engines = {
        '.tif': to_tif,
        '.tiff': to_tif,
        '.hdf': to_hdf,
        '.shp': to_shp,
        '.zarr': to_zarr
    }
    return engines[suffix](data, filename, compress, crs)


def to_cog():
    raise NotImplementedError


def to_hdf():
    raise NotImplementedError


def to_shp():
    raise NotImplementedError


def to_tif(raster, filename: str, compress: str = 'LZW', crs: str = None):
    """
    Save TIFF or TIF files, normally used from raster files.
    """
    assert (pathlib.Path(filename).suffix)[:4] == '.tif', \
        f'to_tif suffix should be one of [.tif, .tiff]'
    if xp.__name__ == 'cupy':
        raster.data = _xarray_to_numpy_(raster.data)
    raster.rio.write_nodata(raster._FillValue)
    if crs is not None:
        raster.rio.write_crs(crs, inplace=True)
    raster.rio.to_raster(filename, BIGTIFF="IF_SAFER", compress=compress)
    return


def to_zarr():
    raise NotImplementedError
