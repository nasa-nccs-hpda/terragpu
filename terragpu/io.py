import os
import logging
import pathlib
import xarray as xr
import rioxarray as rxr
import dask.array as da

from terragpu import engine

xp = engine.array_module('cupy')
xf = engine.df_module('cudf')

CHUNKS = {'band': 1, 'x': 2048, 'y': 2048}

# References
# https://geoexamples.com/other/2019/02/08/cog-tutorial.html/

# -------------------------------------------------------------------------------
# Backend Methods
# -------------------------------------------------------------------------------

def _xarray_to_cupy_(ds, dims: list = ["band", "y", "x"]):
    return xr.DataArray(xp.asarray(ds), dims=dims)

def _xarray_to_numpy_(ds, dims: list = ["band", "y", "x"]):
    return xr.DataArray(xp.asnumpy(ds), dims=dims)

# -------------------------------------------------------------------------------
# Read Methods
# -------------------------------------------------------------------------------

def imread(filename: str = None, backend: str = 'dask'):
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
    return engines[suffix](filename, backend)

def read_tif(filename: str, backend: str = 'dask'):
    """
    Read TIF Imagery to GPU.
    Next Release: cucim support for built-in GPU read.
    """
    raster = xr.open_dataset(filename, engine="rasterio", chunks=CHUNKS)
    if xp.__name__ == 'cupy':
        raster['band_data'] = _xarray_to_cupy_(raster['band_data'])
    if backend == 'dask':
        raster['band_data'] = raster['band_data'].chunk(chunks=CHUNKS)
    return raster

def read_hdf(filename: str):
    # rioxarray or cupy
    raise NotImplementedError

def read_shp(filename: str):
    # cuspatial or geopandas
    raise NotImplementedError

# -------------------------------------------------------------------------------
# Output Methods
# -------------------------------------------------------------------------------

def imsave(format: str = 'tiff'):
    raise NotImplementedError

def to_cog():
    raise NotImplementedError

def to_tiff():
    raise NotImplementedError

def to_zarr():
    raise NotImplementedError
