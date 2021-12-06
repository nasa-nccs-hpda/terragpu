import os
import logging
import pathlib
import xarray as xr
import rioxarray as rxr
import dask.array as da

from terragpu.engine import array_module, df_module

xp = array_module('cupy')
xf = df_module('cudf')

CHUNKS = {'band': 1, 'x': 2048, 'y': 2048}

# References
# https://geoexamples.com/other/2019/02/08/cog-tutorial.html/

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
    raster = xr.open_rasterio(filename)
    if xp.__name__ == 'cupy':
        raster.data = xp.asarray(raster.data)
    if backend == 'dask':
        raster.data = da.from_array(raster.data, chunks=CHUNKS)
    return raster

def read_hdf(filename: str):
    raise NotImplementedError

def read_shp(filename: str):
    raise NotImplementedError

def imsave(format: str = 'tiff'):
    raise NotImplementedError

def to_cog():
    raise NotImplementedError

def to_tiff():
    raise NotImplementedError

def to_zarr():
    raise NotImplementedError
