import os
import logging
import pathlib
import xarray as xr
import rioxarray as rxr
import dask.array as da

from terragpu import engine

xp = engine.array_module()
xf = engine.df_module()

CHUNKS = {'band': 1, 'x': 1024, 'y': 1024}

# References
# https://geoexamples.com/other/2019/02/08/cog-tutorial.html/

# -------------------------------------------------------------------------------
# Backend Methods
# -------------------------------------------------------------------------------

def _xarray_to_cupy_(ds, dims: list = ["band", "y", "x"]):
    return xr.DataArray(xp.asarray(ds), dims=dims) #.chunk(chunks=CHUNKS)

def _xarray_to_numpy_(ds, dims: list = ["band", "y", "x"]):
    return xr.DataArray(xp.asnumpy(ds), dims=dims)

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
    raster = xr.open_dataset(filename, engine="rasterio", chunks=CHUNKS)
    if xp.__name__ == 'cupy':
        raster['band_data'] = _xarray_to_cupy_(raster['band_data'])
    if backend == 'dask':
        raster['band_data'] = raster['band_data'].chunk(chunks=CHUNKS)
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

def imsave(data, filename):
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
    }
    return engines[suffix](data, filename)

def to_cog():
    raise NotImplementedError

def to_hdf():
    raise NotImplementedError

def to_shp():
    raise NotImplementedError

def to_tif(data, filename: str, compress: str = 'LZW'):
    assert (pathlib.Path(filename).suffix)[:4] == '.tif', \
        f'to_tif suffix should be one of [.tif, .tiff]'
    data = data.as_numpy()
    data.band_data.rio.to_raster(filename, compress=compress)
    return

def to_zarr():
    raise NotImplementedError
