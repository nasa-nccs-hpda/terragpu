import sys, os, glob, argparse  # system modifications
import joblib                   # joblib for parallel jobs
from time import time           # tracking time
from datetime import datetime   # tracking date
import numpy as np              # for arrays modifications
import pandas as pd             # csv data frame modifications
import xarray as xr             # read rasters
import dask.array as da         # dask array features
import rioxarray

from sklearn.model_selection import train_test_split # train/test data split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hummingbird.ml import convert # support GPU inference
import torch # import torch to verify available devices

# Fix seed reproducibility.
seed = 21
np.random.seed = seed

def add_DVI(data):
    """
    Difference Vegetation Index (DVI) = (NIR - Red), for 6 bands images, (DVI) = B7 - B5, for 8 bands images
    """
    return ((data[6,:,:] - data[4,:,:])).expand_dims(dim="band", axis=0)

def add_FDI(data):
    """
    Forest Discrimination Index (FDI) = (NIR - (Red + Blue)), for 6 bands images, (FDI) = (B8 - (B6 + B2)), for 8 bands images
    """
    return ((data[7,:,:] - (data[5,:,:] + data[1,:,:]))).expand_dims(dim="band", axis=0)

def add_SI(data):
    """
    Shadow Index (SI) = (1-Blue)*(1-Green)*(1-Red), for 6 bands images, (SI) = (1-B2)*(1-B3)*(1-B5)
    """
    return ((1 - data[1,:,:]) * (1 - data[2,:,:]) * (1 - data[4,:,:])).expand_dims(dim="band", axis=0)
    
# list of rasters
#rasters = ['/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif']
rasters = ['/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/RandomForests/VHR-Stacks/WV02_20140716_M1BS_103001003328DB00-toa_pansharpen.tif']

# open rasters and get both data and coordinates
for rast in rasters:

    # open filename into chunks - memory mapping
    raster_rf = xr.open_rasterio(rast, chunks={'band': 1, 'x': 2048, 'y': 2048})
    nbands = raster_rf.shape[0]
   
    for band in [add_DVI(raster_rf), add_FDI(raster_rf), add_SI(raster_rf)]:
        nbands = nbands + 1
        band.coords['band'] = [nbands]
        raster_rf = xr.concat([raster_rf, band], dim='band')
    print (raster_rf) 
    raster_rf.transpose("y", "x", "band")
    raster_rf.attrs['scales']  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    raster_rf.attrs['offsets'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print (raster_rf.scales)
    print (raster_rf)
    raster_rf.rio.to_raster("new_image.tif")    

    """
    import rasterio as rio
    with rio.open(rast) as src:
        meta = src.profile
    print(meta)

    out_meta = meta # modify profile based on numpy array
    out_meta['count'] = 11 # output is single band
    out_meta['dtype'] = 'int16' # data type is float64

    # write to a raster
    with rio.open("new_img.tif", 'w', **out_meta) as dst:
        dst.write(raster_rf)
    """
