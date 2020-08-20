# -------------------------------------------------------------------------------
# module indices
# This class calculates remote sensing indices given xarray or numpy objects.
# -------------------------------------------------------------------------------
__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# Import System Libraries
# -------------------------------------------------------------------------------
from datetime import datetime
import os, subprocess
import xarray as xr  # read rasters
import rasterio as rio  # geotiff manipulation

# -------------------------------------------------------------------------------
# Module Methods
# -------------------------------------------------------------------------------


def dvi(data, factor=1.0):
    # Difference Vegetation Index (DVI) = B7 - B5, for 8 bands images
    #return ((data[6, :, :] / factor) - (data[4, :, :] / factor)
    #        ).expand_dims(dim="band", axis=0)
    return ((data[3, :, :] / factor) - (data[0, :, :] / factor)
            ).expand_dims(dim="band", axis=0)


def fdi(data, factor=1.0):
    # Forest Discrimination Index (FDI) = (B8 - (B6 + B2)), for 8 bands images
    #return ((data[7, :, :]/factor) - ((data[5, :, :]/factor) + (data[1, :, :]/factor))
    #        ).expand_dims(dim="band", axis=0)
    return ((data[3, :, :]/factor) - ((data[0, :, :]/factor) + (data[2, :, :]/factor))
            ).expand_dims(dim="band", axis=0)

def si(data, factor=1.0):
    # Shadow Index (SI) = (1-Blue)*(1-Green)*(1-Red), for 6 bands images, (SI) = (1-B2)*(1-B3)*(1-B5)
    #return ((1 - (data[1, :, :]/factor)) * (1 - (data[2, :, :]/factor)) * (1 - (data[4, :, :]/factor))
    #        ).expand_dims(dim="band", axis=0)
    return ((1 - (data[0, :, :]/factor)) * (1 - (data[1, :, :]/factor)) * (1 - (data[2, :, :]/factor))
            ).expand_dims(dim="band", axis=0)

def addindices(rastarr, indices, factor=1.0):
    nbands = rastarr.shape[0]  # get number of bands
    for indfunc in indices:  # iterate over each new band
        nbands = nbands + 1  # counter for number of bands
        band = indfunc(rastarr, factor=factor)  # calculate band (indices)
        band.coords['band'] = [nbands]  # add band indices to raster
        rastarr = xr.concat([rastarr, band], dim='band')  # concat new band
    rastarr.attrs['scales'] = [rastarr.attrs['scales'][0]] * nbands
    rastarr.attrs['offsets'] = [rastarr.attrs['offsets'][0]] * nbands
    return rastarr  # return xarray with new bands
