# -------------------------------------------------------------------------------
# module indices
# This class calculates remote sensing indices given xarray or numpy objects.
# -------------------------------------------------------------------------------
__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# Note:
# We assume 8 band nad 6 band imagery has the following order
# ['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge', 'Near-IR1', 'Near-IR2']
# ['Red', 'Green', 'Blue', 'Near-IR1', 'HOM1', 'HOM2']
# In a future implementation we can provide the methods with the bands to calculate
# instead of trying to calculate them using predefined array indices.
# -------------------------------------------------------------------------------
# Import System Libraries
# -------------------------------------------------------------------------------
import xarray as xr  # read rasters

# -------------------------------------------------------------------------------
# Module Methods
# -------------------------------------------------------------------------------


def addindices(rastarr, indices, initbands=8, factor=1.0):
    """
    :param rastarr:
    :param indices:
    :param initbands:
    :param factor:
    :return:
    """
    nbands = rastarr.shape[0]  # get number of bands
    for indfunc in indices:  # iterate over each new band
        nbands = nbands + 1  # counter for number of bands
        band = indfunc(rastarr, initbands=initbands, factor=factor)  # calculate band (indices)
        band.coords['band'] = [nbands]  # add band indices to raster
        rastarr = xr.concat([rastarr, band], dim='band')  # concat new band
    rastarr.attrs['scales'] = [rastarr.attrs['scales'][0]] * nbands
    rastarr.attrs['offsets'] = [rastarr.attrs['offsets'][0]] * nbands
    return rastarr  # return xarray with new bands


# Difference Vegetation Index (DVI)
def dvi(data, initbands=8, factor=1.0):
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param initbands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with DVI calculated
    """
    if initbands > 8:  # 8 band imagery: DVI := NIR1 - Red = B7 - B5
        return ((data[6, :, :] / factor) - (data[4, :, :] / factor)).expand_dims(dim="band", axis=0)
    else:  # 4 band imagery: DVI := NIR - Red = B4 - B1
        print ("SHAPEPEPE: ", ((data[3, :, :] / factor) - (data[0, :, :] / factor)).shape)
        return ((data[3, :, :] / factor) - (data[0, :, :] / factor)).expand_dims(dim="band", axis=0)


# Normalized Difference Vegetation Index (NDVI)
def ndvi(data, initbands=8, factor=1.0):
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param initbands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with NDVI calculated
    """
    if initbands > 8:  # 8 band imagery: NDVI := (B7 - B5) / (B7 + B5)
        return (((data[6, :, :] / factor) - (data[4, :, :] / factor)) /
                ((data[6, :, :] / factor) + (data[4, :, :] / factor))
                ).expand_dims(dim="band", axis=0)
    else:  # 4 band imagery: NDVI := (NIR - Red) / (NIR + RED) = (B4 - B1) / (B4 + B1)
        return (((data[3, :, :] / factor) - (data[0, :, :] / factor)) /
                ((data[3, :, :] / factor) + (data[0, :, :] / factor))
                ).expand_dims(dim="band", axis=0)


# Forest Discrimination Index (FDI)
def fdi(data, initbands=8, factor=1.0):
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param initbands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with FDI calculated
    """
    if initbands > 8:  # 8 band imagery: FDI := NIR2 - (Red_Edge - Blue) = B8 - (B6 + B2)
        return ((data[7, :, :]/factor) - ((data[5, :, :]/factor) + (data[1, :, :]/factor))
                ).expand_dims(dim="band", axis=0)
    else:  # 4 band imagery: FDI := NIR - (Red + Blue) = B4 - (B1 + B3)
        return ((data[3, :, :]/factor) - ((data[0, :, :]/factor) + (data[2, :, :]/factor))
                ).expand_dims(dim="band", axis=0)


# Shadow Index (SI)
def si(data, initbands=8, factor=1.0):
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param initbands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    if initbands > 8:  # 8 band imagery: SI := (1-Blue) * (1-Green) * (1-Red) = (1-B2) * (1-B3) * (1-B5)
        return ((1 - (data[1, :, :]/factor)) * (1 - (data[2, :, :]/factor)) * (1 - (data[4, :, :]/factor))
                ).expand_dims(dim="band", axis=0)
    else:  # 4 band imagery: SI := (1-Blue) * (1-Green) * (1-Red) = (1-B3) * (1-B2) * (1-B1)
        return ((1 - (data[0, :, :]/factor)) * (1 - (data[1, :, :]/factor)) * (1 - (data[2, :, :]/factor))
                ).expand_dims(dim="band", axis=0)

# Shadow Index (SI)
def cs1(data, initbands=8, factor=1.0):
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param initbands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    #if initbands > 8:  # 8 band imagery: SI := (1-Blue) * (1-Green) * (1-Red) = (1-B2) * (1-B3) * (1-B5)
    #    return ((1 - (data[1, :, :]/factor)) * (1 - (data[2, :, :]/factor)) * (1 - (data[4, :, :]/factor))
    #            ).expand_dims(dim="band", axis=0)
    #else:  # 4 band imagery: SI := (1-Blue) * (1-Green) * (1-Red) = (1-B3) * (1-B2) * (1-B1)
    # df['CI1'] = df.apply(lambda row: (3. * row.NearIR1) / (row.Blue + row.Green + row.Red), axis=1)
    return ((3.0 * (data[3, :, :]/factor)) / (data[2, :, :] + data[1, :, :] + data[0, :, :])
                ).expand_dims(dim="band", axis=0)

# Shadow Index (SI)
def cs2(data, initbands=8, factor=1.0):
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param initbands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    #if initbands > 8:  # 8 band imagery: SI := (1-Blue) * (1-Green) * (1-Red) = (1-B2) * (1-B3) * (1-B5)
    #    return ((1 - (data[1, :, :]/factor)) * (1 - (data[2, :, :]/factor)) * (1 - (data[4, :, :]/factor))
    #            ).expand_dims(dim="band", axis=0)
    #else:  # 4 band imagery: SI := (1-Blue) * (1-Green) * (1-Red) = (1-B3) * (1-B2) * (1-B1)
    # df['CI2'] = df.apply(lambda row: (row.Blue + row.Green + row.Red + row.NearIR1) / 4., axis=1)
    return ((data[3, :, :] + data[2, :, :] + data[1, :, :] + data[0, :, :]) / 4.0
                ).expand_dims(dim="band", axis=0)
