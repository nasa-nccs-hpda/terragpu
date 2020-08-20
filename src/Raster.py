# -------------------------------------------------------------------------------
# class Raster
# This class performs operations over raster objects.
# -------------------------------------------------------------------------------
import sys  # system library
import xarray as xr  # read rasters
import rasterio as rio  # geospatial library
import rasterio.features as riofeat

import indices

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587"
__email__  = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


class Raster:

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename=None, logger=None):

        # get raster data - xarray object memory mapped
        if filename is not None:
            self.data = xr.open_rasterio(filename, chunks={'band': 1, 'x': 2048, 'y': 2048})
        else:
            self.data = filename

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------
    def readraster(self, filename):
        self.data = xr.open_rasterio(filename, chunks={'band': 1, 'x': 2048, 'y': 2048})

    def addindices(self, indices, factor=1.0):
        nbands = self.data.shape[0]  # get number of bands
        for indfunc in indices:  # iterate over each new band
            nbands = nbands + 1  # counter for number of bands
            band = indfunc(self.data, factor=factor)  # calculate band (indices)
            band.coords['band'] = [nbands]  # add band indices to raster
            self.data = xr.concat([self.data, band], dim='band')  # concat new band
        self.data.attrs['scales'] = [self.data.attrs['scales'][0]] * nbands
        self.data.attrs['offsets'] = [self.data.attrs['offsets'][0]] * nbands

    def sieve(self, prediction, out, size=350, mask=None, connectivity=8):
        riofeat.sieve(prediction, size, out, mask, connectivity)

    def toraster(self, rast, prediction, output='rfmask.tif'):
        """
        :param rast: raster name to get metadata from
        :param prediction: numpy array with prediction output
        :param output: raster name to save on
        :return: tif file saved to disk
        """
        # get meta features from raster
        with rio.open(rast) as src:
            meta = src.profile
            print(meta)

        out_meta = meta  # modify profile based on numpy array
        out_meta['count'] = 1  # output is single band
        out_meta['dtype'] = 'int16'  # data type is float64

        # write to a raster
        with rio.open(output, 'w', **out_meta) as dst:
            dst.write(prediction, 1)
        print(f'Prediction saved at {output}.')


# -------------------------------------------------------------------------------
# class Raster Unit Tests
# -------------------------------------------------------------------------------


if __name__ == "__main__":

    # Running the script
    # python Raster.py /Users/username/Desktop/cloud-mask-data/WV02_20140716_M1BS_103001003328DB00-toa.tif

    # 1. Create raster object
    raster = Raster(sys.argv[1])
    print(raster.data)

    # 2. Test adding a band (indices) to raster.data - either way is fine
    # raster.data = indices.addindices(raster.data, [indices.si], factor=10000.0)  # call method from indices
    raster.addindices([indices.si], factor=10000.0)  # call Raster method
    print(raster.data)

    # 3. Test adding multiple bands (indices) to raster.data - either way is fine
    # raster.data = indices.addindices(raster.data, [indices.si, indices.fdi, indices.dvi], factor=10000.0)
    raster.addindices([indices.si, indices.fdi, indices.dvi], factor=10000.0)  # call Raster method
    print(raster.data)

    # 4. Read raster file through method
    raster.readraster(sys.argv[1])
    print(raster.data)
