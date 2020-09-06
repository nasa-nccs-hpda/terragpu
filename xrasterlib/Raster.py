import sys  # system library
import os  # system library
import xarray as xr  # array manipulation library, rasterio built-in
import rasterio as rio  # geospatial library
import rasterio.features as riofeat  # rasterio features include sieve filter
import indices  # custom indices calculation module

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# class Raster
#
# This class represents reads and manipulates raster. Currently support TIF
# formatted imagery.
# -------------------------------------------------------------------------------


class Raster:

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename=None, bands=None, logger=None):
        """
        Default Raster initializer
        ----------
        Parameters
        ----------
        filename : str
            Raster filename to read from
        bands : list of str
            Band names - Red Green Blue etc.
        ----------
        Attributes
        ----------
        self.data : xarray, rasterio type
            Raster data stored in xarray type
        self.bands : list of str
            Band names - Red Green Blue etc.
        self.nodataval : int
            Default no-data value used in the
        """

        self.logger = logger

        if filename is not None:  # if filename is provided, read raster into xarray object

            if not os.path.isfile(filename):
                raise RuntimeError('{} does not exist'.format(filename))
            self.data = xr.open_rasterio(filename, chunks={'band': 1, 'x': 2048, 'y': 2048})

            if bands is None:
                raise RuntimeError('Must specify band names. Refer to documentation for details.')
            self.bands = bands

            self.nodataval = self.data.attrs['nodatavals']

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------
    def readraster(self, filename, bands):
        """
        Read raster and append data to existing Raster object
        ----------
        Parameters
        ----------
        filename : str
            Raster filename to read from
        """
        self.data = xr.open_rasterio(filename, chunks={'band': 1, 'x': 2048, 'y': 2048})
        self.bands = bands
        self.nodataval = self.data.attrs['nodatavals']

    def addindices(self, indices, factor=1.0):
        """
        Add multiple indices to existing Raster object self.data
        ----------
        Parameters
        ----------
        indices : list of functions
            Function reference to calculate indices
        factor : float
            Atmospheric factor for indices calculation
        """
        nbands = len(self.bands)  # get initial number of bands
        for indices_function in indices:  # iterate over each new band
            nbands += 1  # counter for number of bands

            # calculate band (indices)
            band, bandid = indices_function(self.data, bands=self.bands, factor=factor)
            self.bands.append(bandid)  # append new band id to list of bands
            band.coords['band'] = [nbands]  # add band indices to raster
            self.data = xr.concat([self.data, band], dim='band')  # concat new band

        # update raster metadata, xarray attributes
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
            nodatavals = src.read_masks(1).astype('int16')
        print(meta)

        nodatavals[nodatavals == 0] = self.nodataval[0]
        prediction[nodatavals == self.nodataval[0]] = nodatavals[nodatavals == self.nodataval[0]]

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

    # Running Unit Tests
    # python raster.py /Users/jacaraba/Desktop/cloudtest/WV02_20181109_M1BS_1030010086582600-toa.tif

    # Local variables
    filename = sys.argv[1]
    bands = ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2']
    unit_tests = [1, 2, 3, 4, 5]

    # 1. Create raster object
    if 1 in unit_tests:
        raster = Raster(filename, bands)
        assert raster.data.shape[0] == 8, "Number of bands should be 8."
        print("Unit Test #1: ", raster.data, raster.bands)

    # 2. Read raster file through method
    if 2 in unit_tests:
        raster = Raster()
        raster.readraster(filename, bands)
        assert raster.data.shape[0] == 8, "Number of bands should be 8."
        print("Unit Test #2: ", raster.data, raster.bands)

    # 3. Test adding a band (indices) to raster.data - either way is fine
    if 3 in unit_tests:
        raster = Raster(filename, bands)
        raster.addindices([indices.fdi, indices.si, indices.ndwi], factor=10000.0)  # call method from Raster
        assert raster.data.shape[0] == 11, "Number of bands should be 11."
        #raster.data = raster.addindices(raster.data, [indices.si], factor=10000.0)  # call method from indices
        print("Unit Test #3: ", raster.data, raster.bands)




