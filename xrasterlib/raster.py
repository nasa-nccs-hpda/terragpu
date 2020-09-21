import os  # system library
import logging  # logging messages
import operator  # operator library
import xarray as xr  # array manipulation library, rasterio built-in
import rasterio as rio  # geospatial library
from scipy.ndimage import median_filter  # scipy includes median filter
from cupyx.scipy.ndimage import median_filter as cp_medfilter  # cupy scipy modules
import cupy as cp
import dask.array as da
import rasterio.features as riofeat  # rasterio features include sieve filter
import xrasterlib.indices as indices  # custom indices calculation module

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# class Raster
#
# This class represents, reads and manipulates raster. Currently support TIF
# formatted imagery.
# -------------------------------------------------------------------------------


class Raster:

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename=None, bands=None, chunks_band=1,
                 chunks_x=2048, chunks_y=2048, logger=None):
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

        if filename is not None:  # if filename is provided, read into xarray

            if not os.path.isfile(filename):
                raise RuntimeError('{} does not exist'.format(filename))

            self.data_chunks = {
                'band': chunks_band,
                'x': chunks_x,
                'y': chunks_y
            }

            self.data = xr.open_rasterio(filename, chunks=self.data_chunks)

            if bands is None:
                raise RuntimeError('Must specify band names.')

            self.bands = bands

            self.nodataval = self.data.attrs['nodatavals']

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # input
    # ---------------------------------------------------------------------------
    def readraster(self, filename, bands, chunks_band=1,
                   chunks_x=2048, chunks_y=2048):
        """
        Read raster and append data to existing Raster object
        ----------
        Parameters
        ----------
        filename : str
            Raster filename to read from
        """
        self.data_chunks = {'band': chunks_band, 'x': chunks_x, 'y': chunks_y}
        self.data = xr.open_rasterio(filename, chunks=self.data_chunks)
        self.bands = bands
        self.nodataval = self.data.attrs['nodatavals']

    # ---------------------------------------------------------------------------
    # preprocessing
    # ---------------------------------------------------------------------------
    def preprocess(self, op='>', boundary=0, subs=0):
        """
        Remove anomalous values from self.data
        ----------
        Parameters
        ----------
        op : str with operator, currently <, and >
            string with operator value
        boundary : int
            boundary for classifying as anomalous
        replace : int, float
            Value to replace with
        ----------
        Example
        ----------
            raster.preprocess(op='>', boundary=0, replace=0) := get all values
            that satisfy the condition self.data > boundary (above 0).
        """
        ops = {'<': operator.lt, '>': operator.gt}
        #with cp.cuda.Device(1):
        #data = (cp.asarray(data[NIR, :, :])
        #self.data = self.data.where(ops[op](self.data, boundary), other=subs)
        #self.data = self.data.where(ops[op](cp.asarray(self.data), boundary), other=subs)
        self.data = da.where(ops[op](self.data, boundary), self.data, subs).compute()

    def addindices(self, indices, factor=1.0):
        """
        Add multiple indices to existing Raster object self.data.
        ----------
        Parameters
        ----------
        indices : list of functions
            Function reference to calculate indices
        factor : float
            Atmospheric factor for indices calculation
        """
        nbands = len(self.bands)  # get initial number of bands
        print("entering indices")
        for indices_function in indices:  # iterate over each new band
            nbands += 1  # counter for number of bands

            # calculate band (indices)
            band, bandid = indices_function(self.data,
                                            bands=self.bands, factor=factor)
            #band, bandid = da.map_blocks(indices_function, self.data, self.bands, dtype='int16')
            #print (band, bandid)

            print("calculated indices")
            self.bands.append(bandid)  # append new band id to list of bands
            #band.coords['band'] = [nbands]  # add band indices to raster
            #self.data = xr.concat([self.data, band], dim='band')  # concat band
            self.data = cp.concatenate((self.data, band), axis=0)
            print("concatenated indices", self.data.shape)
        self.data = cp.nan_to_num(self.data)
        print ("type of data after indices ", type(self.data))
        # update raster metadata, xarray attributes
        #self.data.attrs['scales'] = [self.data.attrs['scales'][0]] * nbands
        #self.data.attrs['offsets'] = [self.data.attrs['offsets'][0]] * nbands

    def dropindices(self, dropindices):
        assert all(band in self.bands for band in dropindices), \
               "Specified band not in raster."
        dropind = [self.bands.index(ind_id)+1 for ind_id in dropindices]
        self.data = self.data.drop(dim="band", labels=dropind, drop=True)
        self.bands = [band for band in self.bands if band not in dropindices]

    # ---------------------------------------------------------------------------
    # post processing
    # ---------------------------------------------------------------------------
    def sieve(self, prediction, out, size=350, mask=None, connectivity=8):
        riofeat.sieve(prediction, size, out, mask, connectivity)

    def median(self, prediction, ksize=20):
        # method for CPU
        # return median_filter(prediction, size=ksize)
        with cp.cuda.Device(1):
            prediction = cp_medfilter(cp.asarray(prediction), size=ksize)
        return cp.asnumpy(prediction)
    # ---------------------------------------------------------------------------
    # output
    # ---------------------------------------------------------------------------
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
        logging.info(meta)

        nodatavals[nodatavals == 0] = self.nodataval[0]
        prediction[nodatavals == self.nodataval[0]] = \
            nodatavals[nodatavals == self.nodataval[0]]

        out_meta = meta  # modify profile based on numpy array
        out_meta['count'] = 1  # output is single band
        out_meta['dtype'] = 'int16'  # data type is float64

        # write to a raster
        with rio.open(output, 'w', **out_meta) as dst:
            dst.write(prediction, 1)
        logging.info(f'Prediction saved at {output}')

# -------------------------------------------------------------------------------
# class Raster Unit Tests
# -------------------------------------------------------------------------------


if __name__ == "__main__":

    # Running Unit Tests
    logging.basicConfig(level=logging.INFO)

    # Local variables
    filename = '/Users/jacaraba/Desktop/cloudtest/' + \
               'WV02_20181109_M1BS_1030010086582600-toa.tif'
    bands = [
        'CoastalBlue', 'Blue', 'Green', 'Yellow',
        'Red', 'RedEdge', 'NIR1', 'NIR2'
    ]
    unit_tests = [1, 2, 3, 4]

    # 1. Create raster object
    if 1 in unit_tests:
        raster = Raster(filename, bands)
        assert raster.data.shape[0] == 8, "Number of bands should be 8."
        logging.info(f"Unit Test #1: {raster.data} {raster.bands}")

    # 2. Read raster file through method
    if 2 in unit_tests:
        raster = Raster()
        raster.readraster(filename, bands)
        assert raster.data.shape[0] == 8, "Number of bands should be 8."
        logging.info(f"Unit Test #2: {raster.data} {raster.bands}")

    # 3. Test adding a band (indices) to raster.data - either way is fine
    if 3 in unit_tests:
        raster = Raster(filename, bands)
        raster.addindices([indices.fdi, indices.si, indices.ndwi],
                          factor=10000.0)
        assert raster.data.shape[0] == 11, "Number of bands should be 11."
        logging.info(f"Unit Test #3: {raster.data} {raster.bands}")

    # 4. Test preprocess function
    if 4 in unit_tests:
        raster = Raster(filename, bands)
        raster.preprocess(op='>', boundary=0, subs=0)
        vmin = raster.data.min().values
        assert vmin == 0, "Minimum should be 0."
        raster.preprocess(op='<', boundary=10000, subs=10000)
        vmax = raster.data.max().values
        assert vmax == 10000, "Maximum should be 10000."
        logging.info(f"Unit Test #4: (min, max) ({vmin},{vmax})")
