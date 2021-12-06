import os
from terragpu import io

# import logging  # logging messages
# import operator  # operator library
# import numpy as np  # array manipulation library
# import xarray as xr  # array manipulation library, rasterio built-in
# import rasterio as rio  # geospatial library
# from scipy.ndimage import median_filter  # scipy includes median filter
# import rasterio.features as riofeat  # rasterio features include sieve filter

# -------------------------------------------------------------------------------
# class Raster
#
# This class represents, reads and manipulates rasters. Currently supports TIF
# formatted imagery only.
# -------------------------------------------------------------------------------


class Raster:

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename: str, bands: list = None,
                 chunks_band: int = 1, chunks_x: int = 2048,
                 chunks_y: int = 2048, logger=None):
        """
        Default Raster initializer
        Args:
            filename (str): raster filename to read from
            bands (str list): list of bands to append to object, e.g ['Red']
            chunks_band (int): integer to map object to memory, z
            chunks_x (int): integer to map object to memory, x
            chunks_y (int): integer to map object to memory, y
            logger (str): log file
        Attributes:
            logger (str): filename to store logs
            has_gpu (bool): global value to determine if GPU is available
            data_chunks (dict): dictionary to feed xarray rasterio object
            data (rasterio xarray): raster data stored in xarray type
            bands (list of str): band names (e.g Red Green Blue)
            nodataval (int): default no-data value used in the
        Return:
            raster (raster object): raster object to manipulate rasters
        ----------
        Example
        ----------
            Raster(filename, bands)
        """
        # Raster filename
        self.filename = filename
        
        # Raster bands to manipulate, useful for calculating indices
        self.bands = bands

        # Raster data chunks for Dask display
        self.data_chunks = {
            'band': chunks_band,
            'x': chunks_x,
            'y': chunks_y
        }

        self.raster = io.imread(filename)


    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # input methods
    # ---------------------------------------------------------------------------
    """
    def readraster(self, filename: str, bands: list, chunks_band: int = 1,
                   chunks_x: int = 2048, chunks_y: int = 2048):
        
        Read raster and append data to existing Raster object
        Args:
            filename (str): raster filename to read from
            bands (str list): list of bands to append to object, e.g ['Red']
            chunks_band (int): integer to map object to memory, z
            chunks_x (int): integer to map object to memory, x
            chunks_y (int): integer to map object to memory, y
        Return:
            raster (raster object): raster object to manipulate rasters
        ----------
        Example
        ----------
            raster.readraster(filename, bands)
        
        self.data_chunks = {'band': chunks_band, 'x': chunks_x, 'y': chunks_y}
        self.data = xr.open_rasterio(filename, chunks=self.data_chunks)
        self.bands = bands
        self.nodataval = self.data.attrs['nodatavals']

    # ---------------------------------------------------------------------------
    # preprocessing
    # ---------------------------------------------------------------------------
    def set_minimum(self, minimum=0):
        
        Remove lower boundary anomalous values from self.data
        Args:
            minimum (int): minimum allowed reflectance value in the dataset
        Return:
            raster object with boundaries fixed to lower boundary value
        ----------
        Example
        ----------
            raster.set_minimum(minimum=0) := get all values
            that satisfy the condition self.data > boundary (above 0).
        
        self.data = self.data.where(self.data > minimum, other=minimum)

    def set_maximum(self, maximum=10000):
        
        Remove upper boundary anomalous values from self.data
        Args:
            maximum (int): maximum allowed reflectance value in the dataset
        Return:
            raster object with boundaries fixed to maximum boundary value
        ----------
        Example
        ----------
            raster.set_maximum(maximum=10000) := get all values
            that satisfy the condition self.data > boundary (above 10000).
        
        self.data = self.data.where(self.data < maximum, other=maximum)

    def preprocess(self, op: str = '>', boundary: int = 0, subs: int = 0):
        
        (Deprecated) Remove anomalous values from self.data
        Args:
            op (str): operator string, currently <, and >
            boundary (int): boundary value for classifying as anomalous
            subs (int): value to replace with (int or float)
        Return:
            raster object with boundaries preprocessed
        ----------
        Example
        ----------
            raster.preprocess(op='>', boundary=0, replace=0) := get all values
            that satisfy the condition self.data > boundary (above 0).
        
        ops = {'<': operator.lt, '>': operator.gt}
        self.data = self.data.where(ops[op](self.data, boundary), other=subs)

    def addindices(self, indices: list, factor: float = 1.0):
        
        Add multiple indices to existing Raster object self.data
        Args:
            indices (int list): list of indices functions
            factor (int or float): atmospheric factor for indices calculation
        Return:
            raster object with new bands appended to self.data
        ----------
        Example
        ----------
            raster.addindices([indices.fdi, indices.si], factor=10000.0)
        
        nbands = len(self.bands)  # get initial number of bands
        for indices_function in indices:  # iterate over each new band
            nbands += 1  # counter for number of bands
            # calculate band (indices)
            band, bandid = \
                indices_function(self.data, bands=self.bands, factor=factor)
            self.bands.append(bandid)  # append new band id to list of bands
            band.coords['band'] = [nbands]  # add band indices to raster
            self.data = xr.concat([self.data, band], dim='band')  # concat band

        # update raster metadata, xarray attributes
        self.data.attrs['scales'] = [self.data.attrs['scales'][0]] * nbands
        self.data.attrs['offsets'] = [self.data.attrs['offsets'][0]] * nbands

    def dropindices(self, dropindices):
        
        Drop multiple indices to existing Raster object self.data.
        Args:
            dropindices (int list): list of indices to drop
        Return:
            raster object with dropped bands on self.data
        ----------
        Example
        ----------
            raster.dropindices(band_ids)
        
        assert all(band in self.bands for band in dropindices), \
               "Specified band not in raster."
        dropind = [self.bands.index(ind_id)+1 for ind_id in dropindices]
        self.data = self.data.drop(dim="band", labels=dropind, drop=True)
        self.bands = [band for band in self.bands if band not in dropindices]

    # ---------------------------------------------------------------------------
    # post processing methods
    # ---------------------------------------------------------------------------
    def sieve(self, prediction: np.array, out: np.array,
              size: int = 350, mask: str = None, connectivity: int = 8):
        
        Apply sieve filter to array object on single band (binary array)
        Args:
            prediction (array): numpy array with prediction output
            out (array): numpy array with prediction output to store on
            size (int): size of sieve
            mask (str): file to save at
            connectivity (int): size of sieve
        Return:
            raster object with sieve filtered output
        ----------
        Example
        ----------
            raster.sieve(raster.prediction, raster.prediction, size=sieve_sz)
       
        riofeat.sieve(prediction, size, out, mask, connectivity)

    def median(self, prediction: np.array, ksize: int = 20) -> np.array:
        
        Apply median filter for postprocessing
        Args:
            prediction (array): numpy array with prediction output
            ksize (int): size of kernel for median filter
        Return:
            raster object with median filtered output
        ----------
        Example
        ----------
            raster.median(raster.prediction, ksize=args.median_sz)
        
        if self.has_gpu:  # method for GPU
            with cp.cuda.Device(1):
                prediction = cp_medfilter(cp.asarray(prediction), size=ksize)
            return cp.asnumpy(prediction)
        else:  # method for CPU
            return median_filter(prediction, size=ksize)

    # ---------------------------------------------------------------------------
    # output methods
    # ---------------------------------------------------------------------------
    def toraster(self, rast: str, prediction: np.array,
                 dtype: str = 'int16', output: str = 'rfmask.tif'):
        
        Save tif file from numpy to disk.
        :param rast: raster name to get metadata from
        :param prediction: numpy array with prediction output
        :param dtype type to store mask on
        :param output: raster name to save on
        :return: None, tif file saved to disk
        ----------
        Example
            raster.toraster(filename, raster_obj.prediction, outname)
        ----------
        
        # get meta features from raster
        with rio.open(rast) as src:
            meta = src.profile
            nodatavals = src.read_masks(1).astype(dtype)
        logging.info(meta)

        nodatavals[nodatavals == 0] = self.nodataval[0]
        prediction[nodatavals == self.nodataval[0]] = \
            nodatavals[nodatavals == self.nodataval[0]]

        out_meta = meta  # modify profile based on numpy array
        out_meta['count'] = 1  # output is single band
        out_meta['dtype'] = dtype  # data type modification

        # write to a raster
        with rio.open(output, 'w', **out_meta) as dst:
            dst.write(prediction, 1)
        logging.info(f'Prediction saved at {output}')
    """

# -------------------------------------------------------------------------------
# class Raster Unit Tests
# -------------------------------------------------------------------------------


if __name__ == "__main__":

    # Running Unit Tests
    print("Unit tests located under xrasterlib/tests/raster.py")
