import logging  # logging messages
import unittest
from xrasterlib.raster import Raster
import xrasterlib.indices as indices  # custom indices calculation module

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Test"

# -------------------------------------------------------------------------------
# tests for class Raster
# This class represents, reads and manipulates raster. Currently support TIF
# formatted imagery.
# -------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO)

# TIF filename, XML filename, and BANDS ids
TIF_FILENAME = '../data/WV02_20181109_M1BS_1030010086582600-toa.tif'
XML_FILENAME = '../data/WV02_20181109_M1BS_1030010086582600-toa.xml'
BANDS = [
    'CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2'
]

# Unit tests to run
UNIT_TESTS = [1, 2, 3, 4]


# -------------------------------------------------------------------------------
# Input:
# WV02_20181109_M1BS_1030010086582600-toa.tif
# WV02_20181109_M1BS_1030010086582600-toa.xml
# -------------------------------------------------------------------------------

class TestRasterMethods(unittest.TestCase):

    # ---------------------------------------------------------------------------
    # class Raster Unit Tests
    # ---------------------------------------------------------------------------
    # 1. Create Raster object, number of bands should be 8, CPU
    def test_raster_init(self):
        raster = Raster(filename=TIF_FILENAME, bands=BANDS)
        self.assertEqual(raster.data.shape[0], 8)

    # 2. Read raster file through method, number of bands should be 8, CPU
    def test_raster_readraster(self):
        raster = Raster()
        raster.readraster(filename=TIF_FILENAME, bands=BANDS)
        self.assertEqual(raster.data.shape[0], 8)

    # 3. Test adding a band (indices) to raster.data - either way is fine
    #    number of bands should be 11, CPU
    def test_raster_addindices(self):
        raster = Raster(filename=TIF_FILENAME, bands=BANDS)
        raster.addindices(
            [indices.fdi, indices.si, indices.ndwi], factor=10000.0
        )
        self.assertEqual(raster.data.shape[0], 11)

    # Deprecated since 0.0.3
    # Test preprocess function, CPU
    # def test_raster_preprocess(self):
    #    raster = Raster(filename=TIF_FILENAME, bands=BANDS)
    #    raster.preprocess(op='>', boundary=0, subs=0)
    #    raster.preprocess(op='<', boundary=10000, subs=10000)
    #    vmin = raster.data.min().values
    #    vmax = raster.data.max().values
    #    self.assertEqual([vmin, vmax], [0, 10000])

    # 4. Test minimum function, CPU
    #    min should be 0, max should be 10000
    def test_raster_set_min_max(self):
        raster = Raster(filename=TIF_FILENAME, bands=BANDS)
        raster.set_minimum(minimum=0)
        raster.set_maximum(maximum=10000)
        vmin = raster.data.min().values
        vmax = raster.data.max().values
        self.assertEqual([vmin, vmax], [0, 10000])


if __name__ == '__main__':
    unittest.main()
