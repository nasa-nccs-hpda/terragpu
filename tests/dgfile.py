import logging  # logging messages
import unittest
from xrasterlib.dgfile import DGFile

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Test"

# -------------------------------------------------------------------------------
# tests for class DGFile
# Read DigiGlobe unique file by parsing XML tags and generating objects.
# Usage requirements are referenced in README.
# Refactored: 07/20/2020
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

class TestDGFileMethods(unittest.TestCase):

    # ---------------------------------------------------------------------------
    # class Raster Unit Tests
    # ---------------------------------------------------------------------------
    # 1. Create raster object with same location XML filename
    #    Verify retrieval of MEANSATAZ value
    def test_dgfile_MEANSATAZ(self):
        raster = DGFile(TIF_FILENAME)
        self.assertEqual(raster.mean_sataz, 74.85)

    # 2. Create raster object with different metadata filename
    #    Verify retrieval of MEANSUNAZ value
    def test_dgfile_MEANSUNAZ(self):
        raster = DGFile(TIF_FILENAME, xml_filename=XML_FILENAME)
        self.assertEqual(raster.mean_sunaz, 146.95)

    # 3. Create raster object with different metadata filename, CPU
    #    number of bands should be 8
    def test_dgfile_readraster(self):
        raster = DGFile(TIF_FILENAME, xml_filename=XML_FILENAME)
        raster.readraster(TIF_FILENAME, BANDS)  # read raster data
        self.assertEqual(raster.data.shape[0], 8)


if __name__ == '__main__':
    unittest.main()
