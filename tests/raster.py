import logging  # logging messages
from xrasterlib.raster import Raster
import xrasterlib.indices as indices  # custom indices calculation module

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# -------------------------------------------------------------------------------
# tests for class Raster
# This class represents, reads and manipulates raster. Currently support TIF
# formatted imagery.
# -------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------------
# Input:
# WV02_20181109_M1BS_1030010086582600-toa.tif
# WV02_20181109_M1BS_1030010086582600-toa.xml
# -------------------------------------------------------------------------------

# TIF filename, XML filename, and BANDS ids
TIF_FILENAME = '../data/WV02_20181109_M1BS_1030010086582600-toa.tif'
XML_FILENAME = '../data/WV02_20181109_M1BS_1030010086582600-toa.xml'
BANDS = [
    'CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2'
]

# Unit tests to run
UNIT_TESTS = [1, 2, 3, 4]

# -------------------------------------------------------------------------------
# class Raster Unit Tests
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Create raster object
    if 1 in UNIT_TESTS:
        raster = Raster(TIF_FILENAME, BANDS)
        assert raster.data.shape[0] == 8, "Number of bands should be 8."
        logging.info(f"Unit Test #1: {raster.data} {raster.bands}")

    # 2. Read raster file through method
    if 2 in UNIT_TESTS:
        raster = Raster()
        raster.readraster(TIF_FILENAME, BANDS)
        assert raster.data.shape[0] == 8, "Number of bands should be 8."
        logging.info(f"Unit Test #2: {raster.data} {raster.bands}")

    # 3. Test adding a band (indices) to raster.data - either way is fine
    if 3 in UNIT_TESTS:
        raster = Raster(TIF_FILENAME, BANDS)
        raster.addindices([indices.fdi, indices.si, indices.ndwi],
                          factor=10000.0)
        assert raster.data.shape[0] == 11, "Number of bands should be 11."
        logging.info(f"Unit Test #3: {raster.data} {raster.bands}")

    # 4. Test preprocess function
    if 4 in UNIT_TESTS:
        raster = Raster(TIF_FILENAME, BANDS)
        raster.preprocess(op='>', boundary=0, subs=0)
        vmin = raster.data.min().values
        assert vmin == 0, "Minimum should be 0."
        raster.preprocess(op='<', boundary=10000, subs=10000)
        vmax = raster.data.max().values
        assert vmax == 10000, "Maximum should be 10000."
        logging.info(f"Unit Test #4: (min, max) ({vmin},{vmax})")
