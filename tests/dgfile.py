import logging
from xrasterlib.dgfile import DGFile

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# -------------------------------------------------------------------------------
# tests for class DGFile
# Read DigiGlobe unique file by parsing XML tags and generating objects.
# Usage requirements are referenced in README.
# Refactored: 07/20/2020
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
UNIT_TESTS = [1, 2, 3]

# -------------------------------------------------------------------------------
# class DGFile Unit Tests
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Create raster object with same location XML filename
    #    Verify retrieval of MEANSATAZ value
    if 1 in UNIT_TESTS:
        raster_obj = DGFile(TIF_FILENAME)
        assert raster_obj.MEANSATAZ == 74.85, \
            f"Mean SAT AZ should be 74.85, got {raster_obj.MEANSATAZ}"
        logging.info(f"UT #1 PASS: {raster_obj.MEANSATAZ}")

    # 2. Create raster object with different metadata filename
    #    Verify retrieval of MEANSUNAZ value
    if 2 in UNIT_TESTS:
        raster_obj = DGFile(TIF_FILENAME, xml_filename=XML_FILENAME)
        assert raster_obj.MEANSUNAZ == 146.95, \
            f"Mean SUN AZ should be 146.95, got {raster_obj.MEANSUNAZ}"
        logging.info(f"UT #2 PASS: {raster_obj.MEANSUNAZ}")

    # 3. Create raster object with different metadata filename
    if 3 in UNIT_TESTS:
        raster_obj = DGFile(TIF_FILENAME, xml_filename=XML_FILENAME)
        raster_obj.readraster(TIF_FILENAME, BANDS)  # read raster data
        assert raster_obj.data.shape[0] == 8, "Number of bands should be 8."
        logging.info(f"UT #3 PASS: {raster_obj.data} {raster_obj.bands}")
