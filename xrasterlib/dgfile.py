import os
import logging
import xml.etree.ElementTree as ET
from xrasterlib.raster import Raster

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# class RF
# Read DigiGlobe unique file by parsing XML tags and generating objects.
# Usage requirements are referenced in README.
# Refactored: 07/20/2020
# -------------------------------------------------------------------------------


class DGFile(Raster):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename, xml_filename='', logger=None):
        super().__init__()

        # Check that the file is NITF or TIFF
        extension = os.path.splitext(filename)[1]

        if extension != '.ntf' and extension != '.tif':
            raise RuntimeError(
                '{} is not a NITF or TIFF file'.format(filename)
            )

        self.extension = extension

        # Ensure the XML file exists.
        if not xml_filename:
            xml_filename = filename.replace(self.extension, '.xml')

        if not os.path.isfile(xml_filename):
            raise RuntimeError('{} does not exist'.format(xml_filename))

        self.xml_filename = xml_filename

        # These data member require the XML file counterpart to the TIF.
        tree = ET.parse(self.xml_filename)
        self.imd_tag = tree.getroot().find('IMD')

        if self.imd_tag is None:
            raise RuntimeError(
                'Unable to locate the "IMD" tag in {}'.format(xml_filename)
            )

        # bandNameList
        try:
            self.bandNameList = \
                 [n.tag for n in self.imd_tag if n.tag.startswith('BAND_')]
        except ValueError:
            self.bandNameList = None

        self.footprintsGml = None

        self.MEANSUNAZ = float(
            self.imd_tag.find('IMAGE').find('MEANSUNAZ').text
        )
        self.MEANSUNEL = float(
            self.imd_tag.find('IMAGE').find('MEANSUNEL').text
        )
        self.MEANSATAZ = float(
            self.imd_tag.find('IMAGE').find('MEANSATAZ').text
        )
        self.MEANSATEL = float(
            self.imd_tag.find('IMAGE').find('MEANSATEL').text
        )
        self.MEANINTRACKVIEWANGLE = float(
            self.imd_tag.find('IMAGE').find('MEANINTRACKVIEWANGLE').text
        )
        self.MEANCROSSTRACKVIEWANGLE = float(
            self.imd_tag.find('IMAGE').find('MEANCROSSTRACKVIEWANGLE').text
        )
        self.MEANOFFNADIRVIEWANGLE = float(
            self.imd_tag.find('IMAGE').find('MEANOFFNADIRVIEWANGLE').text
        )

    # ---------------------------------------------------------------------------
    # getField
    # ---------------------------------------------------------------------------
    # In Development
    def get_field(self, nitf_tag, xml_tag):

        try:
            value = self.dataset.GetMetadataItem(nitf_tag)

            if not value:
                value = self.imd_tag.find('IMAGE').find(xml_tag).text

            return float(value)

        except ValueError:
            return None


# -------------------------------------------------------------------------------
# class DGFile Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Running Unit Tests
    logging.basicConfig(level=logging.INFO)

    # Local variables
    tif = '/Users/jacaraba/Desktop/CLOUD/cloudtest/' + \
        'WV02_20181109_M1BS_1030010086582600-toa.tif'

    xml = '/Users/jacaraba/Desktop/CLOUD/cloudtest/' + \
        'WV02_20181109_M1BS_1030010086582600-toa.xml'

    bands = [
        'CoastalBlue', 'Blue', 'Green', 'Yellow',
        'Red', 'RedEdge', 'NIR1', 'NIR2'
    ]

    # number of unit tests to run
    unit_tests = [1, 2, 3]

    # 1. Create raster object with same metadata filename
    #    (only replaces .tif with .xml)
    if 1 in unit_tests:
        raster_obj = DGFile(tif)
        print(raster_obj.MEANSATAZ)
        assert raster_obj.MEANSATAZ == 74.85, "Mean SAT AZ should be 74.85"
        logging.info(f"UT #1 PASS: {raster_obj.MEANSATAZ}")

    # 2. Create raster object with different metadata filename
    if 2 in unit_tests:
        raster_obj = DGFile(tif, xml_filename=xml)
        print(raster_obj.MEANSATAZ)
        assert raster_obj.MEANSATAZ == 74.85, "Mean SAT AZ should be 74.85"
        logging.info(f"UT #2 PASS: {raster_obj.MEANSATAZ}")

    # 3. Create raster object with different metadata filename
    if 3 in unit_tests:
        raster_obj = DGFile(tif, xml_filename=xml)  # read metadata
        raster_obj.readraster(tif, bands)  # read raster data
        assert raster_obj.data.shape[0] == 8, "Number of bands should be 8."
        logging.info(f"UT #3 PASS: {raster_obj.data} {raster_obj.bands}")
