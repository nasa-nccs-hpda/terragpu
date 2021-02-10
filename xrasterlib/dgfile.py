import os
import warnings
import xml.etree.ElementTree as ET
from xrasterlib.raster import Raster

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# class DGFile
# Read DigiGlobe unique file by parsing XML tags and generating objects.
# Usage requirements are referenced in README.
# Refactored: 07/20/2020
# -------------------------------------------------------------------------------


class DGFile(Raster):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename, xml_filename=None, logger=None):

        # Initialize super class
        super().__init__()

        # Check that the file has NITF or TIFF extension
        extension = os.path.splitext(filename)[1]
        if extension != '.ntf' and extension != '.tif':
            raise RuntimeError(
                '{} is not a NITF or TIFF file'.format(filename)
            )
        self.extension = extension  # Set filename extension

        # Ensure the XML file exists
        if not xml_filename:  # Set matching filename if XML is not provided
            xml_filename = filename.replace(self.extension, '.xml')

        if not os.path.isfile(xml_filename):  # Exit if XML file is not found
            raise RuntimeError('{} does not exist'.format(xml_filename))

        self.xml_filename = xml_filename  # Set XML filename

        # These data member require the XML file counterpart to the TIF
        tree = ET.parse(self.xml_filename)
        self.imd_tag = tree.getroot().find('IMD')

        if self.imd_tag is None:
            raise RuntimeError(
                'Unable to locate the "IMD" tag in {}'.format(xml_filename)
            )

        # Get bandNameList from Digital Globe XML file
        try:
            self.bandNameList = \
                 [n.tag for n in self.imd_tag if n.tag.startswith('BAND_')]
        except ValueError:
            self.bandNameList = None

        self.footprintsGml = None

        self.MEANSUNAZ = self.get_xml_tag(xml_tag='MEANSUNAZ')

        self.MEANSUNEL = self.get_xml_tag(xml_tag='MEANSUNEL')

        self.MEANSATAZ = self.get_xml_tag(xml_tag='MEANSATAZ')

        self.MEANSATEL = self.get_xml_tag(xml_tag='MEANSATEL')

        self.MEANINTRACKVIEWANGLE = \
            self.get_xml_tag(xml_tag='MEANINTRACKVIEWANGLE')

        self.MEANCROSSTRACKVIEWANGLE = \
            self.get_xml_tag(xml_tag='MEANCROSSTRACKVIEWANGLE')

        self.MEANOFFNADIRVIEWANGLE = \
            self.get_xml_tag(xml_tag='MEANOFFNADIRVIEWANGLE')

    # ---------------------------------------------------------------------------
    # getField
    # ---------------------------------------------------------------------------
    # In Development
    def get_xml_tag(self, xml_tag=None):
        """
        :param xml_tag: string refering to XML tag
        :return: float value from XML tag
        """
        value = self.imd_tag.find('IMAGE').find(xml_tag)
        if value is not None:
            return float(value.text)
        else:
            warnings.warn('Unable to locate {}, return None.'.format(xml_tag))
            return value


# -------------------------------------------------------------------------------
# class DGFile Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Running Unit Tests
    print("Unit tests located under xrasterlib/tests/dgfile.py")
