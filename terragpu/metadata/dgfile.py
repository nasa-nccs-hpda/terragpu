import os
import warnings
import xml.etree.ElementTree as ET
from terragpu.array.raster import Raster

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
        """
        Default DGFile initializer
        ----------
        Parameters
        ----------
        :param filename: string refering to NITF or TIF filename
        :param filename: string refering to XML filename
        :param logger: log file
        ----------
        Attributes
        ----------
        self.extension: string for raster filename extension (.nitf or .tif)
        self.xml_filename: string with XML filename
        self.imd_tag: XML object with IMD tag from XML file
        self.bandNameList: list of band names
        self.footprintsGml: footprints from XML file
        self.mean_sunaz: mean sun azimuth angle
        self.mean_sunel: mean sun elevation angle
        self.mean_sataz: mean satellite azimuth angle
        self.mean_satel: mean satellite elevation angle
        self.mean_intrack_viewangle: mean in track view angle
        self.mean_crosstrack_viewangle: mean cross track view angle
        self.mean_offnadir_viewangle: mean off nadir view angle
        ----------
        Example
            DGFile(filename)
        ----------
        """
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

        self.mean_sunaz = self.get_xml_tag(xml_tag='MEANSUNAZ')

        self.mean_sunel = self.get_xml_tag(xml_tag='MEANSUNEL')

        self.mean_sataz = self.get_xml_tag(xml_tag='MEANSATAZ')

        self.mean_satel = self.get_xml_tag(xml_tag='MEANSATEL')

        self.mean_intrack_viewangle = \
            self.get_xml_tag(xml_tag='MEANINTRACKVIEWANGLE')

        self.mean_crosstrack_viewangle = \
            self.get_xml_tag(xml_tag='MEANCROSSTRACKVIEWANGLE')

        self.mean_offnadir_viewangle = \
            self.get_xml_tag(xml_tag='MEANOFFNADIRVIEWANGLE')

    # ---------------------------------------------------------------------------
    # getField
    # ---------------------------------------------------------------------------
    # In Development
    def get_xml_tag(self, xml_tag=None):
        """
        :param xml_tag: string refering to XML tag
        :return: float value from XML tag
        ----------
        Example
            raster.get_xml_tag(xml_tag='MEANOFFNADIRVIEWANGLE')
        ----------
        """
        value = self.imd_tag.find('IMAGE').find(xml_tag)
        if value is not None:
            return float(value.text)
        else:
            warnings.warn('Unable to locate {}, return None.'.format(xml_tag))
            return None


# -------------------------------------------------------------------------------
# class DGFile Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Running Unit Tests
    print("Unit tests located under xrasterlib/tests/dgfile.py")
