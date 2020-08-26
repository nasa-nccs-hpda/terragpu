"""
Refactored: 07/20/2020

Purpose: Read DigiGlobe unique file by parsing XML tags and generating objects.
         Usage requirements are referenced in README.
         
Data Source: This script is intented to be used with DigiGlobe files. There is 
             no intentions on supporting other types of file formats.

Original Author: Margaret Wooten, SCIENTIFIC PROGRAMMER/ANALYST, Code 610
Refactored: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""
#-------------------------------------------------------------------------------
# Import System Libraries
#-------------------------------------------------------------------------------
from datetime import datetime           
import os, subprocess                     
import shutil
import xml.etree.ElementTree as ET

#from osgeo.osr import SpatialReference
#from osgeo import gdal

import xarray as xr            # read rasters
import rasterio as rio         # geotiff manipulation

#-------------------------------------------------------------------------------
# class DgFile
#
# This class represents a Digital Globe file.  It is a single NITF file or a
# GeoTiff with an XML counterpart.  It is unique because of the metadata tags
# within.
#-------------------------------------------------------------------------------
class DGFileMem:

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, filename, logger=None):

        # Check that the file is NITF or TIFF
        extension = os.path.splitext(filename)[1]

        if extension != '.ntf' and extension != '.tif':
            raise RuntimeError('{} is not a NITF or TIFF file'.format(filename))

        self.extension = extension

        # Ensure the XML file exists.
        xml_filename = filename.replace(self.extension, '.xml')

        if not os.path.isfile(xml_filename):
            raise RuntimeError('{} does not exist'.format(xml_filename))

        self.xml_filename = xml_filename

        # These data member require the XML file counterpart to the TIF.
        tree = ET.parse(self.xml_filename)
        self.imd_tag = tree.getroot().find('IMD')

        if self.imd_tag is None:
            raise RuntimeError(f'Unable to locate the "IMD" tag in {self.xml_filename}')

        # bandNameList
        try:
            self.bandNameList = \
                 [n.tag for n in self.imd_tag if n.tag.startswith('BAND_')]
        except:
            self.bandNameList = None

        # get raster data
        self.dataset = xr.open_rasterio(filename, chunks={'band': 1, 'x': 2048, 'y': 2048})

        # get number of bands
        try:
            self.numBands = self.dataset.shape[0]
        except:
            self.numBands = None

        self.footprintsGml = None

    # ---------------------------------------------------------------------------
    # getField
    # ---------------------------------------------------------------------------
    # IN PROGRESS
    def get_field(self, nitf_tag, xml_tag):

        try:

            value = self.dataset.GetMetadataItem(nitf_tag)

            if not value:
                value = self.imd_tag.find('IMAGE').find(xml_tag).text

            return float(value)

        except:
            return None
