from datetime import datetime
import os
import subprocess
import shutil
import xml.etree.ElementTree as ET

from osgeo.osr import SpatialReference
from osgeo import gdal

from django.conf import settings

from EvhrEngine.management.GdalFile import GdalFile
from EvhrEngine.management.SystemCommand import SystemCommand

#-------------------------------------------------------------------------------
# class DgFile
#
# This class represents a Digital Globe file.  It is a single NITF file or a
# GeoTiff with an XML counterpart.  It is uniqure because of the metadata tags
# within.
#-------------------------------------------------------------------------------
class DgFile(GdalFile):

    #---------------------------------------------------------------------------
    # __init__
    #---------------------------------------------------------------------------
    def __init__(self, fileName, logger = None):

        # Check that the file is NITF or TIFF
        extension = os.path.splitext(fileName)[1]

        if extension != '.ntf' and extension != '.tif':
            raise RuntimeError('{} is not a NITF or TIFF file'.format(fileName))

        self.extension = extension

        # Ensure the XML file exists.
        xmlFileName = fileName.replace(self.extension, '.xml')

        if not os.path.isfile(xmlFileName):
            raise RuntimeError('{} does not exist'.format(xmlFileName))

        self.xmlFileName = xmlFileName
        
        # Initialize the base class.
        super(DgFile, self).__init__(fileName, logger)

        # These data member require the XML file counterpart to the TIF.
        tree = ET.parse(self.xmlFileName)
        self.imdTag = tree.getroot().find('IMD')

        if self.imdTag is None:

            raise RuntimeError('Unable to locate the "IMD" tag in ' + \
                               self.xmlFileName)

        # If srs from GdalFile is empty, set srs, and get coords from the .xml
        if not self.srs:

            self.srs = SpatialReference()
            self.srs.ImportFromEPSG(4326)
            
            """
            Below is a temporary fix until ASP fixes dg_mosaic bug:
             dg_mosaic outputs, along with a strip .tif, an aggregate .xml
             file for all scene inputs. The .tif has no projection information,
             so we have to get that from the output .xml. All bands *should* 
             have same extent in the .xml but a bug with ASP does not ensure 
             this is always true
              
             for 4-band mosaics, the output extent is consitent among all bands
             for 8-band mosaics, the first band (BAND_C) is not updated in the
             output .xml, so we have to use second band (BAND_B). 
            
            # if no bug, first BAND tag will work for 8-band, 4-band, 1-band
            bandTag = [n for n in self.imdTag.getchildren() if \
                    n.tag.startswith('BAND_')][0]               
            """                                                  
            try:
                bandTag = [n for n in self.imdTag.getchildren() if \
                    n.tag.startswith('BAND_B')][0] 
                
            except IndexError: # Pan only has BAND_P
                bandTag = [n for n in self.imdTag.getchildren() if \
                    n.tag.startswith('BAND_P')][0]

            self.ulx = min(float(bandTag.find('LLLON').text), \
                                          float(bandTag.find('ULLON').text))

            self.uly = max(float(bandTag.find('ULLAT').text), \
                                          float(bandTag.find('URLAT').text))

            self.lrx = max(float(bandTag.find('LRLON').text), \
                                          float(bandTag.find('URLON').text))

            self.lry = min(float(bandTag.find('LRLAT').text), \
                                          float(bandTag.find('LLLAT').text))

            GdalFile.validateCoordinates(self) # Lastly, validate coordinates

        # bandNameList
        try:
            self.bandNameList = \
                 [n.tag for n in self.imdTag if n.tag.startswith('BAND_')]
        except:
            self.bandNameList = None
    
        # numBands
        try:
            self.numBands = self.dataset.RasterCount

        except:
            self.numBands = None
            
        self.footprintsGml = None
            
    #---------------------------------------------------------------------------
    # abscalFactor()
    #---------------------------------------------------------------------------
    def abscalFactor(self, bandName):

        if isinstance(bandName, str) and bandName.startswith('BAND_'):

            return float(self.imdTag.find(bandName).find('ABSCALFACTOR').text)

        else:
            raise RuntimeError('Could not retrieve abscal factor.')

    #---------------------------------------------------------------------------
    # cloudCover()
    #---------------------------------------------------------------------------
    def cloudCover(self):

        try:
            cc = self.imdTag.find('IMAGE').find('CLOUDCOVER').text
            if cc is None:
                cc = self.dataset.GetMetadataItem('NITF_PIAIMC_CLOUDCVR')
            return float(cc)

        except:
            return None

    #---------------------------------------------------------------------------
    # effectiveBandwidth()
    #---------------------------------------------------------------------------
    def effectiveBandwidth(self, bandName):

        if isinstance(bandName, str) and bandName.startswith('BAND_'):

            return float(self.imdTag.      \
                           find(bandName). \
                           find('EFFECTIVEBANDWIDTH').text)

        else:
            raise RuntimeError('Could not retrieve effective bandwidth.')
          
    #---------------------------------------------------------------------------
    # firstLineTime()
    #---------------------------------------------------------------------------
    def firstLineTime(self):

        try:
            t = self.dataset.GetMetadataItem('NITF_CSDIDA_TIME')
            if t is not None:
                return datetime.strptime(t, "%Y%m%d%H%M%S")
            else:    
                t = self.imdTag.find('IMAGE').find('FIRSTLINETIME').text
                return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
        except:
            return None

    #---------------------------------------------------------------------------
    # getBand()
    #---------------------------------------------------------------------------
    def getBand(self, outputDir, bandName):

        gdalBandIndex = int(self.bandNameList.index(bandName)) + 1
       
        baseName = os.path.basename(self.fileName.replace(self.extension, \
                                            '_b{}.tif'.format(gdalBandIndex)))

        tempBandFile = os.path.join(outputDir, baseName)

        if not os.path.exists(tempBandFile):

            cmd = 'gdal_translate'                      + \
                  ' -b {}'.format(gdalBandIndex)        + \
                  ' -a_nodata 0'                        + \
                  ' -strict'                            + \
                  ' -mo "bandName={}"'.format(bandName) + \
                  ' ' + self.fileName                   + \
                  ' ' + tempBandFile

            sCmd = SystemCommand(cmd, self.fileName, self.logger)

            if sCmd.returnCode:
                tempBandFile = None
        
        # Copy scene .xml to accompany the extracted .tif (needed for dg_mosaic) 
        shutil.copy(self.xmlFileName, tempBandFile.replace('.tif', '.xml'))        

        return tempBandFile

    #---------------------------------------------------------------------------
    # getBandName()
    #---------------------------------------------------------------------------
    def getBandName(self):
        
        try:
            return self.dataset.GetMetadataItem('bandName')
        except:
            return None

    #---------------------------------------------------------------------------
    # getCatalogId()
    #---------------------------------------------------------------------------
    def getCatalogId(self):
        
        return self.imdTag.findall('./IMAGE/CATID')[0].text

    #---------------------------------------------------------------------------
    # getField
    #---------------------------------------------------------------------------
    def getField(self, nitfTag, xmlTag):
        
        try:
            
            value = self.dataset.GetMetadataItem(nitfTag)
            
            if not value:
                value = self.imdTag.find('IMAGE').find(xmlTag).text

            return float(value)

        except:
            return None
        
    #---------------------------------------------------------------------------
    # getStripName()
    #---------------------------------------------------------------------------
    def getStripName(self):
        
        try:
            prodCode = None
            
            if self.specTypeCode() == 'MS': 

                prodCode = 'M1BS'

            else: 
                prodCode = 'P1BS'
            
            dateStr = '{}{}{}'.format(self.year(),                             
                                      str(self.firstLineTime().month).zfill(2), 
                                      str(self.firstLineTime().day).zfill(2))

            return '{}_{}_{}_{}'.format(self.sensor(), 
                                        dateStr, 
                                        prodCode,      
                                        self.getCatalogId())

        except:
            return None   
            
    #---------------------------------------------------------------------------
    # isMultispectral()
    #---------------------------------------------------------------------------
    def isMultispectral(self):

        return self.specTypeCode() == 'MS'

    #---------------------------------------------------------------------------
    # isPanchromatic()
    #---------------------------------------------------------------------------
    def isPanchromatic(self):

        return self.specTypeCode() == 'PAN'

    #---------------------------------------------------------------------------
    # meanSatelliteAzimuth
    #---------------------------------------------------------------------------
    def meanSatelliteAzimuth(self):
        
        return self.getField('NITF_CSEXRA_AZ_OF_OBLIQUITY', 'MEANSATAZ')

    #---------------------------------------------------------------------------
    # meanSatelliteElevation
    #---------------------------------------------------------------------------
    def meanSatelliteElevation(self):
        
        return self.getField('', 'MEANSATEL')

    #---------------------------------------------------------------------------
    # meanSunAzimuth
    #---------------------------------------------------------------------------
    def meanSunAzimuth(self):
    
        return self.getField('NITF_CSEXRA_SUN_AZIMUTH', 'MEANSUNAZ')

    #---------------------------------------------------------------------------
    # meanSunElevation()
    #---------------------------------------------------------------------------
    def meanSunElevation(self):
    
        return self.getField('NITF_CSEXRA_SUN_ELEVATION', 'MEANSUNEL')

    #---------------------------------------------------------------------------
    # prodLevelCode()
    #---------------------------------------------------------------------------
    def prodLevelCode(self):

        try:
            return self.imdTag.find('PRODUCTLEVEL').text

        except:
            return None
     
    #---------------------------------------------------------------------------
    # sensor()
    #---------------------------------------------------------------------------
    def sensor(self):

        try:
            sens = self.dataset.GetMetadataItem('NITF_PIAIMC_SENSNAME')
            if sens is None:
                sens = self.imdTag.find('IMAGE').find('SATID').text

            return sens

        except:
	    return None

    #---------------------------------------------------------------------------
    # setBandName()
    #---------------------------------------------------------------------------
    def setBandName(self, bandName):
        
        self.dataset.SetMetadataItem("bandName", bandName)

    #---------------------------------------------------------------------------
    # specTypeCode()
    #---------------------------------------------------------------------------
    def specTypeCode(self):
        
        try:
            stc = self.dataset.GetMetadataItem('NITF_CSEXRA_SENSOR')
            if stc is None:
                if self.imdTag.find('BANDID').text == 'P':
                    stc = 'PAN'
                elif self.imdTag.find('BANDID').text == 'MS1' or \
                                    self.imdTag.find('BANDID').text == 'Multi':
                    stc = 'MS'
      
            return stc

        except:
          return None          
    
    #---------------------------------------------------------------------------
    # toBandInterleavedBinary()
    #---------------------------------------------------------------------------
    def toBandInterleavedBinary(self, outputDir):
   
        outBin = os.path.join(outputDir, 
               os.path.basename(self.fileName.replace(self.extension, '.bin')))

        try:
            ds = gdal.Open(self.fileName)
            ds = gdal.Translate(outBin, ds, creationOptions = ["INTERLEAVE=BAND"])
            ds = None
            return outBin

        except:
            return None

    #---------------------------------------------------------------------------
    # year()
    #---------------------------------------------------------------------------
    def year(self):

        try:
            yr = self.dataset.GetMetadataItem('NITF_CSDIDA_YEAR')
            if yr is None:
                yr = self.firstLineTime().year

            return yr
  
        except:
            return None