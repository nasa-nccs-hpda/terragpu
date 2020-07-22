# Cloud mask test
# identify polygon
import sys, os            # system modifications
import numpy as np        # for arrays modifications
from time import time     # tracking time
import cv2
from osgeo import gdal

image_file = 'results/Classified/20_log2__WV02_20140716_M1BS_103001003328DB00-toa__classified.tif'

ds         = gdal.Open(image_file)
raster     = np.array(ds.GetRasterBand(1).ReadAsArray())
raster[raster < 0] = 0

print (raster.shape, type(raster))
print (np.unique(raster))

#gray = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY)
gray = raster
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

import matplotlib.pyplot as plt 
plt.imsave('test.png', thresh)
