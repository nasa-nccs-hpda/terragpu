# Cloud mask test
# identify polygon: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
import sys, os            # system modifications
import numpy as np        # for arrays modifications
from time import time     # tracking time
import cv2
from osgeo import gdal

image_file = 'results/Classified/20_log2__WV02_20140716_M1BS_103001003328DB00-toa__classified.tif'

ds = gdal.Open(image_file)
raster = np.array(ds.GetRasterBand(1).ReadAsArray())
raster[raster < 0] = 0

median = cv2.medianBlur(raster, 5)
cv2.imwrite("median.png", median)

imgray = np.uint8(raster * 255)


ret, thresh = cv2.threshold(imgray, 127, 255, 0)
print (np.unique(thresh), "unique tresh", thresh.shape)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print (len(contours))

"""
## Draws contour correctly
conts = np.zeros((raster.shape[0], raster.shape[0], 3))
print (conts.shape)
conts = cv2.drawContours(conts, contours, -1, (0,255,0), 3)

cv2.imwrite("drawCont.png", conts)
print (np.unique(conts), "unique consts")
"""
"""
# trying to find centroid
for c in contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    print (cX, cY)
"""
