from datetime import datetime  # tracking date
from time import time  # tracking time
import sys, os, argparse  # system libraries
import numpy as np  # for arrays modifications
import heapq
import math

# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)

# Add src path to system PATH
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/src')
sys.path.append('/Users/jacaraba/Documents/Development/nccs-rasterlib/src')

import indices
from RandomForest import RandomForest
from DGFileRast import DGFileRast
from Raster import Raster
import cv2

# specify data files
raster = '/Users/jacaraba/Desktop/cloudtest/WV02_20181109_M1BS_1030010086582600-toa.tif'
cloudmask = '/Users/jacaraba/Desktop/cloudtest/cm_WV02_20181109_M1BS_1030010086582600-toa.tif'

# create object to read raster, mask and xml
rfobj_raster = DGFileRast(raster)
rfobj_raster.readraster(raster)

rfobj_mask = Raster(cloudmask)

print(rfobj_raster.data.shape, rfobj_raster.initbands, rfobj_raster.MEANSUNAZ)

# Get contour objects
rfobj_mask = np.uint8(np.squeeze(rfobj_mask.data.values) * 255)
ret, thresh = cv2.threshold(rfobj_mask, 127, 255, 0)
print (np.unique(thresh), "unique tresh", thresh.shape)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))

# draws contour correctly
conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))
print (conts.shape)
conts = cv2.drawContours(conts, contours[5768], -1, (0,255,0), 3)

# trying to find centroid
#sizes = list()
#counter = 0
#for c in contours:
#    M = cv2.moments(c)
#    if M["m00"] != 0:
#        cX = int(M["m10"] / M["m00"])
#        cY = int(M["m01"] / M["m00"])
#    else:
#        cX, cY = 0, 0
#    print (counter, cX, cY, c)
#    print (len(c))
#    sizes.append(len(c))
#    counter = counter + 1

# get n largest elements (might remove some elements with small number of components
# print(heapq.nlargest(5, range(len(sizes)), sizes.__getitem__))
# print(sizes.index(max(sizes)))

# took one cloud to test
cloud1_contour = contours[5768] # get big cloud
M1 = cv2.moments(cloud1_contour) # get moments
cX1 = int(M1["m10"] / M1["m00"]) # centroid x coor
cY1 = int(M1["m01"] / M1["m00"]) # centroid y coor

print ("x,y: ", cX1, cY1, M1)
print ("mean SUN azimuth:   ", rfobj_raster.MEANSUNAZ) # 0 to 360
print ("mean SUN elevation: ", rfobj_raster.MEANSUNEL) # +90 to -90
print ("mean SAT azimuth:   ", rfobj_raster.MEANSATAZ) # 0 to 360
print ("mean SAT elevation: ", rfobj_raster.MEANSATEL) # +90 to -90
print ("MEANINTRACKVIEWANGLE:   ",rfobj_raster.MEANINTRACKVIEWANGLE) # +90 to -90
print("MEANCROSSTRACKVIEWANGLE: ", rfobj_raster.MEANCROSSTRACKVIEWANGLE) # +90 to -90
print ("MEANOFFNADIRVIEWANGLE:  ", rfobj_raster.MEANOFFNADIRVIEWANGLE) # +90 to -90

# MEANINTRACKVIEWANGLE
# Maximum dihedral angle measured at the spacecraft from the nominal spacecraft
# XZ plane to the plane that contains the ground projection of the product center-line
# and the spacecraft X-axis. A positive angle indicates the sensor is looking to the right.
# MEANCROSSTRACKVIEWANGLE
# Mean dihedral angle measured at the spacecraft from the nominal spacecraft XZ plane to the
# plane that contains the ground projection of the product center-line and the spacecraft X-axis.
# A positive angle indicates the sensor is looking to the right.
# MEANOFFNADIRVIEWANGLE
# The minimum spacecraft elevation angle measured from nadir to the product center-line as seen from the spacecraft.
H = 100
x_cld = cX1 + H * math.tan(rfobj_raster.MEANSATEL) * math.sin(rfobj_raster.MEANSATAZ)
y_cld = cY1 + H * math.tan(rfobj_raster.MEANSATEL) * math.cos(rfobj_raster.MEANSATAZ)

x_shd = x_cld + H * math.tan(rfobj_raster.MEANSUNEL) * math.sin(rfobj_raster.MEANSUNAZ)
y_shd = y_cld + H * math.tan(rfobj_raster.MEANSUNEL) * math.cos(rfobj_raster.MEANSUNAZ)

print ("Initial centroid: ", cX1, cY1)
print ("Calc shadow centroid: ", x_shd, y_shd)
print ("Contour: ", contours[5768], M1)

#conts = cv2.drawContours(conts, contours[5768] - 500, -1, (255,0,0), 3)
#cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/drawCont.png", conts)
