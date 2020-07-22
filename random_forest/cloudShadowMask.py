# Cloud mask test
# identify polygon: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
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
raster = np.uint8(raster)
print (np.unique(raster), type(raster))

#gray = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY)
gray = raster
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import imutils

cc = pltc.ListedColormap(['black', 'white'])

plt.imsave('test_blurred_bw.png', blurred, cmap=cc)
#plt.imsave('test_thresh_bw.png', thresh, cmap=cc)

# find contours in the thresholded image
cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#(cnts, _) = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print (len(cnts))

#cv2.drawContours(blurred, [c[0]], -1, (0, 255, 0), cv2.FILLED)
#cv2.drawContours(blurred, cnts, (240, 0, 159), 3)
#cv2.imwrite("Image.png", blurred)


# loop over the contours
for c in cnts[:20]:
    # compute the center of the contour
    print (c)
    M = cv2.moments(c)
    #print ("Center of contour: ", M)

    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    #print (np.unique(blurred))
    cv2.drawContours(blurred, [c], -1, (255, 0, 0), 3)#cv2.FILLED)#2)
    #print (np.unique(blurred))

    #cv2.circle(raster, (cX, cY), 7, (255, 255, 255), -1)
    #cv2.putText(raster, "center", (cX - 20, cY - 20),
    #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    cv2.imwrite("ImageRed.png", blurred)

