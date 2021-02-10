"""
Purpose: Mask out cloud shadows using cloud masks and sun/satellite geometry
         properties. Usage requirements are referenced in README.

Data Source: This script has been tested with very high-resolution NGA data.
             Additional testing will be required to validate the applicability
             of this model for other datasets.

Author: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
# import sys
# import os
import warnings
# from datetime import datetime  # tracking date
from time import time  # tracking time
# import argparse  # system libraries
import numpy as np  # for arrays modifications
import cv2
from math import tan, sin, cos

from xrasterlib.dgfile import DGFile
from xrasterlib.raster import Raster
# import xrasterlib.indices as indices

import matplotlib.pyplot as plt
import numpy.ma as ma
from skimage import color


# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)

# Ignoring true_divide errors since we know they are expected
warnings.filterwarnings(
    "ignore", "invalid value encountered in true_divide", RuntimeWarning
)


def getContourStat(contour, image, raster, points=20):

    mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
    cv2.drawContours(mask, contour, 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))

    # TODO: move from for loop to something else
    # probably select random points from pixelpoints
    #total = 0
    #for pixel in pixelpoints[:points]:
    #    #print(pixel[0], pixel[1], raster[:, pixel[0], pixel[1]].values)
    #    total += np.sum(raster[:, pixel[0], pixel[1]].values, axis=0)
    #return total / (points * 8)

    # first one gives expected output, but slow
    # second one gives (8,10,10) and different results
    # print(raster.values[:, pixelpoints[:10, 0], pixelpoints[:10, 1]])
    #meanval = np.mean(raster[:, pixelpoints[:10, 0], pixelpoints[:10, 1]]).values
    #meanval = np.mean(raster[:, pixelpoints[:100, 0], pixelpoints[:100, 1]]).values
    minval = np.min(raster[:4, pixelpoints[:60, 0], pixelpoints[:60, 1]]).values
    print("min val", minval)
    return minval


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError:
        cx, cy = 0, 0

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------


def main():

    start_time = time()  # record start time

    # --------------------------------------------------------------------------------
    # 2. Instantiate Raster object
    # --------------------------------------------------------------------------------
    # specify data files
    outdir = "/Users/jacaraba/Desktop/CLOUD/cloudtest"
    #print(outdir)

    # Test #1
    # raster = '/Users/jacaraba/Desktop/cloudtest/WV02_20181109_M1BS_1030010086582600-toa.tif'
    # cloudmask = '/Users/jacaraba/Desktop/cloudtest/cm_WV02_20181109_M1BS_1030010086582600-toa.tif'

    # Test #2
    #raster = '/Users/jacaraba/Desktop/CLOUD/cloudtest/WV03_20180804_M1BS_104001003F25AA00-toa.tif'
    #cloudmask = '/Users/jacaraba/Desktop/CLOUD/cloudtest/cm_WV03_20180804_M1BS_104001003F25AA00-toa.tif'
    #bands = ['CoastalBlue', 'Blue', 'Green', 'Yellow',
    #         'Red', 'RedEdge', 'NIR1', 'NIR2'
    #         ]

    # Test #3
    raster = '/Users/jacaraba/Desktop/CLOUD/cloudtest/Keelin00_20120130_data.tif'
    cloudmask = '/Users/jacaraba/Desktop/CLOUD/cloudtest/cm_Keelin00_20120130_data.tif'
    xml = '/Users/jacaraba/Desktop/CLOUD/cloudtest/WV02_20120130_M1BS_1030010011A30B00-toa.xml'
    bands = ['Red', 'Green', 'Blue', 'NIR1', 'HOM1', 'HOM2']

    raster_obj = DGFile(raster, xml_filename=xml)
    raster_obj.readraster(raster, bands)  # read raster
    raster_mask = Raster(cloudmask, ['Gray'])

    print(raster_obj.data.values.min())
    print(raster_obj.data.values.max())

    #print(raster_obj.data.shape, raster_obj.bands, raster_obj.MEANSUNAZ)
    #print(raster_mask.data.shape, raster_mask.bands)

    # Get contour objects
    rfobj_mask = np.uint8(np.squeeze(raster_mask.data.values) * 255)
    ret, thresh = cv2.threshold(rfobj_mask, 127, 255, 0)
    #print(np.unique(thresh), "unique tresh", thresh.shape)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    #print("Len contours: ", len(contours))

    # trying to find centroid and cloud position, remove small contours
    conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))
    conts_shadow = np.zeros((thresh.shape[0], thresh.shape[1], 3))

    # saving this for now since it might be faster to write all contours
    # to the array as a list. Drawing them one by one for now.
    filtered_contours = list()
    filtered_shad_centroids = list()

    for c in contours:
        if len(c) > 100:

            # append new contours based on treshold
            filtered_contours.append(c)

            # get moments and centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            shad_centroids = list()
            shad_reflectance = 10000
            shad_reflectance_contour = None

            # 10. 180. 10
            for H in range(100, 2000, 100):
                try:
                    # contour for each shadow
                    conts_shadow = np.zeros((thresh.shape[0], thresh.shape[1], 3))

                    # or 180.0 - raster_obj.MEANOFFNADIRVIEWANGLE
                    sat_zenith = 90.0 - raster_obj.MEANOFFNADIRVIEWANGLE
                    #sat_zenith = 180.0 - raster_obj.MEANOFFNADIRVIEWANGLE

                    sat_azimuth = raster_obj.MEANSATAZ
                    # raster_obj.MEANSATAZ - 180
                    #true_north = 180 - raster_obj.MEANCROSSTRACKVIEWANGLE
                    true_north = 90 - raster_obj.MEANCROSSTRACKVIEWANGLE


                    x_cld = cX + H * tan(sat_zenith) * \
                        sin(sat_azimuth + true_north)
                    y_cld = cY + H * tan(sat_zenith) * \
                        cos(sat_azimuth + true_north)

                    # https://dg-cms-uploads-production.s3.amazonaws.com/uploads/
                    # document/file/207/Radiometric_Use_of_WorldView-3_v2.pdf
                    sun_zenith = 90.0 - raster_obj.MEANSUNEL
                    sun_azimuth = raster_obj.MEANSUNAZ

                    x_shd = x_cld + H * tan(sun_zenith) * \
                        sin(sun_azimuth + true_north)
                    y_shd = y_cld + H * tan(sun_zenith) * \
                        cos(sun_azimuth + true_north)

                    #print("Initial centroid: ", c.shape, type(c), cX, cY)
                    #print("Calc shadow centroid: ", x_shd, y_shd)
                    shad_centroids.append((int(x_shd), int(y_shd)))

                    # cloud contour
                    c = scale_contour(c, 1.01) #1.05 looks decent
                    conts = cv2.drawContours(conts, [c], -1, (0,255,0), cv2.FILLED)
                    # conts = cv2.drawContours(conts, [c]*10, -1, 255, cv2.FILLED)
                    
                    # shadow contour
                    # conts = cv2.drawContours(
                    #    conts,
                    #    [c + [int(x_shd)-cX, int(y_shd)-cY]], -1, (0, 0, 255),
                    #    cv2.FILLED
                    # )  # shadow contours
                    # mean, stddev = getContourStat([c + [int(x_shd)-cX, int(y_shd)-cY]], conts)
                    print("before maskc")
                    maskc = getContourStat([c + [int(x_shd)-cX, int(y_shd)-cY]], conts_shadow, raster_obj.data)
                    print("before if")
                    if maskc > 0 and maskc < shad_reflectance:
                        shad_reflectance = maskc
                        shad_reflectance_contour = [c + [int(x_shd)-cX, int(y_shd)-cY]]

                    print(shad_reflectance)
                    print("after if")

                    #print("Reflectance value", maskc)
                    # mx = ma.masked_array(raster_obj.data[1,:,:], mask=maskc)
                    # print(raster_obj.data[1,:,:].shape)
                    # print(mx)
                    #conts = color.rgb2gray([c + [int(x_shd)-cX, int(y_shd)-cY]]).astype('int8')
                    #print(conts.shape)
                    
                    #conts_shadow = cv2.drawContours(
                    #    conts,
                    #    [c + [int(x_shd)-cX, int(y_shd)-cY]], -1, (0, 0, 255),
                    #    cv2.FILLED
                    #)  # shadow contours
                    #print(type(conts_shadow), conts_shadow.shape)


                    # print(mean, stddev)

                    # IMPORTANT: Check if contour is part of CLOUD SHADOW
                    # 1. get real values inside contour
                    # 2. Calculate indices to see if it is cloud
                    # 3. get average and check if it is shadow or not,
                    # 4. draw contour then
                except ValueError:
                    pass

            print("Final reflesctance value ", shad_reflectance)
            filtered_shad_centroids.append(shad_centroids)

        conts_shadow = cv2.drawContours(
            conts,
            shad_reflectance_contour, -1, (0, 0, 255),
            cv2.FILLED
        )  # shadow contours

    cv2.imwrite(
        "/Users/jacaraba/Desktop/CLOUD/cloudtest/testing-now1.png",
        conts
    )
    #print("Conts type and shape: ", type(conts), conts.shape)

    # converting mask into raster
    conts = color.rgb2gray(conts).astype('int16')
    #conts_shadow = color.rgb2gray(conts_shadow).astype('int16')

    conts[conts > 0] = 1
    #print(type(conts), conts.shape)
    
    raster_obj.toraster(
        raster, conts,
        "/Users/jacaraba/Desktop/CLOUD/cloudtest/testing-now2.tif"
    )  # save raster with mask
    
    raster_obj.toraster(
        raster, conts_shadow,
        "/Users/jacaraba/Desktop/CLOUD/cloudtest/testing-now33.tif"
    )  # save raster with mask
    
    # output program run time in minutes
    print("Elapsed Time: ", (time() - start_time) / 60.0)


if __name__ == "__main__":
    main()

