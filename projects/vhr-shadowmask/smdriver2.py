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

# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)

# Ignoring true_divide errors since we know they are expected
warnings.filterwarnings(
    "ignore", "invalid value encountered in true_divide", RuntimeWarning
)


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
    print(outdir)

    # Test #1
    # raster = '/Users/jacaraba/Desktop/cloudtest/WV02_20181109_M1BS_1030010086582600-toa.tif'
    # cloudmask = '/Users/jacaraba/Desktop/cloudtest/cm_WV02_20181109_M1BS_1030010086582600-toa.tif'

    # Test #2
    raster = '/Users/jacaraba/Desktop/CLOUD/cloudtest/WV03_20180804_M1BS_104001003F25AA00-toa.tif'
    cloudmask = '/Users/jacaraba/Desktop/CLOUD/cloudtest/cm_WV03_20180804_M1BS_104001003F25AA00-toa.tif'
    bands = ['CoastalBlue', 'Blue', 'Green', 'Yellow',
             'Red', 'RedEdge', 'NIR1', 'NIR2'
             ]

    raster_obj = DGFile(raster)
    raster_obj.readraster(raster, bands)  # read raster
    raster_mask = Raster(cloudmask, ['Gray'])

    print(raster_obj.data.shape, raster_obj.bands, raster_obj.MEANSUNAZ)
    print(raster_mask.data.shape, raster_mask.bands)

    # Get contour objects
    rfobj_mask = np.uint8(np.squeeze(raster_mask.data.values) * 255)
    ret, thresh = cv2.threshold(rfobj_mask, 127, 255, 0)
    print(np.unique(thresh), "unique tresh", thresh.shape)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    print(len(contours))

    # trying to find centroid and cloud position, remove small contours
    conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))

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

            for H in range(10, 120, 10):

                # or 180.0 - raster_obj.MEANOFFNADIRVIEWANGLE
                sat_zenith = 90.0 - raster_obj.MEANOFFNADIRVIEWANGLE
                sat_azimuth = raster_obj.MEANSATAZ
                # raster_obj.MEANSATAZ - 180
                true_north = 180 - raster_obj.MEANCROSSTRACKVIEWANGLE

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

                print("Initial centroid: ", cX, cY)
                print("Calc shadow centroid: ", x_shd, y_shd)
                shad_centroids.append((int(x_shd), int(y_shd)))

                # cloud contour
                # conts = cv2.drawContours(conts, [c], -1, 255, cv2.FILLED)
                conts = cv2.drawContours(conts, [c]*10, -1, 255, cv2.FILLED)
                conts = cv2.drawContours(
                    conts,
                    [c + [int(x_shd)-cX, int(y_shd)-cY]], -1, (0, 0, 255),
                    cv2.FILLED
                )  # shadow contours

                # IMPORTANT: Check if contour is part of CLOUD SHADOW
                # 1. get real values inside contour
                # 2. Calculate indices to see if it is cloud
                # 3. get average and check if it is shadow or not,
                # 4. draw contour then
                
            filtered_shad_centroids.append(shad_centroids)

    cv2.imwrite(
        "/Users/jacaraba/Desktop/CLOUD/cloudtest/testing-now1.png",
        conts
    )
    print(type(conts), conts.shape)

    # converting mask into raster
    from skimage import color
    conts = color.rgb2gray(conts).astype('int16')
    conts[conts > 0] = 1
    print(type(conts), conts.shape)
    raster_obj.toraster(
        raster, conts,
        "/Users/jacaraba/Desktop/CLOUD/cloudtest/testing-now2.tif"
    )  # save raster with mask
    # output program run time in minutes
    print("Elapsed Time: ", (time() - start_time) / 60.0)


if __name__ == "__main__":
    main()
