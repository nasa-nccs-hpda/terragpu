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
import sys
import os
import warnings
from datetime import datetime  # tracking date
from time import time  # tracking time
import argparse  # system libraries
import numpy as np  # for arrays modifications
import cv2
from math import tan, sin, cos

from xrasterlib.dgfile import DGFile
from xrasterlib.raster import Raster
import xrasterlib.indices as indices

# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)

# Ignoring true_divide errors since we know they are expected
warnings.filterwarnings("ignore", "invalid value encountered in true_divide", RuntimeWarning)
# --------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------
def create_logfile(args, logdir='results'):
    """
    :param args: argparser object
    :param logdir: log directory to store log file
    :return: logfile instance, stdour and stderr being logged to file
    """
    logfile = os.path.join(logdir, '{}_log_{}trees_{}.txt'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), args.ntrees, args.maxfeat))
    print('See ', logfile)
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # re-open stdout without buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect stdout and stderr to the log file opened above
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


def getparser():
    """
    :return: argparser object with CLI commands.
    """
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("-l", "--log", required=False, dest='logbool', action='store_true', help="Set logging.")
    parser.add_argument("-o", "--out-directory", type=str, required=True, dest='outdir',
                        default="", help="Specify output directory.")
    parser.add_argument("-m", "--model", type=str, required=False, dest='model',
                        default=None, help="Specify model filename that will be saved or evaluated.")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', help='Specify bands.',
                        required=False, type=str, default=['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge',
                                                           'NIR1', 'NIR2'])
    # Inference
    parser.add_argument("-i", "--rasters", type=str, nargs='*', required=False, dest='rasters',
                        default='*.tif', help="Image or pattern to evaluate images.")
    parser.add_argument('-d', '--drop-bands', nargs='*', dest='dropbands', help='Specify bands to remove.',
                        required=False, type=str, default=['HOM1', 'HOM2'])
    parser.add_argument("-toaf", "--toa-factor", type=float, required=False, dest='toaf',
                        default=10000.0, help="Specify TOA factor for indices calculation.")
    parser.add_argument("-ws", "--window-size", nargs=2, type=int, required=False, dest='windowsize',
                        default=[5000, 5000], help="Specify window size to perform sliding predictions.")
    parser.add_argument("-ps", "--sieve", required=False, dest='sievebool', action='store_true', help="Apply sieve.")
    parser.add_argument("-pm", "--median", required=False, dest='medianbool', action='store_true', help="Apply median.")
    parser.add_argument("-ss", "--sieve-size", type=int, required=False, dest='sieve_sz',
                        default=800, help="Specify size for sieve filter.")
    parser.add_argument("-ms", "--median-size", type=int, required=False, dest='median_sz',
                        default=20, help="Specify size for median filter.")
    return parser.parse_args()


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():

    start_time = time()  # record start time
    #args = getparser()  # initialize arguments parser

    #print('Initializing Random Forest script with the following parameters')
    #print(f'Working Directory: {args.outdir}')
    #print(f'model filename:    {args.model}')

    # --------------------------------------------------------------------------------
    # 1. set log file for script if requested (-l command line option)
    # --------------------------------------------------------------------------------
    #os.system(f'mkdir -p {args.outdir}')  # create output dir
    #if args.logbool:  # if command line option -l was given
    #    logfile = create_logfile(args, logdir=args.outdir)  # create logfile for std
    #print("Command line executed: ", sys.argv)  # saving command into log file

    # --------------------------------------------------------------------------------
    # 2. Instantiate Raster object
    # --------------------------------------------------------------------------------
    # specify data files
    outdir = "/Users/jacaraba/Desktop/cloudtest"

    # Test #1
    #raster = '/Users/jacaraba/Desktop/cloudtest/WV02_20181109_M1BS_1030010086582600-toa.tif'
    #cloudmask = '/Users/jacaraba/Desktop/cloudtest/cm_WV02_20181109_M1BS_1030010086582600-toa.tif'

    # Test #2
    raster = '/Users/jacaraba/Desktop/cloudtest/WV03_20180804_M1BS_104001003F25AA00-toa.tif'
    cloudmask = '/Users/jacaraba/Desktop/cloudtest/cm_WV03_20180804_M1BS_104001003F25AA00-toa.tif'
    bands = ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2']

    raster_obj = DGFile(raster)
    raster_obj.readraster(raster, bands)  # read raster
    raster_mask = Raster(cloudmask, ['Gray'])

    print(raster_obj.data.shape, raster_obj.bands, raster_obj.MEANSUNAZ)
    print(raster_mask.data.shape, raster_mask.bands)

    # Get contour objects
    rfobj_mask = np.uint8(np.squeeze(raster_mask.data.values) * 255)
    ret, thresh = cv2.threshold(rfobj_mask, 127, 255, 0)
    print(np.unique(thresh), "unique tresh", thresh.shape)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    # draws contour correctly
    #conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))
    #print(conts.shape)
    #conts = cv2.drawContours(conts, contours, -1, (0, 255, 0), 3)
    #cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/contours_WV03_20180804_M1BS_104001003F25AA00-toa.png", conts)

    # use this line if you want to draw a single contour
    # conts = cv2.drawContours(conts, contours[5768], -1, (0, 255, 0), 3)

    # trying to find centroid and cloud position, remove small contours
    conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))

    # saving this for now since it might be faster to write all contours
    # to the array as a list. Drawing them one by one for now.
    filtered_contours = list()
    filtered_shad_centroids = list()

    for c in contours:
        if len(c) > 1000:
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

                """
                self.MEANSUNAZ = float(self.imd_tag.find('IMAGE').find('MEANSUNAZ').text)
                self.MEANSUNEL = float(self.imd_tag.find('IMAGE').find('MEANSUNEL').text)
                self.MEANSATAZ = float(self.imd_tag.find('IMAGE').find('MEANSATAZ').text)
                self.MEANSATEL = float(self.imd_tag.find('IMAGE').find('MEANSATEL').text)
                self.MEANINTRACKVIEWANGLE = float(self.imd_tag.find('IMAGE').find('MEANINTRACKVIEWANGLE').text)
                self.MEANCROSSTRACKVIEWANGLE = float(self.imd_tag.find('IMAGE').find('MEANCROSSTRACKVIEWANGLE').text)
                self.MEANOFFNADIRVIEWANGLE = float(self.imd_tag.find('IMAGE').find('MEANOFFNADIRVIEWANGLE').text)
                """

                sat_zenith  = 90.0 - raster_obj.MEANOFFNADIRVIEWANGLE # or 180.0 - raster_obj.MEANOFFNADIRVIEWANGLE
                sat_azimuth = raster_obj.MEANSATAZ
                true_north  = 180 - raster_obj.MEANCROSSTRACKVIEWANGLE #raster_obj.MEANSATAZ - 180

                x_cld = cX + H * tan(sat_zenith) * sin(sat_azimuth + true_north)
                y_cld = cY + H * tan(sat_zenith) * cos(sat_azimuth + true_north)

                sun_zenith  = 90.0 - raster_obj.MEANSUNEL # https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/207/Radiometric_Use_of_WorldView-3_v2.pdf
                sun_azimuth = raster_obj.MEANSUNAZ

                x_shd = x_cld + H * tan(sun_zenith) * sin(sun_azimuth + true_north)
                y_shd = y_cld + H * tan(sun_zenith) * cos(sun_azimuth + true_north)

                print("Initial centroid: ", cX, cY)
                print("Calc shadow centroid: ", x_shd, y_shd)
                shad_centroids.append((int(x_shd), int(y_shd)))

                conts = cv2.drawContours(conts, [c], -1, 255, cv2.FILLED)  # cloud contour
                #conts = cv2.circle(conts, (int(cX), int(cY)), 20, (255, 255, 0), -1)  # cloud centroid
                #conts = cv2.circle(conts, (int(x_shd), int(y_shd)), 20, (0, 0, 255), -1)  # shadow centroids
                conts = cv2.drawContours(conts, [c + [int(x_shd)-cX, int(y_shd)-cY]], -1, (0, 0, 255), cv2.FILLED)  # shadow contours
            filtered_shad_centroids.append(shad_centroids)

    cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/position_WV03_20180804_M1BS_104001003F25AA00-toa.png", conts)
    #cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/position_WV02_20181109_M1BS_1030010086582600-toa.png", conts)

    print (type(conts), conts.shape)

    from skimage import color
    conts = color.rgb2gray(conts).astype('int16')
    conts[conts > 0] = 1

    print (type(conts), conts.shape)


    raster_obj.toraster(raster, conts, "/Users/jacaraba/Desktop/cloudtest/mask_WV03_20180804_M1BS_104001003F25AA00-toa.tif")  # save raster with mask
    #raster_obj.toraster(raster, conts, "/Users/jacaraba/Desktop/cloudtest/mask_WV02_20181109_M1BS_1030010086582600-toa.tif")  # save raster with mask


    #conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))
    #conts = cv2.drawContours(conts, filtered_contours, -1, (0, 255, 0), 3)
    #cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/filt_contours_WV03_20180804_M1BS_104001003F25AA00-toa.png", conts)


    """
    # took one cloud to test
    cloud1_contour = contours[5768]  # get big cloud
    M1 = cv2.moments(cloud1_contour)  # get moments
    cX1 = int(M1["m10"] / M1["m00"])  # centroid x coor
    cY1 = int(M1["m01"] / M1["m00"])  # centroid y coor

    print("x,y: ", cX1, cY1, M1)
    print("mean SUN azimuth:   ", raster_obj.MEANSUNAZ)  # 0 to 360
    print("mean SUN elevation: ", raster_obj.MEANSUNEL)  # +90 to -90
    print("mean SAT azimuth:   ", raster_obj.MEANSATAZ)  # 0 to 360
    print("mean SAT elevation: ", raster_obj.MEANSATEL)  # +90 to -90
    print("MEANINTRACKVIEWANGLE:   ", raster_obj.MEANINTRACKVIEWANGLE)  # +90 to -90
    print("MEANCROSSTRACKVIEWANGLE: ", raster_obj.MEANCROSSTRACKVIEWANGLE)  # +90 to -90
    print("MEANOFFNADIRVIEWANGLE:  ", raster_obj.MEANOFFNADIRVIEWANGLE)  # +90 to -90

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
    for H in range(100, 1500, 100):
        x_cld = cX1 + H * tan(raster_obj.MEANSATEL) * sin(raster_obj.MEANSATAZ)
        y_cld = cY1 + H * tan(raster_obj.MEANSATEL) * cos(raster_obj.MEANSATAZ)

        x_shd = x_cld + H * tan(raster_obj.MEANSUNEL) * sin(raster_obj.MEANSUNAZ)
        y_shd = y_cld + H * tan(raster_obj.MEANSUNEL) * cos(raster_obj.MEANSUNAZ)

        print("Initial centroid: ", cX1, cY1)
        print("Calc shadow centroid: ", x_shd, y_shd)
        print("Contour: ", contours[5768], M1)

        # conts = cv2.drawContours(conts, contours[5768] - 500, -1, (255,0,0), 3)
        # cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/drawCont.png", conts)

        conts = cv2.circle(conts, (int(x_shd), int(y_shd)), 30, (0,0,255), -1)
    cv2.imwrite("/Users/jacaraba/Desktop/cloudtest/drawContCirc.png", conts)
    """


    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time in minutes


if __name__ == "__main__":
    main()
