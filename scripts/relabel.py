"""
Testing OpenCV postprocessing functions for removing artifacts.
Author: Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
"""
import numpy as np
import tifffile as tiff
import rasterio as rio
import cv2


def npy_to_tif(raster_f='image.tif', segments='segment.npy',
               outtif='segment.tif'
               ):
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
    print(meta)

    # load numpy array if file is given
    if type(segments) == str:
        segments = np.load(segments)
    segments = segments.astype('int16')
    print(segments.dtype)  # check datatype

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(outtif, 'w', **out_meta) as dst:
        dst.write(segments, 1)


def int_ov_uni(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    return np.sum(intersection) / np.sum(union)


image = 'subsetLarge.tif'
mask = tiff.imread(image)

x_seg = cv2.resize(mask, (mask.shape[0], mask.shape[1]),
                   interpolation=cv2.INTER_NEAREST)

kk = (50, 50)
kk = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
# kernelc = np.ones(kk,dtype='uint8')
# kerneld = np.ones(kk,dtype='uint8')
# kernele = np.ones(kk,dtype='uint8')

closing = cv2.morphologyEx(x_seg, cv2.MORPH_CLOSE, kk)
dilation = cv2.dilate(x_seg, kk, iterations=5)
erosion = cv2.erode(x_seg, kk, iterations=1)
tophat = cv2.morphologyEx(x_seg, cv2.MORPH_TOPHAT, kk)
gradient = cv2.morphologyEx(x_seg, cv2.MORPH_GRADIENT, kk)
blackhat = cv2.morphologyEx(x_seg, cv2.MORPH_BLACKHAT, kk)
opening = cv2.morphologyEx(x_seg, cv2.MORPH_OPEN, kk)

kkk = np.ones((5, 5), np.float32)/25
smooth = cv2.filter2D(x_seg, -1, kkk)

npy_to_tif(raster_f=image, segments=closing, outtif='relabeled-cv2c.tif')
npy_to_tif(raster_f=image, segments=dilation, outtif='relabeled-cv2d.tif')
npy_to_tif(raster_f=image, segments=erosion, outtif='relabeled-cv2-ers.tif')
npy_to_tif(raster_f=image, segments=tophat, outtif='relabeled-cv2-th.tif')
npy_to_tif(raster_f=image, segments=gradient, outtif='relabeled-cv2-grad.tif')
npy_to_tif(raster_f=image, segments=blackhat, outtif='relabeled-cv2-bh.tif')
npy_to_tif(raster_f=image, segments=opening, outtif='relabeled-cv2-open.tif')
npy_to_tif(raster_f=image, segments=smooth, outtif='relabeled-cv2-smooth.tif')

print("unique labels: ", np.unique(mask))
