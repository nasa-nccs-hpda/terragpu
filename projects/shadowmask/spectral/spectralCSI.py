"""
Purpose: Script to detect cloud/shadow for multi/hyperspectral images.
Adapted from paper: 
Zhai, H., Zhang, H., Zhang, L., & Li, P. (2018). Cloud/shadow detection based on spectral 
indices for multi/hyperspectral optical remote sensing imagery. ISPRS Journal of Photogrammetry 
and Remote Sensing, 144, 235-253.
Author: Jordan A Caraballo Vega, Science Data Processing Branch, 587.0
"""

import rasterio
import xarray as xr

img = rasterio.open('red.tif')
print (img.is_tiled)
print (img.block_shapes)



"""
import PIL.Image
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)
import numpy as np
import tifffile as tiff           # read tiff files
import rasterio
import pandas as pd
image = tiff.imread('image2_20190228_data.tif')

## Identify clouds
# Calculate CI1 without SWIR
# CI1 = (3. * NIR) / (B + G + R)
# Calculate CI1 with SWIR
# CI1 = (NIR + 2 * SWIR-1) / (B + G + R)

# Calculate CI2 without SWIR
# CI2 = (B + G + R + NIR) / 4.
# Calculate CI2 with SWIR
# CI2 = (B + G + R + NIR + SWIR-1 + SWIR-2) / 6.

# Clouds between the following formulation
# (|CI1 - 1 | < T1) or (CI2 > T2), where T1 is a small treshold and T2 a large treshold
## Post-processing with median filter

# Identify cloud shadow
# CSI without SWIR
# CSI = NIR
# CSI with SWIR
# CSI = (NIR + SWIR-1)/2.

# Shadow between the following formulation
# (CSI < T3) and (B < T4), where T3 and T4 are small tresholds
## Spatial matching here
## Post-processing with median filter

## General final segmentation map


image = localStandardization(image, filename='normalization_data', ndata=pd.DataFrame(), strategy='per-batch')
image = np.squeeze(image, axis=0)

#dataset = rasterio.open('image1_20190228_data.tif')
#image = dataset.read()
print (image.shape, image.min(), image.max(), image.mean())

img = image[:,:,:3]
print (image.shape, img.shape, img.min(), img.max(), image[:,:,:3].shape, image[:,:,4:].shape, image[:,:,3:].shape)


flipped_lr = tf.image.flip_left_right(image)
flipped_ud = tf.image.flip_up_down(image)

grayscaled = tf.image.rgb_to_grayscale(image[:,:,:3]) # not using this

saturated = tf.image.adjust_saturation(image[:,:,:3], 3)
bright = tf.concat([tf.image.adjust_brightness(image[:,:,:3], 0.4), image[:,:,3:]], axis=2)

rot90 = tf.image.rot90(image, k=1, name=None)
rot180 = tf.image.rot90(image, k=2, name=None)
rot270 = tf.image.rot90(image, k=3, name=None)

#tf.image.adjust_brightness(image[:,:,:3], 0.4)

print (grayscaled.numpy().min(), grayscaled.numpy().max(), grayscaled.shape)
print (saturated.numpy().min(),saturated.numpy().max())
print (bright.numpy().min(),bright.numpy().max(), bright.shape)

fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(saturated)
plt.show()
"""
