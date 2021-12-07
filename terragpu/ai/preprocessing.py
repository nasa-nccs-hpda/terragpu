import random
from typing import Union
from tqdm import tqdm

import math
import xarray as xr
# import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn import feature_extraction

from terragpu.engine import array_module, df_module

xp = array_module()
xf = df_module()

import numpy as np

# -------------------------------------------------------------------------
# Preprocess methods - Modify
# -------------------------------------------------------------------------

def modify_bands(
        img: xr.core.dataarray.DataArray, input_bands: list,
        output_bands: list, drop_bands: list = []):
    """
    Drop multiple bands to existing rasterio object
    """
    # Do not modify if image has the same number of output bands
    if img.shape[0] == len(output_bands):
        return img

    # Drop any bands from input that should not be on output
    for ind_id in list(set(input_bands) - set(output_bands)):
        drop_bands.append(input_bands.index(ind_id)+1)
    img = img.drop(dim="band", labels=drop_bands, drop=True)
    return img


def modify_roi(
        img: xp.ndarray, label: xp.ndarray,
        ymin: int, ymax: int, xmin: int, xmax: int):
    """
    Crop ROI, from outside to inside based on pixel address
    """
    return img[ymin:ymax, xmin:xmax], label[ymin:ymax, xmin:xmax]


def modify_pixel_extremity(
        img: xp.ndarray, xmin: int = 0, xmax: int = 10000):
    """
    Crop ROI, from outside to inside based on pixel address
    """
    return xp.clip(img, xmin, xmax)


def modify_label_classes(label: xp.ndarray, expressions: str):
    """
    Change pixel label values based on expression
    Example: "x != 1": 0, convert all non-1 values to 0
    """
    for exp in expressions:
        [(k, v)] = exp.items()
        label[eval(k, {k.split(' ')[0]: label})] = v
    return label


# def modify_label_dims(labels, n_classes: int):
#    if n_classes > 2:
#        if labels.min() == 1:
#            labels = labels - 1
#        return tf.keras.utils.to_categorical(
#            labels, num_classes=n_classes, dtype='float32')
#    else:
#        return xp.expand_dims(labels, axis=-1).astype(xp.float32)


# -------------------------------------------------------------------------
# Preprocess methods - Get
# -------------------------------------------------------------------------
def get_std_mean(images, output_filename: str):
    means = xp.mean(images, axis=tuple(range(images.ndim-1)))
    stds = xp.std(images, axis=tuple(range(images.ndim-1)))
    xp.savez(output_filename, mean=means, std=stds)
    return means, stds


def get_class_weights(labels):
    weights = compute_class_weight(
        'balanced',
        xp.unique(xp.ravel(labels, order='C')),
        xp.ravel(labels, order='C'))
    return weights


# -------------------------------------------------------------------------
# Preprocess methods - Calculate
# -------------------------------------------------------------------------
def calc_ntiles(
        width: int, height: int, tile_size: int, max_patches: float = 1):
    if isinstance(max_patches, int):
        return max_patches
    else:
        ntiles = (
            (math.ceil(width / tile_size)) * (math.ceil(height / tile_size)))
        return int(round(ntiles * max_patches))


# -------------------------------------------------------------------------
# Preprocess methods - Generate
# -------------------------------------------------------------------------
def gen_random_tiles(
        image: xp.ndarray, label: xp.ndarray, tile_size: int = 128,
        max_patches: Union[int, float] = None, seed: int = 24):
    """
    Extract small patches for final dataset
    Args:
        img (numpy array - c, y, x): imagery data
        tile_size (tuple): 2D dimensions of tile
        random_state (int): seed for reproducibility (match image and mask)
        n_patches (int): number of tiles to extract
    """
    # Calculate ntiles based on user input
    ntiles = calc_ntiles(
        width=image.shape[0], height=image.shape[1],
        tile_size=tile_size, max_patches=max_patches)

    images_list = []  # list to store data patches
    labels_list = []  # list to store label patches

    for i in tqdm(range(ntiles)):

        # Generate random integers from image
        x = random.randint(0, image.shape[0] - tile_size)
        y = random.randint(0, image.shape[1] - tile_size)

        while image[x: (x + tile_size), y: (y + tile_size), :].min() < 0 \
                or label[x: (x + tile_size), y: (y + tile_size)].min() < 0 \
                or xp.unique(
                    label[x: (x + tile_size), y: (y + tile_size)]).shape[0] < 2:
            x = random.randint(0, image.shape[0] - tile_size)
            y = random.randint(0, image.shape[1] - tile_size)

        # Generate img and mask patches
        image_tile = image[x:(x + tile_size), y:(y + tile_size)]
        label_tile = label[x:(x + tile_size), y:(y + tile_size)]

        # Apply some random transformations
        random_transformation = xp.random.randint(1, 8)
        if random_transformation == 1:
            image_tile = xp.fliplr(image_tile)
            label_tile = xp.fliplr(label_tile)
        if random_transformation == 2:
            image_tile = xp.flipud(image_tile)
            label_tile = xp.flipud(label_tile)
        if random_transformation == 3:
            image_tile = xp.rot90(image_tile, 1)
            label_tile = xp.rot90(label_tile, 1)
        if random_transformation == 4:
            image_tile = xp.rot90(image_tile, 2)
            label_tile = xp.rot90(label_tile, 2)
        if random_transformation == 5:
            image_tile = xp.rot90(image_tile, 3)
            label_tile = xp.rot90(label_tile, 3)
        if random_transformation > 5:
            pass

        images_list.append(image_tile)
        labels_list.append(label_tile)
    return xp.asarray(images_list), xp.asarray(labels_list)


def gen_random_tiles_include():
    raise NotImplementedError


# -------------------------------------------------------------------------
# Standardizing Methods - Calculate
# -------------------------------------------------------------------------
def standardize_global(image, strategy='per-image') -> xp.array:
    """
    Standardize numpy array using global standardization.
    :param image: numpy array in the format (n,w,h,c).
    :param strategy: can select between per-image or per-batch.
    :return: globally standardized numpy array
    """
    if strategy == 'per-batch':
        mean = xp.mean(image)  # global mean of all images
        std = xp.std(image)  # global std of all images
        for i in range(image.shape[0]):  # for each image in images
            image[i, :, :, :] = (image[i, :, :, :] - mean) / std
        return image
    elif strategy == 'per-image':
        return (image - xp.mean(image)) / xp.std(image)


def standardize_local(image, strategy='per-image') -> xp.array:
    """
    Standardize numpy array using global standardization.
    :param image: numpy array in the format (n,w,h,c).
    :param strategy: can select between per-image or per-batch.
    :return: globally standardized numpy array
    """
    if strategy == 'per-batch':
        mean = xp.mean(image)  # global mean of all images
        std = xp.std(image)  # global std of all images
        for i in range(image.shape[0]):  # for each image in images
            image[i, :, :, :] = (image[i, :, :, :] - mean) / std
        return image

    elif strategy == 'per-image':
        for j in range(image.shape[0]):  # for each channel in images
            channel_mean = xp.mean(image[j, :, :])
            channel_std = xp.std(image[j, :, :])
            image[j, :, :] = \
                (image[j, :, :] - channel_mean) / channel_std
        return image
