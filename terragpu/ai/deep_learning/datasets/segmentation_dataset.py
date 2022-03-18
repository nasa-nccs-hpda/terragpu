# ------------------------------------------------------------------------------
# Segmentation Dataset for PyTorch and PyTorch Lighning
# ------------------------------------------------------------------------------
import os
import logging
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from torch.utils.data import Dataset
from torch.utils.dlpack import from_dlpack

import rioxarray as rxr
from terragpu.engine import array_module, df_module
import terragpu.ai.preprocessing as preprocessing

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}

xp = array_module()
xf = df_module()

class SegmentationDataset(Dataset):

    def __init__(
            self,
            input_bands: list = ['CB', 'B', 'G', 'Y', 'R', 'RE', 'N1', 'N2'],
            output_bands: list = ['B', 'G', 'R'],
            tile_size: int = 256,
            max_patches: Union[float, int] = 100,
            test_size: float = 0.20,
            dataset_dir: Optional[str] = None,
            generate_dataset: bool = False,
            images_regex: Optional[str] = None,
            labels_regex: Optional[str] = None,
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = False,
            add_dims: bool = False
        ):

        self.input_bands = input_bands
        self.output_bands = output_bands
        self.tile_size = tile_size
        self.max_patches = max_patches
        self.test_size = test_size

        assert dataset_dir is not None, \
            f'dataset_dir={dataset_dir} should be defined.'

        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.labels_dir = os.path.join(self.dataset_dir, 'labels')

        # Create directories to store datasets, will ignore if dirs exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Set dataset generation variables, only if generate_dataset=True
        self.generate_dataset = generate_dataset

        if self.generate_dataset:

            logging.info(f"Starting to prepare dataset: {self.dataset_dir}")
            
            # Assert images_dir and labels_dir to be not None
            assert images_regex is not None, \
                f'images_regex should be defined with generate_dataset=True.'
            
            assert labels_regex is not None, \
                f'labels_regex should be defined with generate_dataset=True.'
    
            self.images_regex = images_regex
            self.labels_regex = labels_regex

            self.gen_dataset()

        assert len(os.listdir(self.images_dir)) > 0, \
            f'{self.images_dir} is empty. Make sure generate_dataset=True.'

        # Set Dataset metadata
        self.data_filenames = self.get_filenames()
        self.transform = transforms
        self.add_dims = add_dims
    
    # -------------------------------------------------------------------------
    # Dataset methods
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.data_filenames)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def __getitem__(self, idx: int):
        subject = self._load_from_disk(idx)
        if self.transform:
            subject = self.transform(subject)
        return subject

    def get_filenames(self):
        filenames_list: list = []
        for i in os.listdir(self.images_dir):
            filenames_list.append(
                {
                    'image': os.path.join(self.images_dir, i),
                    'label': os.path.join(self.labels_dir, i)
                }
            )
        return filenames_list

    def _load_from_disk(self, idx: int):
        
        # load image from disk
        image = xp.load(self.data_filenames[idx]['image'], allow_pickle=False)
        image = image.transpose((2, 0, 1))
        image = preprocessing.downscale(image)
        image = image / xp.iinfo(image.dtype).max
        image =  from_dlpack(image.toDlpack()).float()
        
        # load label from disk
        mask = xp.load(self.data_filenames[idx]['label'], allow_pickle=False)
        mask = xp.expand_dims(mask, 0) if self.add_dims else mask
        mask = from_dlpack(mask.toDlpack()).long()
        return {'image': image, 'label': mask}

    # -------------------------------------------------------------------------
    # Preprocessing methods
    # -------------------------------------------------------------------------
    def gen_dataset(self):
        """
        Generate training dataset tiles
        """
        logging.info("Preparing dataset...")
        images_list = sorted(glob(self.images_regex))
        labels_list = sorted(glob(self.labels_regex))

        for image, label in zip(images_list, labels_list):

            # Read imagery from disk and process both image and mask
            filename = Path(image).stem
            image = rxr.open_rasterio(image, chunks=CHUNKS).load()
            label = rxr.open_rasterio(label, chunks=CHUNKS).values

            # Modify bands if necessary - in a future version, add indices
            image = preprocessing.modify_bands(
                img=image, input_bands=self.input_bands,
                output_bands=self.output_bands)

            # Asarray option to force array type
            image = xp.asarray(image.values)
            label = xp.asarray(label)

            # Move from chw to hwc, squeze mask if required
            image = xp.moveaxis(image, 0, -1).astype(np.int16)
            label = xp.squeeze(label) if len(label.shape) != 2 else label
            logging.info(f'Label classes from image: {xp.unique(label)}')

            # Generate dataset tiles
            image_tiles, label_tiles = preprocessing.gen_random_tiles(
                image=image, label=label, tile_size=self.tile_size,
                max_patches=self.max_patches)
            logging.info(f"Tiles: {image_tiles.shape}, {label_tiles.shape}")

            # Save to disk
            for id in range(image_tiles.shape[0]):
                xp.save(
                    os.path.join(self.images_dir, f'{filename}_{id}.npy'),
                    image_tiles[id, :, :, :])
                xp.save(
                    os.path.join(self.labels_dir, f'{filename}_{id}.npy'),
                    label_tiles[id, :, :])
        return


if __name__ == '__main__':

    dataset = SegmentationDataset(
        dataset_dir='/lscratch/jacaraba/terragpu/clouds/senegal',
        generate_dataset=True,
        max_patches=750,
        images_regex='/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/training/data/*.tif',
        labels_regex='/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/training/labels/*.tif',
    )
