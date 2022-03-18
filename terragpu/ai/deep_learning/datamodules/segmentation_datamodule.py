# ------------------------------------------------------------------------------
# Segmentation Datamodule for PyTorch and PyTorch Lighning
# ------------------------------------------------------------------------------
import os
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from .base_datamodule import BaseLightningDataModule
from ..datasets.segmentation_dataset \
    import SegmentationDataset


@DATAMODULE_REGISTRY
class SegmentationDataModule(BaseLightningDataModule):

    name = "segmentation"

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
            add_dims: bool = False,
            val_split: float = 0.2,
            test_split: float = 0.1,
            *args: Any,
            **kwargs: Any,
        ) -> None:

        super().__init__(*args, **kwargs)

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
        self.images_regex = images_regex
        self.labels_regex = labels_regex
        self.transforms = transforms
        self.add_dims = add_dims

        # Generate dataset object
        segmentation_dataset = SegmentationDataset(
            input_bands=self.input_bands,
            output_bands=self.output_bands,
            tile_size=self.tile_size,
            max_patches=self.max_patches,
            test_size=self.test_size,
            dataset_dir=self.dataset_dir,
            generate_dataset=self.generate_dataset,
            images_regex=self.images_regex,
            labels_regex=self.labels_regex,
            transforms=self.transforms,
            add_dims=self.add_dims
        )

        # Split into train, val, test
        val_len = round(val_split * len(segmentation_dataset))
        test_len = round(test_split * len(segmentation_dataset))
        train_len = len(segmentation_dataset) - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            segmentation_dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )


if __name__ == '__main__':

    random_forest = SegmentationDataModule(
        dataset_dir='/lscratch/jacaraba/terragpu/clouds/senegal',
        generate_dataset=True,
        max_patches=750,
        images_regex='/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/training/data/*.tif',
        labels_regex='/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/training/labels/*.tif',
    )
