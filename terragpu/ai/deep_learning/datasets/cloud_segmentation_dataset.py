import os
import torch
import numpy as np
import cupy as cp
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

    def __init__(
            self, images_dir: str,
            labels_dir: str,
            pytorch: bool = True,
            transform = None,
            invert: bool = True,
            add_dims: bool = False
        ):

        super().__init__()

        self.files = self.list_files(images_dir, labels_dir)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch = pytorch
        self.transform = transform
        self.invert = invert
        self.add_dims = add_dims

    # -------------------------------------------------------------------------
    # Common methods
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def __getitem__(self, idx):

        # get data
        x = torch.tensor(
            self.open_image(idx), dtype=torch.float32, device=self.device)

        y = torch.tensor(
            self.open_mask(idx), dtype=torch.torch.float32, device=self.device)

        # augment the data
        if augment:

            if np.random.random_sample() > 0.5:  # flip left and right
                x = torch.fliplr(x)
                y = torch.fliplr(y)
            if np.random.random_sample() > 0.5:  # reverse second dimension
                x = torch.flipud(x)
                y = torch.flipud(y)
            if np.random.random_sample() > 0.5:  # rotate 90 degrees
                x = torch.rot90(x, k=1, dims=[1, 2])
                #y = torch.rot90(y, k=1, dims=[0, 1])
                y = torch.rot90(y, k=1, dims=[1, 2])
            if np.random.random_sample() > 0.5:  # rotate 180 degrees
                x = torch.rot90(x, k=2, dims=[1, 2])
                #y = torch.rot90(y, k=2, dims=[0, 1])
                y = torch.rot90(y, k=2, dims=[1, 2])
            if np.random.random_sample() > 0.5:  # rotate 270 degrees
                x = torch.rot90(x, k=3, dims=[1, 2])
                #y = torch.rot90(y, k=3, dims=[0, 1])
                y = torch.rot90(y, k=3, dims=[1, 2])

        # TODO: add class and standardize options
        #y = torch.unsqueeze(y, 0)
        #print(y.max())
        # standardize 0.70, 0.30
        #if np.random.random_sample() > 0.70:
        #    image = preprocess.standardizeLocalCalcTensor(image, means, stds)
        #else:
        #    image = preprocess.standardizeGlobalCalcTensor(image)

        return x, y

    # -------------------------------------------------------------------------
    # IO methods
    # -------------------------------------------------------------------------
    def list_files(self, images_dir: str, labels_dir: str, files_list: list = []):

        for i in os.listdir(images_dir):
            files_list.append(
                {
                    'image': os.path.join(images_dir, i),
                    'label': os.path.join(labels_dir, i)
                }
            )
        return files_list

    def open_image(self, idx: int):
        image = np.load(self.files[idx]['image'])
        return image.transpose((2, 0, 1)) if self.invert else image

    def open_mask(self, idx: int):
        mask = np.load(self.files[idx]['label'])
        return np.expand_dims(mask, 0) if self.add_dims else mask
