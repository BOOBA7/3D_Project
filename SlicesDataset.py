"""
This module defines a custom PyTorch dataset class SlicesDataset for loading 3D medical image data. The dataset is designed to represent 2D slices of a 3D volume, which can be processed individually during training or inference. The dataset can be consumed by the PyTorch DataLoader for batching and shuffling.

SlicesDataset:

This class is a subclass of torch.utils.data.Dataset and represents an indexable dataset that can be used with the PyTorch DataLoader class.
The dataset is initialized with a list of dictionaries, where each dictionary contains 3D image and segmentation data. The class processes these 3D volumes into individual 2D slices.
Each slice of the image and segmentation data is stored in the slices list, which holds tuples of indices corresponding to the volume and slice number.
__getitem__ method:

Retrieves a specific slice of the 3D volume by index (idx), returning a dictionary containing two 3D tensors: one for the image ("image") and one for the segmentation ("seg").
The shape of each tensor is [1, H, W], where H and W are the height and width of the 2D slice.
You can implement caching or data augmentation strategies within this method if needed.
__len__ method:

Returns the total number of 2D slices available in the dataset, which is used by the DataLoader to determine the size of the dataset.
This dataset is useful when working with large 3D medical image volumes and allows for efficient batch processing of individual slices during training or inference.

"""

"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments:
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

      

        i, j = slc
        image_slice = self.data[i]["image"][j]  # shape: (H, W)
        label_slice = self.data[i]["seg"][j]    # shape: (H, W)

        sample["image"] = torch.tensor(image_slice[None, :, :], dtype=torch.float32)
        sample["seg"] = torch.tensor(label_slice[None, :, :], dtype=torch.long)

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)