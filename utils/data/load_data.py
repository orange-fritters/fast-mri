# https://github.com/LISTatSNU/FastMRI_challenge
"""
Data Loader for fastmri challenge
Modified the above repository
"""

from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.data.transform import DataTransform


class SliceData(Dataset):
    """
    Custom Pytorch Dataset class for fastMRI data
    Args:
         root (Union(Path|str): path to the data,
         transform (Any)      : transform  method
         input_key (str)      : key for input data,
         target_key (str)     : key for target data,
         forward: bool=False  : whether to use forward
         part: bool=False     : whether to use part of the data, especially high frequency images
         initialization: bool=False : whether to restart initialization
    """

    def __init__(self, root, transform, input_key, target_key, forward=False, part=False, initialization=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if initialization:  # restart initialization checking the first image's SSIM
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(10)
                ]
                break

            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):
                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(10)
                ]
                break

        if part:  # if part, use only high frequency images
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(10)
                ]

            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):
                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(10)
                ]
        else:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    def _get_metadata(self, fname):
        """
        Get the number of slices in the file
            Args:
                fname (Path): file path
            Returns:
                num_slices (int): number of slices in the file
        """
        with h5py.File(fname, "r") as hf:
            if self.forward:
                if self.input_key in hf.keys():
                    num_slices = hf[self.input_key].shape[0]
                else:
                    num_slices = hf['image_label'].shape[0]
            else:
                if self.input_key in hf.keys():
                    num_slices = hf[self.input_key].shape[0]
                elif self.target_key in hf.keys():
                    num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        """
        Get the i-th item in the dataset
        Args:
            i: index of the data slice
        Returns:
            mask    (torch.Tensor) : mask of the k-space
            kspace  (torch.Tensor) : masked k-space data
            grappa  (torch.Tensor) : grappa reconstruction image
            target  (torch.Tensor) : target image
            maximum (float)        : maximum value of the target image
            fname   (str)          : file name
            slice   (int)          : slice index
            leader_mask (torch.Tensor): mask for morphology operation
        """
        image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])

        if self.forward:
            target = -1
            attrs = -1
            with h5py.File(image_fname, "r") as hf:
                grappa = hf['image_grappa'][dataslice]
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = {'max': hf[self.target_key][:].max()}
                grappa = hf['image_grappa'][dataslice]

        return self.transform(mask, input, grappa, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, isforward=False, part=True, initialization=False):
    """
    Create data loaders for training and validation
    Args:
        data_path (Path) : data path for the data
        args      (Any)  : arguments (max_key, target_key, input_key, batch_size)
        isforward (bool) : whether it is forward phase
        part      (bool) : whether to use part of the data, especially high frequency images
        initialization (bool) : whether to restart initialization

    Returns:
        data_loader (DataLoader) : data loader for training and validation
    """
    if not isforward:  # backward
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward,
        part=part,
        initialization=initialization
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4
    )

    return data_loader
