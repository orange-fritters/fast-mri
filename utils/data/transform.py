# https://github.com/LISTatSNU/FastMRI_challenge
"""
Data transfrom for fastmri challenge
Modified the above repository
"""

import numpy as np
import torch
import cv2


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)


class DataTransform:
    """
    data transform for the model
    """

    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, mask, input, grappa, target, attrs, fname, slice):
        """
        Get the i-th item in the dataset
        Args:
            mask   (np.array) : mask of the k-space
            input  (np.array) : original k-space data
            grappa (np.array) : grappa reconstruction image
            target (np.array) : target image
            attrs  (dict)     : attributes of the target image
            fname  (str)      : file name
            slice  (int)      : slice index

        Returns:
            mask    (torch.Tensor) : mask of the k-space
            kspace  (torch.Tensor) : k-space data
            grappa  (torch.Tensor) : grappa reconstruction image
            target  (torch.Tensor) : target image
            maximum (float)        : maximum value of the target image
            fname   (str)          : file name
            slice   (int)          : slice index
            leader_mask (torch.Tensor): mask for morphology operation
        """

        if not self.isforward:  # backward
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            grappa = to_tensor(grappa)
            leader_mask = torch.from_numpy(self._get_leaderboard_mask(target))
        else:
            target = -1
            maximum = -1
            grappa = to_tensor(grappa)
            leader_mask = 1

        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()

        return mask, kspace, grappa, target, maximum, fname, slice, leader_mask

    @staticmethod
    def _get_leaderboard_mask(target):
        mask = np.zeros(target.shape)
        mask[target > 5e-5] = 1
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=15)
        mask = cv2.erode(mask, kernel, iterations=14)

        return mask
