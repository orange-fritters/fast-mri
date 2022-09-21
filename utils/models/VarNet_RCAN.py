"""
Varnet RCAN for the submission

-- applied self-ensemble
"""

import torch
import torch.nn as nn

from utils.models.RCAN import RCAN
from utils.models.varnet import VarNet


class VarNetRCAN(nn.Module):
    """
    Varnet + RCAN with self-ensemble
    """

    def __init__(self, args):
        super().__init__()
        varnet = VarNet()
        PATH = '../fastMRI/model/pretrained.pt'
        CKPT = torch.load(PATH)
        varnet.load_state_dict(CKPT)
        self.KNet = varnet
        self.INet = RCAN(args)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa: torch.Tensor) -> torch.Tensor:
        output = self.KNet(masked_kspace, mask)
        input = torch.stack((output, grappa))
        output = self.INet(input).squeeze(0)

        return output

    def self_ensemble(self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa: torch.Tensor) -> torch.Tensor:
        """
        Self-ensemble method

        Args:
            masked_kspace (torch.Tensor): masked k-space data
            mask   (torch.Tensor): mask of the k-space
            grappa (torch.Tensor): grappa reconstruction image

        Returns:
            output (torch.Tensor): self-ensembled image
        """
        output = self.KNet(masked_kspace, mask)

        flips = ['original', 'f']
        ensembles = [self._flip(output, grappa, flip=flip) for flip in flips]

        ensembled = []
        for i, (output, grappa) in enumerate(ensembles):
            input = torch.stack((output, grappa))
            img = self.INet(input)
            ensembled.append(self._unflip(img.squeeze(0), flips[i]))

        output = sum(ensembled) / 2

        return output

    @staticmethod
    def _flip(image, grappa, flip):
        """
        Flip the image and grappa

        Args:
            image  (torch.Tensor): image
            grappa (torch.Tensor): grappa reconstruction image
            flip   (str): flip type

        Returns:
            image (torch.Tensor): flipped image

        """
        if flip == 'original':
            return image, grappa

        elif flip == 'f':
            image_f = torch.flip(image, [2])
            grappa_f = torch.flip(grappa, [2])
            return image_f, grappa_f

    @staticmethod
    def _unflip(image, flip):
        """
        Unflip the image

        Args:
            image (torch.Tensor): image
            flip  (str): flip type

        Returns:
            image (torch.Tensor): unflipped image
        """
        if flip == 'original':
            return image

        elif flip == 'f':
            image_original = torch.flip(image, [2])
            return image_original
