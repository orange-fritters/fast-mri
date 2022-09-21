"""
Ordinary Conv Block and Residual Block
"""

import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size):
    """
    Ordinary Conv Block
    Args:
        in_channels  (int) : number of input channels,
        out_channels (int) : number of output channels,
        kernel_size  (int) : size of the kernel

    Returns:
        nn.Conv2d (nn.Module): Convolutional layer
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=1, bias=True)


class ResBlock(nn.Module):
    """
    Residual Block
    Args:
        n_feats (int) : number of features
        res_scale (float) : scale of the residual
    """

    def __init__(self, n_feats, res_scale=0.1):
        super(ResBlock, self).__init__()
        mid_feats = int(n_feats/4)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, mid_feats,
                      kernel_size=1, padding=0, stride=1),
            nn.PReLU(),
            nn.Conv2d(mid_feats, mid_feats,
                      kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(mid_feats, n_feats,
                      kernel_size=1, padding=0, stride=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward pass
        Args:
            x   (torch.Tensor) : input tensor
        Returns:
            res (torch.Tensor) : output tensor

        """
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
