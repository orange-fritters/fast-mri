# https://github.com/sanghyun-son/EDSR-PyTorch
"""
RCAN used in image domain
Modified the above repository
"""

import torch
import torch.nn as nn
from utils.models import common
from torch.utils import checkpoint


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer

    Args:
        channel (int): number of channels
        reduction (int): reduction ratio
    """

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,
                      kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,
                      kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    Args:
        n_feats (int): number of input feature maps
        reduction (int): number of feature maps reduction
        res_scale (float): residual scaling
    """

    def __init__(
            self, n_feats, reduction, res_scale=1):
        super(RCAB, self).__init__()
        mid_feats = int(n_feats / 4)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, mid_feats,
                      kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feats, mid_feats,
                      kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feats, n_feats,
                      kernel_size=1, padding=0, stride=1, bias=True),
            CALayer(n_feats, reduction)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    """
    Residual Group for RIR in the RCAN
    Args:
        conv      (Any): conv layer
        n_feats   (int): number of input feature maps
        reduction (int): number of feature maps reduction
        res_scale (float): residual scaling
        n_resblocks (int): number of residual blocks
    """

    def __init__(self, conv, n_feats, reduction, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feats, reduction, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = checkpoint.checkpoint(self.body, x)
        res = res + x
        return res


# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    """
    RCAN model used in the image domain of the model
    """

    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        reduction = args.reduction

        modules_head = [conv(2, n_feats, kernel_size=3)]
        modules_body = [
            ResidualGroup(
                conv, n_feats, reduction,
                res_scale=args.res_scale,
                n_resblocks=n_resblocks) for _ in range(n_resgroups)]

        # define tail module
        modules_tail = [nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def norm(self, x):
        """
        Normalize the input to have zero mean and unit variance
        Args:
            x   (torch.Tensor) : input tensor
        Returns:
            x   (torch.Tensor) : normalized input tensor
            loc (torch.Tensor) : mean of the input tensors
            scale (torch.Tensor): standard deviation of the input tensor

        """
        x = x.view(1, 2, 384, 384)
        b, c, h, w = x.shape
        x = x.flatten(-2, -1).squeeze(0)
        loc = x.mean(dim=1)
        scale = x.std(dim=1)
        if 0 in scale:
            scale = scale + 1e-20
        x = torch.flatten(x)
        b_loc = x.mean()
        b_scale = x.std()
        x = x.view(b, c, h, w)
        return (x - loc.view(1, 2, 1, 1)) / scale.view(1, 2, 1, 1), b_loc, b_scale

    def unnorm(self, x, loc, scale):
        """
        Unnorm the tensor to the original scale
        Args:
            x   (torch.Tensor) : input tensor
            loc (torch.Tensor) : mean of the input tensors
            scale (torch.Tensor): standard deviation of the input tensor

        Returns:
            x   (torch.Tensor) : unnormed input tensor
        """
        return x * scale + loc

    def forward(self, x):
        x, loc, scale = self.norm(x)
        x = self.head(x)

        res = self.body(x)
        res = res + x

        x = self.tail(res)
        x = self.unnorm(x, loc, scale)

        return x
