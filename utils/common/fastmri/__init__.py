"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

__version__ = "0.1.2a20210721"
__author__ = "Facebook/NYU fastMRI Team"
__author_email__ = "fastmri@fb.com"
__license__ = "MIT"
__homepage__ = "https://fastmri.org/"

import torch
from packaging import version

from utils.common.fastmri.coil_combine import rss, rss_complex
from utils.common.fastmri.fftc import fftshift, ifftshift, roll
from utils.common.fastmri.losses import SSIMLoss
from utils.common.fastmri.math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from utils.common.fastmri.fftc import fft2c_new as fft2c
    from utils.common.fastmri.fftc import ifft2c_new as ifft2c
else:
    from utils.common.fastmri.fftc import fft2c_old as fft2c
    from utils.common.fastmri.fftc import ifft2c_old as ifft2c
