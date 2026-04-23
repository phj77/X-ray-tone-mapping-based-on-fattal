"""
preprocess.py - Pre-processing for HDR images
(Independent of the paper algorithm)

Applies gamma correction to the input HDR image before tone mapping.
"""

import numpy as np


def preprocess(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to the input HDR image.

    Encodes linear light values into perceptual space:
        out = img ^ (1 / gamma)

    Args:
        img   : Input HDR image, float32, range [0, inf)
        gamma : Gamma value (default: 2.2)
    Returns:
        out   : Gamma-corrected image, float32, range [0, 1]
    """
    L_max = np.max(img)
    img_n = img.astype(np.float32) / L_max
    correted_img = np.power(img_n, 1 / gamma).astype(np.float32)
    correted_img = (L_max * correted_img).astype(np.uint8)
    return correted_img
