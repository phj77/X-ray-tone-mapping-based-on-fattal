"""
preprocess.py - Pre-processing for HDR images
(Independent of the paper algorithm)

Applies gamma correction to the input HDR image before tone mapping.
"""

import numpy as np


def preprocess(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma should be large then 0.")
        
    img_hdr = np.asanyarray(img).astype(np.float32)

    img_hdr = np.clip(img_hdr, 0.0, None) 
    max_val = img_hdr.max()
    if max_val > 0:
        img_hdr /= max_val 

    inv_gamma = 1.0 / gamma
    corrected_img = np.power(img_hdr + 1e-8, inv_gamma)

    return np.clip(corrected_img, 0.0, 1.0).astype(np.float32)