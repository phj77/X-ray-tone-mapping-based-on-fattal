"""
postprocess.py - Post-processing for LDR images
(Independent of the paper algorithm)
 
Applies inverse gamma correction to the output LDR image after tone mapping.
"""
 
import numpy as np
import cv2
 
 
def postprocess(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply inverse gamma correction to the output LDR image.
 
    Decodes perceptual values back to linear light space:
        out = img ^ gamma
 
    Args:
        img   : Input LDR image, float32, range [0, 1]
        gamma : Gamma value (default: 2.2)
    Returns:
        out   : Inverse gamma-corrected image, float32, range [0, 1]
    """
    img_n = img.astype(np.float32) / 255.0
    correted_img = np.power(img_n, 1 / gamma).astype(np.float32)
    correted_img = (255.0 * correted_img).astype(np.uint8)
    return correted_img
 