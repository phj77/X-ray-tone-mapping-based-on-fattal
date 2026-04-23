"""
postprocess.py - Post-processing for LDR images
(Independent of the paper algorithm)
 
Applies inverse gamma correction to the output LDR image after tone mapping.
"""
 
import numpy as np
import cv2
 
 
def postprocess(img: np.ndarray, gamma: float) -> np.ndarray:

    """
    imput in [0,1]
    output in [0,1]
    """
    img_f = np.asanyarray(img).astype(np.float32)
    img_clip = np.clip(img_f, 0, 1)
    correted_img = np.power(img_clip, 1 / gamma)
    return correted_img
 