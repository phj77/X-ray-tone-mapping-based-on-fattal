"""
utils.py - File Input/Output Utility
"""

import cv2
import numpy as np


def load_hdr(path: str):
    """
    Read .hdr / .pic files as float32 BGR
    shape: (H, W, 3), dtype: float32, range: [0, ∞)
    """
    # Read HDR image (.hdr format)
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if I.ndim == 3 and I.shape[2] == 3:
        # Convert to grayscale (paper framework is designed for single channel intensity)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    if I is None:
        raise FileNotFoundError(f"Can't open HDR file: {path}")
    return I


def load_ldr(path: str) -> np.ndarray:
    """
    Read standard images (PNG, JPG, etc.) as float32 BGR
    shape: (H, W, 3), dtype: float32, range: [0, 1]
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Can't open LDR file: {path}")
    return img.astype(np.float32) / 255.0


def save_ldr(img: np.ndarray, path: str):
    """
    float32 [0, 1] → uint8 storage after gamma 2.2 encoding
    """
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    cv2.imwrite(path, (img * 255).astype(np.uint8))



# 2026.04.22 - 11:51
def compute_gradient(img):
    """Compute gradient ∇ (using np.gradient)"""
    gx = np.gradient(img, axis=0)
    gy = np.gradient(img, axis=1)
    return gx, gy


# 2026.04.22 - 11:51
def compute_divergence(px, py):
    """Compute divergence (using np.gradient for discretization)"""
    dpx_dx = np.gradient(px, axis=0)
    dpy_dy = np.gradient(py, axis=1)
    return dpx_dx + dpy_dy