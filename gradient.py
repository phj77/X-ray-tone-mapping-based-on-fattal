import numpy as np

# 2026.04.22 - last updated at 11:51
def compute_gradient(img):
    """Compute gradient ∇ (using np.gradient)"""
    gx = np.gradient(img, axis=0)
    gy = np.gradient(img, axis=1)
    return gx, gy


# 2026.04.22 - last updated at 11:51
def compute_divergence(px, py):
    """Compute divergence (using np.gradient for discretization)"""
    dpx_dx = np.gradient(px, axis=0)
    dpy_dy = np.gradient(py, axis=1)
    return dpx_dx + dpy_dy