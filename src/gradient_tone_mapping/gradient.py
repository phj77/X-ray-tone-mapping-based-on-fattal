import numpy as np

# 2026.04.22 - last updated at 11:51
def compute_gradient_central_difference(img):
    """
    Compute gradient ∇ (using np.gradient)
    - Central difference: ∇f(x) ≈ (f(x + 1) - f(x - 1)) / (2)
    """
    gx = np.gradient(img, axis=0)
    gy = np.gradient(img, axis=1)
    return gx, gy


# 2026.04.22 - last updated at 11:51
def compute_divergence_central_difference(px, py):
    """
    Compute divergence (using np.gradient for discretization)
    divF = ∇ ⋅ F = ∂Fx/∂x + ∂Fy/∂y
    """
    dpx_dx = np.gradient(px, axis=0)
    dpy_dy = np.gradient(py, axis=1)
    return dpx_dx + dpy_dy


# 2026.04.22 - last updated at 11:51
def compute_gradient_forward_difference(img):
    """
    Compute gradient ∇ (using Forward difference)
    - Forward difference: ∇f(x) ≈ f(x + 1) - f(x)
    """
    H_pad = np.pad(img, 1, mode='reflect')
    dx_H = H_pad[1:-1, 2:] - H_pad[1:-1, 1:-1]
    dy_H = H_pad[2:, 1:-1] - H_pad[1:-1, 1:-1]
    return dx_H, dy_H


# 2026.04.22 - last updated at 11:51
def compute_divergence_forward_difference(px, py):
    """
    Compute divergence (using Forward difference for discretization)
    divF = ∇ ⋅ F = ∂Fx/∂x + ∂Fy/∂y
    """
    dpx_dx = px[1:-1, 2:] - px[1:-1, 1:-1]
    dpy_dy = py[2:, 1:-1] - py[1:-1, 1:-1]
    return dpx_dx + dpy_dy