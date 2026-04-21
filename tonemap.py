"""
tonemap.py - Full algorithm implementation
Liu et al., "An enhancement framework based on gradient domain tone mapping
and fuzzy logical for X-ray image of complex workpiece", NDT&E Int. 2021

Step 1. Global logarithmic mapping       Section 2.2.1 / Eq. (5)
Step 2. Gradient domain TMO              Section 2.2.2 / Eq. (6)~(15)
Step 3. Fuzzy enhancement & scaling      Section 2.2.3 / Eq. (16), (17)
"""

import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view


def tonemap(img: np.ndarray, gamma: float = 0.51, delta: float = 0.05) -> np.ndarray:
    """
    Full HDR -> LDR pipeline.

    Args:
        img   : Input image, float32, shape (H, W) or (H, W, C)
        gamma : Attenuation factor γ, range [0, 1]  (paper recommendation: 0.43~0.51)
        delta : Iteration stopping threshold δ       (paper default: 0.05)
    Returns:
        fout  : Output LDR image, float32, range [0, 1]
    """
    H    = _step1_log_mapping(img)
    f    = _step2_gradient_tmo(H, gamma, delta)
    fout = _step3_fuzzy_enhance(f)
    return fout


# ══════════════════════════════════════════
# Step 1. Global logarithmic mapping
# ══════════════════════════════════════════

def _step1_log_mapping(img: np.ndarray) -> np.ndarray:
    """
    Eq. (5): H(x,y) = log(I(x,y) + 1) / log(Lmax + 1)
    return: H
    """
    return np.log1p(img) / np.log1p(np.max(img))


# ══════════════════════════════════════════
# Step 2. Gradient domain TMO
# ══════════════════════════════════════════

def _step2_gradient_tmo(
    H: np.ndarray,
    gamma: float,
    delta: float,
) -> np.ndarray:
    """
    Full flow of Step 2.
    """
    E            = _fuzzy_entropy(H)            # Eq. (6), (7)
    K            = _attenuation(E, gamma)       # Eq. (8)
    alpha, beta  = _weights(E)                  # Eq. (10)
    Gx, Gy       = _compressed_gradient(H, K)  # Eq. (2)
    f            = _gradient_descent(H, Gx, Gy, alpha, beta, delta)  # Eq. (12)~(15)
    return f


def _fuzzy_entropy(H: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Compute local fuzzy entropy for each pixel.
    Eq. (6): E(x,y) = (1/N) * Σ [ -μ_H(k,l) * log2(μ_H(k,l)) ]
    Eq. (7): μ_H(k,l) = 1 / (1 + |H(k,l) - H_mean(x,y)|)
    """

    kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.float32) / 8.0
    H_bar = cv2.filter2D(H, -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    H_pad = np.pad(H, pad_width=1, mode='reflect')
    windows = sliding_window_view(H_pad, (window_size, window_size))

    H_bar_extend = H_bar[..., np.newaxis, np.newaxis]
    diff = np.abs(H_bar_extend - windows) #(w,h,3,3)
    mu_h = 1 / (1 + diff)
    entorpy = -mu_h * np.log2(mu_h)
    center_index = window_size//2
    entorpy[...,center_index, center_index] = 0

    w_2 = window_size ** 2
    E = np.mean(entorpy, axis=(-1,-2)) * w_2 / (w_2 - 1) # averaging 'neighborhood' of (x,y)

    return E


def _attenuation(E: np.ndarray, gamma: float) -> np.ndarray:
    """
    Fuzzy entropy-based attenuation function.
    Eq. (8): K_entropy = 1 / E^γ  (E != 0),  0  (E = 0)
    """
    K = np.zeros_like(E)
    mask = (E != 0)
    K[mask] = 1/E[mask]
    K = K ** gamma
    return K


def _weights(E: np.ndarray, eps: float = 1e-6):
    """
    Compute hybrid regularization weights α and β.
    Eq. (10): α = log(1 / (E + ε)), normalized to [0, 1],  β = 1 - α

    Returns:
        alpha, beta : float32 ndarray, range [0, 1]
    """
    alpha = np.log(1 / (E + 1e-10))
    alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    beta = 1- alpha
    return alpha, beta


def _compressed_gradient(H: np.ndarray, K: np.ndarray): # I used forward difference for gradient H referring to original fattal paper
    """
    Compute ideal compressed gradient field G = (Gx, Gy).
    Eq. (2): G(x,y) = ∇H(x,y) * K(x,y)

    Returns:
        Gx, Gy : compressed gradients along x and y directions
    """
    H_pad = np.pad(H, 1, mode='reflect')
    dx_H = H_pad[1:-1, 2:] - H_pad[1:-1, 1:-1]
    dy_H = H_pad[2:, 1:-1] - H_pad[1:-1, 1:-1]
    grad_H = np.concatenate((dx_H,dy_H), axis=0)
    grad_H = np.transpose(grad_H, (1,2,0))

    G = grad_H * K
    return G


def _gradient_descent( # used backward difference for div G referring to original fattal paper
    H: np.ndarray,
    Gx: np.ndarray,
    Gy: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: float,
    dt: float = 0.25,
) -> np.ndarray:
    """
    Minimize energy function Eq. (9) via gradient descent to estimate f.
    Eq. (12): f^{n+1} = f^n + Δt * [ (α+1)∇²f + (β/2)∇·(∇f/|∇f|) - divG ]
    Eq. (14): clamp f to [0, 1] after each iteration
    Eq. (15): stop when MAE(f^n, f^{n-1}) < δ
    """
    # initial f : H
    f = H
    div_G = Gx+Gy
    tv_flow = _tv_flow(f)
    laplacian_kernel = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=np.float32)
    laplacian_f = cv2.filter2D(f, -1, laplacian_kernel, borderType=cv2.BORDER_DEFAULT)
    mae = 1000000
    while mae > delta:
        f_tmp = f + dt *((alpha + 1) * laplacian_f + beta / 2 * tv_flow - div_G)
        f_next = np.maximum(0, np.minimum(1,f_tmp))
        mae = np.mean(np.abs(f_next - f))
        f = f_next



def _tv_flow(f: np.ndarray, eps: float = 1e-8) -> np.ndarray: # used central defference for grad f, second order difference f; is it ok respect to numerical viewpoint?
    """
    Gradient descent flow of TV regularization.
    Eq. (13): (fx²·fyy + fy²·fxx - 2·fx·fy·fxy) / (fx²+fy²+ε)^(3/2)
    """
    f_pad = np.pad(f, 1, mode='reflect')
    dx_f = (f_pad[1:-1, 2:] - f_pad[1:-1, 0:-2]) / 2
    dy_f = (f_pad[2:, 1:-1] - f_pad[0:-2, 1:-1]) / 2
    dxdx_f = f_pad[1:-1, 2:] + f_pad[1:-1, 0:-2] - 2*f
    dydy_f = f_pad[2:, 1:-1] + f_pad[0:-2, 1:-1] - 2*f
    dxdy_kernel = np.array([[-1, 0, 1],
                            [0,0,0],
                            [1,0,-1]], dtype=np.float32) / 4
    
    dxdy_f = cv2.filter2D(f, -1, dxdy_kernel, borderType=cv2.BORDER_DEFAULT)
    
    tv_flow = (dx_f**2 * dydy_f + dy_f**2 * dxdx_f - 2*dx_f*dy_f*dxdy_f) / (dx_f**2 + dy_f**2 + eps)
    return tv_flow

# ══════════════════════════════════════════
# Step 3. Fuzzy enhancement & scaling
# ══════════════════════════════════════════

def _step3_fuzzy_enhance(f: np.ndarray) -> np.ndarray:
    """
    Full flow of Step 3.
    """
    fc   = _find_fc(f)
    fout = _fuzzy_operator(f, fc)
    return fout


def _find_fc(f: np.ndarray) -> float:
    """
    Automatically determine fc from the rightmost trough of the histogram of f.
    Strong edges tend to appear as troughs in the histogram due to fewer pixels.

    TODO: implement
    """
    pass


def _fuzzy_operator(f: np.ndarray, fc: float) -> np.ndarray:
    """
    Apply fuzzy enhancement operator and scale to output.
    Eq. (16): μ'_f = μ_f² / fc              (0 <= μ_f <= fc)
              μ'_f = 1 - (1-μ_f)²/(1-fc)   (fc < μ_f <= 1)
    Eq. (17): fout = Lmax · μ'_f  ->  since Lmax=1, fout = μ'_f

    TODO: implement
    """
    pass