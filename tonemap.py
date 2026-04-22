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
from scipy.ndimage import generic_filter
from scipy.signal import find_peaks
from parameters import Parameters 
params = Parameters()


# ══════════════════════════════════════════
# Step 1. Global logarithmic mapping
# ══════════════════════════════════════════
def _step1_log_mapping(img: np.ndarray):
    """
    Eq. (5): H(x,y) = log(I(x,y) + 1) / log(Lmax + 1)
    return: H
    """
    L_max_img = img.max()
    print(f"Original max intensity (L_max): {L_max_img}")

    H = np.log(img + 1.0) / np.log(L_max_img + 1.0) # H in [0, 1] (sect. 2.2.1)
    return H


# ══════════════════════════════════════════
# Step 2. Gradient domain TMO
# ══════════════════════════════════════════
def _step2_gradient_tmo(
    H: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Full flow of Step 2.
    """
    E            = generic_filter(H, _local_fuzzy_entropy, size=params.neighbor_size) # Eq. (6), (7)
    K            = _attenuation(E, gamma) # Eq. (8)

    # ============
    # for exp: check current issues (while image, many noise) ..
    print(f"\nE min = {E.min():.6f}, E max = {E.max():.6f}, E mean = {E.mean():.6f}")
    print(f"\nK min = {K.min():.2f}, K max = {K.max():.2f}, K mean = {K.mean():.2f}")
    # ============


    alpha, beta  = _weights(E, eps=params.eps) # Eq. (10)
    Gx, Gy       = _compressed_gradient(H, K) # Eq. (2)
    f            = _gradient_descent(H, Gx, Gy, alpha=alpha, beta=beta) # Eq. (12)~(15)
    return f

def _local_fuzzy_entropy(neighborhood):
    """
    Compute local fuzzy entropy according to Eq. (6) + (7) - 3x3 neighborhood:
    
    Compute local fuzzy entropy for each pixel.
    Eq. (6): E(x,y) = (1/N) * Σ [ -μ_H(k,l) * log2(μ_H(k,l)) ]
    Eq. (7): μ_H(k,l) = 1 / (1 + |H(k,l) - H_mean(x,y)|)
    """
    # print("\nComputing local fuzzy entropy with neighborhood: ", neighborhood)
    mean_val = np.mean(neighborhood)
    mu = 1.0 / (1.0 + np.abs(neighborhood - mean_val))
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10) # avoid log(0) or log(1)
    E = np.mean(-mu * np.log2(mu)) # E: entropy
    return E

def _attenuation(E: np.ndarray, gamma: float):
    """
    Fuzzy entropy-based attenuation function.
    Eq. (8): K_entropy = 1 / E^γ  (E != 0),  0  (E = 0)
    """
    print(f"\nProcessing with gamma: ", gamma)
    K = np.where(E > 0, 1.0 / (E ** gamma), 0.0) # E in [0, 1], so E != 0 means E > 0
    return K

def _weights(E: np.ndarray, eps: float=params.eps):
    """
    Compute hybrid regularization weights α and β.
    Eq. (10): α = log(1 / (E + ε)), normalized to [0, 1],  β = 1 - α

    Returns:
        alpha, beta : float32 ndarray, range [0, 1]
    """
    # α (Eq. 10) + normalizeation to [0, 1] by calculating its maximum and minimun values (sect. 2.2.2.2)
    alpha_raw = np.log(1.0 / (E + eps))
    alpha_min, alpha_max = alpha_raw.min(), alpha_raw.max()
    alpha = (alpha_raw - alpha_min) / (alpha_max - alpha_min + eps)

    # β
    beta = 1.0 - alpha
    
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
    Gx = dx_H * K
    Gy = dy_H * K
    
    return Gx, Gy

def _gradient_descent( # used backward difference for div G referring to original fattal paper
    H: np.ndarray,
    Gx: np.ndarray,
    Gy: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: float=params.delta,
    dt: float=params.dt,
) -> np.ndarray:
    """
    Minimize energy function Eq. (9) via gradient descent to estimate f.
    Eq. (12): f^{n+1} = f^n + Δt * [ (α+1)∇²f + (β/2)∇·(∇f/|∇f|) - divG ]
    Eq. (14): clamp f to [0, 1] after each iteration
    Eq. (15): stop when MAE(f^n, f^{n-1}) < δ
    """
    # initial f : H
    f = H.copy()
    div_G = Gx + Gy
    tv_flow = _tv_flow(f)
    laplacian_kernel = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=np.float32)
    laplacian_f = cv2.filter2D(f, -1, laplacian_kernel, borderType=cv2.BORDER_DEFAULT)
    mae = 1000000
    while mae > delta:
        f_tmp = f + dt * ((alpha + 1) * laplacian_f + beta / 2 * tv_flow - div_G)
        f_next = np.maximum(0, np.minimum(1,f_tmp))
        mae = np.mean(np.abs(f_next - f))
        f = f_next

    return f

def _tv_flow(f: np.ndarray, eps: float=params.eps): # used central defference for grad f, second order difference f; is it ok respect to numerical viewpoint?
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
def _step3_fuzzy_enhance(f: np.ndarray):
    """
    Full flow of Step 3.
    """
    fc   = _find_fc(f)
    f_out = _fuzzy_operator(f, fc)
    return f_out

def _find_fc(f: np.ndarray):
    """
    Automatically determine fc from the rightmost trough of the histogram of f.
    Strong edges tend to appear as troughs in the histogram due to fewer pixels.
    """
    # f_c = rightmost trough of histogram of gradient-toned image f
    hist, bin_edges = np.histogram(f.ravel(), bins=256, range=(0.0, 1.0))
    troughs_idx, _ = find_peaks(-hist, distance=3) # local minima

    if len(troughs_idx) > 0:
        rightmost = troughs_idx[-1]
        fc = (bin_edges[rightmost] + bin_edges[rightmost + 1]) / 2.0
    else:
        fc = params.fc

    print(f"f_c (rightmost trough) = {fc:.4f}")
    return fc

def _fuzzy_operator(f: np.ndarray, fc: float):
    """
    Apply fuzzy enhancement operator and scale to output.
    Eq. (16): μ'_f = μ_f² / fc              (0 <= μ_f <= fc)
              μ'_f = 1 - (1-μ_f)²/(1-fc)   (fc < μ_f <= 1)
    Eq. (17): f_out = Lmax · μ'_f  ->  since Lmax=1, fout = μ'_f
    """
    # Fuzzy operator
    mu = f
    if mu.min() < 0 or mu.max() > 1:
        raise ValueError("Input to fuzzy operator must be in [0, 1]")
    
    mu_prime = np.where(mu <= fc,
                        (mu ** 2) / fc,
                        1.0 - (((1.0 - mu) ** 2) / (1.0 - fc)))

    L_max_img = 1.0
    f_out = L_max_img * mu_prime
    return f_out


# ═════════════════════════════════════════
# Main tonemap function
# ═════════════════════════════════════════
def tonemap(img: np.ndarray, gamma: float):
    """
    Full HDR -> LDR pipeline.

    Args:
        I   : Input image, float32, shape (H, W) or (H, W, C)
        gamma : Attenuation factor γ, range [0, 1]  (paper recommendation: 0.43~0.51)
        delta : Iteration stopping threshold δ       (paper default: 0.05)
    Returns:
        f_out  : Output LDR image, float32, range [0, 1]
    """
    H    = _step1_log_mapping(img)
    f    = _step2_gradient_tmo(H, gamma=gamma)
    f_out = _step3_fuzzy_enhance(f)
    return f_out
