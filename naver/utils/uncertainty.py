"""
Uncertainty estimation utilities for DR-Bearing.

PSG α (shape [B,4]) is the softmax similarity weight between the UVP and 4 RSTs.
High entropy of α means the model cannot distinguish which RST the UVP belongs to → high uncertainty.
"""
import numpy as np


def entropy_from_alpha(alpha: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy from PSG α weights.

    Args:
        alpha: [4] numpy array, PSG softmax weights (sum to 1)
    Returns:
        H_norm: float in [0, 1]. 0 = fully confident (one RST dominates), 1 = maximally uncertain (uniform)
    """
    alpha = np.clip(alpha, 1e-9, 1.0)
    H = -np.sum(alpha * np.log(alpha))
    H_norm = H / np.log(4)  # max entropy for 4-class = log(4)
    return float(np.clip(H_norm, 0.0, 1.0))


def u_from_entropy(H_norm: float, mode: str = 'sigmoid', tau: float = 0.5, k: float = 10.0) -> float:
    """
    Map normalized entropy H_norm ∈ [0,1] to uncertainty u ∈ [0,1].

    Args:
        H_norm: normalized entropy from entropy_from_alpha()
        mode:   'sigmoid' (smooth threshold around tau) or 'linear' (identity)
        tau:    sigmoid midpoint (entropy level treated as threshold), default 0.5
        k:      sigmoid steepness, default 10.0
    Returns:
        u: float in [0,1]. 0 = certain, 1 = uncertain
    """
    if mode == 'sigmoid':
        return float(1.0 / (1.0 + np.exp(-k * (H_norm - tau))))
    elif mode == 'linear':
        return float(np.clip(H_norm, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown uncertainty mode: {mode!r}. Choose 'sigmoid' or 'linear'.")
