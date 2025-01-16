"""
    Utilities for Fourier Pricing
"""
import numpy as np


def generate_grid(start: float, step: float, num: int) -> np.ndarray:
    """Generate a grid of equally spaced values."""
    return np.array([start + k * step for k in range(num)])


def compute_weights(num: int, method: str = "trapezoidal") -> np.ndarray:
    """Compute quadrature weights."""
    if method == "trapezoidal":
        w = np.ones(num)
        w[0] = w[-1] = 0.5
    elif method == "simpson":
        w = np.array([3 - (-1)**j for j in range(num)])
        w[0] -= 1
        w = w / 3
    else:
        raise ValueError(f"Unsupported weight computation method: {method}")
    return w
