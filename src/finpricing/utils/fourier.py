"""
Fourier-Based Call Option Pricing
"""
from enum import Enum
import numpy as np
from scipy.fft import fft
from typing import Callable, Tuple
from finpricing.pricing_methods.characteristic_functions import black_scholes_characteristic_function
from finpricing.utils.math_utils import generate_grid, compute_weights


class FourierMethod(Enum):
    CONT_TANKOV = "Cont&Tankov"
    KAR_MADAN = "Kar&Madan"


def cont_tankov_pricer(
        chr_func: Callable[[complex], complex],
        T: float,
        r: float,
        s_0:
        float, N: int,
        Delta: float,
        sigma: float) -> np.ndarray:
    """Fourier pricing using Cont & Tankov method."""
    # Integration grid
    d = 2 * np.pi / (Delta * N)
    A = Delta * (N - 1)
    xs = generate_grid(-0.5 * A, Delta, N)
    us = generate_grid(-N * d / 2 + s_0, d, N)

    # Quadrature weights
    w = compute_weights(N)

    # Modified characteristic function
    zeta_T = np.exp(-r * T) * chr_func(xs - 1j)
    zeta_T -= np.exp(1j * r * T * xs) * black_scholes_characteristic_function(xs - 1j, T, sigma)
    zeta_T /= (1j * xs - xs**2)

    # Inverse Fourier Transform
    zs = fft(zeta_T * w * np.exp(-1j * (s_0 - N * d / 2) * (xs + 0.5 * A)))
    zs = Delta * np.exp(1j * A * us / 2) * zs / (2 * np.pi)

    return zs.real


def kar_madan_pricer(chr_func: Callable[[complex], complex], T: float, r: float, s_0: float, N: int, Delta: float, alpha: float) -> np.ndarray:
    """Fourier pricing using Kar & Madan method."""
    # Integration grid
    xs = generate_grid(0, Delta, N)
    us = generate_grid(-N * Delta / 2 + s_0, Delta, N)

    # Modified characteristic function
    zeta_T = np.exp(-r * T) * chr_func(xs - (alpha + 1) * 1j)
    zeta_T /= alpha**2 + alpha - xs**2 + 1j * (2 * alpha + 1) * xs

    # Quadrature weights (Simpson's rule)
    w = compute_weights(N, method="simpson")

    # Inverse Fourier Transform
    zs = fft(zeta_T * np.exp((N * Delta / 2 - s_0) * 1j * xs) * w) * Delta
    return np.exp(-alpha * us) * zs.real / np.pi


def fourier_call_pricer(
    chr_func: Callable[[complex], complex],
    T: float,
    r: float,
    s_0: float = 0.0,
    method: FourierMethod = FourierMethod.CONT_TANKOV,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier-based call option pricer.
    """
    if method == FourierMethod.CONT_TANKOV:
        N = kwargs.get("N", 1024)
        Delta = kwargs.get("Delta", 0.17)
        sigma = kwargs.get("sigma", 0.3575)
        zs = cont_tankov_pricer(chr_func, T, r, s_0, N, Delta, sigma)
    
    elif method == FourierMethod.KAR_MADAN:
        N = kwargs.get("N", 1024)
        Delta = kwargs.get("Delta", 0.17)
        alpha = kwargs.get("alpha", 0.75)
        zs = kar_madan_pricer(chr_func, T, r, s_0, N, Delta, alpha)
    
    else:
        raise ValueError(f"Unsupported Fourier pricing method: {method}")

    us = generate_grid(-N * Delta / 2 + s_0, Delta, N)
    return us, zs
