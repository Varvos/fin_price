"""
Jump-Diffusion Models: Merton and Kou
"""
import numpy as np
from scipy.integrate import quad_vec
from typing import Callable


def shared_drift_base(T: float, omega: float, k1: float) -> float:
    """
    Computes the shared drift base term for jump-diffusion models.
    """
    return -0.5 * (omega**2) * (1 - np.exp(-2 * k1 * T)) / (2 * k1)


def merton_integrand(s: float, T: float, k2: float, m: float, delta: float) -> float:
    """
    Drift integrand for the Merton model.
    """
    return np.exp(np.exp(-k2 * (T - s)) * m + 0.5 * np.exp(-2 * k2 * (T - s)) * delta**2) - 1


def kou_integrand(s: float, T: float, k2: float, p: float, alpha_plus: float, alpha_minus: float) -> float:
    """
    Drift integrand for the Kou model.
    """
    return (
        p * alpha_plus / (alpha_plus - np.exp(-k2 * (T - s)))
        + (1 - p) * alpha_minus / (alpha_minus + np.exp(-k2 * (T - s)))
        - 1
    )


def merton_drift(T: float, omega: float, lmbd: float, k1: float, k2: float, m: float, delta: float) -> float:
    """
    Computes the drift term for the Merton jump-diffusion model.
    """
    base_drift = shared_drift_base(T, omega, k1)
    integrand = lambda s: merton_integrand(s, T, k2, m, delta)
    jump_term = lmbd * quad_vec(integrand, 0, T)[0]
    return base_drift - jump_term


def kou_drift(T: float, omega: float, lmbd: float, k1: float, k2: float, p: float, alpha_plus: float, alpha_minus: float) -> float:
    """
    Computes the drift term for the Kou jump-diffusion model.
    """
    base_drift = shared_drift_base(T, omega, k1)
    integrand = lambda s: kou_integrand(s, T, k2, p, alpha_plus, alpha_minus)
    jump_term = lmbd * quad_vec(integrand, 0, T)[0]
    return base_drift - jump_term
