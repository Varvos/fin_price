"""
Module for Implied Volatility Calculations
"""
from enum import Enum
from typing import Callable, Union
import numpy as np
from scipy.optimize import root, brentq, newton
from src.finpricing.utils.bs_utils import BlackScholesCalculator


# Enum for Supported Methods
class Method(Enum):
    HYBR = 'hybr'
    LM = 'lm'
    BRENTQ = 'brentq'
    NEWTON = 'newton'
    ITERATIVE = 'iterative'
    ##  TODO: Explore other methods that can be added,
    ## e.g. based on root-finding algorithms below
    # 'broyden1',
    # 'broyden2',
    # 'anderson',
    # 'linearmixing',
    # 'diagbroyden',
    # 'excitingmixing',
    # 'krylov',
    # 'df-sane',


# Helper Functions for Initial Guesses
def corrado_miller_guess(call_value: float, s: float, k: float, t: float, r: float) -> float:
    """
    Calculate an initial guess for implied volatility using the Corrado-Miller approximation.
    """
    exp1 = s - k * np.exp(-r * t)
    exp2 = np.sqrt(2 * np.pi) / (s + k * np.exp(-r * t))
    guess = (exp2 + (call_value - exp1 / 2) +
             np.sqrt((call_value - exp1 / 2) ** 2 - exp1 ** 2 / np.pi)) / np.sqrt(t)
    return max(guess, 1e-8)  # Ensure the guess is positive


def improved_guess(call_value: float, s: float, k: float, t: float, r: float) -> float:
    """
    Calculate an initial guess for implied volatility using an improved approximation.
    """
    x = k * np.exp(-r * t)
    y = 2 * call_value + x - s
    factor1 = np.sqrt(2 * np.pi) / (2 * (s + x))
    expression = y ** 2 - 1.85 * (s + x) * (x - s) ** 2 / (np.pi * np.sqrt(x * s))
    factor2 = y + np.sqrt(max(0, expression))
    guess = factor1 * factor2 / np.sqrt(t)
    return max(guess, 1e-8)


# Iterative Method
def iterative_implied_vol(call_value: float, s: float, k: float, t: float, r: float, tol: float = 1e-8) -> float:
    """
    Calculate implied volatility using an iterative method.
    """
    vol = 0.5  # Initial guess
    max_iter = 300
    lower, upper = 0.0, 5.0

    for _ in range(max_iter):
        price = BlackScholesCalculator.call_price(s, k, t, r, vol)
        if abs(price - call_value) < max(price, call_value) * tol:
            return vol

        if price > call_value:
            upper = vol
            vol = (lower + vol) / 2
        else:
            lower = vol
            vol = (upper + vol) / 2

    raise ValueError("Iterative method failed to converge")


# Root-Finding Method
def root_finding_implied_vol(
    call_value: float, s: float, k: float, t: float, r: float, method: Union[str, Method] = "newton",
    guess_method: str = "improved", tol: float = 1e-8
) -> float:
    """
    Calculate implied volatility using a root-finding method.
    """
    def price_diff(vol: float) -> float:
        return BlackScholesCalculator.call_price(s, k, t, r, vol) - call_value

    # Select initial guess
    if guess_method == 'improved':
        x0 = improved_guess(call_value, s, k, t, r)
    elif guess_method == 'corrado_miller':
        x0 = corrado_miller_guess(call_value, s, k, t, r)
    else:
        x0 = 0.25  # Default guess

    # Root-finding algorithms
    if method in {Method.HYBR, Method.LM}:
        result = root(price_diff, x0, tol=tol, method=method.value)
        if result.success:
            return result.x[0]
        else:
            raise ValueError("Root-finding method failed to converge")
    elif method == Method.BRENTQ:
        return brentq(price_diff, 1e-12, 5.0, xtol=tol)
    elif method == Method.NEWTON:
        fprime: Callable[[float], float] = lambda vol: max(
            BlackScholesCalculator.vega(s, k, t, r, vol), 1e-8
        )
        return newton(price_diff, x0, fprime=fprime, tol=tol)
    else:
        raise ValueError(f"Unsupported root-finding method: {method}")


# Main Dispatcher Function
def implied_volatility(
    call_value: float, s: float, k: float, t: float, r: float,
    method: Union[str, Method] = Method.ITERATIVE, guess_method: str = "improved", tol: float = 1e-8
) -> float:
    """
    Main function to calculate implied volatility using the chosen method.
    """
    if isinstance(method, str):
        method = Method(method)

    if method == Method.ITERATIVE:
        return iterative_implied_vol(call_value, s, k, t, r, tol)
    elif method in {Method.HYBR, Method.LM, Method.BRENTQ, Method.NEWTON}:
        return root_finding_implied_vol(call_value, s, k, t, r, method, guess_method, tol)
    else:
        raise ValueError(f"Unsupported method: {method}")
