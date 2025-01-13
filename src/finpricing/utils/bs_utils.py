"""
Utility module for Black-Scholes calculations
"""
import numpy as np
from scipy.stats import norm
from scipy.special import ndtr


class BlackScholesCalculator:
    """Utility class for Black-Scholes calculations."""

    @staticmethod
    def calculate_d1_d2(s: float, k: float, t: float, r: float, sigma: float):
        """
        Calculate the d1 and d2 terms for the Black-Scholes model.
        """
        if any(param <= 0 for param in (s, k, t, sigma)):
            raise ValueError("Stock price, strike price, time to maturity, and volatility must be positive.")

        log_money = np.log(s / k)
        d1 = (log_money + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return d1, d2

    @staticmethod
    def call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call price"""
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def put_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put price"""
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    @staticmethod
    def vega(s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate Vega of the option"""
        d1, _ = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return s * norm.pdf(d1) * np.sqrt(t)
