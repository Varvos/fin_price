import math
import pytest
import numpy as np
from finpricing.utils.bs_utils import BlackScholesCalculator
from finpricing.models.black_scholes.implied_volatility import (
    ImpliedVolatilityCalculator,
    ImpliedVolatilityParams,
    Method,
)

S, K, T, R = 100.0, 100.0, 1.0, 0.05


def call(s=S, k=K, t=T, r=R, sigma=0.2) -> float:
    return BlackScholesCalculator.call_price(s, k, t, r, sigma)


def put(s=S, k=K, t=T, r=R, sigma=0.2) -> float:
    return BlackScholesCalculator.put_price(s, k, t, r, sigma)


class TestCallPutParity:
    """C - P = S - K * exp(-rT)"""

    @pytest.mark.parametrize("s,k", [(100, 100), (120, 100), (80, 100), (100, 80)])
    def test_parity(self, s, k):
        c = call(s=s, k=k)
        p = put(s=s, k=k)
        expected = s - k * math.exp(-R * T)
        assert abs(c - p - expected) < 1e-10

    @pytest.mark.parametrize("sigma", [0.1, 0.2, 0.5])
    def test_parity_across_vols(self, sigma):
        c = call(sigma=sigma)
        p = put(sigma=sigma)
        expected = S - K * math.exp(-R * T)
        assert abs(c - p - expected) < 1e-10

    def test_parity_short_expiry(self):
        c = call(t=0.01)
        p = put(t=0.01)
        expected = S - K * math.exp(-R * 0.01)
        assert abs(c - p - expected) < 1e-10


class TestIVRoundTrip:
    """price → IV → price recovers original price."""

    @pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4, 0.6])
    def test_call_round_trip(self, sigma):
        target = call(sigma=sigma)
        params = ImpliedVolatilityParams(call_value=target, s=S, k=K, t=T, r=R)
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        recovered = call(sigma=iv)
        assert abs(recovered - target) < 1e-6

    @pytest.mark.parametrize("method", [Method.BRENTQ, Method.NEWTON, Method.ITERATIVE])
    def test_methods_agree(self, method):
        target = call(sigma=0.25)
        params = ImpliedVolatilityParams(call_value=target, s=S, k=K, t=T, r=R, method=method)
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        assert abs(iv - 0.25) < 1e-5

    def test_otm_call(self):
        target = call(s=90.0, k=110.0, sigma=0.3)
        params = ImpliedVolatilityParams(call_value=target, s=90.0, k=110.0, t=T, r=R)
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        recovered = call(s=90.0, k=110.0, sigma=iv)
        assert abs(recovered - target) < 1e-6


class TestGreeks:
    """Finite-difference verification of first-order Greeks."""

    H = 1e-4
    TOL = 1e-4

    def test_delta_call(self):
        _, d2 = BlackScholesCalculator.calculate_d1_d2(S, K, T, R, 0.2)
        from scipy.stats import norm
        d1, _ = BlackScholesCalculator.calculate_d1_d2(S, K, T, R, 0.2)
        analytic_delta = norm.cdf(d1)
        fd_delta = (call(s=S + self.H) - call(s=S - self.H)) / (2 * self.H)
        assert abs(analytic_delta - fd_delta) < self.TOL

    def test_delta_put(self):
        from scipy.stats import norm
        d1, _ = BlackScholesCalculator.calculate_d1_d2(S, K, T, R, 0.2)
        analytic_delta = norm.cdf(d1) - 1
        fd_delta = (put(s=S + self.H) - put(s=S - self.H)) / (2 * self.H)
        assert abs(analytic_delta - fd_delta) < self.TOL

    def test_vega(self):
        analytic_vega = BlackScholesCalculator.vega(S, K, T, R, 0.2)
        fd_vega = (call(sigma=0.2 + self.H) - call(sigma=0.2 - self.H)) / (2 * self.H)
        assert abs(analytic_vega - fd_vega) < self.TOL

    def test_call_put_gamma_equal(self):
        h = 0.5
        gamma_call = (call(s=S + h) - 2 * call() + call(s=S - h)) / h ** 2
        gamma_put = (put(s=S + h) - 2 * put() + put(s=S - h)) / h ** 2
        assert abs(gamma_call - gamma_put) < 1e-4
