"""
Black Scholes model implementations
"""

from typing import Dict
import numpy as np
from scipy.stats import norm
from src.finpricing.models.base import OptionParams, MarketData, PricingModel, ModelConfig, Greeks
from src.finpricing.utils.bs_utils import BlackScholesCalculator


class BlackScholesModel(PricingModel):
    """Black-Scholes option pricing model implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate the model configuration to ensure required parameters are present"""
        required_params = {'calculation_method', 'tolerance'}
        if any(
            param not in self.config.numerical_params for param in required_params
        ):
            raise ValueError(f"Missing required parameters: {required_params}")
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 used in Black-Scholes formula"""
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    def _d2(self, d1: float, sigma: float, T: float) -> float:
        """Calculate d2 used in Black-Scholes formula"""
        return d1 - sigma*np.sqrt(T)
    
    def price(self, market_data: MarketData, instrument: OptionParams) -> float:
        """
        Calculate the price of an option using the Black-Scholes model.
        
        Args:
            market_data: Market data parameters
            instrument: Option parameters
        
        Returns:
            Option price
        
        Raises:
            ValueError: If the option has expired or invalid parameters are provided
        """





        if not isinstance(instrument, OptionParams):
            raise ValueError("Black-Scholes model requires OptionParams")
            
        T = (instrument.expiry - market_data.timestamp).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            raise ValueError("Option has expired")
            
        S = market_data.spot_price
        K = instrument.strike_price
        r = market_data.risk_free_rate
        sigma = self.config.numerical_params.get('volatility', 0.2)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(d1, sigma, T)
        
        if instrument.option_type == 'call':
            return BlackScholesCalculator.call_price(
                market_data.spot_price, instrument.strike_price, T, market_data.risk_free_rate, sigma
            )
        else:
            return BlackScholesCalculator.put_price(
                market_data.spot_price, instrument.strike_price, T, market_data.risk_free_rate, sigma
            )

    def greeks(self, market_data: MarketData, instrument: OptionParams) -> Dict[str, float]:
        """Calculate the Greeks for the option."""
        T = (instrument.expiry - market_data.timestamp).days / 365.25
        sigma = self.config.numerical_params.get('volatility', 0.2)
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(
            market_data.spot_price, instrument.strike_price, T, market_data.risk_free_rate, sigma
        )

        delta = norm.cdf(d1) if instrument.option_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (market_data.spot_price * sigma * np.sqrt(T))
        vega = BlackScholesCalculator.vega(
            market_data.spot_price, instrument.strike_price, T, market_data.risk_free_rate, sigma
        )
        theta = -((market_data.spot_price * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - \
                market_data.risk_free_rate * instrument.strike_price * np.exp(-market_data.risk_free_rate * T) * norm.cdf(d2)
        rho = instrument.strike_price * T * np.exp(-market_data.risk_free_rate * T) * norm.cdf(d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho).dict()
