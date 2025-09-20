"""
Base Models for Parameters and Configuration
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


# Base Models for Parameters and Configuration
class MarketData(BaseModel):
    """Market data parameters"""
    spot_price: float = Field(gt=0, description="Current price of the underlying asset")
    risk_free_rate: float = Field(description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = True


class InstrumentParams(BaseModel):
    """Base instrument parameters"""
    symbol: str = Field(description="Instrument identifier")
    expiry: Optional[datetime] = Field(default=None, description="Expiration date if applicable")
    
    class Config:
        frozen = True

class OptionParams(InstrumentParams):
    """Option-specific parameters"""
    strike_price: float = Field(gt=0, description="Strike price")
    option_type: str = Field(pattern='^(call|put)$', description="Option type")


class ModelConfig(BaseModel):
    """Base configuration for pricing models"""
    model_type: str
    numerical_params: Dict[str, Any] = Field(default_factory=dict)  # type: ignore

    @field_validator("numerical_params")
    def validate_numerical_params(cls, value):
        if not isinstance(value, dict):
            raise ValueError("numerical_params must be a dictionary.")
        return value

    class Config:
        frozen = True


class Greeks(BaseModel):
    """Output structure for Greeks calculation"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# Abstract Base Classes
class PricingModel(ABC):
    """Abstract base class for pricing models"""
    
    @abstractmethod
    def price(self, market_data: MarketData, instrument: InstrumentParams) -> float:
        """Calculate instrument price"""
        pass

    @abstractmethod
    def greeks(self, market_data: MarketData, instrument: InstrumentParams) -> Dict[str, float]:
        """Calculate sensitivity measures (Greeks)"""
        pass


class DataSource(ABC):
    """Abstract base class for market data sources"""

    @abstractmethod
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Any:
        """Fetch market data for given symbol and date range"""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a given symbol"""
        pass

