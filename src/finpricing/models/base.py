"""
Base Models for Parameters and Configuration
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Protocol
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
import yfinance as yf
import yaml
from pathlib import Path

# Base Models for Parameters and Configuration
class MarketData(BaseModel):
    """Market data parameters with validation"""
    spot_price: float = Field(gt=0, description="Current price of the underlying asset")
    risk_free_rate: float = Field(description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    timestamp: datetime = Field(default_factory=datetime.now)
    volatility_surface: Optional[Dict[str, float]] = Field(default=None)

    @validator('risk_free_rate')
    def validate_rate(cls, v: float) -> float:
        if not -1 < v < 1:
            raise ValueError("Risk-free rate must be between -100% and 100%")
        return v

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class InstrumentParams(BaseModel):
    """Base instrument parameters"""
    symbol: str = Field(description="Instrument identifier")
    expiry: Optional[datetime] = Field(default=None, description="Expiration date if applicable")
    
    class Config:
        frozen = True

class OptionParams(InstrumentParams):
    """Option-specific parameters"""
    strike_price: float = Field(gt=0, description="Strike price")
    option_type: str = Field(regex='^(call|put)$', description="Option type")

class ModelConfig(BaseModel):
    """Configuration for pricing models"""
    model_name: str
    parameters: Dict[str, float]
    calibration_method: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ModelConfig':
        """Load configuration from YAML file"""
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    class Config:
        frozen = True

# Abstract Base Classes
class PricingModel(Protocol):
    """Protocol defining interface for pricing models"""
    def price(self, market_data: MarketData, params: InstrumentParams) -> float:
        ...
    
    def calibrate(self, market_data: MarketData) -> None:
        ...

class DataSource(ABC):
    """Abstract base class for market data sources"""
    
    @abstractmethod
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch market data for given symbol and date range"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for given symbol"""
        pass

