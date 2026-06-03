"""
Abstract base class for model calibrators.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Calibrator(ABC):
    """Base class for all calibrators."""

    @abstractmethod
    def calibrate(self, option_surface: pd.DataFrame, **kwargs) -> tuple:
        """
        Fit model parameters to an observed option surface.

        Args:
            option_surface: DataFrame with at minimum columns:
                - 'strike': option strike price
                - 'maturity': time to expiry in years
                - 'price': observed market price
                Optional columns:
                - 'option_type': 'call' or 'put' (default 'call')
                - 'weight': calibration weight (default 1.0)

        Returns:
            Tuple of (fitted_params, auxiliary_params).
        """
        ...

    @staticmethod
    def _validate_surface(df: pd.DataFrame) -> pd.DataFrame:
        required = {'strike', 'maturity', 'price'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"option_surface missing required columns: {missing}")
        if 'option_type' not in df.columns:
            df = df.copy()
            df['option_type'] = 'call'
        if 'weight' not in df.columns:
            df = df.copy()
            df['weight'] = 1.0
        return df
