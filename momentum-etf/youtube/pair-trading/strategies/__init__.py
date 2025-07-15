"""
Strategies package for trading strategies
"""

from .base_strategy import BaseStrategy, StrategyConfig
from .momentum_strategy import MomentumTrendStrategy, AdaptiveMomentumConfig
from .pairs_strategy import PairsStrategy, PairsConfig
from .mean_reversion_strategy import (
    PortfolioMeanReversionStrategy,
    PortfolioMeanReversionConfig,
)
from .statistical_trend_strategy import StatisticalTrendStrategy, StatisticalTrendConfig

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "MomentumTrendStrategy",
    "AdaptiveMomentumConfig",
    "PairsStrategy",
    "PairsConfig",
    "PortfolioMeanReversionStrategy",
    "PortfolioMeanReversionConfig",
    "StatisticalTrendStrategy",
    "StatisticalTrendConfig",
]
