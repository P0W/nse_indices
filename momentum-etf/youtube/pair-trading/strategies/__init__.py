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
from .pmv_momentum_strategy import PMVMomentumStrategy, PMVMomentumConfig
from .nifty_shop_strategy import NiftyShopStrategy, NiftyShopConfig

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
    "PMVMomentumStrategy",
    "PMVMomentumConfig",
    "NiftyShopStrategy",
    "NiftyShopConfig",
]
