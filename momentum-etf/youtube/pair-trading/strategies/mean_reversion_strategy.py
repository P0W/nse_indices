"""
Portfolio Mean Reversion Strategy

A strategy that identifies and trades mean reversion opportunities in a portfolio of stocks.
This strategy looks for stocks that have deviated significantly from their historical mean
and trades on the expectation that prices will revert to the mean.
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .base_strategy import BaseStrategy, StrategyConfig


class PortfolioMeanReversionStrategy(BaseStrategy):
    """
    Portfolio Mean Reversion Strategy

    This strategy:
    1. Calculates rolling mean and standard deviation for each stock
    2. Identifies oversold/overbought conditions based on z-scores
    3. Takes long positions in oversold stocks and short positions in overbought stocks
    4. Exits positions when prices revert towards the mean
    """

    params = (
        ("lookback_period", 20),  # Period for calculating mean and std
        ("entry_zscore", 2.0),  # Z-score threshold for entry
        ("exit_zscore", 0.5),  # Z-score threshold for exit
        ("max_positions", 5),  # Maximum number of positions
        ("rebalance_freq", 5),  # Rebalance frequency in days
        ("risk_per_trade", 0.02),  # Risk per trade (2%)
        ("stop_loss_zscore", 3.0),  # Stop loss z-score threshold
        ("min_volume_ratio", 0.5),  # Minimum volume ratio vs average
        ("printlog", False),
    )

    # Now using log method from BaseStrategy

    def __init__(self):
        """Initialize the strategy"""
        super().__init__()
        self.order_dict = {}  # Track orders for each data feed
        self.position_sizes = {}  # Track position sizes
        self.rebalance_counter = 0

        # Initialize indicators for each data feed
        self.sma_dict = {}
        self.std_dict = {}
        self.volume_sma_dict = {}

        for i, data in enumerate(self.datas):
            self.sma_dict[data._name] = bt.indicators.SMA(
                data.close, period=self.params.lookback_period
            )
            self.std_dict[data._name] = bt.indicators.StandardDeviation(
                data.close, period=self.params.lookback_period
            )
            self.volume_sma_dict[data._name] = bt.indicators.SMA(
                data.volume, period=self.params.lookback_period
            )

    def calculate_zscore(self, data):
        """Calculate z-score for a given data feed"""
        if data._name not in self.sma_dict or data._name not in self.std_dict:
            return 0.0

        current_price = data.close[0]
        mean_price = self.sma_dict[data._name][0]
        std_price = self.std_dict[data._name][0]

        if std_price == 0:
            return 0.0

        zscore = (current_price - mean_price) / std_price
        return zscore

    def check_volume_condition(self, data):
        """Check if volume condition is met"""
        if data._name not in self.volume_sma_dict:
            return True  # Default to True if no volume data

        current_volume = data.volume[0]
        avg_volume = self.volume_sma_dict[data._name][0]

        if avg_volume == 0:
            return True

        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.params.min_volume_ratio

    def get_position_size(self, data):
        """Calculate position size based on risk management"""
        available_cash = self.broker.get_cash()
        risk_amount = available_cash * self.params.risk_per_trade

        # Simple position sizing based on price
        current_price = data.close[0]
        if current_price <= 0:
            return 0

        size = int(risk_amount / current_price)
        return max(size, 1)  # Minimum size of 1

    def should_enter_long(self, data):
        """Check if we should enter a long position"""
        zscore = self.calculate_zscore(data)
        volume_ok = self.check_volume_condition(data)

        # Enter long if price is significantly below mean (oversold)
        return (
            zscore <= -self.params.entry_zscore
            and volume_ok
            and self.getposition(data).size == 0
        )

    def should_enter_short(self, data):
        """Check if we should enter a short position"""
        zscore = self.calculate_zscore(data)
        volume_ok = self.check_volume_condition(data)

        # Enter short if price is significantly above mean (overbought)
        return (
            zscore >= self.params.entry_zscore
            and volume_ok
            and self.getposition(data).size == 0
        )

    def should_exit_position(self, data):
        """Check if we should exit current position"""
        position = self.getposition(data)
        if position.size == 0:
            return False

        zscore = self.calculate_zscore(data)

        # Exit long position if price reverted towards mean
        if position.size > 0:  # Long position
            return (
                zscore >= -self.params.exit_zscore
                or zscore <= -self.params.stop_loss_zscore
            )

        # Exit short position if price reverted towards mean
        else:  # Short position
            return (
                zscore <= self.params.exit_zscore
                or zscore >= self.params.stop_loss_zscore
            )

    def count_current_positions(self):
        """Count current number of positions"""
        count = 0
        for data in self.datas:
            if self.getposition(data).size != 0:
                count += 1
        return count

    def execute_strategy(self):
        """Execute mean reversion strategy logic"""
        # Only rebalance on specified frequency
        if self.rebalance_counter % self.params.rebalance_freq != 0:
            self.rebalance_counter += 1
            return

        # Check minimum data requirement
        if len(self.datas[0]) < self.params.lookback_period:
            self.rebalance_counter += 1
            return

        # First, check exit conditions for existing positions
        for data in self.datas:
            if self.should_exit_position(data):
                self.close(data=data)
                self.log(f"Closing position for {data._name}")

        # Then, look for new entry opportunities
        current_positions = self.count_current_positions()

        if current_positions < self.params.max_positions:
            # Find potential trades
            potential_longs = []
            potential_shorts = []

            for data in self.datas:
                if self.should_enter_long(data):
                    zscore = self.calculate_zscore(data)
                    potential_longs.append((data, abs(zscore)))

                elif self.should_enter_short(data):
                    zscore = self.calculate_zscore(data)
                    potential_shorts.append((data, abs(zscore)))

            # Sort by z-score magnitude (highest deviation first)
            potential_longs.sort(key=lambda x: x[1], reverse=True)
            potential_shorts.sort(key=lambda x: x[1], reverse=True)

            # Enter positions up to max limit
            remaining_slots = self.params.max_positions - current_positions
            all_potential = potential_longs + potential_shorts
            all_potential.sort(key=lambda x: x[1], reverse=True)

            for data, zscore_mag in all_potential[:remaining_slots]:
                position_size = self.get_position_size(data)

                if self.should_enter_long(data):
                    self.buy(data=data, size=position_size)
                    self.log(
                        f"Entering LONG position for {data._name}, Z-score: {-zscore_mag:.2f}"
                    )

                elif self.should_enter_short(data):
                    self.sell(data=data, size=position_size)
                    self.log(
                        f"Entering SHORT position for {data._name}, Z-score: {zscore_mag:.2f}"
                    )

        self.rebalance_counter += 1


class PortfolioMeanReversionConfig(StrategyConfig):
    """
    Configuration for Portfolio Mean Reversion Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for mean reversion strategy experiments
        """
        return {
            "lookback_period": [10, 15, 20, 30, 40],
            "entry_zscore": [1.5, 2.0, 2.5, 3.0],
            "exit_zscore": [0.2, 0.5, 0.8, 1.0],
            "max_positions": [3, 5, 7, 10],
            "rebalance_freq": [1, 3, 5, 10],
            "risk_per_trade": [0.01, 0.02, 0.03, 0.05],
            "stop_loss_zscore": [2.5, 3.0, 3.5, 4.0],
            "min_volume_ratio": [0.3, 0.5, 0.7, 1.0],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for mean reversion strategy
        """
        return {
            "lookback_period": 20,
            "entry_zscore": 2.0,
            "exit_zscore": 0.5,
            "max_positions": 5,
            "rebalance_freq": 5,
            "risk_per_trade": 0.02,
            "stop_loss_zscore": 3.0,
            "min_volume_ratio": 0.5,
            "printlog": False,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate mean reversion strategy parameters
        """
        # Entry z-score should be higher than exit z-score
        if params.get("entry_zscore", 0) <= params.get("exit_zscore", 0):
            return False

        # Stop loss should be higher than entry
        if params.get("stop_loss_zscore", 0) <= params.get("entry_zscore", 0):
            return False

        # All numeric parameters must be positive
        numeric_params = [
            "lookback_period",
            "entry_zscore",
            "exit_zscore",
            "max_positions",
            "rebalance_freq",
            "risk_per_trade",
            "stop_loss_zscore",
            "min_volume_ratio",
        ]

        for param in numeric_params:
            value = params.get(param, 0)
            if isinstance(value, (int, float)) and value <= 0:
                return False

        # Risk per trade should be reasonable (less than 10%)
        if params.get("risk_per_trade", 0) > 0.1:
            return False

        # Max positions should be reasonable
        if params.get("max_positions", 0) > 15:
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the mean reversion strategy class
        """
        return PortfolioMeanReversionStrategy

    def get_required_data_feeds(self) -> int:
        """
        Mean reversion strategy works with multiple stocks
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights optimized for mean reversion strategy
        """
        return {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,  # Higher weight on risk-adjusted returns
            "max_drawdown": 0.3,
        }
