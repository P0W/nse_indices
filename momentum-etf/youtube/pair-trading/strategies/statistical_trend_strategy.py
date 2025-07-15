"""
Pure-equity Stat-Trend Hybrid (Nifty-50 cash segment)
Author:  <you>
Date:    2024-07-15

A strategy that uses statistical measures and trend indicators to identify
and follow trends in stock prices. This strategy combines z-score mean reversion
with momentum filters using EMA and ADX indicators.
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import time

from .base_strategy import BaseStrategy, StrategyConfig


class StatisticalTrendStrategy(BaseStrategy):
    """
    Pure-equity Stat-Trend Hybrid Strategy

    This strategy combines:
    1. Z-score based mean reversion signals
    2. EMA trend filters (fast vs slow)
    3. ADX momentum strength filter
    4. ATR-based position sizing
    5. Risk management with maximum positions
    """

    params = (
        ("lookback", 20),  # Lookback period for z-score calculation
        ("z_entry", 2.0),  # Z-score threshold to trigger entry
        ("z_exit", 1.0),  # Z-score threshold to trigger exit
        ("ema_fast", 20),  # Fast EMA period
        ("ema_slow", 50),  # Slow EMA period
        ("adx_period", 14),  # ADX period
        ("adx_level", 25),  # Minimum ADX level for trend strength
        ("max_risk", 0.008),  # Maximum risk per trade (0.8% of equity)
        ("slippage", 0.0002),  # Slippage percentage
        ("comm", 0.0003),  # Commission (3 bps one-way for equity delivery)
        ("short", False),  # Enable short selling (set True if you have SLB)
        ("square_off_hour", 15),  # Square off hour
        ("square_off_minute", 20),  # Square off minute
        ("max_positions", 5),  # Maximum concurrent positions
        ("printlog", False),
    )

    # Now using log method from BaseStrategy

    def __init__(self):
        """Initialize indicators for all data feeds"""
        super().__init__()

        # Dictionary to store indicators for each data feed
        self.indicators = {}

        for data in self.datas:
            self.indicators[data._name] = {
                "ema_fast": bt.indicators.EMA(data.close, period=self.params.ema_fast),
                "ema_slow": bt.indicators.EMA(data.close, period=self.params.ema_slow),
                "adx": bt.indicators.ADX(data, period=self.params.adx_period),
                "atr": bt.indicators.ATR(data, period=14),
                # Daily returns for z-score calculation
                "ret": bt.indicators.PctChange(data.close, period=1),
                # Z-score on daily returns
                "z": self.create_zscore_indicator(data),
            }

    def create_zscore_indicator(self, data):
        """Create a custom z-score indicator for the data"""

        class ZScore(bt.Indicator):
            lines = ("zscore",)
            params = (("period", self.params.lookback),)

            def __init__(self):
                self.returns = bt.indicators.PctChange(data.close, period=1)
                self.addminperiod(self.params.period)

            def next(self):
                returns_array = np.array(
                    [self.returns[-i] for i in range(self.params.period)]
                )
                returns_array = returns_array[::-1]  # Reverse to chronological order

                if len(returns_array) >= self.params.period:
                    mean_ret = np.mean(returns_array)
                    std_ret = np.std(returns_array)

                    if std_ret > 0:
                        current_ret = self.returns[0]
                        self.lines.zscore[0] = (current_ret - mean_ret) / std_ret
                    else:
                        self.lines.zscore[0] = 0.0
                else:
                    self.lines.zscore[0] = 0.0

        return ZScore()

    def is_square_off_time(self):
        """Check if it's time to square off all positions"""
        current_time = self.datas[0].datetime.time(0)
        square_off_time = time(
            self.params.square_off_hour, self.params.square_off_minute
        )
        return current_time >= square_off_time

    def close_all_positions(self):
        """Close all open positions"""
        for data in self.datas:
            position = self.getposition(data)
            if position.size != 0:
                self.close(data=data)
                self.log(f"Square off: Closing position for {data._name}")

    def get_candidates(self):
        """Get candidate stocks for trading based on z-score and momentum filters"""
        candidates = []
        debug_info = []

        for data in self.datas:
            # Need sufficient data for indicators
            if len(data) < max(self.params.lookback, self.params.ema_slow) + 1:
                debug_info.append(f"{data._name}: Insufficient data ({len(data)} bars)")
                continue

            indicators = self.indicators[data._name]

            # Get current values
            z_score = (
                indicators["z"].zscore[0] if len(indicators["z"].zscore) > 0 else 0
            )
            ema_fast = (
                indicators["ema_fast"][0] if len(indicators["ema_fast"]) > 0 else 0
            )
            ema_slow = (
                indicators["ema_slow"][0] if len(indicators["ema_slow"]) > 0 else 0
            )
            adx = indicators["adx"][0] if len(indicators["adx"]) > 0 else 0

            debug_info.append(
                f"{data._name}: z={z_score:.3f}, adx={adx:.1f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}"
            )

            # Check entry conditions - RELAXED CONDITIONS
            z_threshold = self.params.z_entry
            adx_threshold = self.params.adx_level

            if abs(z_score) > z_threshold and adx > adx_threshold:
                if self.params.short:
                    # Long on negative z-score with uptrend, short on positive z-score with downtrend
                    if z_score < -z_threshold and ema_fast > ema_slow:
                        candidates.append((data, 1, z_score))  # Long
                        debug_info.append(
                            f"{data._name}: LONG candidate (z={z_score:.3f})"
                        )
                    elif z_score > z_threshold and ema_fast < ema_slow:
                        candidates.append((data, -1, z_score))  # Short
                        debug_info.append(
                            f"{data._name}: SHORT candidate (z={z_score:.3f})"
                        )
                else:
                    # Long-only: only buy on negative z-score with uptrend OR positive z-score in strong downtrend (contrarian)
                    if z_score < -z_threshold and ema_fast > ema_slow:
                        candidates.append((data, 1, z_score))  # Long only
                        debug_info.append(
                            f"{data._name}: LONG candidate (z={z_score:.3f})"
                        )
                    elif z_score > z_threshold and ema_fast < ema_slow:
                        # Allow contrarian long entries when stock is oversold in downtrend
                        candidates.append((data, 1, z_score))  # Contrarian long
                        debug_info.append(
                            f"{data._name}: CONTRARIAN LONG candidate (z={z_score:.3f})"
                        )

        # Log debug info occasionally
        if len(debug_info) > 0 and self.params.printlog:
            self.log(f"Scan results: {len(candidates)} candidates")
            for info in debug_info[:5]:  # Show first 5 for brevity
                self.log(info)

        return candidates

    def calculate_position_size(self, data):
        """Calculate position size based on ATR and risk management"""
        try:
            indicators = self.indicators[data._name]
            atr = indicators["atr"][0] if len(indicators["atr"]) > 0 else 0

            if atr <= 0:
                return 0

            # Risk amount based on portfolio value
            portfolio_value = self.broker.getvalue()
            risk_amount = portfolio_value * self.params.max_risk

            # Position size based on ATR stop loss (0.75 * ATR as in original)
            stop_distance = 0.75 * atr

            if stop_distance > 0:
                size = int(risk_amount / stop_distance)
                return max(size, 0)
            else:
                return 0

        except Exception as e:
            self.log(f"Error calculating position size for {data._name}: {e}")
            return 0

    def should_exit_position(self, data):
        """Check if existing position should be exited based on z-score reversal"""
        position = self.getposition(data)
        if position.size == 0:
            return False

        indicators = self.indicators[data._name]
        z_score = indicators["z"].zscore[0] if len(indicators["z"].zscore) > 0 else 0

        # Exit if z-score has reverted below exit threshold
        return abs(z_score) < self.params.z_exit

    def count_positions(self):
        """Count current number of open positions"""
        count = 0
        for data in self.datas:
            if self.getposition(data).size != 0:
                count += 1
        return count

    def execute_strategy(self):
        """Execute the main strategy logic"""
        # Check if it's time to square off all positions
        if self.is_square_off_time():
            self.close_all_positions()
            return

        # Exit existing positions if z-score has reverted
        for data in self.datas:
            if self.should_exit_position(data):
                self.close(data=data)
                self.log(f"Exiting position for {data._name} - z-score reverted")

        # Get trading candidates
        candidates = self.get_candidates()

        if not candidates:
            return

        # Limit to maximum positions
        current_positions = self.count_positions()
        available_slots = max(0, self.params.max_positions - current_positions)

        if available_slots == 0:
            return

        # Sort by absolute z-score (strongest signals first)
        candidates.sort(key=lambda x: abs(x[2]), reverse=True)

        # Take positions up to available slots
        for data, direction, z_score in candidates[:available_slots]:
            # Skip if already have position in this stock
            if self.getposition(data).size != 0:
                continue

            size = self.calculate_position_size(data)

            if size > 0:
                if direction > 0:  # Long
                    self.buy(data=data, size=size)
                    self.log(
                        f"Entering LONG {data._name}, z-score: {z_score:.3f}, size: {size}"
                    )
                elif direction < 0:  # Short
                    self.sell(data=data, size=size)
                    self.log(
                        f"Entering SHORT {data._name}, z-score: {z_score:.3f}, size: {size}"
                    )


class StatisticalTrendConfig(StrategyConfig):
    """
    Configuration for Pure-equity Stat-Trend Hybrid Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for statistical trend strategy experiments
        """
        return {
            "lookback": [15, 20, 25, 30, 40],  # Extended for intraday
            "z_entry": [1.0, 1.5, 2.0, 2.5],  # More lenient starting from 1.0
            "z_exit": [0.5, 0.75, 1.0, 1.5],
            "ema_fast": [15, 20, 25, 30, 40],  # Extended for intraday
            "ema_slow": [40, 50, 60, 80, 100],  # Extended for intraday
            "adx_period": [10, 14, 18, 24, 30],  # Extended for intraday
            "adx_level": [15, 20, 25, 30],  # More lenient starting from 15
            "max_risk": [0.005, 0.008, 0.01, 0.012],
            "max_positions": [5, 8, 10, 12],  # Increased maximum positions
            "square_off_hour": [15],
            "square_off_minute": [20],
            "short": [False, True],
        }

    def get_intraday_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define parameter grid optimized for intraday (5m, 15m) trading
        """
        return {
            "lookback": [48, 60, 72, 96],  # 4-8 hours of 5m data
            "z_entry": [0.8, 1.0, 1.2, 1.5],  # More sensitive for intraday
            "z_exit": [0.3, 0.5, 0.7, 1.0],  # Quicker exits for intraday
            "ema_fast": [24, 36, 48, 60],  # 2-5 hours of 5m data
            "ema_slow": [72, 96, 120, 144],  # 6-12 hours of 5m data
            "adx_period": [24, 36, 48],  # 2-4 hours of 5m data
            "adx_level": [20, 25, 30, 35],  # Higher ADX threshold for intraday noise
            "max_risk": [0.003, 0.005, 0.008],  # Lower risk for intraday volatility
            "max_positions": [3, 5, 8],  # Fewer positions for intraday management
            "square_off_hour": [15],
            "square_off_minute": [20],
            "short": [False, True],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for statistical trend strategy
        """
        return {
            "lookback": 20,
            "z_entry": 1.5,  # Reduced from 2.0 to get more trades
            "z_exit": 0.75,  # Reduced from 1.0
            "ema_fast": 20,
            "ema_slow": 50,
            "adx_period": 14,
            "adx_level": 20,  # Reduced from 25 to be more lenient
            "max_risk": 0.008,
            "slippage": 0.0002,
            "comm": 0.0003,
            "short": False,
            "square_off_hour": 15,
            "square_off_minute": 20,
            "max_positions": 8,  # Increased from 5 to allow more positions
            "printlog": True,  # Enable logging to debug
        }

    def get_intraday_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters optimized for intraday trading
        """
        return {
            "lookback": 60,  # 5 hours of 5m data
            "z_entry": 1.0,  # More sensitive for intraday
            "z_exit": 0.5,  # Quicker exits
            "ema_fast": 36,  # 3 hours of 5m data
            "ema_slow": 96,  # 8 hours of 5m data
            "adx_period": 36,  # 3 hours of 5m data
            "adx_level": 25,  # Higher threshold for noise
            "max_risk": 0.005,  # Lower risk for volatility
            "slippage": 0.0005,  # Higher slippage for intraday
            "comm": 0.0003,
            "short": False,
            "square_off_hour": 15,
            "square_off_minute": 20,
            "max_positions": 5,  # Fewer positions for management
            "printlog": True,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate statistical trend strategy parameters
        """
        # EMA fast should be less than slow
        if params.get("ema_fast", 0) >= params.get("ema_slow", 0):
            return False

        # Z-score thresholds should be valid
        z_entry = params.get("z_entry", 2.0)
        z_exit = params.get("z_exit", 1.0)
        if z_exit >= z_entry or z_exit <= 0 or z_entry <= 0:
            return False

        # All positive parameters must be positive
        positive_params = [
            "lookback",
            "z_entry",
            "z_exit",
            "ema_fast",
            "ema_slow",
            "adx_period",
            "adx_level",
            "max_risk",
            "max_positions",
        ]

        for param in positive_params:
            if params.get(param, 0) <= 0:
                return False

        # Risk per trade should be reasonable
        if params.get("max_risk", 0) > 0.02:  # Max 2% risk per trade
            return False

        # Max positions should be reasonable
        if params.get("max_positions", 0) > 15:
            return False

        # ADX level should be between 10 and 50
        adx_level = params.get("adx_level", 25)
        if adx_level < 10 or adx_level > 50:
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the statistical trend strategy class
        """
        return StatisticalTrendStrategy

    def get_required_data_feeds(self) -> int:
        """
        Statistical trend strategy works with multiple stocks
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights optimized for stat-trend hybrid strategy
        """
        return {
            "total_return": 0.35,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.25,  # Balanced approach for mean reversion + trend following
        }
