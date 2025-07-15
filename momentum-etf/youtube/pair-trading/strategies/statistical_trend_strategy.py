"""
Statistical Trend Following Strategy

A strategy that uses statistical measures and trend indicators to identify
and follow trends in stock prices. This strategy combines multiple statistical
indicators to create robust trend-following signals.
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .base_strategy import BaseStrategy, StrategyConfig


class StatisticalTrendStrategy(BaseStrategy):
    """
    Statistical Trend Following Strategy

    This strategy combines multiple statistical indicators:
    1. Moving Average Convergence Divergence (MACD)
    2. Relative Strength Index (RSI)
    3. Bollinger Bands
    4. Average True Range (ATR) for position sizing
    5. Linear regression slope for trend strength
    """

    params = (
        ("macd_fast", 12),  # MACD fast period
        ("macd_slow", 26),  # MACD slow period
        ("macd_signal", 9),  # MACD signal period
        ("rsi_period", 14),  # RSI period
        ("bb_period", 20),  # Bollinger Bands period
        ("bb_std", 2.0),  # Bollinger Bands standard deviations
        ("atr_period", 14),  # ATR period for volatility
        ("trend_period", 20),  # Period for trend strength calculation
        ("rsi_oversold", 30),  # RSI oversold threshold
        ("rsi_overbought", 70),  # RSI overbought threshold
        ("trend_threshold", 0.001),  # Minimum trend slope threshold
        ("risk_per_trade", 0.02),  # Risk per trade
        ("max_positions", 8),  # Maximum positions
        ("rebalance_freq", 3),  # Rebalance frequency
        ("printlog", False),
    )

    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}: {txt}")

    def __init__(self):
        """Initialize indicators for all data feeds"""
        super().__init__()
        self.rebalance_counter = 0

        # Dictionary to store indicators for each data feed
        self.indicators = {}

        for data in self.datas:
            self.indicators[data._name] = {
                "macd": bt.indicators.MACD(
                    data.close,
                    period_me1=self.params.macd_fast,
                    period_me2=self.params.macd_slow,
                    period_signal=self.params.macd_signal,
                ),
                "rsi": bt.indicators.RSI(data.close, period=self.params.rsi_period),
                "bb": bt.indicators.BollingerBands(
                    data.close,
                    period=self.params.bb_period,
                    devfactor=self.params.bb_std,
                ),
                "atr": bt.indicators.ATR(data, period=self.params.atr_period),
                "sma": bt.indicators.SMA(data.close, period=self.params.trend_period),
            }

    def calculate_trend_strength(self, data):
        """Calculate trend strength using linear regression slope"""
        if len(data.close) < self.params.trend_period:
            return 0.0

        # Get recent prices
        prices = []
        for i in range(self.params.trend_period):
            prices.append(data.close[-i])

        # Reverse to get chronological order
        prices = prices[::-1]

        # Calculate linear regression slope
        x = np.arange(len(prices))
        y = np.array(prices)

        # Handle edge cases
        if len(y) < 2 or np.std(y) == 0:
            return 0.0

        try:
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            # Normalize by average price to get percentage slope
            avg_price = np.mean(y)
            if avg_price > 0:
                normalized_slope = slope / avg_price
            else:
                normalized_slope = 0.0

            return normalized_slope
        except:
            return 0.0

    def get_signal_strength(self, data):
        """Calculate combined signal strength for a data feed"""
        if data._name not in self.indicators:
            return 0.0, {}

        ind = self.indicators[data._name]

        # Initialize signal components
        signals = {"macd_signal": 0, "rsi_signal": 0, "bb_signal": 0, "trend_signal": 0}

        try:
            # MACD Signal
            if len(ind["macd"].macd) > 0 and len(ind["macd"].signal) > 0:
                if ind["macd"].macd[0] > ind["macd"].signal[0]:
                    signals["macd_signal"] = 1  # Bullish
                else:
                    signals["macd_signal"] = -1  # Bearish

            # RSI Signal
            if len(ind["rsi"]) > 0:
                rsi_val = ind["rsi"][0]
                if rsi_val < self.params.rsi_oversold:
                    signals["rsi_signal"] = 1  # Oversold - bullish
                elif rsi_val > self.params.rsi_overbought:
                    signals["rsi_signal"] = -1  # Overbought - bearish
                else:
                    signals["rsi_signal"] = 0  # Neutral

            # Bollinger Bands Signal
            if (
                len(ind["bb"].top) > 0
                and len(ind["bb"].bot) > 0
                and len(ind["bb"].mid) > 0
            ):

                current_price = data.close[0]
                if current_price > ind["bb"].top[0]:
                    signals["bb_signal"] = -1  # Above upper band - bearish
                elif current_price < ind["bb"].bot[0]:
                    signals["bb_signal"] = 1  # Below lower band - bullish
                elif current_price > ind["bb"].mid[0]:
                    signals["bb_signal"] = 0.5  # Above middle - mildly bullish
                else:
                    signals["bb_signal"] = -0.5  # Below middle - mildly bearish

            # Trend Signal
            trend_strength = self.calculate_trend_strength(data)
            if abs(trend_strength) > self.params.trend_threshold:
                signals["trend_signal"] = 1 if trend_strength > 0 else -1
            else:
                signals["trend_signal"] = 0

            # Combine signals (weighted average)
            weights = {
                "macd_signal": 0.3,
                "rsi_signal": 0.25,
                "bb_signal": 0.25,
                "trend_signal": 0.2,
            }

            combined_signal = sum(signals[key] * weights[key] for key in signals)

            return combined_signal, signals

        except Exception as e:
            self.log(f"Error calculating signals for {data._name}: {e}")
            return 0.0, signals

    def get_position_size(self, data):
        """Calculate position size based on ATR and risk management"""
        if data._name not in self.indicators:
            return 0

        try:
            available_cash = self.broker.get_cash()
            risk_amount = available_cash * self.params.risk_per_trade

            # Use ATR for volatility-based sizing
            atr = self.indicators[data._name]["atr"]
            if len(atr) > 0 and atr[0] > 0:
                # Size based on ATR - risk one ATR per trade
                current_price = data.close[0]
                if current_price > 0:
                    size = int(risk_amount / (atr[0] * 2))  # 2x ATR for stop loss
                    return max(size, 1)

            # Fallback to simple sizing
            current_price = data.close[0]
            if current_price > 0:
                size = int(risk_amount / current_price)
                return max(size, 1)

        except Exception as e:
            self.log(f"Error calculating position size for {data._name}: {e}")

        return 1

    def should_enter_long(self, data):
        """Check if we should enter a long position"""
        if self.getposition(data).size != 0:
            return False

        signal_strength, signals = self.get_signal_strength(data)

        # Strong bullish signal (threshold can be adjusted)
        return signal_strength > 0.3

    def should_enter_short(self, data):
        """Check if we should enter a short position"""
        if self.getposition(data).size != 0:
            return False

        signal_strength, signals = self.get_signal_strength(data)

        # Strong bearish signal
        return signal_strength < -0.3

    def should_exit_position(self, data):
        """Check if we should exit current position"""
        position = self.getposition(data)
        if position.size == 0:
            return False

        signal_strength, signals = self.get_signal_strength(data)

        # Exit long position if signal turns bearish or weak
        if position.size > 0:
            return signal_strength < 0.1

        # Exit short position if signal turns bullish or weak
        else:
            return signal_strength > -0.1

    def count_positions(self):
        """Count current number of positions"""
        count = 0
        for data in self.datas:
            if self.getposition(data).size != 0:
                count += 1
        return count

    def execute_strategy(self):
        """Execute statistical trend strategy logic"""
        # Rebalance only on specified frequency
        if self.rebalance_counter % self.params.rebalance_freq != 0:
            self.rebalance_counter += 1
            return

        # Need minimum data for indicators
        min_data_needed = max(
            self.params.macd_slow + self.params.macd_signal,
            self.params.rsi_period,
            self.params.bb_period,
            self.params.trend_period,
        )

        if len(self.datas[0]) < min_data_needed:
            self.rebalance_counter += 1
            return

        # First, check exit conditions for existing positions
        for data in self.datas:
            if self.should_exit_position(data):
                self.close(data=data)
                self.log(f"Closing position for {data._name}")

        # Then, look for new entry opportunities
        current_positions = self.count_positions()

        if current_positions < self.params.max_positions:
            # Find potential trades
            potential_trades = []

            for data in self.datas:
                if self.getposition(data).size == 0:  # No current position
                    signal_strength, signals = self.get_signal_strength(data)

                    if abs(signal_strength) > 0.3:  # Strong signal
                        potential_trades.append((data, signal_strength, signals))

            # Sort by signal strength (strongest first)
            potential_trades.sort(key=lambda x: abs(x[1]), reverse=True)

            # Enter positions up to max limit
            remaining_slots = self.params.max_positions - current_positions

            for data, signal_strength, signals in potential_trades[:remaining_slots]:
                position_size = self.get_position_size(data)

                if signal_strength > 0.3:  # Long signal
                    self.buy(data=data, size=position_size)
                    self.log(
                        f"Entering LONG {data._name}, Signal: {signal_strength:.3f}"
                    )

                elif signal_strength < -0.3:  # Short signal
                    self.sell(data=data, size=position_size)
                    self.log(
                        f"Entering SHORT {data._name}, Signal: {signal_strength:.3f}"
                    )

        self.rebalance_counter += 1


class StatisticalTrendConfig(StrategyConfig):
    """
    Configuration for Statistical Trend Following Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for statistical trend strategy experiments
        """
        return {
            "macd_fast": [8, 12, 16, 20],
            "macd_slow": [21, 26, 30, 35],
            "macd_signal": [6, 9, 12, 15],
            "rsi_period": [10, 14, 18, 21],
            "bb_period": [15, 20, 25, 30],
            "bb_std": [1.5, 2.0, 2.5, 3.0],
            "atr_period": [10, 14, 18, 21],
            "trend_period": [15, 20, 25, 30],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "trend_threshold": [0.0005, 0.001, 0.002, 0.003],
            "risk_per_trade": [0.01, 0.015, 0.02, 0.025],
            "max_positions": [5, 8, 10, 12],
            "rebalance_freq": [1, 3, 5, 7],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for statistical trend strategy
        """
        return {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "trend_period": 20,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "trend_threshold": 0.001,
            "risk_per_trade": 0.02,
            "max_positions": 8,
            "rebalance_freq": 3,
            "printlog": False,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate statistical trend strategy parameters
        """
        # MACD fast should be less than slow
        if params.get("macd_fast", 0) >= params.get("macd_slow", 0):
            return False

        # RSI thresholds should be valid
        oversold = params.get("rsi_oversold", 30)
        overbought = params.get("rsi_overbought", 70)
        if oversold >= overbought or oversold < 10 or overbought > 90:
            return False

        # All positive parameters must be positive
        positive_params = [
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "rsi_period",
            "bb_period",
            "bb_std",
            "atr_period",
            "trend_period",
            "trend_threshold",
            "risk_per_trade",
            "max_positions",
            "rebalance_freq",
        ]

        for param in positive_params:
            if params.get(param, 0) <= 0:
                return False

        # Risk per trade should be reasonable
        if params.get("risk_per_trade", 0) > 0.05:
            return False

        # Max positions should be reasonable
        if params.get("max_positions", 0) > 20:
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
        Weights optimized for trend following strategy
        """
        return {
            "total_return": 0.4,
            "sharpe_ratio": 0.35,
            "max_drawdown": 0.25,  # Lower weight since trend following can have larger drawdowns
        }
