"""
P=MV Momentum Strategy

A momentum-based strategy that calculates momentum as 'p=mv' where:
- m is determined using composite scoring based on RSI, VWAP and returns
- v is based on volume dynamics
- Positions are exited after a week of holding or based on stop-loss conditions

This strategy identifies momentum using a combination of price action, volume,
technical indicators like RSI, and comparative metrics like VWAP.
"""

import backtrader as bt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .base_strategy import BaseStrategy, StrategyConfig


class PMVMomentumStrategy(BaseStrategy):
    """
    P=MV Momentum Strategy Implementation
    """

    params = (
        ("rsi_period", 14),
        ("vwap_period", 20),  # Days to calculate VWAP
        ("top_n_stocks", 15),  # Number of stocks to hold in portfolio
        ("position_size", 0.2),  # Maximum position size per stock (% of portfolio)
        ("weekly_exit", True),  # Whether to exit after a week of holding
        ("stop_loss", 5.0),  # Stop loss percentage
        ("take_profit", 10.0),  # Take profit percentage
        ("printlog", False),  # Whether to print log messages
    )

    def __init__(self):
        super().__init__()
        self.inds = {}
        self.entry_dates = {}  # Track entry dates for each position
        self.peak_prices = {}  # Track peak prices for trailing stops
        self.monthly_returns = {}
        self.peak_value = self.broker.getvalue()
        self.drawdowns = []
        self.bar_executed = 0  # Counter for executed bars

        # Create indicators for each data feed
        for d in self.datas:
            # Calculate RSI
            rsi = bt.ind.RSI(d.close, period=self.p.rsi_period)

            # Create return indicators using ROC (Rate of Change)
            returns_1w = bt.ind.ROC(d.close, period=5)  # 1-week return
            returns_1mo = bt.ind.ROC(d.close, period=20)  # 1-month return
            returns_1y = bt.ind.ROC(d.close, period=252)  # 1-year return

            # VWAP calculations for different timeframes
            # For simplicity, we'll use SMA as a proxy for VWAP in backtrader
            vwap_1w = bt.ind.SMA(d.close * d.volume, period=5) / bt.ind.SMA(
                d.volume, period=5
            )
            vwap_1mo = bt.ind.SMA(d.close * d.volume, period=20) / bt.ind.SMA(
                d.volume, period=20
            )
            vwap_1y = bt.ind.SMA(d.close * d.volume, period=252) / bt.ind.SMA(
                d.volume, period=252
            )

            # Store all indicators
            self.inds[d._name] = {
                "rsi": rsi,
                "returns_1w": returns_1w,
                "returns_1mo": returns_1mo,
                "returns_1y": returns_1y,
                "vwap_1w": vwap_1w,
                "vwap_1mo": vwap_1mo,
                "vwap_1y": vwap_1y,
            }

    def _calculate_composite_score(self, data):
        """Calculate the composite score for a stock using the p=mv formula"""
        if len(data) < 252 or self.bar_executed < 252:  # Need at least a year of data
            return -999  # Invalid score

        d_name = data._name

        # Check if we have indicators for this data
        if d_name not in self.inds:
            return -999

        # Get current values - using try/except for safety
        try:
            rsi = self.inds[d_name]["rsi"][0]

            # Extract price and volume data
            price = data.close[0]
            volume = data.volume[0]

            # Ensure we have numeric values for our calculations
            returns_1w = float(self.inds[d_name]["returns_1w"][0])
            returns_1mo = float(self.inds[d_name]["returns_1mo"][0])
            returns_1y = float(self.inds[d_name]["returns_1y"][0])

            vwap_1w = float(self.inds[d_name]["vwap_1w"][0])
            vwap_1mo = float(self.inds[d_name]["vwap_1mo"][0])
            vwap_1y = float(self.inds[d_name]["vwap_1y"][0])

            # Prepare returns dict for the composite score function
            returns = {
                "1w": {"return": returns_1w, "vwap": vwap_1w, "rsi": rsi},
                "1mo": {"return": returns_1mo, "vwap": vwap_1mo, "rsi": rsi},
                "1y": {"return": returns_1y, "vwap": vwap_1y, "rsi": rsi},
            }

            # Calculate momentum score
            score, _, _, _ = self._composite_score(returns, price)

            # Adjust with volume factor (normalized volume relative to average)
            # Safely calculate average volume
            volumes = []
            for i in range(1, min(21, len(data))):
                if data.volume[-i] > 0:  # Ensure positive volume
                    volumes.append(data.volume[-i])

            avg_volume = np.mean(volumes) if volumes else 1.0
            volume_factor = volume / avg_volume if avg_volume > 0 else 1.0

            # Final p=mv score: momentum × volume factor
            final_score = score * volume_factor

            return final_score

        except Exception as e:
            print(f"Error calculating score for {d_name}: {str(e)}")
            return -999  # Return invalid score on error

    def _composite_score(self, returns, price):
        """Implementation of the composite score calculation"""
        # Define weights for each metric
        weight_returns = 0.4
        weight_vwap = 0.3
        weight_rsi = 0.3

        normalized_returns = [
            returns["1y"]["return"],
            100.0 * ((1 + returns["1mo"]["return"] / 100.0) ** 12 - 1),
            100.0 * ((1 + returns["1w"]["return"] / 100.0) ** 52 - 1),
        ]

        # Weighing the recent value more, difference from current price
        weighted_vwap = [0.2, 0.3, 0.5]
        normalized_vwap = [
            (price - returns["1y"]["vwap"]) * weighted_vwap[0] / price,
            (price - returns["1mo"]["vwap"]) * weighted_vwap[1] / price,
            (price - returns["1w"]["vwap"]) * weighted_vwap[2] / price,
        ]

        normalized_rsi = [
            returns["1y"]["rsi"],
            returns["1mo"]["rsi"],
            returns["1w"]["rsi"],
        ]

        # Calculate time-weighted components (more weight to recent periods)
        time_weights = [0.2, 0.3, 0.5]  # 1y, 1mo, 1w

        returns_component = sum(r * w for r, w in zip(normalized_returns, time_weights))
        vwap_component = sum(v * w for v, w in zip(normalized_vwap, time_weights))
        rsi_component = sum(r * w for r, w in zip(normalized_rsi, time_weights))

        # Calculate the composite score with revised weights
        composite_score_result = (
            weight_returns * returns_component
            + weight_vwap * vwap_component
            + weight_rsi * rsi_component
        )

        return (
            composite_score_result,
            normalized_returns,
            normalized_vwap,
            normalized_rsi,
        )

    def _should_exit(self, d):
        """Determine if we should exit a position"""
        pos = self.getposition(d)
        if pos.size == 0:  # No position to exit
            return False

        current_date = self.datas[0].datetime.date(0)
        d_name = d._name

        # Check if we've held for a week (7 calendar days)
        if self.p.weekly_exit and d_name in self.entry_dates:
            entry_date = self.entry_dates[d_name]
            days_held = (current_date - entry_date).days
            if days_held >= 7:
                return True

        # Check stop loss
        price = d.close[0]
        if price <= pos.price * (1 - self.p.stop_loss / 100):
            return True

        # Check take profit
        if d_name in self.peak_prices:
            if price >= pos.price * (1 + self.p.take_profit / 100):
                return True

        # Update peak price for trailing stops
        if d_name not in self.peak_prices or price > self.peak_prices[d_name]:
            self.peak_prices[d_name] = price

        return False

    def _should_enter(self, d):
        """Determine if this stock should be considered for entry"""
        # Only consider entry if we have enough data and no current position
        if len(d) < 252 or self.getposition(d).size > 0:
            return False

        # Calculate momentum score
        score = self._calculate_composite_score(d)
        return score > 0  # Positive momentum is required for entry

    def execute_strategy(self):
        """Execute P=MV momentum strategy logic"""
        # Track portfolio performance
        current_value = self.broker.getvalue()
        current_date = self.datas[0].datetime.date(0)

        # Update peak and calculate drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value

        drawdown = (self.peak_value - current_value) / self.peak_value * 100
        self.drawdowns.append(drawdown)

        # Check for exits first
        for d in self.datas:
            if self.getposition(d).size > 0 and self._should_exit(d):
                self.log(f"EXIT {d._name} at {d.close[0]:.2f}")
                self.close(d)

                # Remove from entry tracking
                if d._name in self.entry_dates:
                    del self.entry_dates[d._name]
                if d._name in self.peak_prices:
                    del self.peak_prices[d._name]

        # Calculate scores for potential entries
        scores = []
        for d in self.datas:
            if not self.getposition(d).size > 0 and self._should_enter(d):
                score = self._calculate_composite_score(d)
                scores.append((d, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Calculate available cash for new positions
        available_cash = self.broker.getcash()

        # Enter top N positions not already in portfolio
        current_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)
        positions_needed = max(0, self.p.top_n_stocks - current_positions)

        for i, (d, score) in enumerate(scores):
            if i < positions_needed:
                # Calculate position size
                target_value = current_value * self.p.position_size
                price = d.close[0]
                size = int(target_value / price)

                if size > 0 and target_value <= available_cash:
                    self.log(f"BUY {d._name} at {price:.2f}, Score: {score:.2f}")
                    self.buy(d, size=size)
                    self.entry_dates[d._name] = current_date
                    self.peak_prices[d._name] = price
                    available_cash -= target_value

    def next(self):
        """Called for each bar of data"""
        # Increment bar counter
        self.bar_executed += 1

        # Skip until we have enough data for all indicators
        if self.bar_executed < 252:
            return

        # Execute the strategy logic
        self.execute_strategy()

        # Store portfolio value for performance tracking
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0))


class PMVMomentumConfig(StrategyConfig):
    """
    Configuration for P=MV Momentum Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for P=MV momentum strategy experiments
        """
        return {
            "rsi_period": [9, 14, 21],
            "vwap_period": [10, 20, 30],
            "top_n_stocks": [5, 10, 15],
            "position_size": [0.1, 0.15, 0.2],
            "weekly_exit": [True],  # Always exit after a week
            "stop_loss": [3.0, 5.0, 7.0],
            "take_profit": [7.0, 10.0, 15.0],
            "printlog": [False],  # Don't include in optimization
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for P=MV momentum strategy
        """
        return {
            "rsi_period": 14,
            "vwap_period": 20,
            "top_n_stocks": 15,
            "position_size": 0.2,
            "weekly_exit": True,
            "stop_loss": 5.0,
            "take_profit": 10.0,
            "printlog": False,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate P=MV momentum strategy parameters
        """
        # All numeric parameters must be positive
        for key in [
            "rsi_period",
            "vwap_period",
            "top_n_stocks",
            "stop_loss",
            "take_profit",
        ]:
            if params.get(key, 0) <= 0:
                return False

        # Position size must be between 0 and 1
        if not 0 < params.get("position_size", 0) <= 1.0:
            return False

        # Top N stocks should be reasonable
        if params.get("top_n_stocks", 0) > 20:
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the P=MV momentum strategy class
        """
        return PMVMomentumStrategy

    def get_required_data_feeds(self) -> int:
        """
        P=MV Momentum strategy works with multiple stocks
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights for the composite score
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.3}
