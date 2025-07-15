"""
Pairs Trading Strategy

A statistical arbitrage strategy that trades correlated asset pairs based on mean reversion
of their price spread. This strategy identifies when the spread between two correlated assets
deviates significantly from its historical mean and trades on the expectation that the spread
will revert to the mean.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import logging
from typing import Dict, Any, List

from .base_strategy import BaseStrategy, StrategyConfig


class PairsStrategy(BaseStrategy):
    """
    Pairs trading strategy based on statistical arbitrage
    """

    params = (
        ("beta_window", 60),
        ("std_period", 20),
        ("devfactor_entry", 1.5),
        ("devfactor_exit", 0.5),
        ("capital", 100000),
        ("base_size", 100),
        ("printlog", True),
        ("min_corr", 0.85),
        ("stop_dev", 4),
        ("max_hold_days", 15),
        ("vol_factor", True),
        ("min_profit_target", 0.002),
        ("use_returns", True),
        ("atr_period", 14),
        ("risk_per_trade", 0.01),
    )

    def __init__(self):
        super().__init__()
        self.data0 = self.datas[0]  # First asset
        self.data1 = self.datas[1]  # Second asset

        # Initialize indicators
        self.atr0 = bt.indicators.ATR(self.data0, period=self.p.atr_period)
        self.atr1 = bt.indicators.ATR(self.data1, period=self.p.atr_period)

        # Track state
        self.in_position = False
        self.last_trade_bar = 0
        self.entry_price0 = 0
        self.entry_price1 = 0
        self.entry_spread = 0
        self.entry_date = None
        self.position_type = None  # 'long_spread' or 'short_spread'

        # Data storage for analysis
        self.spreads = []
        self.betas = []
        self.correlation_history = []

    # Now using log method from BaseStrategy

    def calculate_beta(self, returns0, returns1):
        """Calculate beta using linear regression"""
        if len(returns0) < 2 or len(returns1) < 2:
            return 1.0

        try:
            if self.p.use_returns:
                # Use returns for beta calculation
                x = np.array(returns1)
                y = np.array(returns0)
            else:
                # Use price levels
                x = np.array(returns1)
                y = np.array(returns0)

            # Remove any infinite or NaN values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) < 2:
                return 1.0

            # Calculate beta using OLS
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            beta = model.params[1] if len(model.params) > 1 else 1.0

            # Ensure reasonable beta range
            beta = max(0.1, min(beta, 10.0))

            return beta
        except Exception as e:
            return 1.0

    def calculate_correlation(self, returns0, returns1):
        """Calculate correlation between two return series"""
        if len(returns0) < 2 or len(returns1) < 2:
            return 0.0

        try:
            corr = np.corrcoef(returns0, returns1)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0

    def get_spread_data(self):
        """Get recent price and return data for spread calculation"""
        if len(self.data0) < self.p.beta_window:
            return None, None, None, None, None

        # Get recent prices
        prices0 = [self.data0.close[-i] for i in range(self.p.beta_window - 1, -1, -1)]
        prices1 = [self.data1.close[-i] for i in range(self.p.beta_window - 1, -1, -1)]

        # Calculate returns
        returns0 = [np.log(prices0[i] / prices0[i - 1]) for i in range(1, len(prices0))]
        returns1 = [np.log(prices1[i] / prices1[i - 1]) for i in range(1, len(prices1))]

        # Calculate current beta
        beta = self.calculate_beta(returns0, returns1)

        # Calculate spread (price0 - beta * price1)
        current_spread = self.data0.close[0] - beta * self.data1.close[0]

        # Calculate historical spreads for volatility
        spreads = [prices0[i] - beta * prices1[i] for i in range(len(prices0))]

        return beta, current_spread, spreads, returns0, returns1

    def execute_strategy(self):
        """Execute pairs trading strategy logic"""
        # Need minimum data
        if len(self.data0) < self.p.beta_window or len(self.data1) < self.p.beta_window:
            return

        # Get spread data
        result = self.get_spread_data()
        if result[0] is None:
            return

        beta, current_spread, spreads, returns0, returns1 = result

        # Store for analysis
        self.spreads.append(current_spread)
        self.betas.append(beta)

        # Calculate correlation
        correlation = self.calculate_correlation(returns0, returns1)
        self.correlation_history.append(correlation)

        # Check minimum correlation requirement
        if correlation < self.p.min_corr:
            if self.in_position:
                self.log(f"Closing position due to low correlation: {correlation:.3f}")
                self.close_position("Low correlation")
            return

        # Calculate spread statistics
        spread_window = (
            spreads[-self.p.std_period :]
            if len(spreads) >= self.p.std_period
            else spreads
        )
        spread_mean = np.mean(spread_window)
        spread_std = np.std(spread_window)

        if spread_std == 0:
            return

        # Z-score of current spread
        z_score = (current_spread - spread_mean) / spread_std

        # Position management
        if self.in_position:
            self.manage_position(z_score, current_spread)
        else:
            self.check_entry(z_score, current_spread, beta)

    def check_entry(self, z_score, current_spread, beta):
        """Check for entry signals"""
        # Entry conditions
        if abs(z_score) > self.p.devfactor_entry:
            # Determine position type
            if z_score > 0:  # Spread is too high, expect reversion down
                position_type = "short_spread"  # Short asset0, long asset1
                action = "SHORT SPREAD"
            else:  # Spread is too low, expect reversion up
                position_type = "long_spread"  # Long asset0, short asset1
                action = "LONG SPREAD"

            self.enter_position(position_type, current_spread, beta, z_score, action)

    def enter_position(self, position_type, current_spread, beta, z_score, action):
        """Enter a pairs trade"""
        try:
            # Calculate position sizes
            available_cash = self.broker.get_cash()

            if self.p.vol_factor:
                # Use volatility-based sizing
                atr0_val = (
                    self.atr0[0] if len(self.atr0) > 0 else self.data0.close[0] * 0.02
                )
                atr1_val = (
                    self.atr1[0] if len(self.atr1) > 0 else self.data1.close[0] * 0.02
                )

                # Risk per trade based sizing
                risk_amount = available_cash * self.p.risk_per_trade
                size0 = int(risk_amount / (atr0_val * 2))  # Half risk on each leg
                size1 = int(risk_amount / (atr1_val * 2))
            else:
                # Simple equal dollar sizing
                target_investment = available_cash * 0.4  # Use 40% of available cash
                size0 = int(target_investment / (2 * self.data0.close[0]))
                size1 = int(target_investment / (2 * self.data1.close[0]))

            # Minimum size check
            size0 = max(size0, self.p.base_size)
            size1 = max(size1, int(self.p.base_size * beta))

            if position_type == "long_spread":
                # Long asset0, short asset1
                self.buy(data=self.data0, size=size0)
                self.sell(data=self.data1, size=size1)
            else:
                # Short asset0, long asset1
                self.sell(data=self.data0, size=size0)
                self.buy(data=self.data1, size=size1)

            # Record entry
            self.in_position = True
            self.last_trade_bar = len(self.data0)
            self.entry_price0 = self.data0.close[0]
            self.entry_price1 = self.data1.close[0]
            self.entry_spread = current_spread
            self.entry_date = self.data0.datetime.date(0)
            self.position_type = position_type

            self.log(
                f"{action}: Z-score={z_score:.2f}, Spread={current_spread:.2f}, Beta={beta:.3f}"
            )

        except Exception as e:
            self.log(f"Error entering position: {e}")

    def manage_position(self, z_score, current_spread):
        """Manage existing position"""
        days_in_position = len(self.data0) - self.last_trade_bar

        # Exit conditions
        exit_signal = False
        exit_reason = ""

        # 1. Mean reversion exit
        if abs(z_score) < self.p.devfactor_exit:
            exit_signal = True
            exit_reason = f"Mean reversion (z-score: {z_score:.2f})"

        # 2. Stop loss - spread moved too far against us
        elif abs(z_score) > self.p.stop_dev:
            exit_signal = True
            exit_reason = f"Stop loss (z-score: {z_score:.2f})"

        # 3. Maximum holding period
        elif days_in_position > self.p.max_hold_days:
            exit_signal = True
            exit_reason = f"Max hold period ({days_in_position} days)"

        # 4. Profit target check
        spread_change = current_spread - self.entry_spread
        if self.position_type == "long_spread":
            profit_pct = (
                spread_change / abs(self.entry_spread) if self.entry_spread != 0 else 0
            )
        else:
            profit_pct = (
                -spread_change / abs(self.entry_spread) if self.entry_spread != 0 else 0
            )

        if profit_pct > self.p.min_profit_target:
            exit_signal = True
            exit_reason = f"Profit target (profit: {profit_pct:.1%})"

        if exit_signal:
            self.close_position(exit_reason)

    def close_position(self, reason):
        """Close the pairs position"""
        try:
            # Close all positions
            if self.getposition(self.data0).size != 0:
                self.close(data=self.data0)
            if self.getposition(self.data1).size != 0:
                self.close(data=self.data1)

            # Reset state
            self.in_position = False
            self.position_type = None

            self.log(f"Position closed: {reason}")

        except Exception as e:
            self.log(f"Error closing position: {e}")


class PairsConfig(StrategyConfig):
    """
    Configuration for Pairs Trading Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for pairs trading experiments
        """
        return {
            "beta_window": [30, 60, 90, 120],
            "std_period": [10, 20, 30, 40],
            "devfactor_entry": [1.0, 1.5, 2.0, 2.5],
            "devfactor_exit": [0.3, 0.5, 0.7, 1.0],
            "min_corr": [0.7, 0.8, 0.85, 0.9],
            "stop_dev": [3, 4, 5, 6],
            "max_hold_days": [10, 15, 20, 30],
            "risk_per_trade": [0.005, 0.01, 0.015, 0.02],
            "min_profit_target": [0.001, 0.002, 0.003, 0.005],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for pairs trading strategy
        """
        return {
            "beta_window": 60,
            "std_period": 20,
            "devfactor_entry": 1.5,
            "devfactor_exit": 0.5,
            "capital": 100000,
            "base_size": 100,
            "printlog": False,
            "min_corr": 0.85,
            "stop_dev": 4,
            "max_hold_days": 15,
            "vol_factor": True,
            "min_profit_target": 0.002,
            "use_returns": True,
            "atr_period": 14,
            "risk_per_trade": 0.01,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate pairs trading parameters
        """
        # Entry threshold should be higher than exit threshold
        if params.get("devfactor_entry", 0) <= params.get("devfactor_exit", 0):
            return False

        # Correlation should be between 0 and 1
        min_corr = params.get("min_corr", 0)
        if min_corr < 0 or min_corr > 1:
            return False

        # All positive parameters must be positive
        positive_params = [
            "beta_window",
            "std_period",
            "devfactor_entry",
            "devfactor_exit",
            "stop_dev",
            "max_hold_days",
            "risk_per_trade",
            "min_profit_target",
        ]

        for param in positive_params:
            if params.get(param, 0) <= 0:
                return False

        # Risk per trade should be reasonable (less than 5%)
        if params.get("risk_per_trade", 0) > 0.05:
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the pairs trading strategy class
        """
        return PairsStrategy

    def get_required_data_feeds(self) -> int:
        """
        Pairs trading requires exactly 2 data feeds
        """
        return 2

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights optimized for pairs trading strategy
        """
        return {"total_return": 0.3, "sharpe_ratio": 0.4, "max_drawdown": 0.3}
