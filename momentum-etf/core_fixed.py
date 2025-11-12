"""
Fixed version of MomentumCalculator with date-based lookback and forward-fill.
This addresses the critical issues identified in the momentum calculation.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional
from core import StrategyConfig


class MomentumCalculatorFixed:
    """Fixed momentum calculator with date-based lookback."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def calculate_returns(self, prices: pd.Series, period_days: int) -> Optional[float]:
        """
        Calculate percentage return over specified period using DATE-based lookback.

        FIXES:
        1. Uses actual calendar days instead of index positions
        2. Handles data gaps correctly
        3. Validates lookback date is within reasonable range
        """
        if prices.empty or len(prices) < 50:  # Minimum data requirement
            return None

        try:
            current_price = prices.iloc[-1]
            current_date = prices.index[-1]

            # Calculate target lookback date (FIX #1: Date-based instead of index-based)
            lookback_date = current_date - timedelta(days=period_days)

            # Find closest available date
            time_diffs = abs(prices.index - lookback_date)
            closest_idx = time_diffs.argmin()

            # Ensure we found a date within reasonable range (±30 days tolerance)
            actual_lookback_days = abs((prices.index[closest_idx] - lookback_date).days)
            if actual_lookback_days > 30:
                # Lookback date is too far from requested period
                return None

            past_price = prices.iloc[closest_idx]

            if past_price == 0 or np.isnan(past_price) or np.isnan(current_price):
                return None

            return (current_price / past_price) - 1

        except (IndexError, ZeroDivisionError, KeyError):
            return None

    def calculate_momentum_scores(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for all ETFs with fixed methodology.

        FIXES:
        1. Uses forward-fill instead of dropna()
        2. Date-based lookback periods
        3. Consistent time periods across all ETFs
        """
        scores = {}
        long_weight, short_weight = self.config.momentum_weights

        if self.config.adaptive_weights:
            # Adaptive weights based on market volatility
            volatility_window = min(30, len(prices_df))
            market_vol = prices_df.mean(axis=1).pct_change().tail(
                volatility_window
            ).std() * np.sqrt(252)
            if market_vol > 0.2:
                long_weight, short_weight = 0.5, 0.5

        for ticker in prices_df.columns:
            # FIX #2: Use forward-fill instead of dropna to maintain time alignment
            prices = prices_df[ticker].fillna(method='ffill')

            # Remove leading NaN values if any
            prices = prices.dropna()

            if len(prices) < self.config.min_data_points:
                continue

            # Calculate returns with fixed method
            long_return = self.calculate_returns(
                prices, self.config.long_term_period_days
            )
            short_return = self.calculate_returns(
                prices, self.config.short_term_period_days
            )

            if long_return is None:
                continue

            # Weighted momentum score
            if short_return is not None:
                momentum_score = long_return * long_weight + short_return * short_weight
            else:
                momentum_score = long_return

            scores[ticker] = momentum_score

        return pd.Series(scores).sort_values(ascending=False)

    def apply_filters(
        self, prices_df: pd.DataFrame, volume_df: Optional[pd.DataFrame] = None
    ) -> list[str]:
        """
        Apply filters with forward-fill for consistency.
        """
        filtered_tickers = []

        for ticker in prices_df.columns:
            prices = prices_df[ticker].fillna(method='ffill').dropna()

            if len(prices) < self.config.min_data_points:
                continue

            # Retracement filter
            if self.config.use_retracement_filter:
                if not self._passes_retracement_filter(prices):
                    continue

            # Moving average filter
            if self.config.use_moving_average_filter:
                if not self._passes_ma_filter(prices):
                    continue

            # Volume filter
            if self.config.use_volume_filter and volume_df is not None:
                if ticker in volume_df.columns:
                    volumes = volume_df[ticker].fillna(method='ffill').dropna()
                    if not self._passes_volume_filter(volumes):
                        continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    def _passes_retracement_filter(self, prices: pd.Series) -> bool:
        """Check if ETF passes retracement filter."""
        try:
            # Use date-based lookback for consistency
            current_date = prices.index[-1]
            lookback_date = current_date - timedelta(days=self.config.long_term_period_days)

            # Get prices within lookback period
            lookback_prices = prices[prices.index >= lookback_date]

            if len(lookback_prices) < 50:  # Minimum lookback
                return False

            peak = lookback_prices.max()
            current_price = prices.iloc[-1]

            if peak <= 0:
                return False

            retracement = (peak - current_price) / peak
            return retracement <= self.config.max_retracement_percentage
        except Exception:
            return False

    def _passes_ma_filter(self, prices: pd.Series) -> bool:
        """Check if current price is above moving average."""
        try:
            if len(prices) < self.config.moving_average_period:
                return False

            # Calculate EMA
            ema = (
                prices.ewm(span=self.config.moving_average_period, adjust=False)
                .mean()
                .iloc[-1]
            )
            current_price = prices.iloc[-1]

            return current_price > ema
        except Exception:
            return False

    def _passes_volume_filter(self, volumes: pd.Series) -> bool:
        """Check if average volume meets minimum requirement."""
        try:
            avg_volume = volumes.iloc[-30:].mean()  # 30-day average volume
            return avg_volume >= self.config.min_avg_volume
        except Exception:
            return False
