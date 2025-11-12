# Updated core.py with realistic_execution added to StrategyConfig
# Also added to address criticisms: Momentum Weighting Bias (made weights configurable and adaptive option), Metric Gaps (not here, but in metrics calculation in etf_momentum_strategy.py)

"""
Core components for the ETF Momentum Strategy.

This module contains the core logic for data handling, momentum calculation,
and strategy configuration, designed to be shared across different backtesting
and execution environments.

Author : Prashant Srivastava
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# Ensure logs directory exists with proper error handling
# Check if file logging is disabled (e.g., when running as web server)
disable_file_logging = os.environ.get("DISABLE_FILE_LOGGING", "0") == "1"

if not disable_file_logging:
    try:
        os.makedirs("logs", exist_ok=True)
        log_dir = "logs"
    except PermissionError:
        # Fallback to temp directory if we can't create logs directory
        import tempfile

        log_dir = tempfile.gettempdir()
        print(f"Warning: Unable to create logs directory, using {log_dir}")

    # Configure logging
    log_filename = (
        f"{log_dir}/momentum_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add color to log messages based on log level."""

    COLORS = {
        "DEBUG": "\x1b[36m",  # Cyan
        "INFO": "\x1b[32m",  # Green
        "WARNING": "\x1b[33m",  # Yellow
        "ERROR": "\x1b[31m",  # Red
        "CRITICAL": "\x1b[31;1m",  # Red bold
        "RESET": "\x1b[0m",  # Reset color
    }

    def format(self, record):
        log_message = super().format(record)
        return (
            self.COLORS.get(record.levelname, self.COLORS["RESET"])
            + log_message
            + self.COLORS["RESET"]
        )


# Configure logging handlers based on environment
handlers = []

if not disable_file_logging:
    # Create a file handler only if file logging is enabled
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
    )
    handlers.append(file_handler)

# Always create a stream handler with the custom colored formatter
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    ColoredFormatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
)
handlers.append(stream_handler)

logging.basicConfig(
    level=logging.INFO,
    handlers=handlers,
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration class for ETF momentum strategy parameters."""

    # ETF Universe - Highly Liquid NSE ETFs
    etf_universe: List[str] = field(
        default_factory=lambda: [
            "NIFTYBEES.NS",  # Nifty 50 ETF
            "SETFNN50.NS",  # Nifty Next 50 ETF
            "GOLDBEES.NS",  # Gold ETF
            "SILVERBEES.NS",  # Silver ETF
            "CPSEETF.NS",  # CPSE ETF
            "PSUBNKBEES.NS",  # PSU Bank ETF
            "PHARMABEES.NS",  # Pharma ETF
            "ITBEES.NS",  # IT ETF
            "AUTOBEES.NS",  # Auto ETF
            "INFRAIETF.NS",  # Infra ETF
            "SHARIABEES.NS",  # Shariah ETF
            "DIVOPPBEES.NS",  # Dividend Opportunities ETF
            "CONSUMBEES.NS",  # Consumer Goods ETF
        ]
    )

    # Portfolio Parameters
    portfolio_size: int = 7
    exit_rank_buffer_multiplier: float = 2.0

    # Rebalancing Parameters
    rebalance_frequency: str = "monthly"
    rebalance_day_of_month: int = 5

    # Threshold-based rebalancing
    use_threshold_rebalancing: bool = False
    profit_threshold_pct: float = 10.0
    loss_threshold_pct: float = -5.0

    # Momentum Calculation Parameters
    long_term_period_days: int = 180
    short_term_period_days: int = 60
    momentum_weights: Tuple[float, float] = (0.6, 0.4)
    adaptive_weights: bool = (
        False  # New: Option for adaptive weights based on market volatility
    )

    # Risk Management and Filters
    use_retracement_filter: bool = True
    max_retracement_percentage: float = 0.50
    use_moving_average_filter: bool = True
    moving_average_period: int = 50
    use_volume_filter: bool = False
    min_avg_volume: int = 100000
    max_position_size: float = 0.20
    min_data_points: int = 200

    # Trading Parameters
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001

    # Trading Cost Parameters
    brokerage_rate: float = 0.0003
    max_brokerage: float = 20.0
    stt_sell_rate: float = 0.00025
    exchange_fee_rate: float = 0.0000325
    gst_rate: float = 0.18
    sebi_fee_rate: float = 0.0000001
    stamp_duty_rate: float = 0.00003

    # Market Impact Parameters
    linear_impact_coefficient: float = 0.002
    sqrt_impact_coefficient: float = 0.0005

    # New: Toggle for realistic execution
    realistic_execution: bool = False


class DataProvider:
    """Handles data fetching and caching for ETF prices."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._data_cache = {}

    def fetch_etf_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple ETFs concurrently.

        Returns:
            DataFrame with MultiIndex columns (ticker, ohlcv)
        """
        logger.info(
            f"Fetching data for {len(tickers)} ETFs from {start_date.date()} to {end_date.date()}"
        )

        def fetch_single_etf(ticker: str) -> Tuple[str, pd.DataFrame]:
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(start=start_date, end=end_date, auto_adjust=True)
                if hist.empty:
                    logger.warning(f"No data found for {ticker}")
                    return ticker, pd.DataFrame()
                return ticker, hist
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                return ticker, pd.DataFrame()

        # Fetch data concurrently
        all_data = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(fetch_single_etf, ticker): ticker for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker, data = future.result()
                if not data.empty:
                    all_data[ticker] = data

        if not all_data:
            raise ValueError("No data could be fetched for any ETF")

        # Create MultiIndex DataFrame
        combined_data = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)
        return combined_data

    def get_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract adjusted close prices from the fetched data."""
        if data.empty:
            return pd.DataFrame()

        # Extract Close prices (yfinance auto_adjust=True gives adjusted prices)
        prices = data.xs("Close", level=1, axis=1)
        return prices.fillna(method="ffill").dropna(how="all")

    def get_volumes(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract volume data from the fetched data."""
        if data.empty:
            return None
        if "Volume" not in data.columns.get_level_values(1):
            return None
        volumes = data.xs("Volume", level=1, axis=1)
        return volumes.fillna(method="ffill").dropna(how="all")


class MomentumCalculator:
    """Handles momentum score calculations and filtering."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def calculate_returns(self, prices: pd.Series, period_days: int) -> Optional[float]:
        """
        Calculate percentage return over specified period using DATE-based lookback.

        FIXED: Uses actual calendar days instead of index positions to ensure
        consistent time periods across all ETFs, especially when data has gaps.
        """
        if prices.empty or len(prices) < 50:  # Minimum data requirement
            return None

        try:
            current_price = prices.iloc[-1]
            current_date = prices.index[-1]

            # Calculate target lookback date (DATE-based instead of index-based)
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
        Calculate momentum scores for all ETFs using vectorized operations.

        FIXED: Uses forward-fill instead of dropna to maintain time alignment
        and ensure consistent lookback periods across all ETFs.

        Returns:
            Series with ETF tickers as index and momentum scores as values
        """
        scores = {}
        long_weight, short_weight = self.config.momentum_weights

        if self.config.adaptive_weights:
            # Adaptive weights based on market volatility (example: higher short weight in high vol)
            volatility_window = min(30, len(prices_df))
            market_vol = prices_df.mean(axis=1).pct_change().tail(
                volatility_window
            ).std() * np.sqrt(252)
            # If vol > 20%, increase short weight to 0.5, else keep default
            if market_vol > 0.2:
                long_weight, short_weight = 0.5, 0.5

        for ticker in prices_df.columns:
            # FIXED: Use forward-fill instead of dropna to maintain time alignment
            prices = prices_df[ticker].fillna(method='ffill')

            # Remove leading NaN values if any
            prices = prices.dropna()

            if len(prices) < self.config.min_data_points:
                continue

            # Calculate returns with fixed date-based method
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
    ) -> List[str]:
        """
        Apply various filters to screen ETFs.

        FIXED: Uses forward-fill for consistency with momentum calculation.

        Returns:
            List of ticker symbols that pass all filters
        """
        filtered_tickers = []

        for ticker in prices_df.columns:
            # FIXED: Use forward-fill instead of dropna
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
                    # FIXED: Use forward-fill for volumes too
                    volumes = volume_df[ticker].fillna(method='ffill').dropna()
                    if not self._passes_volume_filter(volumes):
                        continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    def _passes_retracement_filter(self, prices: pd.Series) -> bool:
        """
        Check if ETF passes retracement filter.

        FIXED: Uses date-based lookback for consistency with momentum calculation.
        """
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


class TradingCostCalculator:
    """Calculate comprehensive trading costs for Indian equity markets."""

    def __init__(self, config: StrategyConfig):
        # Indian market cost structure
        self.brokerage_rate = config.brokerage_rate
        self.max_brokerage = config.max_brokerage
        self.stt_sell_rate = config.stt_sell_rate
        self.exchange_fee_rate = config.exchange_fee_rate
        self.gst_rate = config.gst_rate
        self.sebi_fee_rate = config.sebi_fee_rate
        self.stamp_duty_rate = config.stamp_duty_rate

    def calculate_costs(
        self, ticker: str, shares: float, price: float, action: str
    ) -> Dict[str, float]:
        """Calculate all trading costs for a transaction."""
        value = shares * price

        # Brokerage
        brokerage = min(value * self.brokerage_rate, self.max_brokerage)

        # STT (Securities Transaction Tax) - only on sell side for delivery
        stt = value * self.stt_sell_rate if action.upper() == "SELL" else 0

        # Exchange charges
        exchange_fee = value * self.exchange_fee_rate

        # GST on brokerage and exchange fees
        gst = (brokerage + exchange_fee) * self.gst_rate

        # SEBI charges
        sebi_fee = value * self.sebi_fee_rate

        # Stamp duty (on buy side)
        stamp_duty = value * self.stamp_duty_rate if action.upper() == "BUY" else 0

        total_cost = brokerage + stt + exchange_fee + gst + sebi_fee + stamp_duty

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_fee": exchange_fee,
            "gst": gst,
            "sebi_fee": sebi_fee,
            "stamp_duty": stamp_duty,
            "total_cost": total_cost,
        }


class MarketImpactModel:
    """Model market impact and slippage for order execution."""

    def __init__(self, config: StrategyConfig):
        # More realistic impact coefficients for Indian ETF market
        self.linear_impact_coefficient = config.linear_impact_coefficient
        self.sqrt_impact_coefficient = config.sqrt_impact_coefficient

    def estimate_slippage(
        self,
        ticker: str,
        order_size: float,
        daily_volume: float,
        volatility: float,
        action: str,
    ) -> float:
        """Estimate slippage based on order size and market conditions."""
        if daily_volume <= 0:
            return 0.015  # 1.5% slippage for illiquid securities (reduced from 5%)

        # Calculate order size as percentage of daily volume
        volume_participation = order_size / daily_volume

        # More realistic linear impact model
        linear_impact = self.linear_impact_coefficient * volume_participation

        # Square root law (for larger orders) - reduced impact
        sqrt_impact = self.sqrt_impact_coefficient * np.sqrt(volume_participation)

        # Reduced volatility adjustment for ETFs (they're generally more stable)
        volatility_adjustment = volatility * 0.02  # 2% of volatility (reduced from 10%)

        # Smaller direction adjustment for ETFs
        direction_multiplier = 1.1 if action.upper() == "SELL" else 1.0

        total_slippage = (
            linear_impact + sqrt_impact + volatility_adjustment
        ) * direction_multiplier

        # Cap slippage at more reasonable levels for ETFs
        return min(total_slippage, 0.03)  # Max 3% slippage (reduced from 10%)
