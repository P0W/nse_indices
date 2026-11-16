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
import time
import warnings
import os

import pandas as pd
import numpy as np
import pytz
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
            "SBINEXT50.NS",  # SBI Nifty Next 50 ETF (formerly SETFNN50.NS)
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

    # Signal proxy map: score/filter renamed ETFs via liquid same-index benchmark ETF
    # while pricing and trading the held ticker directly.
    signal_proxy_map: Dict[str, str] = field(
        default_factory=lambda: {
            "SBINEXT50.NS": "JUNIORBEES.NS",
            "SETFNN50.NS": "JUNIORBEES.NS",
        }
    )

    # Legacy symbol aliases for holdings files (e.g., NSE ticker rename SETFNN50 -> SBINEXT50)
    ticker_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "SETFNN50.NS": "SBINEXT50.NS",
            "SETFNN50": "SBINEXT50.NS",
        }
    )

    # Portfolio Parameters (aligned with rolling 200d/90d ETF momentum strategy)
    portfolio_size: int = 5
    exit_rank_buffer_multiplier: float = 1.0

    # Rebalancing Parameters
    rebalance_frequency: str = "rolling"
    rebalance_period_days: int = 30
    rebalance_day_of_month: int = 5

    # Threshold-based rebalancing
    use_threshold_rebalancing: bool = False
    profit_threshold_pct: float = 10.0
    loss_threshold_pct: float = -5.0

    # Momentum Calculation Parameters
    long_term_period_days: int = 200
    short_term_period_days: int = 90
    momentum_weights: Tuple[float, float] = (0.7, 0.3)
    adaptive_weights: bool = True
    min_momentum_threshold: float = 0.0

    # Risk Management and Filters
    use_retracement_filter: bool = True
    max_retracement_percentage: float = 0.25
    use_moving_average_filter: bool = True
    moving_average_period: int = 20
    use_volume_filter: bool = True
    min_avg_volume: int = 200000
    max_position_size: float = 1.0
    min_data_points: int = 201

    # Trading Parameters
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001

    # Trading Cost Parameters (Indian NSE ETF delivery)
    brokerage_rate: float = 0.0003
    max_brokerage: float = 20.0
    stt_buy_rate: float = 0.0  # 0% STT on ETF delivery buys
    stt_sell_rate: float = 0.00001  # 0.001% (₹1/lakh) STT on equity ETF delivery sells
    exchange_fee_rate: float = 0.0000297
    gst_rate: float = 0.18
    sebi_fee_rate: float = 0.000001
    stamp_duty_rate: float = 0.00015
    dp_charge_per_sell: float = 15.34  # CDSL DP charge per sell scrip (₹13 + 18% GST)

    # Market Impact Parameters
    linear_impact_coefficient: float = 0.002
    sqrt_impact_coefficient: float = 0.0005

    # Toggle for realistic execution
    realistic_execution: bool = False


class DataProvider:
    """Handles deterministic data fetching, retry, and proxy stitching for ETF prices."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._data_cache = {}

    @staticmethod
    def _normalize_ohlcv_df(df: pd.DataFrame, end_date: datetime) -> pd.DataFrame:
        """Normalize OHLCV DataFrame index to IST date-naive midnight and deduplicate."""
        if df is None or df.empty:
            return pd.DataFrame()

        cleaned = df.copy()
        if isinstance(cleaned.columns, pd.MultiIndex):
            cleaned.columns = cleaned.columns.get_level_values(-1)

        idx = pd.to_datetime(cleaned.index)
        if idx.tz is not None:
            ist_tz = pytz.timezone("Asia/Kolkata")
            idx = idx.tz_convert(ist_tz).tz_localize(None)
        cleaned.index = idx.normalize()
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()

        if "Close" in cleaned.columns:
            cleaned = cleaned.dropna(subset=["Close"])

        end_day = pd.Timestamp(end_date).normalize()
        cleaned = cleaned[cleaned.index <= end_day]
        return cleaned

    def fetch_etf_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple ETFs reliably without Yahoo Finance thread rate-limits,
        with sequential retry on missing symbols and signal-proxy backfill for renamed ETFs.

        Returns:
            DataFrame with MultiIndex columns (ticker, field)
        """
        requested_tickers = list(dict.fromkeys(tickers))
        proxy_map = {
            t: self.config.signal_proxy_map[t]
            for t in requested_tickers
            if t in self.config.signal_proxy_map
            and self.config.signal_proxy_map[t] != t
        }
        symbols_to_fetch = list(
            dict.fromkeys(requested_tickers + list(proxy_map.values()))
        )

        # yfinance treats midnight `end` as exclusive; add 1 day so `end_date` is inclusive
        yf_end = pd.Timestamp(end_date).normalize() + timedelta(days=1)

        logger.info(
            f"Fetching data for {len(requested_tickers)} ETFs "
            f"(+{len(symbols_to_fetch) - len(requested_tickers)} signal proxies) "
            f"from {start_date.date()} to {end_date.date()}"
        )

        raw_by_symbol: Dict[str, pd.DataFrame] = {}

        # 1. Primary single-session batch download (threads=False avoids Yahoocrumb/SQLite races)
        try:
            batch_df = yf.download(
                tickers=symbols_to_fetch,
                start=start_date,
                end=yf_end,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=False,
            )
            if batch_df is not None and not batch_df.empty:
                if len(symbols_to_fetch) == 1:
                    sym = symbols_to_fetch[0]
                    sub = batch_df
                    if (
                        isinstance(batch_df.columns, pd.MultiIndex)
                        and sym in batch_df.columns.get_level_values(0)
                    ):
                        sub = batch_df[sym]
                    norm = self._normalize_ohlcv_df(sub, end_date)
                    if not norm.empty:
                        raw_by_symbol[sym] = norm
                elif isinstance(batch_df.columns, pd.MultiIndex):
                    available_lvl0 = set(batch_df.columns.get_level_values(0))
                    for sym in symbols_to_fetch:
                        if sym in available_lvl0:
                            norm = self._normalize_ohlcv_df(batch_df[sym], end_date)
                            if not norm.empty:
                                raw_by_symbol[sym] = norm
        except Exception as e:
            logger.warning(
                f"Batch yf.download encountered an error ({e}); falling back to sequential fetch"
            )

        # 2. Sequential retry with backoff for any missing symbol
        missing_symbols = [s for s in symbols_to_fetch if s not in raw_by_symbol]
        for sym in missing_symbols:
            for attempt in range(1, 3):
                try:
                    if attempt > 1:
                        time.sleep(0.5 * attempt)
                    hist = yf.Ticker(sym).history(
                        start=start_date, end=yf_end, auto_adjust=True
                    )
                    norm = self._normalize_ohlcv_df(hist, end_date)
                    if not norm.empty:
                        raw_by_symbol[sym] = norm
                        logger.info(
                            f"Recovered {sym} on sequential retry #{attempt} ({len(norm)} rows)"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Retry #{attempt} failed for {sym}: {e}")

        # 3. Assemble output for requested_tickers, applying proxy signal/history where configured
        all_data: Dict[str, pd.DataFrame] = {}
        ohlc_cols = ["Open", "High", "Low", "Close"]

        for ticker in requested_tickers:
            t_df = raw_by_symbol.get(ticker, pd.DataFrame())
            proxy_sym = proxy_map.get(ticker)
            p_df = raw_by_symbol.get(proxy_sym, pd.DataFrame()) if proxy_sym else pd.DataFrame()

            if proxy_sym and p_df.empty:
                logger.warning(
                    f"Signal proxy {proxy_sym} returned no data for {ticker}; falling back to direct series ({len(t_df)} rows)"
                )

            if proxy_sym and not p_df.empty:
                if t_df.empty:
                    # No direct market quotes for target ticker: expose proxy for signal/liquidity only,
                    # leaving tradeable OHLC as NaN so proxy prices are never mistaken for live quotes.
                    merged = pd.DataFrame(index=p_df.index)
                    for col in ohlc_cols:
                        merged[col] = np.nan
                    merged["SignalClose"] = p_df["Close"]
                    if "Volume" in p_df.columns:
                        merged["Volume"] = p_df["Volume"]
                    all_data[ticker] = merged
                    logger.warning(
                        f"No direct market quotes for {ticker}; populated SignalClose/Volume via {proxy_sym} (Close left NaN)"
                    )
                    continue

                # Pure causal separation:
                # - OHLC (`Close`, etc.) uses ONLY real quotes from `t_df` (never proxy prices or future scale factors).
                # - `SignalClose` and `Volume` use ONLY pure unscaled proxy history from `p_df` (scale-invariant & leak-free).
                union_idx = p_df.index.union(t_df.index).sort_values()
                merged = pd.DataFrame(index=union_idx)

                for col in ohlc_cols:
                    if col in t_df.columns:
                        merged[col] = t_df[col].reindex(union_idx)
                    else:
                        merged[col] = np.nan

                merged["SignalClose"] = p_df["Close"].reindex(union_idx)
                if "Volume" in p_df.columns:
                    merged["Volume"] = p_df["Volume"].reindex(union_idx)
                elif "Volume" in t_df.columns:
                    merged["Volume"] = t_df["Volume"].reindex(union_idx)

                all_data[ticker] = merged
                logger.info(
                    f"Mapped {ticker} signal/liquidity via {proxy_sym} "
                    f"(direct price rows={int(merged['Close'].notna().sum())}, signal rows={int(merged['SignalClose'].notna().sum())})"
                )
            elif not t_df.empty:
                merged = t_df.copy()
                merged["SignalClose"] = t_df["Close"]
                all_data[ticker] = merged
            else:
                logger.error(
                    f"Failed to fetch market data for {ticker} after retries — excluded from universe!"
                )

        if not all_data:
            raise ValueError("No data could be fetched for any ETF")

        still_missing = [t for t in requested_tickers if t not in all_data]
        if still_missing:
            logger.warning(f"Missing market data for {len(still_missing)} ETF(s): {still_missing}")

        combined_data = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)
        return combined_data

    def get_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract tradeable adjusted close prices from the fetched data."""
        if data.empty:
            return pd.DataFrame()

        prices = data.xs("Close", level=1, axis=1)
        return prices.ffill(limit=5).dropna(how="all")

    def get_signal_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract signal close prices (uses proxy close for renamed ETFs when configured)."""
        if data.empty:
            return pd.DataFrame()

        if "SignalClose" in data.columns.get_level_values(1):
            sig = data.xs("SignalClose", level=1, axis=1)
            return sig.ffill(limit=5).dropna(how="all")
        return self.get_prices(data)

    def get_volumes(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract volume data from the fetched data."""
        if data.empty:
            return None
        if "Volume" not in data.columns.get_level_values(1):
            return None
        volumes = data.xs("Volume", level=1, axis=1)
        return volumes.dropna(how="all")


class MomentumCalculator:
    """Handles momentum score calculations and filtering."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def calculate_returns(self, prices: pd.Series, period_days: int) -> Optional[float]:
        """Calculate percentage return over `period_days` trading bars."""
        if period_days <= 0 or len(prices) < period_days + 1:
            return None

        try:
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-(period_days + 1)]

            if past_price <= 0 or np.isnan(past_price) or np.isnan(current_price):
                return None

            return float((current_price / past_price) - 1.0)
        except (IndexError, ZeroDivisionError):
            return None

    def calculate_momentum_scores(
        self,
        prices_df: pd.DataFrame,
        market_prices_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Calculate composite momentum scores for all candidate ETFs.

        Args:
            prices_df: Candidate ETF close prices
            market_prices_df: Optional full-universe prices used to measure market volatility
                              (NIFTYBEES.NS if present, else equal-weighted daily return)

        Returns:
            Series with ETF tickers as index and momentum scores as values
        """
        scores = {}
        long_weight, short_weight = self.config.momentum_weights

        if self.config.adaptive_weights:
            ref_df = market_prices_df if market_prices_df is not None else prices_df
            market_vol = 0.0
            if "NIFTYBEES.NS" in ref_df.columns:
                nifty_ret = ref_df["NIFTYBEES.NS"].dropna().pct_change().dropna().tail(30)
                if len(nifty_ret) >= 10:
                    market_vol = float(nifty_ret.std() * np.sqrt(252))
            elif not ref_df.empty:
                ew_ret = ref_df.pct_change().mean(axis=1).dropna().tail(30)
                if len(ew_ret) >= 10:
                    market_vol = float(ew_ret.std() * np.sqrt(252))

            if market_vol > 0.20:
                long_weight, short_weight = 0.5, 0.5

        min_required = max(
            self.config.min_data_points, self.config.long_term_period_days + 1
        )

        for ticker in prices_df.columns:
            if not self._has_fresh_observation(prices_df[ticker], prices_df.index):
                continue

            prices = prices_df[ticker].dropna()

            if len(prices) < min_required:
                continue

            long_return = self.calculate_returns(
                prices, self.config.long_term_period_days
            )
            short_return = self.calculate_returns(
                prices, self.config.short_term_period_days
            )

            if long_return is None:
                continue

            if short_return is not None:
                momentum_score = long_return * long_weight + short_return * short_weight
            else:
                momentum_score = long_return

            scores[ticker] = momentum_score

        return pd.Series(scores, dtype=float).sort_values(ascending=False)

    @staticmethod
    def _has_fresh_observation(
        series: pd.Series, ref_index: pd.Index, max_stale_bars: int = 5
    ) -> bool:
        """Reject halted/stale series whose last valid observation lags the universe end date."""
        if series is None or len(ref_index) == 0:
            return False
        last_valid = series.last_valid_index()
        if last_valid is None:
            return False
        try:
            pos = ref_index.get_indexer([last_valid])[0]
            if pos < 0:
                cutoff_idx = max(0, len(ref_index) - 1 - max_stale_bars)
                return pd.Timestamp(last_valid) >= pd.Timestamp(ref_index[cutoff_idx])
            return pos >= (len(ref_index) - 1 - max_stale_bars)
        except Exception:
            return False

    def apply_filters(
        self, prices_df: pd.DataFrame, volume_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Apply regime and liquidity filters to screen ETFs.

        Returns:
            List of ticker symbols that pass all filters
        """
        filtered_tickers = []
        min_required = max(
            self.config.min_data_points, self.config.long_term_period_days + 1
        )

        for ticker in prices_df.columns:
            if not self._has_fresh_observation(prices_df[ticker], prices_df.index):
                continue

            prices = prices_df[ticker].dropna()

            if len(prices) < min_required:
                continue

            # Retracement filter
            if self.config.use_retracement_filter:
                if not self._passes_retracement_filter(prices):
                    continue

            # Moving average filter
            if self.config.use_moving_average_filter:
                if not self._passes_ma_filter(prices):
                    continue

            # Volume filter (fail closed when enabled)
            if self.config.use_volume_filter:
                if volume_df is None or ticker not in volume_df.columns:
                    logger.warning(
                        f"Volume filter enabled but no volume data for {ticker}; excluding"
                    )
                    continue
                if not self._has_fresh_observation(volume_df[ticker], prices_df.index):
                    continue
                volumes = volume_df[ticker].dropna()
                if not self._passes_volume_filter(volumes):
                    continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    def _passes_retracement_filter(self, prices: pd.Series) -> bool:
        """Check if ETF drawdown from rolling long-term close high is within threshold."""
        try:
            lookback_prices = prices.iloc[-self.config.long_term_period_days :]
            if len(lookback_prices) < 50:
                return False

            peak = lookback_prices.max()
            current_price = prices.iloc[-1]

            if peak <= 0:
                return False

            retracement = (peak - current_price) / peak
            return bool(retracement <= self.config.max_retracement_percentage)
        except Exception:
            return False

    def _passes_ma_filter(self, prices: pd.Series) -> bool:
        """Check if current price is above exponential moving average."""
        try:
            if len(prices) < self.config.moving_average_period:
                return False

            ema = (
                prices.ewm(span=self.config.moving_average_period, adjust=False)
                .mean()
                .iloc[-1]
            )
            current_price = prices.iloc[-1]

            return bool(current_price > ema)
        except Exception:
            return False

    def _passes_volume_filter(self, volumes: pd.Series) -> bool:
        """Check if 30-day average completed daily volume meets minimum requirement."""
        try:
            if len(volumes) == 0:
                return False

            # Drop incomplete intraday bar first if run during live NSE trading hours (before 15:35 IST)
            ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
            last_idx = pd.Timestamp(volumes.index[-1]).date()
            market_in_progress = (
                last_idx == ist_now.date()
                and (ist_now.hour, ist_now.minute) < (15, 35)
            )
            completed_volumes = volumes.iloc[:-1] if market_in_progress else volumes
            if len(completed_volumes) < 30:
                return False

            window = completed_volumes.iloc[-30:]
            avg_volume = float(window.mean())
            return avg_volume >= self.config.min_avg_volume
        except Exception:
            return False


class TradingCostCalculator:
    """Calculate comprehensive trading costs for Indian equity/ETF delivery markets."""

    def __init__(self, config: StrategyConfig):
        self.brokerage_rate = config.brokerage_rate
        self.max_brokerage = config.max_brokerage
        self.stt_buy_rate = getattr(config, "stt_buy_rate", config.stt_sell_rate)
        self.stt_sell_rate = config.stt_sell_rate
        self.exchange_fee_rate = config.exchange_fee_rate
        self.gst_rate = config.gst_rate
        self.sebi_fee_rate = config.sebi_fee_rate
        self.stamp_duty_rate = config.stamp_duty_rate
        self.dp_charge_per_sell = getattr(config, "dp_charge_per_sell", 0.0)

    def calculate_costs(
        self,
        ticker: str,
        shares: float,
        price: float,
        action: str,
        include_dp_charge: bool = False,
        zero_delivery_brokerage: bool = False,
    ) -> Dict[str, float]:
        """Calculate all statutory trading costs for a transaction."""
        value = shares * price
        is_sell = action.upper() == "SELL"

        brokerage = (
            0.0
            if zero_delivery_brokerage
            else min(value * self.brokerage_rate, self.max_brokerage)
        )

        # Equity-oriented ETF delivery STT is 0% on buy and 0.001% on sell;
        # Gold/Silver/Liquid/International ETFs have 0% STT on both sides.
        non_equity_etf = any(
            k in ticker.upper()
            for k in ("GOLD", "SILVER", "LIQUID", "MON100", "MAFANG", "HNGSNG", "MASPTOP")
        )
        if non_equity_etf:
            stt = 0.0
        else:
            stt = value * (self.stt_sell_rate if is_sell else self.stt_buy_rate)

        exchange_fee = value * self.exchange_fee_rate
        sebi_fee = value * self.sebi_fee_rate
        gst = (brokerage + exchange_fee + sebi_fee) * self.gst_rate
        stamp_duty = value * self.stamp_duty_rate if not is_sell else 0.0
        dp_charge = self.dp_charge_per_sell if (include_dp_charge and is_sell and value > 0) else 0.0

        total_cost = (
            brokerage + stt + exchange_fee + gst + sebi_fee + stamp_duty + dp_charge
        )

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_fee": exchange_fee,
            "gst": gst,
            "sebi_fee": sebi_fee,
            "stamp_duty": stamp_duty,
            "dp_charge": dp_charge,
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
