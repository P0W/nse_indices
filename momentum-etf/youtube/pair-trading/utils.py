"""
Market Data Utilities for Nifty Trading Strategy

This module provides a clean, reusable class for loading and caching market data
from various sources including stocks, indices, and derivatives.
"""

import os
import pickle
import pandas as pd
import yfinance as yf
import backtrader as bt
import logging
from datetime import datetime, date
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class IndianBrokerageCommission(bt.CommInfoBase):
    """
    Indian Brokerage Commission Scheme as per SEBI regulations and typical broker charges

    Based on:
    - SEBI regulations for transaction charges
    - Typical discount broker charges (like Zerodha, Upstox, etc.)
    - Exchange charges (NSE/BSE)
    - Regulatory charges (STT, GST, etc.)

    Usage:
        commission_scheme = IndianBrokerageCommission()
        cerebro.broker.addcommissioninfo(commission_scheme)
    """

    params = (
        # Basic commission structure
        ("commission", 0.0003),  # 0.03% or ₹20 per trade, whichever is lower
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        # Indian specific charges
        ("brokerage_flat", 20.0),  # Flat ₹20 per trade for discount brokers
        ("stt_rate", 0.001),  # STT: 0.1% on sell side (equity delivery)
        ("transaction_charge", 0.0000345),  # NSE transaction charge: 0.00345%
        ("gst_rate", 0.18),  # GST: 18% on brokerage + transaction charges
        ("sebi_charge", 0.0000001),  # SEBI charge: ₹10 per crore
        ("stamp_duty", 0.00003),  # Stamp duty: 0.003% on buy side
    )

    def getcommission(self, size, price):
        """
        Calculate total commission including all Indian regulatory charges

        Args:
            size: Number of shares
            price: Price per share

        Returns:
            float: Total commission amount
        """
        # Trade value
        trade_value = abs(size) * price

        # 1. Brokerage: 0.03% or ₹20, whichever is lower
        brokerage_perc = trade_value * self.p.commission
        brokerage = min(brokerage_perc, self.p.brokerage_flat)

        # 2. STT (Securities Transaction Tax) - only on sell side
        # For delivery: 0.1% on sell, 0% on buy
        # For intraday: 0.025% on both buy and sell
        # We'll use delivery STT model (conservative approach)
        stt = (
            0  # Applied separately in backtrader through slippage or manual calculation
        )

        # 3. Transaction charges (NSE: 0.00345%)
        transaction_charge = trade_value * self.p.transaction_charge

        # 4. GST on brokerage and transaction charges
        gst_base = brokerage + transaction_charge
        gst = gst_base * self.p.gst_rate

        # 5. SEBI charges (₹10 per crore)
        sebi_charge = (trade_value / 10000000) * 10  # ₹10 per crore

        # 6. Stamp duty (only on buy side) - 0.003%
        stamp_duty = 0  # Applied separately as it's buy-side only

        # Total commission
        total_commission = brokerage + transaction_charge + gst + sebi_charge

        return total_commission

    def getoperationcost(self, size, price):
        """
        Additional operational costs including STT and stamp duty

        Args:
            size: Number of shares
            price: Price per share

        Returns:
            float: Total operational cost amount
        """
        trade_value = abs(size) * price

        # STT: 0.1% on sell side (we'll apply 0.05% average for both sides)
        stt = trade_value * (
            self.p.stt_rate / 2
        )  # Average between buy (0%) and sell (0.1%)

        # Stamp duty: 0.003% on buy side (we'll apply 0.0015% average)
        stamp_duty = trade_value * (self.p.stamp_duty / 2)

        return stt + stamp_duty


def setup_logger():
    """Setup logging configuration for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    )
    return logging.getLogger(__name__)


class MarketDataLoader:
    """
    Universal market data loader with intelligent caching for stocks, indices, and derivatives.

    Features:
    - Date-based cache filenames to avoid cache conflicts
    - Support for multiple instrument types (equity, index, derivative)
    - Automatic data validation and processing for backtrader compatibility
    - Intelligent cache validation based on date ranges
    """

    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize the MarketDataLoader.

        Args:
            cache_dir (str): Directory for cached data files
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def load_market_data(
        self,
        symbols: List[str],
        start_date: Union[datetime, date],
        end_date: Union[datetime, date],
        force_refresh: bool = False,
        use_parallel: bool = True,
        max_workers: int = 8,
        interval: str = "1d",
    ) -> List[bt.feeds.PandasData]:
        """
        Load market data for multiple symbols with intelligent caching and parallel processing.

        Args:
            symbols (List[str]): List of symbols to download (e.g., ['^NSEI', '^NSEBANK', 'RELIANCE.NS'])
            start_date (Union[datetime, date]): Start date for data
            end_date (Union[datetime, date]): End date for data
            force_refresh (bool): Force re-download even if cache exists (default: False)
            use_parallel (bool): Use parallel processing for faster downloads (default: True)
            max_workers (int): Maximum number of parallel workers (default: 8)
            interval (str): Data interval ('1d', '5m', '15m', '1h', etc.) (default: '1d')

        Returns:
            List[bt.feeds.PandasData]: List of backtrader data feeds

        Note:
            For intraday intervals (5m, 15m, 1h), data availability is limited:
            - Free tier: Last 60 days for 5m data, last 730 days for hourly
            - Daily data: Available for longer historical periods
        """
        # Create date-based cache identifiers to avoid incorrect cache usage
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        date_range = f"{start_str}_{end_str}"

        if use_parallel and len(symbols) > 1:
            return self._load_parallel(
                symbols,
                start_date,
                end_date,
                date_range,
                force_refresh,
                max_workers,
                interval,
            )
        else:
            return self._load_sequential(
                symbols, start_date, end_date, date_range, force_refresh, interval
            )

    def _load_sequential(
        self, symbols, start_date, end_date, date_range, force_refresh, interval="1d"
    ):
        """Sequential loading with progress bar"""
        data_feeds = []
        successful_loads = 0
        failed_loads = []

        # Use tqdm for progress tracking
        for symbol in tqdm(
            symbols,
            desc="📈 Loading market data",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            # Auto-detect instrument type based on symbol pattern
            instrument_type = self._detect_instrument_type(symbol)

            symbol_data = self._load_single_instrument(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                date_range=date_range,
                force_refresh=force_refresh,
                instrument_type=instrument_type,
                interval=interval,
            )

            if symbol_data is not None:
                symbol_feed = bt.feeds.PandasData(dataname=symbol_data, name=symbol)
                data_feeds.append(symbol_feed)
                successful_loads += 1
            else:
                failed_loads.append(symbol)

        print(f"✅ Successfully loaded {successful_loads}/{len(symbols)} symbols")
        return data_feeds

    def _load_parallel(
        self,
        symbols,
        start_date,
        end_date,
        date_range,
        force_refresh,
        max_workers,
        interval="1d",
    ):
        """Parallel loading with progress bar and thread safety"""
        data_feeds = []

        def _load_single_symbol(symbol):
            """Load a single symbol - optimized for parallel execution"""
            try:
                # Auto-detect instrument type based on symbol pattern
                instrument_type = self._detect_instrument_type(symbol)

                symbol_data = self._load_single_instrument(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    date_range=date_range,
                    force_refresh=force_refresh,
                    instrument_type=instrument_type,
                    interval=interval,
                )

                if symbol_data is not None:
                    symbol_feed = bt.feeds.PandasData(dataname=symbol_data, name=symbol)
                    return symbol, symbol_feed, True
                else:
                    return symbol, None, False
            except Exception as e:
                return symbol, None, False

        successful_loads = 0
        failed_loads = []

        print(
            f"🚀 Loading {len(symbols)} symbols using {max_workers} parallel workers..."
        )

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(_load_single_symbol, symbol): symbol
                for symbol in symbols
            }

            # Process results with progress bar
            with tqdm(
                total=len(symbols),
                desc="📈 Loading market data",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100,
            ) as pbar:

                for future in as_completed(future_to_symbol):
                    symbol, symbol_feed, success = future.result()

                    if success and symbol_feed is not None:
                        data_feeds.append(symbol_feed)
                        successful_loads += 1
                        pbar.set_postfix_str(f"✅ {symbol}")
                    else:
                        failed_loads.append(symbol)
                        pbar.set_postfix_str(f"❌ {symbol}")

                    pbar.update(1)

        print(f"✅ Successfully loaded {successful_loads}/{len(symbols)} symbols")
        if failed_loads:
            print(f"❌ Failed to load: {len(failed_loads)} symbols")

        return data_feeds

    def _detect_instrument_type(self, symbol: str) -> str:
        """
        Auto-detect instrument type based on symbol pattern.

        Args:
            symbol (str): Symbol to analyze

        Returns:
            str: Instrument type ('index', 'equity', 'derivative')
        """
        if symbol.startswith("^"):
            return "index"
        elif symbol.endswith(".NS") or symbol.endswith(".BO"):
            return "equity"
        elif (
            "FUT" in symbol.upper() or "CE" in symbol.upper() or "PE" in symbol.upper()
        ):
            return "derivative"
        else:
            return "equity"  # Default to equity

    def _load_single_instrument(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        date_range: str,
        force_refresh: bool,
        instrument_type: str,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Helper method to load a single instrument with intelligent caching.

        Args:
            symbol (str): Symbol to load
            start_date (datetime): Start date
            end_date (datetime): End date
            date_range (str): Date range string for cache filename
            force_refresh (bool): Force refresh flag
            instrument_type (str): Type of instrument ('equity', 'index', 'derivative')
            interval (str): Data interval ('1d', '5m', '15m', '1h', etc.)

        Returns:
            pd.DataFrame or None: Loaded data or None if failed
        """
        # Create cache filename with date range, instrument type, and interval
        safe_symbol = symbol.replace("^", "").replace(".", "_").replace("&", "AND")
        cache_filename = f"{safe_symbol}_{instrument_type}_{interval}_{date_range}.pkl"
        cache_file = os.path.join(self.cache_dir, cache_filename)

        # Check for existing cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                # Validate cached data has the expected date range
                if self._validate_cached_data(cached_data, start_date, end_date):
                    return cached_data

            except Exception as e:
                print(f"Error loading cache for {symbol}: {e}, re-downloading...")

        # Download fresh data
        try:
            data_df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if data_df.empty:
                return None

            # Process data for backtrader compatibility
            processed_data = self._process_yfinance_data(data_df)

            # Additional validation for backtrader compatibility
            if processed_data is None or processed_data.empty:
                return None

            # Ensure we have enough data points
            if len(processed_data) < 10:
                print(
                    f"Warning: {symbol} has insufficient data ({len(processed_data)} days)"
                )
                return None

            # Cache the processed data
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(processed_data, f)
            except Exception as e:
                print(f"Warning: Could not cache data for {symbol}: {e}")

            return processed_data

        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None

    def _validate_cached_data(
        self, data: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> bool:
        """
        Validate that cached data covers the requested date range.

        Args:
            data (pd.DataFrame): Cached data
            start_date (datetime): Requested start date
            end_date (datetime): Requested end date

        Returns:
            bool: True if cache is valid, False otherwise
        """
        if data is None or data.empty:
            return False

        try:
            data_start = data.index.min().date()
            data_end = data.index.max().date()

            # Allow some tolerance for weekends/holidays - handle both datetime and date objects
            requested_start = (
                start_date.date() if hasattr(start_date, "date") else start_date
            )
            requested_end = end_date.date() if hasattr(end_date, "date") else end_date

            # Cache is valid if it covers the requested range (within 5 days tolerance)
            start_diff = (data_start - requested_start).days
            end_diff = (requested_end - data_end).days

            return start_diff <= 5 and end_diff <= 5

        except Exception:
            return False

    def _process_yfinance_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process yfinance data for backtrader compatibility.

        Args:
            data_df (pd.DataFrame): Raw yfinance data

        Returns:
            pd.DataFrame: Processed data ready for backtrader
        """
        try:
            # Handle MultiIndex columns from yfinance
            if isinstance(data_df.columns, pd.MultiIndex):
                data_df.columns = data_df.columns.droplevel(1)

            # Ensure proper column names for backtrader (lowercase)
            data_df.columns = [col.lower() for col in data_df.columns]

            # Ensure we have required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data_df.columns
            ]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Sort by date to ensure proper order
            data_df = data_df.sort_index()

            # Remove any duplicate dates
            data_df = data_df[~data_df.index.duplicated(keep="first")]

            # Ensure index is datetime type, not float (critical for backtrader)
            if not isinstance(data_df.index, pd.DatetimeIndex):
                data_df.index = pd.to_datetime(data_df.index)

            # Remove any rows with NaN values in OHLCV data
            data_df = data_df.dropna()

            # Validate numeric data types
            for col in required_columns:
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

            # Remove any rows that became NaN after numeric conversion
            data_df = data_df.dropna()

            return data_df

        except Exception as e:
            print(f"Error processing data: {e}")
            return None

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached data files.

        Args:
            pattern (str, optional): Pattern to match files (e.g., "RELIANCE" to clear only RELIANCE files)

        Returns:
            int: Number of files deleted
        """
        if not os.path.exists(self.cache_dir):
            return 0

        deleted_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                if pattern is None or pattern in filename:
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"Deleted cache file: {filename}")
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")

        print(f"Deleted {deleted_count} cache files")
        return deleted_count
