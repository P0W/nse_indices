"""
Market Data Utilities for Nifty Trading Strategy

Fast, smart market data loader with intelligent caching and parallel processing.
"""

import os
import pandas as pd
import yfinance as yf
import backtrader as bt
import logging
from datetime import datetime, date
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq


class IndianBrokerageCommission(bt.CommInfoBase):
    """
    Realistic cost model for NSE **delivery (CNC)** trades.

    Assumptions
    -----------
    • Broker: Discount broker (Zerodha / Upstox style)
    • Plan: 0.03 % or ₹20 per executed order, whichever is lower
    • Product: **Equity Delivery** (CNC, holding > 1 day)
    • Charges based on SEBI / NSE circulars effective 1-Jan-2024
    """

    params = dict(
        brokerage_pct=0.0003,  # 0.03 %
        brokerage_cap=20.0,  # ₹20 per side
        stt_pct=0.001,  # 0.1 % on SELL side only
        txn_charge_pct=0.0000345,  # NSE: 0.00345 %
        gst_rate=0.18,  # 18 % on (brokerage + txn + SEBI)
        sebi_pct=0.0000001,  # ₹10 / crore  ≈ 0.000001 %
        stamp_pct=0.00015,  # 0.015 % on BUY side
    )

    def getcommission(self, size, price):
        """Calculate commission for each side (buy and sell separately)."""
        turnover = abs(size) * price
        brokerage = min(turnover * self.p.brokerage_pct, self.p.brokerage_cap)
        txn = turnover * self.p.txn_charge_pct
        sebi = turnover * self.p.sebi_pct
        gst = (brokerage + txn + sebi) * self.p.gst_rate
        return brokerage + txn + sebi + gst

    def getoperationcost(self, size, price):
        """Calculate additional per-side costs (stamp duty and STT)."""
        turnover = abs(size) * price
        stamp = turnover * self.p.stamp_pct if size > 0 else 0.0  # BUY only
        stt = turnover * self.p.stt_pct if size < 0 else 0.0  # SELL only
        return stamp + stt


def setup_logger():
    """Setup logging configuration for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    )
    return logging.getLogger(__name__)


class MarketDataLoader:
    """
    Fast, smart market data loader with intelligent caching:
    • Smart cache checking - only downloads missing data
    • Uses yfinance with optimizations for speed
    • Incremental updates - no redundant downloads
    • Parallel processing for multiple symbols
    """

    def __init__(self, cache_dir: str = "data_parquet", verbose: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def load_market_data(
        self,
        symbols: List[str],
        start_date: Union[datetime, date],
        end_date: Union[datetime, date],
        force_refresh: bool = False,
        use_parallel: bool = True,
        max_workers: int = 4,
        interval: str = "1d",
    ) -> List[bt.feeds.PandasData]:
        """Load market data with intelligent caching."""
        return self.load(
            symbols=symbols,
            start=start_date,
            end=end_date,
            interval=interval,
            force_refresh=force_refresh,
            use_parallel=use_parallel,
            max_workers=max_workers,
        )

    def load(
        self,
        symbols: List[str],
        start: Union[str, datetime, date],
        end: Union[str, datetime, date],
        interval: str = "1d",
        force_refresh: bool = False,
        use_parallel: bool = True,
        max_workers: int = 4,
    ) -> List[bt.feeds.PandasData]:
        """Smart loading that only downloads what's needed."""
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()

        if self.verbose:
            self.logger.info(
                f"Loading {len(symbols)} symbols for {start_ts.date()} to {end_ts.date()}"
            )

        if use_parallel and len(symbols) > 1:
            return self._load_parallel(
                symbols, start_ts, end_ts, interval, force_refresh, max_workers
            )
        else:
            return self._load_sequential(
                symbols, start_ts, end_ts, interval, force_refresh
            )

    def _load_sequential(
        self,
        symbols: List[str],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
        force_refresh: bool,
    ) -> List[bt.feeds.PandasData]:
        """Sequential loading with smart caching."""
        feeds = []
        iterator = tqdm(symbols, desc="Loading data")

        for sym in iterator:
            try:
                df = self._get_symbol_data(
                    sym, start_ts, end_ts, interval, force_refresh
                )
                if df is not None and not df.empty:
                    feeds.append(bt.feeds.PandasData(dataname=df, name=sym))
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to load {sym}: {e}")
                continue

        return feeds

    def _load_parallel(
        self,
        symbols: List[str],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
        force_refresh: bool,
        max_workers: int,
    ) -> List[bt.feeds.PandasData]:
        """Parallel loading for faster processing."""
        feeds = []

        def load_single(sym):
            try:
                df = self._get_symbol_data(
                    sym, start_ts, end_ts, interval, force_refresh
                )
                if df is not None and not df.empty:
                    return bt.feeds.PandasData(dataname=df, name=sym)
                return None
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(load_single, sym): sym for sym in symbols
            }

            iterator = as_completed(future_to_symbol)
            iterator = tqdm(
                iterator, total=len(symbols), desc="Loading data (parallel)"
            )
            for future in iterator:
                result = future.result()
                if result is not None:
                    feeds.append(result)

        return feeds

    def _get_symbol_data(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
        force_refresh: bool,
    ) -> pd.DataFrame:
        """Smart data retrieval with intelligent caching."""
        pq_file = self._pq_path(symbol, interval)

        # Check cache
        if not force_refresh and pq_file.exists():
            try:
                cached_meta = pq.read_table(pq_file, columns=["index"]).to_pandas()

                if not cached_meta.empty:
                    cached_start = pd.Timestamp(cached_meta["index"].min()).tz_localize(
                        None
                    )
                    cached_end = pd.Timestamp(cached_meta["index"].max()).tz_localize(
                        None
                    )
                    start_ts = start_ts.tz_localize(None) if start_ts.tz else start_ts
                    end_ts = end_ts.tz_localize(None) if end_ts.tz else end_ts

                    # Cache hit
                    if cached_start <= start_ts and cached_end >= end_ts:
                        df = pq.read_table(
                            pq_file,
                            filters=[
                                ("index", ">=", start_ts),
                                ("index", "<=", end_ts),
                            ],
                        ).to_pandas()

                        if not df.empty:
                            df = df.set_index("index")
                            return self._standardise(df)

                    # Cache extension needed
                    else:
                        return self._extend_cache(
                            symbol,
                            pq_file,
                            start_ts,
                            end_ts,
                            interval,
                            cached_start,
                            cached_end,
                        )

            except Exception:
                pass  # Fall through to fresh download

        # Fresh download
        return self._download_and_cache(symbol, pq_file, start_ts, end_ts, interval)

    def _extend_cache(
        self,
        symbol: str,
        pq_file: Path,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
        cached_start: pd.Timestamp,
        cached_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Extend existing cache with only missing data."""
        existing_df = pq.read_table(pq_file).to_pandas().set_index("index")

        # Determine extended range with buffer
        download_start = min(start_ts, cached_start) - pd.Timedelta(days=30)
        download_end = max(end_ts, cached_end) + pd.Timedelta(days=30)

        new_df = self._download_yfinance(symbol, download_start, download_end)

        if new_df is not None and not new_df.empty:
            combined_df = pd.concat([existing_df, new_df]).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            self._save_to_cache(combined_df, pq_file)

            mask = (combined_df.index >= start_ts) & (combined_df.index <= end_ts)
            return combined_df[mask]

        # Fallback to existing cache
        mask = (existing_df.index >= start_ts) & (existing_df.index <= end_ts)
        return existing_df[mask]

    def _download_and_cache(
        self,
        symbol: str,
        pq_file: Path,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
    ) -> pd.DataFrame:
        """Download data and cache it with buffer."""
        buffer_days = 90
        download_start = max(
            start_ts - pd.Timedelta(days=buffer_days),
            pd.Timestamp.today() - pd.Timedelta(days=5 * 365),
        )
        download_end = min(
            end_ts + pd.Timedelta(days=buffer_days), pd.Timestamp.today()
        )

        df = self._download_yfinance(symbol, download_start, download_end)

        if df is not None and not df.empty:
            self._save_to_cache(df, pq_file)
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            return df[mask]

        raise ValueError(f"Failed to download data for {symbol}")

    def _download_yfinance(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Download data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                auto_adjust=True,
                back_adjust=False,
                repair=True,
                keepna=False,
                actions=False,
            )

            if df.empty:
                return None

            df.columns = [col.lower() for col in df.columns]
            required_cols = ["open", "high", "low", "close", "volume"]
            df = df[required_cols]
            return self._standardise(df)

        except Exception:
            return None

    def _save_to_cache(self, df: pd.DataFrame, pq_file: Path):
        """Save dataframe to parquet cache."""
        cache_df = df.reset_index()
        cache_df = cache_df.rename(columns={cache_df.columns[0]: "index"})
        cache_df.to_parquet(pq_file, compression="zstd", engine="pyarrow", index=False)

    def _pq_path(self, symbol: str, interval: str) -> Path:
        """Get parquet file path for symbol."""
        return self.cache_dir / f"{symbol}_{interval}.parquet"

    @staticmethod
    def _standardise(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataframe format."""
        df.columns = [c.lower() for c in df.columns]

        if df.index.name:
            df.index.name = df.index.name.lower()

        # Remove timezone for consistent caching
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.dropna()
        return df

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cached parquet files."""
        deleted_count = 0
        for file_path in self.cache_dir.glob("*.parquet"):
            if pattern is None or pattern in file_path.name:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception:
                    pass
        return deleted_count
