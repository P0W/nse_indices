"""
Adaptive Momentum Strategy

A multi-stock momentum-based trading strategy that identifies and follows trends using
technical indicators like moving averages, ATR, and momentum calculations. The strategy
dynamically selects the top-performing stocks and rebalances the portfolio periodically.
"""

import backtrader as bt
import os
import json
import logging
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from .base_strategy import BaseStrategy, StrategyConfig


class MomentumTrendStrategy(BaseStrategy):
    """
    Momentum-based trend following strategy
    """

    params = (
        ("sma_fast", 30),
        ("sma_slow", 100),
        ("atr_period", 14),
        ("atr_multiplier", 2.5),
        ("momentum_period", 20),
        ("top_n_stocks", 5),
        ("rebalance_days", 10),
    )

    def __init__(self):
        super().__init__()
        self.inds = {}
        self.rebalance_counter = 0
        self.monthly_returns = {}
        self.peak_value = self.broker.getvalue()
        self.drawdowns = []

        self.log(
            f"Initializing MomentumTrendStrategy with {len(self.datas)} data feeds"
        )
        self.log(
            f"Parameters: SMA_fast={self.p.sma_fast}, SMA_slow={self.p.sma_slow}, ATR_period={self.p.atr_period}, Momentum_period={self.p.momentum_period}"
        )
        self.log(
            f"Rebalance_days={self.p.rebalance_days}, Top_n_stocks={self.p.top_n_stocks}, ATR_multiplier={self.p.atr_multiplier}"
        )

        for d in self.datas:
            self.log(f"Setting up indicators for symbol: {d._name}")
            try:
                self.inds[d._name] = {
                    "sma_fast": bt.ind.SMA(d.close, period=int(self.p.sma_fast)),
                    "sma_slow": bt.ind.SMA(d.close, period=int(self.p.sma_slow)),
                    "atr": bt.ind.ATR(d, period=int(self.p.atr_period)),
                    "momentum": bt.ind.ROC(d.close, period=int(self.p.momentum_period)),
                }
                self.log(f"Successfully created indicators for {d._name}")
            except Exception as e:
                self.log(f"ERROR creating indicators for {d._name}: {str(e)}")
                import traceback

                self.log(f"Indicator creation traceback for {d._name}:")
                self.log(traceback.format_exc())
                # Create a dummy indicator set to avoid KeyError later
                self.inds[d._name] = {
                    "sma_fast": None,
                    "sma_slow": None,
                    "atr": None,
                    "momentum": None,
                }

        self.log(f"Indicators initialized for {len(self.inds)} symbols")

    def _should_exit(self, d):
        """Determine if we should exit a position"""
        pos = self.getposition(d)
        if pos.size == 0:  # No position to exit
            return False

        try:
            self.log(
                f"Checking exit conditions for {d._name}, position size: {pos.size}"
            )

            # Check if indicators have enough data
            ind = self.inds[d._name]
            if (
                len(ind["atr"]) == 0
                or len(ind["sma_fast"]) == 0
                or len(ind["sma_slow"]) == 0
                or len(ind["momentum"]) == 0
            ):
                self.log(f"WARNING: Not enough data for indicators on {d._name}")
                return False

            price = d.close[0]
            atr_value = ind["atr"][0]
            sma_fast_value = ind["sma_fast"][0]
            sma_slow_value = ind["sma_slow"][0]
            momentum_value = ind["momentum"][0]

            stop_loss_price = pos.price - self.p.atr_multiplier * atr_value

            self.log(
                f"Exit check for {d._name}: price={price:.2f}, stop_loss={stop_loss_price:.2f}, "
                f"sma_fast={sma_fast_value:.2f}, sma_slow={sma_slow_value:.2f}, momentum={momentum_value:.4f}"
            )

            should_exit = (
                price < stop_loss_price
                or sma_fast_value < sma_slow_value
                or momentum_value < 0
            )

            if should_exit:
                self.log(
                    f"EXIT SIGNAL for {d._name}: price_below_stop={price < stop_loss_price}, "
                    f"sma_bearish={sma_fast_value < sma_slow_value}, momentum_negative={momentum_value < 0}"
                )

            return should_exit

        except (IndexError, ValueError, KeyError) as e:
            self.log(f"ERROR in _should_exit for {d._name}: {str(e)}")
            return False

    def _should_enter(self, d):
        """Determine if we should enter a position"""
        try:
            ind = self.inds[d._name]

            # Check if indicators have enough data
            if len(ind["sma_fast"]) == 0 or len(ind["sma_slow"]) == 0:
                self.log(f"WARNING: Not enough SMA data for entry check on {d._name}")
                return False

            sma_fast_value = ind["sma_fast"][0]
            sma_slow_value = ind["sma_slow"][0]

            should_enter = sma_fast_value > sma_slow_value

            self.log(
                f"Entry check for {d._name}: sma_fast={sma_fast_value:.2f}, "
                f"sma_slow={sma_slow_value:.2f}, should_enter={should_enter}"
            )

            return should_enter

        except (IndexError, ValueError, KeyError) as e:
            self.log(f"ERROR in _should_enter for {d._name}: {str(e)}")
            return False

    def execute_strategy(self):
        """Execute momentum strategy logic"""
        try:
            # Track portfolio performance
            current_value = self.broker.getvalue()
            current_date = self.datas[0].datetime.date(0)

            # Check if we have enough data to run the strategy
            # We need at least max(sma_slow, momentum_period, atr_period) bars
            min_bars_needed = max(
                self.p.sma_slow, self.p.momentum_period, self.p.atr_period
            )
            current_bar = len(self.datas[0])

            if current_bar < min_bars_needed:
                self.log(
                    f"Not enough data yet: {current_bar}/{min_bars_needed} bars, skipping strategy execution"
                )
                return

            self.log(
                f"Strategy execution on {current_date}, portfolio value: {current_value:.2f}, rebalance_counter: {self.rebalance_counter}, bar: {current_bar}"
            )

            # Update peak and calculate drawdown
            if current_value > self.peak_value:
                self.peak_value = current_value

            drawdown = (self.peak_value - current_value) / self.peak_value * 100
            self.drawdowns.append(drawdown)

            # Track monthly returns
            month_year = current_date.strftime("%Y-%m")
            if month_year not in self.monthly_returns:
                if len(self.portfolio_values) > 1:
                    # Find the last value from previous month
                    prev_month_values = [
                        v
                        for i, v in enumerate(self.portfolio_values[:-1])
                        if self.dates[i].strftime("%Y-%m") != month_year
                    ]
                    if prev_month_values:
                        prev_value = prev_month_values[-1]
                        monthly_return = (current_value - prev_value) / prev_value * 100
                        self.monthly_returns[month_year] = monthly_return

            if self.rebalance_counter % self.p.rebalance_days != 0:
                self.rebalance_counter += 1
                return

            self.log(f"Rebalancing portfolio on day {self.rebalance_counter}")

            # Filter valid data feeds (not NaN) and sort by momentum
            valid_datas = []
            for d in self.datas:
                try:
                    # First check if we have indicators for this symbol
                    if d._name not in self.inds:
                        self.log(f"No indicators found for {d._name}, skipping")
                        continue

                    # Check basic data availability
                    if not hasattr(d, "close") or len(d.close) == 0:
                        self.log(f"No close data for {d._name}")
                        continue

                    # Check if current price is valid
                    try:
                        current_price = d.close[0]
                        if current_price != current_price:  # Check for NaN
                            self.log(f"NaN price for {d._name}")
                            continue
                    except IndexError:
                        self.log(f"Cannot access current price for {d._name}")
                        continue

                    # Check if momentum indicator has data and is valid
                    momentum_ind = self.inds[d._name]["momentum"]
                    if momentum_ind is None or len(momentum_ind) == 0:
                        self.log(f"Momentum indicator has no data for {d._name}")
                        continue

                    try:
                        momentum_val = momentum_ind[0]
                        if momentum_val != momentum_val:  # Check for NaN
                            self.log(f"NaN momentum value for {d._name}")
                            continue
                    except IndexError:
                        self.log(
                            f"Cannot access momentum[0] for {d._name}, length: {len(momentum_ind)}"
                        )
                        continue

                    # If we get here, the data is valid
                    self.log(f"Valid data for {d._name}, momentum: {momentum_val:.4f}")
                    valid_datas.append(d)

                except Exception as e:
                    self.log(f"Error checking data validity for {d._name}: {str(e)}")
                    continue  # Skip this data feed if there's an issue

            self.log(
                f"Found {len(valid_datas)} valid data feeds out of {len(self.datas)} total"
            )

            if not valid_datas:
                self.log("No valid data feeds found, skipping rebalancing")
                self.rebalance_counter += 1
                return

            # Sort by momentum and select top stocks
            try:
                # Additional safety check before sorting
                safe_valid_datas = []
                for d in valid_datas:
                    try:
                        momentum_val = self.inds[d._name]["momentum"][0]
                        if momentum_val == momentum_val:  # Not NaN
                            safe_valid_datas.append(d)
                        else:
                            self.log(
                                f"Filtering out {d._name} due to NaN momentum in sorting phase"
                            )
                    except (IndexError, KeyError) as e:
                        self.log(
                            f"Error accessing momentum for {d._name} in sorting phase: {str(e)}"
                        )
                        continue

                if not safe_valid_datas:
                    self.log("No safe valid data feeds for sorting")
                    self.rebalance_counter += 1
                    return

                top_stocks = sorted(
                    safe_valid_datas,
                    key=lambda d: self.inds[d._name]["momentum"][0],
                    reverse=True,
                )[: self.p.top_n_stocks]

                top_symbols = [d._name for d in top_stocks]
                top_momentums = [self.inds[d._name]["momentum"][0] for d in top_stocks]
                self.log(
                    f"Top {len(top_stocks)} momentum stocks: {dict(zip(top_symbols, top_momentums))}"
                )

            except Exception as e:
                self.log(f"Error sorting stocks by momentum: {str(e)}")
                import traceback

                self.log(f"Sorting traceback: {traceback.format_exc()}")
                self.rebalance_counter += 1
                return

            # Exit positions that should be closed
            exits_made = 0
            for data_feed in self.datas:
                pos = self.getposition(data_feed)
                if pos.size != 0:
                    if self._should_exit(data_feed):
                        self.log(
                            f"Closing position in {data_feed._name}, size: {pos.size}"
                        )
                        self.close(data_feed)
                        exits_made += 1

            self.log(f"Made {exits_made} position exits")

            # Enter new positions for top momentum stocks
            entries_made = 0
            for d in top_stocks:
                if self.getposition(d).size == 0 and self._should_enter(d):
                    available_cash = self.broker.get_cash()
                    if available_cash > 0 and d.close[0] > 0:
                        size = int(available_cash / (self.p.top_n_stocks * d.close[0]))
                        if size > 0:
                            self.log(
                                f"Buying {size} shares of {d._name} at {d.close[0]:.2f}, cash available: {available_cash:.2f}"
                            )
                            self.buy(data=d, size=size)
                            entries_made += 1
                        else:
                            self.log(
                                f"Insufficient cash to buy {d._name}, required: {d.close[0]:.2f}, available: {available_cash:.2f}"
                            )
                    else:
                        self.log(
                            f"Cannot buy {d._name}: cash={available_cash:.2f}, price={d.close[0] if len(d.close) > 0 else 'N/A'}"
                        )

            self.log(f"Made {entries_made} new position entries")
            self.rebalance_counter += 1

        except Exception as e:
            self.log(f"CRITICAL ERROR in execute_strategy: {str(e)}")
            import traceback

            self.log(f"Traceback: {traceback.format_exc()}")
            raise


class AdaptiveMomentumConfig(StrategyConfig):
    """
    Configuration for Adaptive Momentum Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for momentum strategy experiments
        """
        return {
            "sma_fast": [20, 30, 50, 60],
            "sma_slow": [100, 150, 200, 250],
            "atr_period": [10, 14, 20],
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
            "momentum_period": [10, 15, 20, 25],
            "top_n_stocks": [3, 5, 7, 10],
            "rebalance_days": [5, 10, 15, 20, 30],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for momentum strategy
        """
        return {
            "sma_fast": 30,
            "sma_slow": 100,
            "atr_period": 14,
            "atr_multiplier": 2.5,
            "momentum_period": 20,
            "top_n_stocks": 5,
            "rebalance_days": 10,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate momentum strategy parameters
        """
        # SMA fast must be less than SMA slow
        if params.get("sma_fast", 0) >= params.get("sma_slow", 0):
            return False

        # All parameters must be positive
        for key, value in params.items():
            if isinstance(value, (int, float)) and value <= 0:
                return False

        # Top N stocks should be reasonable
        if params.get("top_n_stocks", 0) > 20:
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the momentum strategy class
        """
        return MomentumTrendStrategy

    def get_required_data_feeds(self) -> int:
        """
        Momentum strategy works with multiple stocks
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights optimized for momentum strategy
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.3}


def fetch_nifty_200_from_nse():
    """
    Fetch Nifty 200 stocks from NSE India API
    """
    try:
        print("📡 Fetching Nifty 200 stocks from NSE India...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # NSE API endpoint for Nifty 200 constituents
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200"

        session = requests.Session()
        session.headers.update(headers)

        # First, get the main page to establish session
        session.get("https://www.nseindia.com", timeout=10)

        # Now fetch the Nifty 200 data
        response = session.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            stocks = []

            if "data" in data:
                for stock_info in data["data"]:
                    symbol = stock_info.get("symbol", "")
                    if symbol and symbol != "NIFTY 200":  # Exclude the index itself
                        stocks.append(f"{symbol}.NS")

            if stocks:
                print(
                    f"✅ Successfully fetched {len(stocks)} Nifty 200 stocks from NSE"
                )
                return stocks
            else:
                print("⚠️ No stock data found in NSE response")
                return None
        else:
            print(f"⚠️ NSE API returned status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"⚠️ Error fetching from NSE: {e}")
        return None


def get_nifty_200_stocks():
    """
    Get Nifty 200 stocks from NSE API with fallback to comprehensive hardcoded list
    """
    # Try to fetch from NSE first
    nse_stocks = fetch_nifty_200_from_nse()

    if nse_stocks:
        return nse_stocks

    print("🔄 Using comprehensive fallback Nifty 200 list...")

    # Comprehensive Nifty 200 fallback list (based on actual Nifty 200 constituents)
    nifty_200_stocks = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "KOTAKBANK.NS",
        "ITC.NS",
        "LT.NS",
        "ASIANPAINT.NS",
        "AXISBANK.NS",
        "MARUTI.NS",
        "TITAN.NS",
        "NESTLEIND.NS",
        "BAJFINANCE.NS",
        "HCLTECH.NS",
        "SUNPHARMA.NS",
        "ULTRACEMCO.NS",
        # ... (rest of the list for brevity)
    ]

    # Remove duplicates and return
    unique_stocks = list(dict.fromkeys(nifty_200_stocks))
    print(f"✅ Using fallback list with {len(unique_stocks)} Nifty 200 stocks")
    return unique_stocks


def filter_stocks_with_data(
    symbols, start_date, end_date, max_stocks_to_test=50, max_workers=8
):
    """
    Filter stocks that have data available for the requested period using parallel processing
    """
    print(
        f"\n🔍 Filtering {max_stocks_to_test} stocks for data availability using {max_workers} parallel workers..."
    )

    from utils import MarketDataLoader

    def _check_single_stock(symbol):
        """Check data availability for a single stock - optimized for parallel execution"""
        if not symbol:
            return symbol, 0, "Empty symbol"

        # Create a new loader instance for each worker to avoid threading issues
        loader = MarketDataLoader()

        try:
            # Use 1-year test period
            test_end = (datetime.now() - timedelta(days=3)).date()
            test_start = test_end.replace(year=test_end.year - 1)

            start_str = test_start.strftime("%Y%m%d")
            end_str = test_end.strftime("%Y%m%d")
            date_range = f"{start_str}_{end_str}"

            instrument_type = loader._detect_instrument_type(symbol)
            symbol_data = loader._load_single_instrument(
                symbol=symbol,
                start_date=test_start,
                end_date=test_end,
                date_range=date_range,
                force_refresh=False,
                instrument_type=instrument_type,
            )

            data_length = (
                len(symbol_data)
                if symbol_data is not None and not symbol_data.empty
                else 0
            )

            return (
                symbol,
                data_length,
                "Success" if data_length > 100 else "Insufficient data",
            )

        except Exception as e:
            return symbol, 0, f"Error: {str(e)[:50]}"

    # Test stocks in parallel
    test_symbols = symbols[:max_stocks_to_test]
    available_stocks = []
    failed_stocks = []

    print(f"🔄 Processing {len(test_symbols)} stocks in parallel...")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(_check_single_stock, symbol): symbol
            for symbol in test_symbols
        }

        # Process results with progress bar
        with tqdm(
            total=len(test_symbols),
            desc="📊 Checking stocks",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100,
        ) as pbar:

            for future in as_completed(future_to_symbol):
                symbol, data_length, status = future.result()

                if data_length > 100:  # At least 100 days of data
                    available_stocks.append(symbol)
                    pbar.set_postfix_str(f"✅ {symbol}: {data_length} days")
                else:
                    failed_stocks.append(symbol)
                    pbar.set_postfix_str(f"❌ {symbol}: {data_length} days")

                pbar.update(1)

                # Small delay to show progress updates
                time.sleep(0.01)

    print(f"\n📈 Results Summary:")
    print(f"  ✅ Available stocks: {len(available_stocks)}")
    print(f"  ❌ Failed stocks: {len(failed_stocks)}")
    print(f"  📊 Success rate: {len(available_stocks)/len(test_symbols)*100:.1f}%")

    if len(available_stocks) < 10:
        print("\n⚠️ Warning: Less than 10 stocks available. Consider:")
        print("   - Using a more recent date range")
        print("   - Checking internet connection")
        print("   - Increasing max_stocks_to_test parameter")

    # Show top performing stocks
    if available_stocks:
        print(f"\n🏆 Top available stocks (showing first 10):")
        for i, stock in enumerate(available_stocks[:10], 1):
            print(f"  {i:2d}. {stock}")

    return available_stocks
