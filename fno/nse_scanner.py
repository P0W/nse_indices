#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Short Iron Condor Scanner for NSE Stocks
This script provides a simple approach to identify stocks with low volatility
suitable for short iron condor option strategies
"""

import os
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
from tqdm import tqdm

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
plt.style.use("ggplot")

# Setup logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# List of commonly traded NSE F&O stocks
NIFTY_FO_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "HCLTECH.NS",
    "BAJFINANCE.NS",
]


def download_stock_data(
    symbols: List[str],
    period: str = "6mo",
    cache_dir: str = "stock_cache",
    cache_days: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for the given stock symbols with caching

    Args:
        symbols: List of stock symbols
        period: Time period for data download
        cache_dir: Directory to cache stock data
        cache_days: Number of days before refreshing cache

    Returns:
        Dictionary mapping symbols to their historical data
    """
    logger.info("Checking data for %d stocks...", len(symbols))

    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    all_data = {}
    max_retries = 3
    today = datetime.now().date()

    # Create progress bar
    pbar = tqdm(
        total=len(symbols),
        desc="Processing stocks",
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for symbol in symbols:
        cache_file = os.path.join(cache_dir, f"{symbol.replace('.', '_')}.pkl")
        use_cache = False

        # Update progress bar description
        pbar.set_description(f"Processing {symbol}")

        # Check if we have a valid cache file
        if os.path.exists(cache_file):
            file_modified = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
            days_old = (today - file_modified).days

            if days_old <= cache_days:
                try:
                    data = pd.read_pickle(cache_file)
                    if not data.empty and len(data) > 30:
                        all_data[symbol] = data
                        # Simplified output for clean progress bar display
                        use_cache = True
                except Exception as exc:
                    tqdm.write(f"Error reading cache for {symbol}: {exc}")

        # Download data if cache is invalid or missing
        if not use_cache:
            retries = 0
            while retries < max_retries:
                try:
                    # Update progress bar during download
                    pbar.set_description(f"Downloading {symbol}")

                    data = yf.download(
                        symbol,
                        period=period,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                        timeout=30,
                    )

                    if not data.empty and len(data) > 30:
                        # Save to cache
                        all_data[symbol] = data
                        data.to_pickle(cache_file)
                        # Use tqdm.write for clean output
                        tqdm.write(
                            f"✅ Downloaded {len(data)} days of data for {symbol}"
                        )
                        break
                    else:
                        tqdm.write(
                            f"⚠️ Warning: Insufficient data for {symbol}, retrying..."
                        )
                except Exception as exc:
                    tqdm.write(
                        f"❌ Error downloading {symbol} (attempt {retries+1}/{max_retries}): {exc}"
                    )

                retries += 1
                if retries < max_retries:
                    pbar.set_description(f"Retrying {symbol}")
                    time.sleep(2)

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    logger.info(
        "✅ Successfully loaded data for %d/%d stocks", len(all_data), len(symbols)
    )
    return all_data


def main(days_to_expiry: int = 30) -> Optional[pd.DataFrame]:
    """
    Main execution function for the Iron Condor Scanner

    Args:
        days_to_expiry: Number of days to expiration for option calculations

    Returns:
        DataFrame with final analysis results or None if there was an error
    """
    logger.info("=== Enhanced Iron Condor Stock Scanner ===")
    logger.info("Using %d days to expiry for calculations", days_to_expiry)

    # Track data quality issues
    data_quality_issues = {
        "missing_values": {},
        "default_values_used": {},
        "warning_count": 0,
    }

    # Read stock list from CSV if available
    symbols = read_stock_list("nse_fo_stocks.csv")

    # Download data with caching enabled
    stock_data = download_stock_data(symbols, period="6mo", cache_days=1)

    if not stock_data:
        logger.error("No stock data available. Exiting.")
        return None

    # Add IV skew analysis
    logger.info("\nAnalyzing implied volatility skew...")
    iv_metrics = {}
    # for symbol in tqdm(
    #     symbols, desc="Analyzing IV skew"
    # ):
    #     avg_iv, iv_skew = analyze_iv_skew(stock_data, symbol)
    #     if avg_iv is not None and iv_skew is not None:
    #         iv_metrics[symbol] = {"Avg_IV": avg_iv, "IV_Skew": iv_skew}
    #         logger.info(
    #             "IV for %s: Avg_IV = %.2f, IV_Skew = %.2f", symbol, avg_iv, iv_skew
    #         )

    # Calculate mean reversion metrics
    logger.info("\nCalculating mean reversion indicators...")
    mean_reversion_metrics = calculate_mean_reversion_metrics(stock_data)

    # Analyze volume profile
    logger.info("\nAnalyzing volume profiles...")
    volume_metrics = analyze_volume_profile(stock_data)

    # Analyze volatility
    logger.info("\nAnalyzing volatility metrics...")
    analysis_df = analyze_volatility(stock_data)

    # Track NaN values from volatility analysis
    for col in ["Volatility_21d", "Volatility_63d", "ATR_Percent", "Range_52w_Pct"]:
        if col in analysis_df.columns and analysis_df[col].isna().any():
            count = analysis_df[col].isna().sum()
            data_quality_issues["missing_values"][col] = count
            data_quality_issues["warning_count"] += 1

    if analysis_df.empty:
        logger.error("No analysis results available. Exiting.")
        return None

    # Identify best candidates with enhanced metrics
    logger.info("\nIdentifying best iron condor candidates...")
    candidates = identify_iron_condor_candidates(
        analysis_df, mean_reversion_metrics, volume_metrics, iv_metrics
    )

    # Track NaN values in rank components
    rank_components = [
        "Volatility_Rank",
        "ATR_Rank",
        "Range_Rank",
        "RSI_Rank",
        "BB_Width_Rank",
        "Volume_Rank",
    ]
    for col in rank_components:
        if col in candidates.columns and candidates[col].isna().any():
            count = candidates[col].isna().sum()
            data_quality_issues["missing_values"][col] = count
            data_quality_issues["warning_count"] += 1

    # Check if IC_Score had issues
    if "IC_Score" in candidates.columns and candidates["IC_Score"].isna().any():
        count = candidates["IC_Score"].isna().sum()
        data_quality_issues["missing_values"]["IC_Score"] = count
        data_quality_issues["warning_count"] += 1

    # Calculate optimal strikes with the specified expiry
    logger.info(
        "\nCalculating optimal strike prices for %d-day expiry...", days_to_expiry
    )
    final_df = calculate_strikes(candidates, days_to_expiry=days_to_expiry)

    # Calculate probability of profit
    logger.info("\nCalculating probability of profit...")
    final_df = calculate_probability_of_profit(final_df)

    # Check for upcoming events
    logger.info("\nChecking for upcoming earnings/events...")
    events = check_upcoming_events(
        [s + ".NS" for s in final_df["Symbol"].head(10).tolist()]
    )

    # Display results
    display_cols = [
        "Symbol",
        "Price",
        "Expected_Move_Pct",
        "Premium_Collected",
        "Max_Risk",
        "Risk_Reward",
        "Prob_of_Profit",
        "IC_Score",
        ## Call and Put strikes
    ]

    top_5 = final_df.head(5)
    logger.info(
        "\nTop 5 candidates for short iron condor strategy (for %d-day expiry):",
        days_to_expiry,
    )
    logger.info("\n%s", top_5[display_cols].round(2).to_string())

    # Display data quality report if issues were found
    if data_quality_issues["warning_count"] > 0:
        logger.warning("\n⚠️ DATA QUALITY ISSUES SUMMARY ⚠️")
        logger.warning("The following issues were automatically fixed during analysis:")

        if data_quality_issues["missing_values"]:
            logger.warning("\n  Missing Values (NaN) Detected and Fixed:")
            for col, count in data_quality_issues["missing_values"].items():
                fix_method = "median" if "Rank" in col else "default values"
                logger.warning(
                    "    - %s: %d missing values replaced with %s",
                    col,
                    count,
                    fix_method,
                )

        logger.warning(
            "\nNOTE: While these issues were automatically fixed, they may affect result quality."
        )
        logger.warning(
            "Consider checking data sources or extending the analysis period for better results."
        )

    # Save full results to CSV
    save_results_to_csv(final_df, f"iron_condor_candidates_{days_to_expiry}d.csv")

    # Also save just the top candidates for quick reference
    save_results_to_csv(top_5, f"top_iron_condor_candidates_{days_to_expiry}d.csv")

    # Show any upcoming events
    if events:
        logger.info("\nUpcoming Events (Next 30 days):")
        for symbol, event_info in events.items():
            logger.info(
                "%s: %s in %d days (%s)",
                symbol,
                event_info["Event"],
                event_info["Days_Away"],
                event_info["Date"],
            )

    # Plot top candidates
    logger.info("\nGenerating price chart for top candidates...")
    top_symbols = top_5["Symbol"].tolist()
    # plot_top_candidates(stock_data, top_symbols)

    return final_df


def analyze_iv_skew(
    stock_data: Dict[str, pd.DataFrame], symbol: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Analyze the implied volatility vs historical volatility

    Args:
        stock_data: Dictionary of stock data
        symbol: Stock symbol to analyze

    Returns:
        Tuple of average IV and IV skew, or None if data not available
    """
    try:
        # Get options data using yfinance
        stock = yf.Ticker(symbol)
        options = stock.options

        if not options:
            logger.warning("No options data available for %s", symbol)
            return None, None

        # Get nearest expiration date
        nearest_exp = options[0]

        # Get option chain
        opt = stock.option_chain(nearest_exp)
        calls = opt.calls
        puts = opt.puts

        # Calculate IV skew (difference between put IV and call IV at ATM strikes)
        current_price = stock_data[symbol]["Close"].iloc[-1]
        atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]]

        iv_skew = (
            atm_put["impliedVolatility"].iloc[0] - atm_call["impliedVolatility"].iloc[0]
        )
        avg_iv = (
            atm_put["impliedVolatility"].iloc[0] + atm_call["impliedVolatility"].iloc[0]
        ) / 2

        logger.info("IV for %s: Avg_IV = %.2f, IV_Skew = %.2f", symbol, avg_iv, iv_skew)

        return avg_iv, iv_skew
    except Exception as exc:
        logger.error("Error analyzing IV for %s: %s", symbol, exc)
        return None, None


def calculate_mean_reversion_metrics(
    stock_data: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate mean reversion metrics for each stock

    Args:
        stock_data: Dictionary of stock data

    Returns:
        Dictionary mapping symbols to their mean reversion metrics
    """
    mean_reversion_metrics = {}

    for symbol, data in stock_data.items():
        try:
            # RSI (Relative Strength Index)
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Bollinger Bands
            sma_20 = data["Close"].rolling(window=20).mean()
            std_20 = data["Close"].rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)

            # %B indicator (position within Bollinger Bands)
            percent_b = (data["Close"] - lower_band) / (upper_band - lower_band)

            # Stochastic Oscillator
            low_14 = data["Low"].rolling(window=14).min()
            high_14 = data["High"].rolling(window=14).max()
            k = 100 * ((data["Close"] - low_14) / (high_14 - low_14))
            d = k.rolling(window=3).mean()

            # Store the most recent values
            mean_reversion_metrics[symbol] = {
                "RSI": rsi.iloc[-1],
                "Percent_B": percent_b.iloc[-1],
                "Stoch_K": k.iloc[-1],
                "Stoch_D": d.iloc[-1],
                "BB_Width": (upper_band.iloc[-1] - lower_band.iloc[-1])
                / sma_20.iloc[-1]
                * 100,
            }

        except Exception as exc:
            logger.error(
                "Error calculating mean reversion metrics for %s: %s", symbol, exc
            )

    return mean_reversion_metrics


def analyze_volume_profile(
    stock_data: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    """
    Analyze volume profile to identify strong support/resistance levels

    Args:
        stock_data: Dictionary of stock data

    Returns:
        Dictionary mapping symbols to their volume metrics
    """
    volume_metrics = {}

    for symbol, data in stock_data.items():
        try:
            # Create a simplified volume profile using price bins
            # Use .iloc[0] instead of float() to extract scalars
            price_min = data["Low"].min()
            if isinstance(price_min, pd.Series):
                price_min = price_min.iloc[0]

            price_max = data["High"].max()
            if isinstance(price_max, pd.Series):
                price_max = price_max.iloc[0]

            # Skip stocks with invalid price data
            if (
                np.isnan(price_min)
                or np.isnan(price_max)
                or price_min == price_max
                or price_max <= price_min
            ):
                continue

            # Create bins
            bins = 20
            bin_size = (price_max - price_min) / bins

            # Initialize volume bins
            volume_bins = np.zeros(bins)
            price_bins = np.linspace(price_min, price_max, bins + 1)

            # Iterate through data to accumulate volume in bins
            for idx, row in data.iterrows():
                try:
                    price = row["Close"]
                    # Extract scalar from Series if needed
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]

                    volume = row["Volume"]
                    if isinstance(volume, pd.Series):
                        volume = volume.iloc[0]

                    # Skip NaN values
                    if np.isnan(price) or np.isnan(volume):
                        continue

                    # Find the bin this price belongs to
                    bin_idx = min(int((price - price_min) / bin_size), bins - 1)
                    if 0 <= bin_idx < bins:  # Safety check
                        volume_bins[bin_idx] += volume
                except Exception:
                    # Skip problematic rows
                    continue

            # Find highest volume bin
            max_vol_bin = np.argmax(volume_bins)
            value_area_low = float(price_bins[max_vol_bin])
            value_area_high = float(price_bins[max_vol_bin + 1])

            # Calculate volume ratio (recent vs average)
            recent_vol = data["Volume"].iloc[-5:].mean()
            if isinstance(recent_vol, pd.Series):
                recent_vol = recent_vol.iloc[0]

            avg_vol = data["Volume"].mean()
            if isinstance(avg_vol, pd.Series):
                avg_vol = avg_vol.iloc[0]

            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

            # Store metrics
            volume_metrics[symbol] = {
                "Value_Area_Low": value_area_low,
                "Value_Area_High": value_area_high,
                "Volume_Ratio": vol_ratio,
            }

        except Exception as exc:
            logger.error("Error analyzing volume profile for %s: %s", symbol, exc)

    return volume_metrics


def read_stock_list(csv_file: str = "nse_fo_stocks.csv") -> List[str]:
    """
    Read list of stocks from CSV file or use default if file not found

    Args:
        csv_file: Path to the CSV file containing stock symbols

    Returns:
        List of stock symbols
    """
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # Assuming the CSV has a column named 'Symbol' or 'Ticker'
            symbol_col = "Symbol" if "Symbol" in df.columns else "Ticker"

            # Get the list of symbols and add '.NS' suffix if not present
            symbols = df[symbol_col].tolist()
            symbols = [s if s.endswith(".NS") else f"{s}.NS" for s in symbols]

            logger.info("Loaded %d stocks from %s", len(symbols), csv_file)
            return symbols
        else:
            logger.info("CSV file %s not found, using default stock list", csv_file)
            return NIFTY_FO_STOCKS
    except Exception as exc:
        logger.error("Error reading stock list: %s", exc)
        return NIFTY_FO_STOCKS


def analyze_volatility(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze volatility metrics for each stock

    Args:
        stock_data: Dictionary of stock data

    Returns:
        DataFrame with volatility analysis results
    """
    results = []

    for symbol, data in stock_data.items():
        try:
            # Make sure we're using the right price column
            price_col = "Close"  # Keeping it simple by just using Close

            # Calculate returns
            data["Returns"] = data[price_col].pct_change()

            # Skip if not enough data
            if len(data) < 50:
                logger.info("Skipping %s: insufficient data", symbol)
                continue

            # Calculate metrics
            current_price = data[price_col].iloc[-1]
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]

            # Historical volatility (21-day, 63-day)
            vol_21d = data["Returns"].rolling(window=21).std() * np.sqrt(252) * 100
            vol_63d = data["Returns"].rolling(window=63).std() * np.sqrt(252) * 100

            # Price range metrics - prevent NaN values by checking data length
            if len(data) >= 252:  # We have at least 1 year of data
                high_52w = data[price_col].rolling(window=252).max().iloc[-1]
                low_52w = data[price_col].rolling(window=252).min().iloc[-1]

                # Handle Series objects
                if isinstance(high_52w, pd.Series):
                    high_52w = high_52w.iloc[0]
                if isinstance(low_52w, pd.Series):
                    low_52w = low_52w.iloc[0]

                # Calculate range only if both values are valid and low is not zero
                if not np.isnan(high_52w) and not np.isnan(low_52w) and low_52w > 0:
                    range_52w_pct = (high_52w - low_52w) / low_52w * 100
                else:
                    # Use a fallback if we can't calculate the 52-week range
                    # Use the available data to calculate an approximate range
                    available_high = data[price_col].max()
                    available_low = data[price_col].min()
                    if isinstance(available_high, pd.Series):
                        available_high = available_high.iloc[0]
                    if isinstance(available_low, pd.Series):
                        available_low = available_low.iloc[0]

                    if available_low > 0:
                        range_52w_pct = (
                            (available_high - available_low) / available_low * 100
                        )
                    else:
                        range_52w_pct = (
                            30.0  # Default value based on typical market range
                        )
            else:
                # Not enough data for 52-week calculation, use available data
                available_high = data[price_col].max()
                available_low = data[price_col].min()
                if isinstance(available_high, pd.Series):
                    available_high = available_high.iloc[0]
                if isinstance(available_low, pd.Series):
                    available_low = available_low.iloc[0]

                if available_low > 0:
                    range_52w_pct = (
                        (available_high - available_low) / available_low * 100
                    )
                else:
                    range_52w_pct = 30.0  # Default value

            # Simple ATR
            data["TR"] = np.maximum(
                data["High"] - data["Low"],
                np.maximum(
                    abs(data["High"] - data[price_col].shift(1)),
                    abs(data["Low"] - data[price_col].shift(1)),
                ),
            )
            atr_14d = data["TR"].rolling(window=14).mean().iloc[-1]

            # Process values with proper Series handling
            # Check if values are Series and extract with iloc if needed
            if isinstance(atr_14d, pd.Series):
                atr_14d = atr_14d.iloc[0]

            # Calculate derived values after extraction
            atr_percent = (atr_14d / current_price) * 100
            daily_volume = data["Volume"].mean()
            if isinstance(daily_volume, pd.Series):
                daily_volume = daily_volume.iloc[0]

            # Get final values for volatility
            vol_21d_val = vol_21d.iloc[-1]
            if isinstance(vol_21d_val, pd.Series):
                vol_21d_val = vol_21d_val.iloc[0]

            vol_63d_val = vol_63d.iloc[-1]
            if isinstance(vol_63d_val, pd.Series):
                vol_63d_val = vol_63d_val.iloc[0]

            # Make sure all values are valid
            if (
                np.isnan(vol_21d_val)
                or np.isnan(vol_63d_val)
                or np.isnan(atr_percent)
                or np.isnan(range_52w_pct)
            ):
                logger.warning("Some values are NaN for %s", symbol)
                # Fill NaN values with reasonable defaults
                vol_21d_val = 30.0 if np.isnan(vol_21d_val) else vol_21d_val
                vol_63d_val = 25.0 if np.isnan(vol_63d_val) else vol_63d_val
                atr_percent = 2.0 if np.isnan(atr_percent) else atr_percent
                range_52w_pct = 30.0 if np.isnan(range_52w_pct) else range_52w_pct

            # Store results with properly extracted scalar values
            results.append(
                {
                    "Symbol": symbol.replace(".NS", ""),
                    "Price": current_price,
                    "Volatility_21d": vol_21d_val,
                    "Volatility_63d": vol_63d_val,
                    "ATR_14d": atr_14d,
                    "ATR_Percent": atr_percent,
                    "Range_52w_Pct": range_52w_pct,
                    "Daily_Volume": daily_volume,
                }
            )

        except Exception as exc:
            logger.error("Error analyzing %s: %s", symbol, exc)

    # Create DataFrame
    if not results:
        logger.error("No valid results found.")
        return pd.DataFrame()

    return pd.DataFrame(results)


def identify_iron_condor_candidates(
    analysis_df: pd.DataFrame,
    mean_reversion_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    volume_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    iv_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Identify the best stocks for iron condor strategy with enhanced metrics

    Args:
        analysis_df: DataFrame with volatility analysis results
        mean_reversion_metrics: Dictionary of mean reversion metrics
        volume_metrics: Dictionary of volume metrics

    Returns:
        DataFrame with candidate stocks sorted by score
    """
    if analysis_df.empty:
        logger.error("No data available for analysis.")
        return pd.DataFrame()

    # Create a copy to avoid modifying original
    df = analysis_df.copy()

    # Add columns for metrics if they don't exist
    all_columns = [
        "RSI",
        "Percent_B",
        "Stoch_K",
        "Stoch_D",
        "BB_Width",
        "Value_Area_Low",
        "Value_Area_High",
        "Volume_Ratio",
    ]

    for col in all_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Add IV skew metrics if available
    if iv_metrics:
        logger.info("Adding implied volatility metrics...")
        for symbol, metrics in iv_metrics.items():
            symbol_short = symbol.replace(".NS", "")
            mask = df["Symbol"] == symbol_short
            if mask.any():
                idx = df.index[mask][0]
                for metric, value in metrics.items():
                    if isinstance(value, pd.Series):
                        value = float(value.iloc[0])
                    df.at[idx, metric] = value

    # Add mean reversion metrics if available
    if mean_reversion_metrics:
        logger.info("Adding mean reversion metrics...")
        for symbol, metrics in mean_reversion_metrics.items():
            symbol_short = symbol.replace(".NS", "")
            mask = df["Symbol"] == symbol_short
            if mask.any():
                idx = df.index[mask][0]
                for metric, value in metrics.items():
                    if metric in df.columns:
                        # Make sure to convert pandas Series to float
                        if isinstance(value, pd.Series):
                            value = float(value.iloc[0])
                        df.at[idx, metric] = value

    # Add volume metrics if available
    if volume_metrics:
        logger.info("Adding volume metrics...")
        for symbol, metrics in volume_metrics.items():
            symbol_short = symbol.replace(".NS", "")
            mask = df["Symbol"] == symbol_short
            if mask.any():
                idx = df.index[mask][0]
                for metric, value in metrics.items():
                    if metric in df.columns:
                        # Make sure to convert pandas Series to float
                        if isinstance(value, pd.Series):
                            value = float(value.iloc[0])
                        df.at[idx, metric] = value

    # Debug missing values
    for col in ["Volatility_21d", "ATR_Percent", "Range_52w_Pct"]:
        if df[col].isna().any():
            logger.warning("Found %d NaN values in %s", df[col].isna().sum(), col)
            # Replace NaNs with median value
            df[col] = df[col].fillna(df[col].median())

    # Calculate ranks for volatility metrics (explicitly handle missing values)
    df["Volatility_Rank"] = df["Volatility_21d"].rank(method="min")
    df["ATR_Rank"] = df["ATR_Percent"].rank(method="min")
    df["Range_Rank"] = df["Range_52w_Pct"].rank(method="min")

    # Print debug info after ranking
    logger.debug(
        "Ranked rows with Range_Rank values: %d/%d", df["Range_Rank"].count(), len(df)
    )

    # Add RSI factor (prefer middle range 40-60)
    if "RSI" in df.columns and df["RSI"].notna().any():
        # Fill any NaN values with 50 (neutral)
        df["RSI"] = df["RSI"].fillna(50)
        df["RSI_Score"] = abs(df["RSI"] - 50)
        df["RSI_Rank"] = df["RSI_Score"].rank(method="min")
    else:
        logger.warning("No valid RSI data found, using volatility rank as substitute")
        df["RSI_Rank"] = df["Volatility_Rank"]  # Default to volatility rank

    # Add Bollinger Band width factor (prefer narrower bands)
    if "BB_Width" in df.columns and df["BB_Width"].notna().any():
        df["BB_Width"] = df["BB_Width"].fillna(df["BB_Width"].median())
        df["BB_Width_Rank"] = df["BB_Width"].rank(method="min")
    else:
        logger.warning(
            "No valid BB_Width data found, using volatility rank as substitute"
        )
        df["BB_Width_Rank"] = df["Volatility_Rank"]  # Default to volatility rank

    # Add Volume stability factor (prefer stable volume)
    if "Volume_Ratio" in df.columns and df["Volume_Ratio"].notna().any():
        df["Volume_Ratio"] = df["Volume_Ratio"].fillna(1)
        df["Volume_Stability"] = abs(df["Volume_Ratio"] - 1)
        df["Volume_Rank"] = df["Volume_Stability"].rank(method="min")
    else:
        logger.warning(
            "No valid Volume_Ratio data found, using volatility rank as substitute"
        )
        df["Volume_Rank"] = df["Volatility_Rank"]  # Default to volatility rank

    # Make sure all rank columns have values
    rank_columns = [
        "Volatility_Rank",
        "ATR_Rank",
        "Range_Rank",
        "RSI_Rank",
        "BB_Width_Rank",
        "Volume_Rank",
    ]
    for col in rank_columns:
        if df[col].isna().any():
            missing_count = df[col].isna().sum()
            logger.warning(
                "Found %d NaN values in %s, filling with median rank",
                missing_count,
                col,
            )
            df[col] = df[col].fillna(df[col].median())

    # Include IV skew in the ranking if available
    if "IV_Skew" in df.columns and df["IV_Skew"].notna().any():
        # For iron condors, prefer neutral to slightly negative skew
        df["IV_Skew"] = df["IV_Skew"].fillna(0)
        df["IV_Skew_Score"] = abs(df["IV_Skew"] + 0.05)  # Slight negative skew is ideal
        df["IV_Skew_Rank"] = df["IV_Skew_Score"].rank(method="min")

        # Add IV Skew to rank columns list
        rank_columns.append("IV_Skew_Rank")

        # Update the IC_Score calculation to include IV skew
        df["IC_Score"] = (
            df["Volatility_Rank"] * 0.35
            + df["ATR_Rank"] * 0.15
            + df["Range_Rank"] * 0.1
            + df["RSI_Rank"] * 0.15
            + df["BB_Width_Rank"] * 0.1
            + df["Volume_Rank"] * 0.05
            + df["IV_Skew_Rank"] * 0.1  # Add 10% weight to IV skew
        )
    else:
        logger.warning("No valid IV skew data found, using default scoring")
        # Original score calculation if IV skew data not available
        df["IC_Score"] = (
            df["Volatility_Rank"] * 0.4
            + df["ATR_Rank"] * 0.2
            + df["Range_Rank"] * 0.1
            + df["RSI_Rank"] * 0.15
            + df["BB_Width_Rank"] * 0.1
            + df["Volume_Rank"] * 0.05
        )

    # Debug the score calculation
    logger.debug(
        "IC_Score calculated, NaN values: %d/%d", df["IC_Score"].isna().sum(), len(df)
    )
    logger.debug("IC_Score range: %f to %f", df["IC_Score"].min(), df["IC_Score"].max())

    # Make sure IC_Score is computed and not NaN
    if df["IC_Score"].isna().any():
        logger.warning(
            "Some IC_Score values are NaN, filling with max score (worst case)"
        )
        df["IC_Score"] = df["IC_Score"].fillna(df["IC_Score"].max())

    # Sort by score (lower is better)
    result_df = df.sort_values("IC_Score")

    # Final verification
    logger.debug("Final dataframe has %d rows", len(result_df))
    logger.debug(
        "IC_Score in output: %d/%d", result_df["IC_Score"].count(), len(result_df)
    )

    return result_df


def plot_top_candidates(
    stock_data: Dict[str, pd.DataFrame], symbols: List[str], days: int = 90
) -> None:
    """
    Plot normalized price action for top candidates

    Args:
        stock_data: Dictionary of stock data
        symbols: List of stock symbols to plot
        days: Number of days to plot
    """
    plt.figure(figsize=(14, 7))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for symbol in symbols:
        if symbol + ".NS" in stock_data:
            data = stock_data[symbol + ".NS"]

            # Get data for the specified period
            mask = (data.index >= start_date) & (data.index <= end_date)
            plot_data = data.loc[mask].copy()

            if not plot_data.empty:
                # Normalize to starting price
                norm_price = plot_data["Close"] / plot_data["Close"].iloc[0]
                plt.plot(plot_data.index, norm_price, label=symbol)

    plt.title(f"Normalized Price Movement (Last {days} days)")
    plt.ylabel("Price (Normalized to 1.0)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Add a horizontal line at y=1
    plt.axhline(y=1, color="grey", linestyle="--", alpha=0.5)

    plt.show()


def save_results_to_csv(
    dataframe: pd.DataFrame, filename: str = "iron_condor_candidates.csv"
) -> Optional[str]:
    """
    Save analysis results to a CSV file

    Args:
        dataframe: DataFrame to save
        filename: Name of the output file

    Returns:
        Path to the saved file or None if there was an error
    """
    try:
        # Create 'results' directory if it doesn't exist
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(results_dir, f"{timestamp}_{filename}")

        # Save to CSV
        dataframe.to_csv(filepath, index=False)
        logger.info("✅ Results saved to %s", filepath)
        return filepath
    except Exception as exc:
        logger.error("❌ Error saving results to CSV: %s", exc)
        return None


def check_upcoming_events(
    symbols: List[str], days_ahead: int = 30
) -> Dict[str, Dict[str, Any]]:
    """
    Check for upcoming earnings or other significant events

    Args:
        symbols: List of stock symbols
        days_ahead: Number of days to look ahead for events

    Returns:
        Dictionary mapping symbols to their upcoming events
    """
    events = {}
    today = datetime.now().date()

    for symbol in symbols:
        try:
            # Get calendar data
            stock = yf.Ticker(symbol)
            calendar = stock.calendar

            if calendar is not None and hasattr(calendar, "loc"):
                # Check for earnings date
                if "Earnings Date" in calendar.index:
                    earnings_date = calendar.loc["Earnings Date", 0].date()
                    days_to_earnings = (earnings_date - today).days

                    if 0 <= days_to_earnings <= days_ahead:
                        events[symbol] = {
                            "Event": "Earnings",
                            "Date": earnings_date,
                            "Days_Away": days_to_earnings,
                        }

        except Exception as exc:
            logger.error("Error checking events for %s: %s", symbol, exc)

    return events


def calculate_strikes(
    candidates_df: pd.DataFrame,
    days_to_expiry: int = 30,
    confidence: float = 0.70,
    risk_reward_target: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate recommended strikes for iron condor setup with premium estimates

    Args:
        candidates_df: DataFrame with iron condor candidates
        days_to_expiry: Number of days to expiration
        confidence: Confidence level for strike calculation
        risk_reward_target: Target risk-reward ratio

    Returns:
        DataFrame with added strike price and premium calculations
    """
    if candidates_df.empty:
        return candidates_df

    # Create a deep copy to avoid modifying the original
    df = candidates_df.copy()

    # Debug input columns
    logger.debug("calculate_strikes received %d rows", len(df))

    # Define round_strike function outside the loop
    def round_strike(price_val: float) -> float:
        """Round strike price to appropriate level"""
        if isinstance(price_val, (pd.Series, pd.DataFrame)):
            price_val = price_val.iloc[0]

        if price_val > 1000:
            return round(price_val / 100) * 100
        else:
            return round(price_val / 5) * 5

    # Calculate strike distances based on volatility
    for idx, row in df.iterrows():
        current_price = row["Price"]
        volatility_21d = row["Volatility_21d"]

        # Monthly volatility (convert annual to monthly)
        monthly_vol = volatility_21d / 100 * np.sqrt(days_to_expiry / 252)

        # Try different confidence levels to find optimal risk-reward
        found_optimal = False
        test_confidences = [0.70, 0.65, 0.60, 0.55, 0.50]

        for test_conf in test_confidences:
            # Z-score corresponding to confidence level
            z_score = norm.ppf(test_conf)
            expected_move = current_price * monthly_vol * abs(z_score)

            # Calculate strikes
            call_short = round_strike(current_price + expected_move)
            put_short = round_strike(current_price - expected_move)

            # Tighter spreads for better premium collection
            wing_width = max(5, round(0.2 * expected_move))
            call_long = round_strike(call_short + wing_width)
            put_long = round_strike(put_short - wing_width)

            # Estimate option premiums using Black-Scholes approximation
            call_short_premium = bs_option_price(
                current_price,
                call_short,
                days_to_expiry / 365,
                volatility_21d / 100,
                0.06,
                "call",
            )
            call_long_premium = bs_option_price(
                current_price,
                call_long,
                days_to_expiry / 365,
                volatility_21d / 100,
                0.06,
                "call",
            )
            put_short_premium = bs_option_price(
                current_price,
                put_short,
                days_to_expiry / 365,
                volatility_21d / 100,
                0.06,
                "put",
            )
            put_long_premium = bs_option_price(
                current_price,
                put_long,
                days_to_expiry / 365,
                volatility_21d / 100,
                0.06,
                "put",
            )

            # Total premium collected
            net_premium = (
                call_short_premium
                - call_long_premium
                + put_short_premium
                - put_long_premium
            )

            # Maximum risk (width of either spread minus net premium)
            max_risk = max(call_long - call_short, put_short - put_long) - net_premium

            # Risk-reward ratio
            risk_reward = net_premium / max_risk if max_risk > 0 else 0

            # If risk-reward is favorable, use these strikes
            if risk_reward >= risk_reward_target:
                found_optimal = True
                break

        # If we couldn't find optimal strikes, use the last calculated ones
        df.at[idx, "Call_Short"] = call_short
        df.at[idx, "Call_Long"] = call_long
        df.at[idx, "Put_Short"] = put_short
        df.at[idx, "Put_Long"] = put_long
        df.at[idx, "Premium_Collected"] = round(net_premium, 2)
        df.at[idx, "Max_Risk"] = round(max_risk, 2)
        df.at[idx, "Risk_Reward"] = round(risk_reward, 2)
        df.at[idx, "Confidence"] = (
            test_conf if found_optimal else test_confidences[-1]
        ) * 100
        df.at[idx, "Expected_Move_Pct"] = monthly_vol * abs(z_score) * 100
        df.at[idx, "Expected_Range"] = f"{put_short} - {call_short}"

    return df


def calculate_probability_of_profit(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate more precise probability of profit for iron condor

    Args:
        candidates_df: DataFrame with iron condor candidates

    Returns:
        DataFrame with added probability of profit calculations
    """
    if candidates_df.empty:
        return candidates_df

    df = candidates_df.copy()

    for idx, row in df.iterrows():
        current_price = row["Price"]
        volatility = row["Volatility_21d"] / 100  # Convert to decimal
        call_short = row["Call_Short"]
        put_short = row["Put_Short"]
        days = 30  # Typical days to expiry

        # Standard deviation for the period
        std_dev = current_price * volatility * np.sqrt(days / 252)

        # Probability using normal distribution
        prob_below_call = 1 - norm.cdf((call_short - current_price) / std_dev)
        prob_above_put = 1 - norm.cdf((current_price - put_short) / std_dev)

        # Probability of profit (both strikes holding)
        pop = (1 - prob_below_call - prob_above_put) * 100

        # Store the probability
        df.at[idx, "Prob_of_Profit"] = pop

    return df


def bs_option_price(
    s: float, k: float, t: float, v: float, r: float, option_type: str
) -> float:
    """
    Calculate Black-Scholes option price

    Args:
        s: Stock price
        k: Strike price
        t: Time to expiration in years
        v: Volatility (decimal)
        r: Risk-free rate (decimal)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    d1 = (np.log(s / k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)

    if option_type.lower() == "call":
        price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    else:
        price = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    return max(0.05, price)  # Minimum price of 0.05 to be realistic


def parse_expiry_date(date_str):
    """
    Parse date string in either DD-BB-YYYY or DD-MM-YYYY format

    Args:
        date_str: Date string in DD-MM-YYYY or DD-BB-YYYY format

    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None

    formats = ["%d-%m-%Y", "%d-%b-%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    logger.error(
        f"Could not parse date: {date_str}. Use format DD-MM-YYYY or DD-MM-YYYY"
    )
    return None


## Add argparse for command line arguments


parser = argparse.ArgumentParser(description="NSE Iron Condor Scanner")
parser.add_argument(
    "--days_to_expiry", type=int, default=30, help="Days to expiry for options"
)
parser.add_argument(
    "--expiry_date", type=str, help="Expiry date in format DD-MM-YYYY or DD-MM-YYYY"
)

if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()

    # Calculate days to expiry if expiry_date is specified
    if args.expiry_date:
        expiry_date = parse_expiry_date(args.expiry_date)
        if expiry_date:
            today = datetime.now()
            days_to_expiry = (expiry_date - today).days
            if days_to_expiry <= 0:
                logger.error("Expiry date must be in the future")
                days_to_expiry = args.days_to_expiry
            else:
                logger.info(
                    f"Using expiry date: {expiry_date.strftime('%d-%b-%Y')} ({days_to_expiry} days from now)"
                )
        else:
            days_to_expiry = args.days_to_expiry
    else:
        days_to_expiry = args.days_to_expiry

    # Run main function with calculated days_to_expiry
    main(days_to_expiry)
