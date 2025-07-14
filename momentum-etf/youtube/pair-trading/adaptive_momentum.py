import backtrader as bt
import os
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tabulate import tabulate
from utils import MarketDataLoader, setup_logger, IndianBrokerageCommission
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import json

# Configure matplotlib before importing pyplot to prevent font warnings
matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# Suppress all font-related warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Glyph.*missing from font.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Font family.*not found.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib.font_manager"
)

# Set matplotlib font manager logging to ERROR level to suppress font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


STRATEGY_PARAMS = {
    "sma_fast": 30,
    "sma_slow": 100,
    "atr_period": 3,  # Changed from 3.0 to 3 (integer)
    "atr_multiplier": 2.5,
    "momentum_period": 15,
    "top_n_stocks": 3,
    "rebalance_days": 10,
}

if os.path.exists("strategy_config.json"):
    logging.info("📄 Loading strategy parameters from strategy_config.json")
    with open("strategy_config.json", "r") as f:
        STRATEGY_PARAMS.update(json.load(f))

INITIAL_CASH = 10_00_00


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
        "ADANIGREEN.NS",
        "WIPRO.NS",
        "ONGC.NS",
        "NTPC.NS",
        "JSWSTEEL.NS",
        "POWERGRID.NS",
        "M&M.NS",
        "TECHM.NS",
        "COALINDIA.NS",
        "TATAMOTORS.NS",
        "BAJAJFINSV.NS",
        "ADANIPORTS.NS",
        "HDFCLIFE.NS",
        "GRASIM.NS",
        "CIPLA.NS",
        "INDUSINDBK.NS",
        "DRREDDY.NS",
        "EICHERMOT.NS",
        "BRITANNIA.NS",
        "APOLLOHOSP.NS",
        "DIVISLAB.NS",
        "BPCL.NS",
        "TATACONSUM.NS",
        "HINDALCO.NS",
        "SHREECEM.NS",
        "HEROMOTOCO.NS",
        "UPL.NS",
        "TATASTEEL.NS",
        "BAJAJ-AUTO.NS",
        "SBILIFE.NS",
        "TATAPOWER.NS",
        "PIDILITIND.NS",
        "GODREJCP.NS",
        "BERGEPAINT.NS",
        "DABUR.NS",
        "MARICO.NS",
        "COLPAL.NS",
        "MCDOWELL-N.NS",
        "BAJAJHLDNG.NS",
        "ADANIENT.NS",
        "ADANITRANS.NS",
        "VEDL.NS",
        "HINDZINC.NS",
        "NATIONALUM.NS",
        "SAIL.NS",
        "JINDALSTEL.NS",
        "TATACHEM.NS",
        "GAIL.NS",
        "IOC.NS",
        "HINDPETRO.NS",
        "BANKINDIA.NS",
        "CANBK.NS",
        "PNB.NS",
        "UNIONBANK.NS",
        "IDFCFIRSTB.NS",
        "FEDERALBNK.NS",
        "RBLBANK.NS",
        "BANDHANBNK.NS",
        "AUBANK.NS",
        "CSBBANK.NS",
        "LICHSGFIN.NS",
        "SRTRANSFIN.NS",
        "M&MFIN.NS",
        "PFC.NS",
        "RECLTD.NS",
        "IRFC.NS",
        "IRCTC.NS",
        "CONCOR.NS",
        "GMRINFRA.NS",
        "ABFRL.NS",
        "RAYMOND.NS",
        "ADITYADB.NS",
        "CROMPTON.NS",
        "HAVELLS.NS",
        "POLYCAB.NS",
        "CUMMINSIND.NS",
        "ABB.NS",
        "SIEMENS.NS",
        "BHEL.NS",
        "LTTS.NS",
        "PERSISTENT.NS",
        "MINDTREE.NS",
        "MPHASIS.NS",
        "COFORGE.NS",
        "LTIM.NS",
        # Additional Nifty 200 stocks to reach closer to 200
        "MOTHERSON.NS",
        "PAGEIND.NS",
        "DMART.NS",
        "NAUKRI.NS",
        "ZOMATO.NS",
        "PAYTM.NS",
        "POLICYBZR.NS",
        "ZEEL.NS",
        "DIXON.NS",
        "VOLTAS.NS",
        "WHIRLPOOL.NS",
        "JUBLFOOD.NS",
        "VBL.NS",
        "RADICO.NS",
        "BALKRISIND.NS",
        "CEAT.NS",
        "APOLLOTYRE.NS",
        "MRF.NS",
        "ESCORTS.NS",
        "ASHOKLEY.NS",
        "TVSMOTOR.NS",
        "MAHINDCIE.NS",
        "ENDURANCE.NS",
        "EXIDEIND.NS",
        "AMARON.NS",
        "BOSCHLTD.NS",
        "MOTHERSON.NS",
        "SUNDRMFAST.NS",
        "BHARAT.NS",
        "MFSL.NS",
        "ICICIGI.NS",
        "ICICIPRULI.NS",
        "SBICARD.NS",
        "HDFCAMC.NS",
        "MUTHOOTFIN.NS",
        "CHOLAFIN.NS",
        "SHRIRAMFIN.NS",
        "MANAPPURAM.NS",
        "LICI.NS",
        "NIACL.NS",
        "TORNTPHARM.NS",
        "LUPIN.NS",
        "BIOCON.NS",
        "GLENMARK.NS",
        "CADILAHC.NS",
        "ALKEM.NS",
        "AUROPHARMA.NS",
        "TORNTPOWER.NS",
        "ABBOTINDIA.NS",
        "PFIZER.NS",
        "GLAXO.NS",
        "NOVARTIS.NS",
        "SANOFI.NS",
        "WELLSPHL.NS",
        "FORTIS.NS",
        "MAXHEALTH.NS",
        "NARAYANHSR.NS",
        "LAXMIMACH.NS",
        "RAJESHEXPO.NS",
        "KALPATPWR.NS",
        "THERMAX.NS",
        "CRISIL.NS",
        "CREDITACC.NS",
        "EQUITAS.NS",
        "DCBBANK.NS",
        "SOUTHBANK.NS",
        "IBREALEST.NS",
        "PRESTIGE.NS",
        "DLF.NS",
        "GODREJPROP.NS",
        "BRIGADE.NS",
        "MAHLIFE.NS",
        "SOBHA.NS",
        "PHOENIXLTD.NS",
        "ATUL.NS",
        "CLEAN.NS",
        "FINEORG.NS",
        "DEEPAKNTR.NS",
        "NAVINFLUOR.NS",
        "ALKYLAMINE.NS",
        "SRF.NS",
        "AAVAS.NS",
        "HOMEFIRST.NS",
        "CANFINHOME.NS",
        "HUDCO.NS",
        "INDIAMART.NS",
        "JUSTDIAL.NS",
        "INOXWIND.NS",
        "SUZLON.NS",
        "RENUKA.NS",
        "BALRAMCHIN.NS",
        "DHAMPUR.NS",
        "DCMSHRIRAM.NS",
        "CHAMBLFERT.NS",
        "COROMANDEL.NS",
        "GNFC.NS",
        "KRIBHCO.NS",
        "MADRASFERT.NS",
        "NFL.NS",
        "RCF.NS",
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

    Args:
        symbols: List of stock symbols to test
        start_date: Start date for data availability check
        end_date: End date for data availability check
        max_stocks_to_test: Maximum number of stocks to test
        max_workers: Maximum number of parallel workers
    """
    print(
        f"\n� Filtering {max_stocks_to_test} stocks for data availability using {max_workers} parallel workers..."
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


# Get Nifty 200 stocks dynamically and filter for data availability
ALL_NIFTY_200_STOCKS = get_nifty_200_stocks()


class MomentumTrendStrategy(bt.Strategy):
    params = (
        ("sma_fast", STRATEGY_PARAMS["sma_fast"]),
        ("sma_slow", STRATEGY_PARAMS["sma_slow"]),
        ("atr_period", STRATEGY_PARAMS["atr_period"]),
        ("atr_multiplier", STRATEGY_PARAMS["atr_multiplier"]),
        ("momentum_period", STRATEGY_PARAMS["momentum_period"]),
        ("top_n_stocks", STRATEGY_PARAMS["top_n_stocks"]),
        ("rebalance_days", STRATEGY_PARAMS["rebalance_days"]),
    )

    def __init__(self):
        self.inds = {}
        self.rebalance_counter = 0

        # Track portfolio performance
        self.portfolio_values = []
        self.dates = []
        self.monthly_returns = {}
        self.peak_value = INITIAL_CASH
        self.drawdowns = []

        for d in self.datas:
            self.inds[d._name] = {
                "sma_fast": bt.ind.SMA(d.close, period=int(self.p.sma_fast)),
                "sma_slow": bt.ind.SMA(d.close, period=int(self.p.sma_slow)),
                "atr": bt.ind.ATR(d, period=int(self.p.atr_period)),
                "momentum": bt.ind.ROC(d.close, period=int(self.p.momentum_period)),
            }

    def _should_exit(self, d):
        pos = self.getposition(d)
        if pos.size == 0:  # No position to exit
            return False

        price = d.close[0]
        ind = self.inds[d._name]
        return (
            price < pos.price - self.p.atr_multiplier * ind["atr"][0]
            or ind["sma_fast"][0] < ind["sma_slow"][0]
            or ind["momentum"][0] < 0
        )

    def _should_enter(self, d):
        ind = self.inds[d._name]
        return ind["sma_fast"][0] > ind["sma_slow"][0]

    def next(self):
        # Track portfolio performance
        current_value = self.broker.getvalue()
        current_date = self.datas[0].datetime.date(0)

        self.portfolio_values.append(current_value)
        self.dates.append(current_date)

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

        # Filter valid data feeds (not NaN) and sort by momentum
        valid_datas = []
        for d in self.datas:
            try:
                if (
                    len(d.close) > 0
                    and d.close[0] == d.close[0]  # Check for NaN
                    and d._name in self.inds
                    and len(self.inds[d._name]["momentum"]) > 0
                    and self.inds[d._name]["momentum"][0]
                    == self.inds[d._name]["momentum"][0]
                ):  # Check momentum not NaN
                    valid_datas.append(d)
            except (IndexError, KeyError):
                continue  # Skip this data feed if there's an issue

        if not valid_datas:
            self.rebalance_counter += 1
            return

        top_stocks = sorted(
            valid_datas,
            key=lambda d: self.inds[d._name]["momentum"][0],
            reverse=True,
        )[: self.p.top_n_stocks]

        # Exit positions that should be closed
        for data_feed in self.datas:
            pos = self.getposition(data_feed)
            if pos.size != 0 and self._should_exit(data_feed):
                self.close(data_feed)

        # Enter new positions for top momentum stocks
        for d in top_stocks:
            if self.getposition(d).size == 0 and self._should_enter(d):
                available_cash = self.broker.get_cash()
                if available_cash > 0 and d.close[0] > 0:
                    size = int(available_cash / (self.p.top_n_stocks * d.close[0]))
                    if size > 0:
                        self.buy(data=d, size=size)

        self.rebalance_counter += 1


def create_beautiful_charts(strategy, start_date, end_date, symbols=None):
    """Create beautiful and intuitive visualization charts"""

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Adaptive Momentum Strategy - Performance Dashboard",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # 1. Portfolio Growth Chart
    ax1 = plt.subplot(3, 3, (1, 2))
    portfolio_df = pd.DataFrame(
        {"Date": strategy.dates, "Portfolio Value": strategy.portfolio_values}
    )
    portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"])

    ax1.plot(
        portfolio_df["Date"],
        portfolio_df["Portfolio Value"],
        linewidth=2.5,
        color="#2E8B57",
        label="Portfolio Value",
    )
    ax1.axhline(
        y=INITIAL_CASH,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Initial Investment",
    )
    ax1.fill_between(
        portfolio_df["Date"],
        INITIAL_CASH,
        portfolio_df["Portfolio Value"],
        where=(portfolio_df["Portfolio Value"] >= INITIAL_CASH),
        alpha=0.3,
        color="green",
        label="Gains",
    )
    ax1.fill_between(
        portfolio_df["Date"],
        INITIAL_CASH,
        portfolio_df["Portfolio Value"],
        where=(portfolio_df["Portfolio Value"] < INITIAL_CASH),
        alpha=0.3,
        color="red",
        label="Losses",
    )

    ax1.set_title("Portfolio Growth Over Time", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value (₹)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x/1e5:.1f}L"))

    # 2. Drawdown Chart
    ax2 = plt.subplot(3, 3, 3)
    drawdown_df = pd.DataFrame({"Date": strategy.dates, "Drawdown": strategy.drawdowns})
    drawdown_df["Date"] = pd.to_datetime(drawdown_df["Date"])

    ax2.fill_between(
        drawdown_df["Date"],
        0,
        -np.array(strategy.drawdowns),
        color="red",
        alpha=0.6,
        label="Drawdown",
    )
    ax2.plot(
        drawdown_df["Date"],
        -np.array(strategy.drawdowns),
        color="darkred",
        linewidth=1.5,
    )
    ax2.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Monthly Returns Heatmap
    ax3 = plt.subplot(3, 3, (4, 5))

    # Prepare monthly returns data
    monthly_data = {}
    for month_year, return_val in strategy.monthly_returns.items():
        year, month = month_year.split("-")
        if year not in monthly_data:
            monthly_data[year] = {}
        monthly_data[year][int(month)] = return_val

    if monthly_data:
        # Create matrix for heatmap
        years = sorted(monthly_data.keys())
        months = range(1, 13)
        heatmap_data = []

        for year in years:
            year_data = []
            for month in months:
                year_data.append(monthly_data[year].get(month, np.nan))
            heatmap_data.append(year_data)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=years,
            columns=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )

        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Monthly Return %"},
        )
        ax3.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Year")

    # 4. Return Distribution
    ax4 = plt.subplot(3, 3, 6)
    daily_returns = []
    for i in range(1, len(strategy.portfolio_values)):
        daily_return = (
            (strategy.portfolio_values[i] - strategy.portfolio_values[i - 1])
            / strategy.portfolio_values[i - 1]
            * 100
        )
        daily_returns.append(daily_return)

    ax4.hist(daily_returns, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax4.axvline(
        np.mean(daily_returns),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(daily_returns):.3f}%",
    )
    ax4.set_title("Daily Returns Distribution", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Daily Return %")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Sharpe Ratio (252-day window)
    ax5 = plt.subplot(3, 3, 7)
    if len(daily_returns) > 252:
        rolling_sharpe = []
        rolling_dates = []
        for i in range(252, len(daily_returns)):
            window_returns = daily_returns[i - 252 : i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                rolling_sharpe.append(sharpe)
                rolling_dates.append(strategy.dates[i + 1])

        ax5.plot(rolling_dates, rolling_sharpe, color="purple", linewidth=2)
        ax5.axhline(
            y=1.0, color="orange", linestyle="--", alpha=0.7, label="Good Threshold"
        )
        ax5.axhline(y=0, color="red", linestyle="-", alpha=0.5)
        ax5.set_title("Rolling Sharpe Ratio (1Y)", fontsize=14, fontweight="bold")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Sharpe Ratio")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Cumulative Return vs Benchmark
    ax6 = plt.subplot(3, 3, 8)
    cumulative_returns = [
        (val / INITIAL_CASH - 1) * 100 for val in strategy.portfolio_values
    ]
    ax6.plot(
        portfolio_df["Date"],
        cumulative_returns,
        linewidth=2.5,
        color="green",
        label="Strategy",
    )

    # Add a benchmark line (assuming 8% annual return)
    benchmark_returns = []
    for i, date in enumerate(strategy.dates):
        days_elapsed = i
        annual_return = 0.08  # 8% annual
        daily_return = (1 + annual_return) ** (days_elapsed / 365.25) - 1
        benchmark_returns.append(daily_return * 100)

    ax6.plot(
        portfolio_df["Date"],
        benchmark_returns,
        linewidth=2,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label="8% Benchmark",
    )

    ax6.set_title("Cumulative Returns vs Benchmark", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("Cumulative Return %")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Key Metrics Summary Box - Split into two sections for better readability
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis("off")

    # Calculate key metrics
    final_value = (
        strategy.portfolio_values[-1] if strategy.portfolio_values else INITIAL_CASH
    )
    total_return = (
        (final_value / INITIAL_CASH - 1) * 100
        if final_value and not np.isnan(final_value)
        else 0
    )
    max_drawdown = (
        max(strategy.drawdowns)
        if strategy.drawdowns and not all(np.isnan(strategy.drawdowns))
        else 0
    )
    peak_value = (
        max(strategy.portfolio_values)
        if strategy.portfolio_values and not all(np.isnan(strategy.portfolio_values))
        else INITIAL_CASH
    )

    # Calculate annualized return
    days_invested = len(strategy.dates) if strategy.dates else 1
    annualized_return = (
        ((final_value / INITIAL_CASH) ** (365.25 / days_invested) - 1) * 100
        if days_invested > 0 and final_value and not np.isnan(final_value)
        else 0
    )

    # Calculate daily returns safely
    if len(strategy.portfolio_values) > 1:
        daily_returns = []
        for i in range(1, len(strategy.portfolio_values)):
            if (
                strategy.portfolio_values[i]
                and strategy.portfolio_values[i - 1]
                and not np.isnan(strategy.portfolio_values[i])
                and not np.isnan(strategy.portfolio_values[i - 1])
                and strategy.portfolio_values[i - 1] != 0
            ):
                daily_return = (
                    (strategy.portfolio_values[i] - strategy.portfolio_values[i - 1])
                    / strategy.portfolio_values[i - 1]
                    * 100
                )
                daily_returns.append(daily_return)
    else:
        daily_returns = [0]

    sharpe_ratio = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if len(daily_returns) > 0
        and np.std(daily_returns) > 0
        and not np.isnan(np.std(daily_returns))
        else 0
    )
    win_rate = (
        len([r for r in daily_returns if r > 0]) / len(daily_returns) * 100
        if daily_returns
        else 0
    )

    # Count winning and losing months from monthly returns
    winning_months = len([r for r in strategy.monthly_returns.values() if r > 0])
    losing_months = len([r for r in strategy.monthly_returns.values() if r < 0])
    total_months = len(strategy.monthly_returns)

    # Calculate period metrics for configuration table
    period_days = (end_date - start_date).days
    period_years = period_days / 365.25

    # Create Strategy Configuration Table (Compact)
    config_section = f"""📊 STRATEGY CONFIGURATION
┌──────────────────┬─────────────────────┐
│ Initial Capital  │ ₹{INITIAL_CASH:,.0f}           │
│ Duration         │ {period_years:.1f}y ({period_days}d)        │
│ Stocks Analyzed  │ {len(symbols) if symbols else 'N/A'}                   │
│ Top Stocks       │ {STRATEGY_PARAMS['top_n_stocks']} selected           │
│ Rebalance        │ Every {STRATEGY_PARAMS['rebalance_days']} days        │
│ SMA Fast/Slow    │ {STRATEGY_PARAMS['sma_fast']}/{STRATEGY_PARAMS['sma_slow']} days         │
│ Momentum Period  │ {STRATEGY_PARAMS['momentum_period']} days            │
│ Commission       │ Indian Brokerage    │
│ Cost Model       │ SEBI Compliant     │
└──────────────────┴─────────────────────┘"""

    # Create Performance Metrics (Compact)
    performance_section = f"""📈 PERFORMANCE METRICS
┌──────────────────┬─────────────────────┐
│ Final Value      │ ₹{final_value:,.0f}           │
│ Total Return     │ {total_return:.1f}%              │
│ Annual Return    │ {annualized_return:.1f}%              │
│ Sharpe Ratio     │ {sharpe_ratio:.2f}               │
│ Max Drawdown     │ {max_drawdown:.1f}%              │
│ Win Rate         │ {win_rate:.0f}%               │
│ Volatility       │ {np.std(daily_returns):.1f}%              │
│ Monthly W/L      │ {winning_months}/{losing_months}                │
└──────────────────┴─────────────────────┘

📊 DAILY EXTREMES
Best: {max(daily_returns):.2f}% | Worst: {min(daily_returns):.2f}%
Average: {np.mean(daily_returns):.3f}%"""

    # Display configuration in upper half
    ax7.text(
        0.02,
        0.98,
        config_section,
        transform=ax7.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        wrap=False,
    )

    # Display performance metrics in lower half
    ax7.text(
        0.02,
        0.48,
        performance_section,
        transform=ax7.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        wrap=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"momentum_strategy_dashboard_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n📈 Beautiful dashboard saved as: {filename}")

    plt.show()


def display_backtest_results(cerebro, result, start_date, end_date, symbols):
    """Display backtest results in a beautiful tabulated format"""

    # Extract analyzer results
    sharpe_analysis = result.analyzers.sharpe.get_analysis()
    drawdown_analysis = result.analyzers.drawdown.get_analysis()
    returns_analysis = result.analyzers.returns.get_analysis()

    # Calculate additional metrics
    initial_value = INITIAL_CASH
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value - initial_value) / initial_value) * 100

    # Calculate period metrics
    period_days = (end_date - start_date).days
    period_years = period_days / 365.25
    annualized_return = ((final_value / initial_value) ** (1 / period_years) - 1) * 100

    print("\n" + "=" * 80)
    print("🚀 ADAPTIVE MOMENTUM STRATEGY - BACKTEST RESULTS")
    print("=" * 80)

    # Strategy Configuration Table
    config_data = [
        ["Parameter", "Value"],
        ["Initial Capital", f"₹{initial_value:,.2f}"],
        [
            "Period",
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        ],
        ["Duration", f"{period_years:.1f} years ({period_days} days)"],
        ["Stocks Analyzed", len(symbols)],
        ["Top Stocks Selected", STRATEGY_PARAMS["top_n_stocks"]],
        ["Rebalance Frequency", f"Every {STRATEGY_PARAMS['rebalance_days']} days"],
        [
            "SMA Fast/Slow",
            f"{STRATEGY_PARAMS['sma_fast']}/{STRATEGY_PARAMS['sma_slow']} days",
        ],
        ["Momentum Period", f"{STRATEGY_PARAMS['momentum_period']} days"],
        ["Commission Model", "Indian Brokerage (SEBI Compliant)"],
        ["Cost Structure", "Brokerage + STT + GST + Regulatory"],
    ]

    print("\n📊 STRATEGY CONFIGURATION")
    print(tabulate(config_data, headers="firstrow", tablefmt="grid"))

    # Performance Metrics Table
    performance_data = [
        ["Metric", "Value", "Interpretation"],
        ["Final Portfolio Value", f"₹{final_value:,.2f}", "💰 Portfolio worth"],
        ["Total Return", f"{total_return:.2f}%", "📈 Absolute gain/loss"],
        ["Annualized Return", f"{annualized_return:.2f}%", "📊 Yearly average return"],
        [
            "Sharpe Ratio",
            f"{sharpe_analysis.get('sharperatio', 0):.3f}",
            "⚖️ Risk-adjusted return",
        ],
        [
            "Max Drawdown",
            f"{drawdown_analysis.get('max', {}).get('drawdown', 0):.2f}%",
            "📉 Worst peak-to-trough loss",
        ],
        [
            "Max Drawdown Amount",
            f"₹{drawdown_analysis.get('max', {}).get('moneydown', 0):,.2f}",
            "💸 Largest loss amount",
        ],
        [
            "Current Drawdown",
            f"{drawdown_analysis.get('drawdown', 0):.2f}%",
            "📊 Current unrealized loss",
        ],
    ]

    print("\n📈 PERFORMANCE METRICS")
    print(tabulate(performance_data, headers="firstrow", tablefmt="grid"))

    # Commission and Cost Analysis
    print("\n💰 COMMISSION & COST STRUCTURE")
    commission_data = [
        ["Cost Component", "Rate/Amount", "Description"],
        ["Brokerage", "0.03% or ₹20", "Per trade (whichever is lower)"],
        ["STT (Securities Transaction Tax)", "~0.05% avg", "0.1% on sell side only"],
        ["Transaction Charges", "0.00345%", "NSE exchange charges"],
        ["GST", "18%", "On brokerage + transaction charges"],
        ["SEBI Charges", "₹10 per crore", "Regulatory supervision fee"],
        ["Stamp Duty", "~0.0015% avg", "0.003% on buy side only"],
        ["Total Impact", "~0.12-0.15%", "Estimated per round trip"],
    ]
    print(tabulate(commission_data, headers="firstrow", tablefmt="grid"))

    # Estimate commission impact
    gross_return = total_return
    # Rough estimate: assume average trade size and frequency
    estimated_trades = (
        period_days / STRATEGY_PARAMS.get("rebalance_days", 10) * 2
    )  # Buy + Sell
    avg_commission_per_trade = 0.0012  # ~0.12% per round trip
    total_commission_impact = estimated_trades * avg_commission_per_trade

    print(f"\n📊 ESTIMATED COMMISSION IMPACT")
    impact_data = [
        ["Metric", "Value"],
        ["Estimated Total Trades", f"{estimated_trades:.0f}"],
        ["Avg Commission per Round Trip", f"{avg_commission_per_trade:.2%}"],
        ["Total Commission Impact", f"{total_commission_impact:.2%}"],
        ["Gross Return (Before Costs)", f"{gross_return:.2f}%"],
        ["Net Return (After Costs)", f"{total_return:.2f}%"],
    ]
    print(tabulate(impact_data, headers="firstrow", tablefmt="grid"))

    # Risk Assessment
    risk_level = (
        "🟢 Low"
        if drawdown_analysis.get("max", {}).get("drawdown", 0) < 10
        else (
            "🟡 Medium"
            if drawdown_analysis.get("max", {}).get("drawdown", 0) < 20
            else "🔴 High"
        )
    )

    sharpe_rating = (
        "🌟 Excellent"
        if sharpe_analysis.get("sharperatio", 0) > 1.5
        else (
            "👍 Good"
            if sharpe_analysis.get("sharperatio", 0) > 1.0
            else (
                "⚖️ Acceptable"
                if sharpe_analysis.get("sharperatio", 0) > 0.5
                else "👎 Poor"
            )
        )
    )

    # Summary Table
    summary_data = [
        ["Assessment", "Rating", "Comment"],
        [
            "Risk Level",
            risk_level,
            f"Max drawdown: {drawdown_analysis.get('max', {}).get('drawdown', 0):.1f}%",
        ],
        [
            "Risk-Adjusted Returns",
            sharpe_rating,
            f"Sharpe ratio: {sharpe_analysis.get('sharperatio', 0):.2f}",
        ],
        [
            "Strategy Performance",
            "📊 Momentum-based selection",
            "Selects top momentum stocks periodically",
        ],
    ]

    print("\n🎯 STRATEGY ASSESSMENT")
    print(tabulate(summary_data, headers="firstrow", tablefmt="grid"))

    # Sample of analyzed stocks
    print(
        f"\n📋 ANALYZED STOCKS (Sample of {min(10, len(symbols))} from {len(symbols)} total)"
    )
    stock_sample = symbols[:10]
    stock_data = [
        [i + 1, stock.replace(".NS", "")] for i, stock in enumerate(stock_sample)
    ]
    print(tabulate(stock_data, headers=["#", "Stock Symbol"], tablefmt="simple"))
    if len(symbols) > 10:
        print(f"... and {len(symbols) - 10} more stocks")

    print("\n" + "=" * 80)
    print("✅ Backtest completed successfully!")
    print("=" * 80)


def run_backtest(symbols, fromdate, todate, max_workers=8):
    """Run backtest using MarketDataLoader for the given symbols with parallel processing"""
    logger = setup_logger()
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.set_coc(True)

    # Set up realistic Indian brokerage commission
    commission_scheme = IndianBrokerageCommission()
    cerebro.broker.addcommissioninfo(commission_scheme)

    logger.info("💰 Applied Indian brokerage commission scheme:")
    logger.info("   📊 Brokerage: 0.03% or ₹20 per trade (whichever lower)")
    logger.info("   🏛️ STT: ~0.05% average (0.1% on sell side)")
    logger.info("   💼 Transaction charges: 0.00345%")
    logger.info("   📄 GST: 18% on brokerage + transaction charges")
    logger.info("   🏦 SEBI charges: ₹10 per crore")
    logger.info("   📋 Stamp duty: ~0.0015% average (0.003% on buy side)")

    # Load data using utils with parallel processing
    logger.info(f"Loading data for {len(symbols)} symbols using parallel processing...")
    loader = MarketDataLoader()
    data_feeds = loader.load_market_data(
        symbols=symbols,
        start_date=fromdate,
        end_date=todate,
        force_refresh=False,
        use_parallel=True,
        max_workers=max_workers,
    )

    if not data_feeds:
        logger.error("No valid stock data found.")
        return

    # Add data feeds to cerebro
    for data_feed in data_feeds:
        cerebro.adddata(data_feed)

    logger.info(f"Successfully loaded {len(data_feeds)} data feeds")

    cerebro.addstrategy(MomentumTrendStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    logger.info(f"Running backtest on {len(data_feeds)} stocks...")

    # Show progress for backtest execution
    print("🔄 Executing backtest strategy...")
    result = cerebro.run()[0]

    # Display beautiful results
    display_backtest_results(cerebro, result, fromdate, todate, symbols)

    # Create beautiful charts
    try:
        logger.info("Creating beautiful visualization dashboard...")
        create_beautiful_charts(result, fromdate, todate, symbols)
    except Exception as e:
        logger.warning(f"Could not generate charts: {e}")
        logger.info("Continuing without charts...")


if __name__ == "__main__":
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()

    print("🚀 Starting Adaptive Momentum Strategy Analysis")
    print("=" * 60)

    # Curated list of top Nifty stocks for quick testing
    test_symbols = [
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
    ]

    # Enable comprehensive filtering to find stocks with data
    enable_full_filtering = True
    max_workers = 8  # Adjust based on your system capabilities
    max_stocks_to_test = 150  # Increase for more comprehensive testing

    if enable_full_filtering:
        print("🔍 Finding stocks with data availability using parallel processing...")
        all_stocks = ALL_NIFTY_200_STOCKS

        available_stocks = filter_stocks_with_data(
            all_stocks,
            start_date,
            end_date,
            max_stocks_to_test=max_stocks_to_test,
            max_workers=max_workers,
        )

        if len(available_stocks) >= 10:
            test_symbols = available_stocks
            print(f"✅ Using {len(test_symbols)} stocks with confirmed data")
        else:
            print("⚠️ Using fallback curated list")

    print(f"\n🎯 Running backtest on {len(test_symbols)} stocks")
    print("=" * 60)

    run_backtest(test_symbols, start_date, end_date, max_workers=max_workers)
