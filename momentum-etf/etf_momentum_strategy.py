import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from tabulate import tabulate
import itertools  # Import itertools for permutations

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration class for ETF momentum strategy parameters."""

    # List of Highly Liquid, High-Volume NSE ETF Symbols (Minimal Overlap, Trading >5 Years, No Debt)
    etf_universe: List[str] = field(
        default_factory=lambda: [
            "NIFTYBEES.NS",  # Nifty 50 ETF
            "SETFNN50.NS",  # Nifty Next 50 ETF
            #  "BANKBEES.NS",  # Bank Nifty ETF
            "GOLDBEES.NS",  # Gold ETF,
            "SILVERBEES.NS",  # Silver ETF,
            "CPSEETF.NS",  # CPSE ETF
            "PSUBNKBEES.NS",  # PSU Bank ETF
            "PHARMABEES.NS",  # Pharma ETF
            "ITBEES.NS",  # IT ETF
            "AUTOBEES.NS",  # Auto ETF
            "INFRAIETF.NS",  # Infra ETF,
            "SHARIABEES.NS",  # Shariah ETF
            "DIVOPPBEES.NS",  # Dividend Opportunities ETF
            "CONSUMBEES.NS",  # Consumer Goods - Nifty India Consumption
        ]
    )

    # Portfolio Parameters
    portfolio_size: int = 7
    exit_rank_buffer_multiplier: float = 2.0

    # Rebalancing
    rebalance_frequency: str = "monthly"  # "monthly" or "weekly"
    rebalance_day_of_month: int = 5
    # Threshold-based rebalancing options
    use_threshold_rebalancing: bool = False  # If True, use profit/loss thresholds
    profit_threshold_pct: float = 10.0  # Rebalance if portfolio profit >= 10%
    loss_threshold_pct: float = -5.0  # Rebalance if portfolio loss <= -5%

    # Momentum Calculation (adjusted for real-time use)
    long_term_period_days: int = 180  # ~6 months (reduced from 252)
    short_term_period_days: int = 60  # ~3 months
    momentum_weights: Tuple[float, float] = (0.6, 0.4)  # (long_term, short_term)

    # Filters
    use_retracement_filter: bool = True
    max_retracement_percentage: float = 0.50
    use_moving_average_filter: bool = True
    moving_average_period: int = 50
    use_volume_filter: bool = False  # Disabled by default for real-time use
    min_avg_volume: int = 100000  # Minimum average daily volume for Indian ETFs (1cr)

    # Risk Management
    max_position_size: float = 0.20  # Max 20% in any single ETF
    min_data_points: int = 200  # Minimum historical data required (reduced from 300)

    # Performance (Indian Rupees)
    initial_capital: float = 1000000.0  # 10 Lakh INR
    commission_rate: float = 0.001  # 0.1% commission per trade


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
        """Calculate percentage return over specified period."""
        if len(prices) < period_days:
            return None

        try:
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-period_days]

            if past_price == 0 or np.isnan(past_price) or np.isnan(current_price):
                return None

            return (current_price / past_price) - 1
        except (IndexError, ZeroDivisionError):
            return None

    def calculate_momentum_scores(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for all ETFs using vectorized operations.

        Returns:
            Series with ETF tickers as index and momentum scores as values
        """
        scores = {}
        long_weight, short_weight = self.config.momentum_weights

        for ticker in prices_df.columns:
            prices = prices_df[ticker].dropna()

            if len(prices) < self.config.min_data_points:
                continue

            # Calculate returns
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

        Returns:
            List of ticker symbols that pass all filters
        """
        filtered_tickers = []

        for ticker in prices_df.columns:
            prices = prices_df[ticker].dropna()

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
                    volumes = volume_df[ticker].dropna()
                    if not self._passes_volume_filter(volumes):
                        continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    def _passes_retracement_filter(self, prices: pd.Series) -> bool:
        """Check if ETF passes retracement filter."""
        try:
            lookback_prices = prices.iloc[-self.config.long_term_period_days :]
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


class Portfolio:
    """Manages portfolio holdings and rebalancing logic."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.holdings = {}  # {ticker: shares}
        self.cash = config.initial_capital
        self.total_value = config.initial_capital
        self.transaction_costs = 0.0
        self.trade_log = []

    def rebalance(
        self,
        ranked_etfs: List[Tuple[str, float]],
        current_prices: pd.Series,
        rebalance_date: datetime,
    ) -> None:
        """
        Rebalance portfolio based on new rankings.

        Args:
            ranked_etfs: List of (ticker, momentum_score) tuples, sorted by score
            current_prices: Current prices for all ETFs
            rebalance_date: Date of rebalancing
        """
        # Update portfolio value
        self._update_portfolio_value(current_prices)

        # Determine target portfolio
        target_tickers = [
            ticker for ticker, _ in ranked_etfs[: self.config.portfolio_size]
        ]

        # Calculate exit threshold
        exit_rank_threshold = int(
            self.config.portfolio_size * self.config.exit_rank_buffer_multiplier
        )

        # Determine which holdings to exit
        tickers_to_exit = []
        for ticker in self.holdings.keys():
            current_rank = next(
                (i for i, (t, _) in enumerate(ranked_etfs) if t == ticker), float("inf")
            )
            if current_rank >= exit_rank_threshold:
                tickers_to_exit.append(ticker)

        # Exit positions
        for ticker in tickers_to_exit:
            # Ensure the ticker is still in current_prices before attempting to sell
            if ticker in current_prices.index:
                self._sell_position(
                    ticker, current_prices[ticker], rebalance_date, "rank_exit"
                )
            else:
                logger.warning(
                    f"Cannot sell {ticker} on {rebalance_date.date()}: Price data not available."
                )

        # Calculate target allocation for remaining + new positions
        target_positions = {}
        # Only consider tickers that are in current_prices for buying
        remaining_tickers = [
            t
            for t in target_tickers
            if t in current_prices.index
            and (t in self.holdings or t not in tickers_to_exit)
        ]

        if remaining_tickers:
            allocation_per_etf = min(
                self.total_value / len(remaining_tickers),
                self.total_value * self.config.max_position_size,
            )

            for ticker in remaining_tickers:
                if ticker in current_prices.index:
                    target_shares = allocation_per_etf / current_prices[ticker]
                    target_positions[ticker] = target_shares

        # Rebalance positions
        for ticker, target_shares in target_positions.items():
            current_shares = self.holdings.get(ticker, 0)
            shares_diff = target_shares - current_shares

            if abs(shares_diff) > 0.01:  # Minimum trade threshold
                if shares_diff > 0:
                    # Ensure price data is available for buying
                    if ticker in current_prices.index:
                        self._buy_shares(
                            ticker, shares_diff, current_prices[ticker], rebalance_date
                        )
                    else:
                        logger.warning(
                            f"Cannot buy {ticker} on {rebalance_date.date()}: Price data not available."
                        )

                else:
                    # Ensure price data is available for selling
                    if ticker in current_prices.index:
                        self._sell_shares(
                            ticker,
                            abs(shares_diff),
                            current_prices[ticker],
                            rebalance_date,
                        )
                    else:
                        logger.warning(
                            f"Cannot sell {ticker} on {rebalance_date.date()}: Price data not available."
                        )

    def _update_portfolio_value(self, current_prices: pd.Series) -> None:
        """Update total portfolio value based on current prices."""
        holdings_value = 0
        for ticker, shares in self.holdings.items():
            if ticker in current_prices.index:
                holdings_value += shares * current_prices[ticker]

        self.total_value = holdings_value + self.cash

    def _buy_shares(
        self, ticker: str, shares: float, price: float, date: datetime
    ) -> None:
        """Buy shares of an ETF."""
        cost = shares * price
        commission = cost * self.config.commission_rate
        total_cost = cost + commission

        if total_cost <= self.cash:
            self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
            self.cash -= total_cost
            self.transaction_costs += commission

            self.trade_log.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "buy",
                    "shares": shares,
                    "price": price,
                    "cost": total_cost,
                }
            )

    def _sell_shares(
        self, ticker: str, shares: float, price: float, date: datetime
    ) -> None:
        """Sell shares of an ETF."""
        if ticker in self.holdings and self.holdings[ticker] >= shares:
            proceeds = shares * price
            commission = proceeds * self.config.commission_rate
            net_proceeds = proceeds - commission

            self.holdings[ticker] -= shares
            if self.holdings[ticker] < 0.01:  # Remove very small positions
                del self.holdings[ticker]

            self.cash += net_proceeds
            self.transaction_costs += commission

            self.trade_log.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "sell",
                    "shares": shares,
                    "price": price,
                    "proceeds": net_proceeds,
                }
            )

    def _sell_position(
        self, ticker: str, price: float, date: datetime, reason: str
    ) -> None:
        """Sell entire position in an ETF."""
        if ticker in self.holdings:
            shares = self.holdings[ticker]
            self._sell_shares(ticker, shares, price, date)
            logger.info(
                f"Sold {ticker} position ({shares:.2f} shares) - Reason: {reason}"
            )


class ETFMomentumStrategy:
    """Main strategy orchestrator."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_provider = DataProvider(config)
        self.momentum_calculator = MomentumCalculator(config)
        self.portfolio = Portfolio(config)
        self.performance_metrics = {}

    def get_current_portfolio_status(self) -> Dict:
        """
        Get current portfolio status and check if rebalancing is needed.

        Returns:
            Dictionary containing current portfolio status and rebalancing recommendation
        """
        logger.info("Checking current portfolio status...")

        # Get current date and calculate data fetch period
        current_date = datetime.now()
        data_start = current_date - timedelta(
            days=self.config.min_data_points + 100
        )  # Increased buffer

        try:
            # Fetch current market data
            all_data = self.data_provider.fetch_etf_data(
                self.config.etf_universe, data_start, current_date
            )

            prices_df = self.data_provider.get_prices(all_data)

            # Convert timezone-aware index to timezone-naive
            if prices_df.index.tz is not None:
                prices_df.index = prices_df.index.tz_localize(None)

            volume_df = self.data_provider.get_volumes(all_data)
            if volume_df is not None and volume_df.index.tz is not None:
                volume_df.index = volume_df.index.tz_localize(None)

            # Get current prices (latest available)
            current_prices = prices_df.iloc[-1]
            latest_date = prices_df.index[-1]

            # Apply filters to find eligible ETFs
            eligible_tickers = self.momentum_calculator.apply_filters(
                prices_df, volume_df
            )

            if not eligible_tickers:
                return {
                    "status": "error",
                    "message": "No eligible ETFs found with current filters",
                    "date": latest_date,
                }

            # Calculate current momentum scores
            eligible_data = prices_df[eligible_tickers]
            momentum_scores = self.momentum_calculator.calculate_momentum_scores(
                eligible_data
            )

            if momentum_scores.empty:
                return {
                    "status": "error",
                    "message": "Could not calculate momentum scores",
                    "date": latest_date,
                }

            # Get top ranked ETFs
            ranked_etfs = [(ticker, score) for ticker, score in momentum_scores.items()]
            top_etfs = ranked_etfs[: self.config.portfolio_size]

            # Analyze current portfolio vs optimal portfolio
            current_holdings = (
                list(self.portfolio.holdings.keys())
                if hasattr(self.portfolio, "holdings")
                else []
            )
            optimal_tickers = [ticker for ticker, _ in top_etfs]

            # Calculate exit threshold
            exit_rank_threshold = int(
                self.config.portfolio_size * self.config.exit_rank_buffer_multiplier
            )

            # Check which holdings should be exited
            holdings_to_exit = []
            if current_holdings:
                for ticker in current_holdings:
                    current_rank = next(
                        (i for i, (t, _) in enumerate(ranked_etfs) if t == ticker),
                        float("inf"),
                    )
                    if current_rank >= exit_rank_threshold:
                        holdings_to_exit.append((ticker, current_rank))

            # Determine if rebalancing is needed
            needs_rebalancing = False
            rebalancing_reasons = []

            # Check if any holdings need to be exited
            if holdings_to_exit:
                needs_rebalancing = True
                rebalancing_reasons.append(
                    f"Holdings to exit due to poor ranking: {[h[0] for h in holdings_to_exit]}"
                )

            # Check if there are new top performers not in current portfolio
            new_opportunities = [
                ticker for ticker in optimal_tickers if ticker not in current_holdings
            ]
            if new_opportunities:
                needs_rebalancing = True
                rebalancing_reasons.append(
                    f"New high-momentum opportunities: {new_opportunities}"
                )

            # Check next scheduled rebalancing date
            next_rebalance_date = self._get_next_rebalance_date(current_date)
            days_to_rebalance = (next_rebalance_date - current_date).days

            # Threshold-based rebalancing logic
            threshold_triggered = False
            threshold_reason = None
            if self.config.use_threshold_rebalancing:
                # Calculate portfolio return since start
                portfolio_return = (
                    self.portfolio.total_value / self.config.initial_capital - 1
                ) * 100
                if portfolio_return >= self.config.profit_threshold_pct:
                    threshold_triggered = True
                    threshold_reason = f"Profit threshold reached: {portfolio_return:.2f}% >= {self.config.profit_threshold_pct}%"
                elif portfolio_return <= self.config.loss_threshold_pct:
                    threshold_triggered = True
                    threshold_reason = f"Loss threshold reached: {portfolio_return:.2f}% <= {self.config.loss_threshold_pct}%"
                if threshold_triggered:
                    needs_rebalancing = True
                    rebalancing_reasons.append(threshold_reason)

            # Date-based rebalancing logic
            if not self.config.use_threshold_rebalancing or (
                self.config.use_threshold_rebalancing
                and self.config.rebalance_frequency
            ):
                if days_to_rebalance <= 0:
                    needs_rebalancing = True
                    rebalancing_reasons.append("Scheduled rebalancing date reached")

            return {
                "status": "success",
                "date": latest_date,
                "current_date": current_date,
                "next_rebalance_date": next_rebalance_date,
                "days_to_next_rebalance": max(0, days_to_rebalance),
                "needs_rebalancing": needs_rebalancing,
                "rebalancing_reasons": rebalancing_reasons,
                "current_holdings": current_holdings,
                "optimal_portfolio": optimal_tickers,
                "holdings_to_exit": holdings_to_exit,
                "new_opportunities": new_opportunities,
                "top_10_momentum_scores": ranked_etfs[:10],
                "current_prices": current_prices.to_dict(),
                "market_data_date": latest_date.strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.error(f"Error checking portfolio status: {e}")
            return {
                "status": "error",
                "message": f"Failed to check portfolio status: {str(e)}",
                "date": current_date,
            }

    def _get_next_rebalance_date(self, current_date: datetime) -> datetime:
        """Calculate the next scheduled rebalancing date."""
        # Start with current month
        next_date = current_date.replace(day=self.config.rebalance_day_of_month)

        # If we've passed this month's rebalance date, move to next month
        if current_date.day > self.config.rebalance_day_of_month:
            if next_date.month == 12:
                next_date = next_date.replace(year=next_date.year + 1, month=1)
            else:
                next_date = next_date.replace(month=next_date.month + 1)

        return next_date

    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Run complete backtest of the momentum strategy.

        Returns:
            Dictionary containing performance metrics and trade history
        """
        logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")

        # Generate rebalancing dates
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date)

        # Fetch all historical data
        # Fetch enough data to cover the longest lookback period plus the backtest range
        max_lookback = max(
            self.config.long_term_period_days, self.config.moving_average_period
        )
        data_fetch_start = start_date - timedelta(days=max_lookback + 50)  # Add buffer

        all_data = self.data_provider.fetch_etf_data(
            self.config.etf_universe, data_fetch_start, end_date
        )

        prices_df_full = self.data_provider.get_prices(all_data)
        # Convert timezone-aware index to timezone-naive to avoid comparison issues
        if prices_df_full.index.tz is not None:
            prices_df_full.index = prices_df_full.index.tz_localize(None)

        volume_df_full = self.data_provider.get_volumes(all_data)
        # Convert timezone-aware index to timezone-naive for volume data too
        if volume_df_full is not None and volume_df_full.index.tz is not None:
            volume_df_full.index = volume_df_full.index.tz_localize(None)

        portfolio_history = []
        # Track portfolio value at last rebalance for threshold comparisons
        last_rebalance_value = self.config.initial_capital
        last_rebalance_date = start_date
        rebalance_idx = 0
        current_date = rebalance_dates[0] if rebalance_dates else start_date

        # Log configuration info
        if self.config.use_threshold_rebalancing:
            logger.info(
                f"Threshold rebalancing enabled: profit {self.config.profit_threshold_pct}%, loss {self.config.loss_threshold_pct}%"
            )

        while rebalance_idx < len(rebalance_dates):
            current_date = rebalance_dates[rebalance_idx]

            # Ensure current_date is within the fetched data range
            if (
                current_date < prices_df_full.index.min()
                or current_date > prices_df_full.index.max()
            ):
                logger.warning(
                    f"Rebalance date {current_date.date()} is outside fetched data range."
                )
                rebalance_idx += 1
                continue

            # Get data strictly up to current_date for calculations (addressing look-ahead bias)
            current_data_prices = prices_df_full[prices_df_full.index <= current_date]
            current_data_volume = (
                volume_df_full[volume_df_full.index <= current_date]
                if volume_df_full is not None
                else None
            )

            if current_data_prices.empty:
                logger.warning(
                    f"No data available up to {current_date.date()}. Skipping rebalance."
                )
                rebalance_idx += 1
                continue

            # Apply filters
            eligible_tickers = self.momentum_calculator.apply_filters(
                current_data_prices, current_data_volume
            )

            if not eligible_tickers:
                logger.warning(f"No eligible ETFs found for {current_date.date()}")
                # Still record portfolio value on rebalance date even if no trades
                if not current_data_prices.empty:
                    current_prices_for_value = current_data_prices.iloc[-1]
                    self.portfolio._update_portfolio_value(current_prices_for_value)
                    portfolio_history.append(
                        {
                            "date": current_date,
                            "total_value": self.portfolio.total_value,
                            "cash": self.portfolio.cash,
                            "holdings": dict(self.portfolio.holdings),
                            "top_etfs": [],  # No eligible ETFs means no top ETFs
                        }
                    )
                rebalance_idx += 1
                continue

            # Calculate momentum scores
            eligible_data = current_data_prices[eligible_tickers]
            momentum_scores = self.momentum_calculator.calculate_momentum_scores(
                eligible_data
            )

            if momentum_scores.empty:
                logger.warning(
                    f"Could not calculate momentum scores for {current_date.date()}"
                )
                # Record portfolio value
                if not current_data_prices.empty:
                    current_prices_for_value = current_data_prices.iloc[-1]
                    self.portfolio._update_portfolio_value(current_prices_for_value)
                    portfolio_history.append(
                        {
                            "date": current_date,
                            "total_value": self.portfolio.total_value,
                            "cash": self.portfolio.cash,
                            "holdings": dict(self.portfolio.holdings),
                            "top_etfs": [],  # No momentum scores means no top ETFs
                        }
                    )
                rebalance_idx += 1
                continue

            # Rank ETFs
            ranked_etfs = [(ticker, score) for ticker, score in momentum_scores.items()]

            # Get current prices for rebalancing (using the latest available price on or before current_date)
            current_prices_for_rebalance = current_data_prices.iloc[-1]

            # Rebalance portfolio
            self.portfolio.rebalance(
                ranked_etfs, current_prices_for_rebalance, current_date
            )

            # Record portfolio state AFTER rebalancing
            self.portfolio._update_portfolio_value(
                current_prices_for_rebalance
            )  # Update value after trades
            portfolio_history.append(
                {
                    "date": current_date,
                    "total_value": self.portfolio.total_value,
                    "cash": self.portfolio.cash,
                    "holdings": dict(self.portfolio.holdings),
                    "top_etfs": ranked_etfs[:10],  # Top 10 for analysis
                }
            )
            logger.info(
                f"Rebalanced on {current_date.date()}. Portfolio value: ₹{self.portfolio.total_value:,.2f}"
            )

            # Threshold-based rebalancing: check after each scheduled rebalance
            threshold_triggered = False
            if self.config.use_threshold_rebalancing:
                # Calculate return since LAST rebalance (not just from initial capital)
                period_return = (
                    self.portfolio.total_value / last_rebalance_value - 1
                ) * 100
                threshold_triggered = False

                if period_return >= self.config.profit_threshold_pct:
                    threshold_triggered = True
                    logger.info(
                        f"Profit threshold reached: {period_return:.2f}% >= {self.config.profit_threshold_pct}% on {current_date.date()}"
                    )
                    logger.info(
                        f"  Portfolio value: ₹{self.portfolio.total_value:,.2f} vs last: ₹{last_rebalance_value:,.2f}"
                    )
                elif period_return <= self.config.loss_threshold_pct:
                    threshold_triggered = True
                    logger.info(
                        f"Loss threshold reached: {period_return:.2f}% <= {self.config.loss_threshold_pct}% on {current_date.date()}"
                    )
                    logger.info(
                        f"  Portfolio value: ₹{self.portfolio.total_value:,.2f} vs last: ₹{last_rebalance_value:,.2f}"
                    )

            # Update last rebalance value and date if a threshold was triggered or it's a scheduled rebalance
            if threshold_triggered or not self.config.use_threshold_rebalancing:
                last_rebalance_value = self.portfolio.total_value
                last_rebalance_date = current_date

            # If threshold was triggered, find the next trading day to rebalance again
            if threshold_triggered:
                next_trading_date = current_date + timedelta(days=1)
                while (
                    next_trading_date not in prices_df_full.index
                    and next_trading_date <= end_date
                ):
                    next_trading_date += timedelta(days=1)

                if next_trading_date <= end_date:
                    # Insert the next trading date into the rebalance_dates list
                    # Find the correct position to maintain sorted order
                    insert_pos = len(rebalance_dates)
                    for i, date in enumerate(rebalance_dates[rebalance_idx + 1 :]):
                        if next_trading_date < date:
                            insert_pos = rebalance_idx + 1 + i
                            break
                    rebalance_dates.insert(insert_pos, next_trading_date)
                    logger.info(
                        f"Scheduled next threshold rebalance on {next_trading_date.date()}"
                    )
                else:
                    logger.info(
                        "End of backtest period reached after threshold trigger."
                    )

            # Move to next scheduled rebalance date (or the inserted threshold rebalance date)
            rebalance_idx += 1

        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(
            portfolio_history
        )
        return {
            "portfolio_history": portfolio_history,
            "trade_log": self.portfolio.trade_log,
            "performance_metrics": self.performance_metrics,
            "final_value": self.portfolio.total_value,
            "total_return": (
                self.portfolio.total_value / self.config.initial_capital - 1
            )
            * 100,
        }

    def _generate_rebalance_dates(
        self, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """Generate list of rebalancing dates."""
        dates = []
        current = start_date.replace(day=self.config.rebalance_day_of_month)

        # If start date is after rebalance day, move to next month
        if start_date.day > self.config.rebalance_day_of_month:
            # Handle potential ValueError if day is out of range for the next month
            try:
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            except ValueError:
                # If day is out of range (e.g., 31st of Jan to Feb), set to last day of month
                if current.month == 12:
                    current = current.replace(
                        year=current.year + 1,
                        month=1,
                        day=pd.Timestamp(current.year + 1, 1, 1).days_in_month,
                    )
                else:
                    current = current.replace(
                        month=current.month + 1,
                        day=pd.Timestamp(
                            current.year, current.month + 1, 1
                        ).days_in_month,
                    )

        while current <= end_date:
            dates.append(current)
            # Move to next month
            try:
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            except ValueError:
                # If day is out of range (e.g., 31st of Jan to Feb), set to last day of month
                if current.month == 12:
                    current = current.replace(
                        year=current.year + 1,
                        month=1,
                        day=pd.Timestamp(current.year + 1, 1, 1).days_in_month,
                    )
                else:
                    current = current.replace(
                        month=current.month + 1,
                        day=pd.Timestamp(
                            current.year, current.month + 1, 1
                        ).days_in_month,
                    )

        return dates

    def _calculate_performance_metrics(self, portfolio_history: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not portfolio_history:
            return {}

        # Ensure portfolio_history is sorted by date
        portfolio_history = sorted(portfolio_history, key=lambda x: x["date"])

        values = [p["total_value"] for p in portfolio_history]
        dates = [p["date"] for p in portfolio_history]

        # Convert to pandas Series for easier calculation
        value_series = pd.Series(values, index=dates)
        returns = value_series.pct_change().dropna()

        # Basic metrics
        total_return = (values[-1] / self.config.initial_capital - 1) * 100
        # Ensure enough history for annualized return calculation
        if (dates[-1] - dates[0]).days > 0:
            annualized_return = (
                (values[-1] / self.config.initial_capital)
                ** (365.25 / (dates[-1] - dates[0]).days)
                - 1
            ) * 100
        else:
            annualized_return = 0.0  # Or NaN, depending on desired behavior

        # Risk metrics
        # Annualize volatility based on the frequency of portfolio history points
        # Assuming portfolio_history points are roughly monthly for monthly rebalancing
        if len(returns) > 1:
            # Estimate frequency based on average difference between dates
            avg_days_between_rebalances = (
                (dates[-1] - dates[0]).days / (len(dates) - 1) if len(dates) > 1 else 30
            )
            annualization_factor = (
                365.25 / avg_days_between_rebalances
                if avg_days_between_rebalances > 0
                else 12
            )  # Default to monthly if only one data point

            volatility = returns.std() * np.sqrt(annualization_factor) * 100
        else:
            volatility = 0.0  # Not enough data points to calculate volatility

        max_drawdown = ((value_series / value_series.expanding().max()) - 1).min() * 100

        # Sharpe ratio (assuming risk-free rate of 3%)
        risk_free_rate = 0.03
        # Adjust risk-free rate to the frequency of returns
        if len(returns) > 0:
            # Assuming returns are approximately monthly
            risk_free_rate_per_period = (
                (1 + risk_free_rate) ** (avg_days_between_rebalances / 365.25) - 1
                if "avg_days_between_rebalances" in locals()
                and avg_days_between_rebalances > 0
                else risk_free_rate / 12
            )
            excess_returns = returns - risk_free_rate_per_period

            sharpe_ratio = (
                excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)
                if returns.std() > 0
                and "annualization_factor" in locals()
                and annualization_factor > 0
                else 0
            )
        else:
            sharpe_ratio = 0.0

        # Win ratio: count profitable trades vs total trades
        trade_log = getattr(self.portfolio, "trade_log", [])
        # Create a dictionary to track buy prices for each ticker
        buy_prices = {}
        win_trades = 0
        total_trades = 0

        # First pass: collect buy prices
        for trade in trade_log:
            if trade["action"] == "buy":
                ticker = trade["ticker"]
                if ticker not in buy_prices:
                    buy_prices[ticker] = []
                # Store price and shares as a tuple
                buy_prices[ticker].append((trade["price"], trade["shares"]))

        # Second pass: analyze sell trades
        for trade in trade_log:
            if trade["action"] == "sell":
                ticker = trade["ticker"]
                sell_price = trade["price"]
                sell_shares = trade["shares"]

                # If we have buy price data for this ticker
                if ticker in buy_prices and buy_prices[ticker]:
                    # Use FIFO (First In, First Out) for matching buys and sells
                    avg_buy_price = 0
                    total_shares_matched = 0
                    shares_to_match = sell_shares

                    # Create a temporary list to hold shares from buys being used
                    used_shares_list = []

                    # Find and use shares from buy_prices
                    i = 0
                    while shares_to_match > 0 and i < len(buy_prices[ticker]):
                        buy_price, buy_shares = buy_prices[ticker][i]
                        shares_from_this_buy = min(buy_shares, shares_to_match)

                        avg_buy_price += buy_price * shares_from_this_buy
                        total_shares_matched += shares_from_this_buy
                        shares_to_match -= shares_from_this_buy

                        # Record how many shares were used from this specific buy transaction
                        used_shares_list.append((i, shares_from_this_buy))

                        i += 1

                    # Update or remove used buy entries in reverse order
                    for idx, used_shares in reversed(used_shares_list):
                        buy_price, buy_shares = buy_prices[ticker][idx]
                        remaining_shares = buy_shares - used_shares
                        if (
                            remaining_shares <= 0.01
                        ):  # Use a small threshold for floating point comparison
                            buy_prices[ticker].pop(idx)
                        else:
                            buy_prices[ticker][idx] = (buy_price, remaining_shares)

                    if total_shares_matched > 0:
                        avg_buy_price /= total_shares_matched
                        # Determine if this was a winning trade
                        if sell_price > avg_buy_price:
                            win_trades += 1
                        total_trades += 1

        win_ratio = (win_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "volatility_pct": volatility,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(self.portfolio.trade_log),
            "transaction_costs": self.portfolio.transaction_costs,
            "win_ratio_pct": win_ratio,
        }


# Modified function to run backtest with multiple initial investment amounts
# and permute portfolio_size and rebalance_day_of_month
def run_multi_investment_backtest(
    investment_amounts: List[float] = None,
    portfolio_sizes: List[int] = None,  # New parameter: list of portfolio sizes
    rebalance_days: List[int] = None,  # New parameter: list of rebalance days
    exit_rank_buffers: List[
        float
    ] = None,  # New parameter: list of exit rank buffer multipliers
    use_threshold_rebalancing: bool = False,
    profit_threshold_pct: float = 10.0,
    loss_threshold_pct: float = -5.0,
) -> List[Dict]:
    """
    Run backtest with multiple initial investment amounts and permute
    portfolio_size, rebalance_day_of_month, and exit_rank_buffer_multiplier.
    """

    investment_amounts = investment_amounts or [
        1000000
    ]  # Default to 1M if not provided
    portfolio_sizes = portfolio_sizes or [5, 7]  # Default portfolio sizes
    rebalance_days = rebalance_days or [1, 5, 10, 15]  # Default rebalance days
    exit_rank_buffers = exit_rank_buffers or [
        1.5,
        2.0,
        2.5,
    ]  # Default exit rank buffers

    all_results = []

    # Iterate through all combinations of portfolio_size, rebalance_day_of_month, and exit_rank_buffer_multiplier
    for p_size, r_day, exit_buffer in itertools.product(
        portfolio_sizes, rebalance_days, exit_rank_buffers
    ):
        print(f"\n{'='*80}")
        print(
            f"RUNNING PERMUTATION: Portfolio Size = {p_size}, Rebalance Day = {r_day}, Exit Buffer = {exit_buffer}"
        )
        print(f"{'='*80}")

        results_for_permutation = []

        for initial_capital in investment_amounts:
            print(f"\n{'='*60}")
            print(f"TESTING WITH INITIAL CAPITAL: ₹{initial_capital:,.2f}")
            print(f"{'='*60}")
            # Create configuration with specific initial capital and rebalancing options
            config = StrategyConfig(
                portfolio_size=p_size,  # Use permuted portfolio size
                rebalance_day_of_month=r_day,  # Use permuted rebalance day
                exit_rank_buffer_multiplier=exit_buffer,  # Use permuted exit buffer
                long_term_period_days=252,
                short_term_period_days=60,
                initial_capital=initial_capital,
                use_threshold_rebalancing=use_threshold_rebalancing,
                profit_threshold_pct=profit_threshold_pct,
                loss_threshold_pct=loss_threshold_pct,
            )

            # Initialize strategy
            strategy = ETFMomentumStrategy(config)

            # Run backtest
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2024, 12, 31)

            try:
                results = strategy.run_backtest(start_date, end_date)

                print(f"\n=== Backtest Results for this Configuration ===")
                print(f"Initial Capital: ₹{config.initial_capital:,.2f}")
                print(f"Final Value: ₹{results['final_value']:,.2f}")
                print(
                    f"Absolute Gain: ₹{results['final_value'] - config.initial_capital:,.2f}"
                )
                print(f"Total Return: {results['total_return']:.2f}%")
                print(
                    f"Annualized Return: {results['performance_metrics']['annualized_return_pct']:.2f}%"
                )
                print(
                    f"Max Drawdown: {results['performance_metrics']['max_drawdown_pct']:.2f}%"
                )
                print(
                    f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}"
                )
                print(f"Total Trades: {results['performance_metrics']['total_trades']}")
                print(
                    f"Transaction Costs: ₹{results['performance_metrics']['transaction_costs']:,.2f}"
                )
                print(
                    f"Transaction Costs %: {(results['performance_metrics']['transaction_costs']/results['final_value']*100):.2f}%"
                )
                print(
                    f"Win Ratio: {results['performance_metrics'].get('win_ratio_pct', 0):.2f}%"
                )

                # Store results for comparison, including permutation parameters
                results_for_permutation.append(
                    {
                        "portfolio_size": p_size,
                        "rebalance_day": r_day,
                        "exit_buffer": exit_buffer,  # Include exit buffer in results
                        "initial_capital": initial_capital,
                        "final_value": results["final_value"],
                        "total_return_pct": results["total_return"],
                        "annualized_return_pct": results["performance_metrics"][
                            "annualized_return_pct"
                        ],
                        "sharpe_ratio": results["performance_metrics"]["sharpe_ratio"],
                        "max_drawdown_pct": results["performance_metrics"][
                            "max_drawdown_pct"
                        ],
                        "total_trades": results["performance_metrics"]["total_trades"],
                        "transaction_costs": results["performance_metrics"][
                            "transaction_costs"
                        ],
                        "transaction_costs_pct": (
                            (
                                results["performance_metrics"]["transaction_costs"]
                                / results["final_value"]
                                * 100
                            )
                            if results["final_value"] > 0
                            else 0.0
                        ),  # Handle division by zero
                        "win_ratio_pct": results["performance_metrics"].get(
                            "win_ratio_pct", 0
                        ),
                    }
                )

            except Exception as e:
                logger.error(
                    f"Backtest failed for Portfolio Size {p_size}, Rebalance Day {r_day}, Exit Buffer {exit_buffer}, Capital ₹{initial_capital:,.2f}: {e}"
                )
                continue

        all_results.extend(results_for_permutation)

    # Print overall comparison summary for all permutations
    print(f"\n{'='*100}")
    print("OVERALL PERMUTATION COMPARISON SUMMARY")
    print(f"{'='*100}")

    # Prepare data for tabulate
    summary_table_data = []
    headers = [
        "Port. Size",
        "Rebal. Day",
        "Exit Buffer",  # Add Exit Buffer to headers
        "Initial Capital",
        "Final Value",
        "Total Return",
        "Ann. Return",
        "Sharpe",
        "Max DD",
        "Tx Costs %",
        "Win Ratio %",
    ]
    for result in all_results:
        summary_table_data.append(
            [
                result["portfolio_size"],
                result["rebalance_day"],
                result["exit_buffer"],  # Add Exit Buffer to data
                f"₹{result['initial_capital']/100000:.0f}L",
                f"₹{result['final_value']/100000:.1f}L",
                f"{result['total_return_pct']:.1f}%",
                f"{result['annualized_return_pct']:.1f}%",
                f"{result['sharpe_ratio']:.2f}",
                f"{result['max_drawdown_pct']:.1f}%",
                f"{result['transaction_costs_pct']:.2f}%",
                f"{result['win_ratio_pct']:.2f}%",
            ]
        )

    # Sort results for better comparison in the table (e.g., by Annualized Return descending)
    summary_table_data_sorted = sorted(
        summary_table_data, key=lambda x: float(x[6].replace("%", "")), reverse=True
    )

    print(tabulate(summary_table_data_sorted, headers=headers, tablefmt="grid"))

    return all_results


if __name__ == "__main__":
    # Define the lists of parameters you want to permute
    test_investment_amounts = [1000000]  # Example: Test with 1M INR
    test_portfolio_sizes = [3, 5, 7]  # Example: Test portfolio sizes 3, 5 and 7
    test_rebalance_days = [
        1,
        5,
        10,
    ]  # Example: Test rebalancing on 1st, 5th, and 10th of month
    test_exit_rank_buffers = [
        1.5,
        2.0,
        2.5,
    ]  # Example: Test exit rank buffers 1.5, 2.0, and 2.5

    # Run the backtest with permutations
    run_multi_investment_backtest(
        investment_amounts=test_investment_amounts,
        portfolio_sizes=test_portfolio_sizes,
        rebalance_days=test_rebalance_days,
        exit_rank_buffers=test_exit_rank_buffers,  # Pass the list of exit rank buffers
        use_threshold_rebalancing=False,  # Set to True if you want to include threshold rebalancing
    )
