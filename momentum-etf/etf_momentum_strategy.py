"""
Main strategy logic for ETF momentum strategy.
Author : Prashant Srivastava
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import itertools
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

from core import (
    StrategyConfig,
    DataProvider,
    MomentumCalculator,
    logger,
    TradingCostCalculator,
    MarketImpactModel,
)


class Portfolio:
    """Manages portfolio holdings and rebalancing logic."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.holdings = {}  # {ticker: shares}
        self.cash = config.initial_capital
        self.total_value = config.initial_capital
        self.transaction_costs = 0.0
        self.trade_log = []
        # For realistic execution if enabled
        self.cost_calculator = (
            TradingCostCalculator(config) if config.realistic_execution else None
        )
        self.market_impact_model = (
            MarketImpactModel(config) if config.realistic_execution else None
        )

    def rebalance(
        self,
        ranked_etfs: List[Tuple[str, float]],
        current_prices: pd.Series,
        rebalance_date: datetime,
        current_volumes: pd.Series = None,  # Added for realism
        volatilities: Dict[str, float] = None,  # Added for realism
    ) -> None:
        """
        Rebalance portfolio based on new rankings.

        Args:
            ranked_etfs: List of (ticker, momentum_score) tuples, sorted by score
            current_prices: Current prices for all ETFs
            rebalance_date: Date of rebalancing
            current_volumes: Current volumes for liquidity checks (if realistic)
            volatilities: Volatilities for market impact (if realistic)
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
                    ticker,
                    current_prices[ticker],
                    rebalance_date,
                    "rank_exit",
                    (
                        current_volumes.get(ticker, 100000)
                        if current_volumes is not None
                        else 100000
                    ),
                    volatilities.get(ticker, 0.2) if volatilities else 0.2,
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
                            ticker,
                            shares_diff,
                            current_prices[ticker],
                            rebalance_date,
                            (
                                current_volumes.get(ticker, 100000)
                                if current_volumes is not None
                                else 100000
                            ),
                            volatilities.get(ticker, 0.2) if volatilities else 0.2,
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
                            (
                                current_volumes.get(ticker, 100000)
                                if current_volumes is not None
                                else 100000
                            ),
                            volatilities.get(ticker, 0.2) if volatilities else 0.2,
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
        self,
        ticker: str,
        shares: float,
        price: float,
        date: datetime,
        daily_volume: float,
        volatility: float,
    ) -> None:
        """Buy shares of an ETF with optional realism."""
        if self.config.realistic_execution:
            # Realistic execution
            slippage = self.market_impact_model.estimate_slippage(
                ticker, shares, daily_volume, volatility, "buy"
            )
            execution_price = price * (1 + slippage)
            cost_breakdown = self.cost_calculator.calculate_costs(
                ticker, shares, execution_price, "buy"
            )
            total_cost = shares * execution_price + cost_breakdown["total_cost"]
        else:
            # Simple execution
            cost = shares * price
            commission = cost * self.config.commission_rate
            total_cost = cost + commission

        if total_cost <= self.cash:
            self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
            self.cash -= total_cost
            self.transaction_costs += (
                cost_breakdown["total_cost"]
                if self.config.realistic_execution
                else commission
            )

            self.trade_log.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "buy",
                    "shares": shares,
                    "price": (
                        execution_price if self.config.realistic_execution else price
                    ),
                    "cost": total_cost,
                }
            )

    def _sell_shares(
        self,
        ticker: str,
        shares: float,
        price: float,
        date: datetime,
        daily_volume: float,
        volatility: float,
    ) -> None:
        """Sell shares of an ETF with optional realism."""
        if ticker in self.holdings and self.holdings[ticker] >= shares:
            if self.config.realistic_execution:
                # Realistic execution
                slippage = self.market_impact_model.estimate_slippage(
                    ticker, shares, daily_volume, volatility, "sell"
                )
                execution_price = price * (1 - slippage)
                cost_breakdown = self.cost_calculator.calculate_costs(
                    ticker, shares, execution_price, "sell"
                )
                proceeds = shares * execution_price
                net_proceeds = proceeds - cost_breakdown["total_cost"]
            else:
                # Simple execution
                proceeds = shares * price
                commission = proceeds * self.config.commission_rate
                net_proceeds = proceeds - commission

            self.holdings[ticker] -= shares
            if self.holdings[ticker] < 0.01:  # Remove very small positions
                del self.holdings[ticker]

            self.cash += net_proceeds
            self.transaction_costs += (
                cost_breakdown["total_cost"]
                if self.config.realistic_execution
                else commission
            )

            self.trade_log.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "sell",
                    "shares": shares,
                    "price": (
                        execution_price if self.config.realistic_execution else price
                    ),
                    "proceeds": net_proceeds,
                }
            )

    def _sell_position(
        self,
        ticker: str,
        price: float,
        date: datetime,
        reason: str,
        daily_volume: float,
        volatility: float,
    ) -> None:
        """Sell entire position in an ETF."""
        if ticker in self.holdings:
            shares = self.holdings[ticker]
            self._sell_shares(ticker, shares, price, date, daily_volume, volatility)
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

        # Get current date in IST timezone
        ist_tz = pytz.timezone('Asia/Kolkata')
        current_date = datetime.now(ist_tz).replace(tzinfo=None)  # Convert to naive datetime in IST
        data_start = current_date - timedelta(
            days=self.config.min_data_points + 100
        )  # Increased buffer

        try:
            # Fetch current market data
            all_data = self.data_provider.fetch_etf_data(
                self.config.etf_universe, data_start, current_date
            )

            prices_df = self.data_provider.get_prices(all_data)

            # Convert timezone-aware index to IST timezone, then to timezone-naive
            if prices_df.index.tz is not None:
                ist_tz = pytz.timezone('Asia/Kolkata')
                prices_df.index = prices_df.index.tz_convert(ist_tz).tz_localize(None)
            else:
                # If already timezone-naive, assume it's in UTC and convert to IST
                prices_df.index = pd.to_datetime(prices_df.index).tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

            # Forward-fill prices to handle gaps
            prices_df = prices_df.fillna(method="ffill")

            volume_df = self.data_provider.get_volumes(all_data)
            if volume_df is not None:
                if volume_df.index.tz is not None:
                    ist_tz = pytz.timezone('Asia/Kolkata')
                    volume_df.index = volume_df.index.tz_convert(ist_tz).tz_localize(None)
                else:
                    # If already timezone-naive, assume it's in UTC and convert to IST
                    volume_df.index = pd.to_datetime(volume_df.index).tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

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
            logger.info(f"Calculating momentum scores for eligible ETFs: {eligible_tickers}")
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
            ranked_etfs = sorted(
                [(ticker, score) for ticker, score in momentum_scores.items()],
                key=lambda x: x[1],
                reverse=True,
            )
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
        # Convert timezone-aware index to IST timezone, then to timezone-naive
        if prices_df_full.index.tz is not None:
            ist_tz = pytz.timezone('Asia/Kolkata')
            prices_df_full.index = prices_df_full.index.tz_convert(ist_tz).tz_localize(None)
        else:
            # If already timezone-naive, assume it's in UTC and convert to IST
            prices_df_full.index = pd.to_datetime(prices_df_full.index).tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

        # Forward-fill prices to handle gaps
        prices_df_full = prices_df_full.fillna(method="ffill")

        volume_df_full = self.data_provider.get_volumes(all_data)
        # Convert timezone-aware index to IST timezone, then to timezone-naive for volume data too
        if volume_df_full is not None:
            if volume_df_full.index.tz is not None:
                ist_tz = pytz.timezone('Asia/Kolkata')
                volume_df_full.index = volume_df_full.index.tz_convert(ist_tz).tz_localize(None)
            else:
                # If already timezone-naive, assume it's in UTC and convert to IST
                volume_df_full.index = pd.to_datetime(volume_df_full.index).tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

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
            ranked_etfs = sorted(
                [(ticker, score) for ticker, score in momentum_scores.items()],
                key=lambda x: x[1],
                reverse=True,
            )

            # Get current prices for rebalancing (using the latest available price on or before current_date)
            current_prices_for_rebalance = current_data_prices.iloc[-1]

            # For realism, get volumes and volatilities if enabled
            current_volumes = (
                current_data_volume.iloc[-1]
                if current_data_volume is not None
                else pd.Series(index=eligible_tickers, data=100000)
            )
            volatilities = {}
            if self.config.realistic_execution:
                volatility_window = min(30, len(current_data_prices))
                for ticker in eligible_tickers:
                    returns = (
                        current_data_prices[ticker].pct_change().tail(volatility_window)
                    )
                    volatilities[ticker] = (
                        returns.std() * np.sqrt(252) if not returns.empty else 0.2
                    )

            # Rebalance portfolio
            self.portfolio.rebalance(
                ranked_etfs,
                current_prices_for_rebalance,
                current_date,
                current_volumes,
                volatilities,
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
            try:
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            except ValueError:
                # Handle day out of range (e.g., 31st of Jan to Feb)
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
            try:
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            except ValueError:
                # Handle day out of range
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

        # Max Drawdown Amount and Recovery
        drawdown_series = (value_series / value_series.expanding().max()) - 1
        max_drawdown_pct = drawdown_series.min() * 100
        mdd_end_date = drawdown_series.idxmin()
        peak_value_before_mdd = value_series.loc[:mdd_end_date].max()
        peak_date_before_mdd = value_series.loc[:mdd_end_date].idxmax()
        trough_value_at_mdd = value_series[mdd_end_date]
        max_drawdown_amount = peak_value_before_mdd - trough_value_at_mdd

        # Recovery
        recovery_series = value_series.loc[mdd_end_date:]
        recovery_date_series = recovery_series[recovery_series >= peak_value_before_mdd]
        if not recovery_date_series.empty:
            recovery_date = recovery_date_series.index[0]
            days_to_recovery = (recovery_date - peak_date_before_mdd).days
        else:
            days_to_recovery = "Not Recovered"
            recovery_date = "N/A"

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
        gross_profit = 0.0
        gross_loss = 0.0
        wins = []
        losses = []
        trade_outcomes = []

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
                        profit_loss = (
                            sell_price - avg_buy_price
                        ) * total_shares_matched
                        # Determine if this was a winning trade
                        if sell_price > avg_buy_price:
                            win_trades += 1
                            trade_outcomes.append(1)
                            gross_profit += profit_loss
                            wins.append(profit_loss / total_shares_matched)
                        else:
                            trade_outcomes.append(-1)
                            gross_loss += abs(profit_loss)
                            losses.append(abs(profit_loss) / total_shares_matched)
                        total_trades += 1

        win_ratio = (win_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = (avg_win / avg_loss) if avg_loss != 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else 0

        # Calculate streaks
        max_win_streak = 0
        max_loss_streak = 0
        if trade_outcomes:
            current_win_streak = 0
            current_loss_streak = 0
            for outcome in trade_outcomes:
                if outcome == 1:
                    current_win_streak += 1
                    current_loss_streak = 0
                else:  # outcome == -1
                    current_loss_streak += 1
                    current_win_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        # Ulcer Index
        drawdowns_squared = drawdown_series**2
        ulcer_index = np.sqrt(drawdowns_squared.mean()) * 100  # As percentage

        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "volatility_pct": volatility,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "transaction_costs": self.portfolio.transaction_costs,
            "win_ratio_pct": win_ratio,
            "max_drawdown_amount": max_drawdown_amount,
            "days_to_recovery": days_to_recovery,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "mdd_end_date": mdd_end_date,
            "trough_value_at_mdd": trough_value_at_mdd,
            "recovery_date": (
                recovery_date if isinstance(days_to_recovery, int) else "Not Recovered"
            ),
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "ulcer_index": ulcer_index,
        }


def run_parameter_experiments(
    investment_amounts: List[float] = None,
    portfolio_sizes: List[int] = None,
    use_threshold_rebalancing_values: List[bool] = None,
    profit_threshold_pct: float = 10.0,
    loss_threshold_pct: float = -5.0,
) -> List[Dict]:
    """
    Run backtest with multiple parameter permutations in parallel.
    """

    # Default values for experiments
    investment_amounts = investment_amounts or [1000000]
    portfolio_sizes = portfolio_sizes or [5]
    use_threshold_rebalancing_values = use_threshold_rebalancing_values or [False]

    all_results = []

    # Create all combinations of parameters
    param_combinations = list(
        itertools.product(
            portfolio_sizes,
            use_threshold_rebalancing_values,
            investment_amounts,
        )
    )

    print(f"Running {len(param_combinations)} backtest permutations in parallel...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_params = {
            executor.submit(
                run_single_backtest,
                p_size,
                use_threshold,
                initial_capital,
                profit_threshold_pct,
                loss_threshold_pct,
            ): (p_size, use_threshold, initial_capital)
            for p_size, use_threshold, initial_capital in param_combinations
        }

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Backtest failed for params {params}: {e}")

    # Print overall comparison summary for all permutations
    print(f"\n{'='*120}")
    print("OVERALL PERMUTATION COMPARISON SUMMARY")
    print(f"{'='*120}")

    # Prepare data for tabulate
    summary_table_data = []
    headers = [
        "Port. Size",
        "Threshold Rebal",
        "Ann. Return",
        "Sharpe",
        "Max DD",
        "Win Ratio %",
        "Trans. Costs %",
        "Expectancy",
    ]
    for result in all_results:
        summary_table_data.append(
            [
                result["portfolio_size"],
                result["use_threshold_rebalancing"],
                f"{result['annualized_return_pct']:.1f}%",
                f"{result['sharpe_ratio']:.2f}",
                f"{result['max_drawdown_pct']:.1f}%",
                f"{result['win_ratio_pct']:.2f}%",
                f"{result['transaction_costs_pct']:.2f}%",
                f"{result['expectancy']:.2f}",
            ]
        )

    # Sort results by annualized return (descending)
    summary_table_data_sorted = sorted(
        summary_table_data, key=lambda x: float(x[2].replace("%", "")), reverse=True
    )

    print(tabulate(summary_table_data_sorted, headers=headers, tablefmt="grid"))

    return all_results


def run_single_backtest(
    p_size,
    use_threshold,
    initial_capital,
    profit_threshold_pct,
    loss_threshold_pct,
):
    """Helper function to run a single backtest instance."""
    config = StrategyConfig(
        portfolio_size=p_size,
        initial_capital=initial_capital,
        use_threshold_rebalancing=use_threshold,
        profit_threshold_pct=profit_threshold_pct,
        loss_threshold_pct=loss_threshold_pct,
        realistic_execution=False,  # Default off for consistency, can toggle
    )

    strategy = ETFMomentumStrategy(config)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    results = strategy.run_backtest(start_date, end_date)

    return {
        "portfolio_size": p_size,
        "use_threshold_rebalancing": use_threshold,
        "initial_capital": initial_capital,
        "final_value": results["final_value"],
        "total_return_pct": results["total_return"],
        "annualized_return_pct": results["performance_metrics"][
            "annualized_return_pct"
        ],
        "sharpe_ratio": results["performance_metrics"]["sharpe_ratio"],
        "expectancy": results["performance_metrics"]["expectancy"],
        "max_drawdown_pct": results["performance_metrics"]["max_drawdown_pct"],
        "total_trades": results["performance_metrics"]["total_trades"],
        "transaction_costs": results["performance_metrics"]["transaction_costs"],
        "transaction_costs_pct": (
            (
                results["performance_metrics"]["transaction_costs"]
                / results["final_value"]
                * 100
            )
            if results["final_value"] > 0
            else 0.0
        ),
        "win_ratio_pct": results["performance_metrics"].get("win_ratio_pct", 0),
    }


if __name__ == "__main__":
    # If run directly, use the unified CLI
    from cli import unified_cli
    unified_cli()
