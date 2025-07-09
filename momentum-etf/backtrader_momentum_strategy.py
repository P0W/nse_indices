"""
Backtrader implementation of ETF Momentum Strategy.

"""

import sys
import codecs
import argparse
import itertools

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)

import backtrader as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings

# Import utilities from the new core and visualizer modules
from core import (
    StrategyConfig,
    DataProvider,
    MomentumCalculator,
    TradingCostCalculator,
    MarketImpactModel,
    logger,
)

from visualizer import create_performance_charts

warnings.filterwarnings("ignore")


class MomentumPortfolioStrategy(bt.Strategy):
    """
    Real-world backtrader strategy with realistic trading constraints.
    Incorporates market impact, liquidity constraints, and comprehensive cost modeling.
    """

    params = (
        ("config", None),  # StrategyConfig object
        ("rebalance_dates", None),  # List of rebalancing dates
        ("etf_universe", None),  # List of ETF symbols
        ("all_data_prices", None),  # Pre-fetched price data
        ("all_data_volumes", None),  # Pre-fetched volume data
    )

    def __init__(self):
        self.config = self.params.config
        self.rebalance_dates = self.params.rebalance_dates
        self.etf_universe = self.params.etf_universe
        self.all_data_prices = self.params.all_data_prices
        self.all_data_volumes = self.params.all_data_volumes

        # Initialize momentum calculator with same config
        self.momentum_calculator = MomentumCalculator(self.config)

        # Initialize real-world trading components
        self.cost_calculator = TradingCostCalculator(self.config)
        self.market_impact_model = MarketImpactModel(self.config)

        # Track portfolio state
        self.portfolio_history = []
        self.trade_log = []
        self.transaction_costs = 0.0
        self.total_slippage_costs = 0.0

        # Rebalancing tracking
        self.rebalance_idx = 0
        self.last_rebalance_value = self.config.initial_capital
        self.last_rebalance_date = None

        # Real-world constraints (more realistic for Indian ETF market)
        self.max_volume_participation = (
            0.15  # Max 15% of daily volume (higher for ETFs)
        )
        self.min_daily_volume = (
            10000  # Minimum daily volume threshold (lower for Indian market)
        )
        self.max_position_size = 0.25  # Max 25% in any single position

        # Create data feeds mapping
        self.data_feeds = {}
        for i, ticker in enumerate(self.etf_universe):
            if i < len(self.datas):
                self.data_feeds[ticker] = self.datas[i]

        logger.info(
            f"Initialized real-world momentum strategy with {len(self.data_feeds)} data feeds"
        )
        logger.info(
            f"Rebalance dates: {[d.date() for d in self.rebalance_dates[:5]]}..."
        )  # First 5 dates
        logger.info(f"ETF universe: {self.etf_universe}")
        logger.info(f"Data feeds mapping: {list(self.data_feeds.keys())}")
        logger.info(
            "Real-world constraints enabled: market impact, liquidity limits, comprehensive costs"
        )

        # Add EMA indicator for plotting
        self.emas = {}
        for ticker, data_feed in self.data_feeds.items():
            self.emas[ticker] = bt.indicators.ExponentialMovingAverage(
                data_feed.close, period=self.config.moving_average_period
            )

    def start(self):
        """Called before the strategy starts."""
        logger.info(f"Strategy started. Data feed count: {len(self.datas)}")
        if len(self.datas) > 0:
            logger.info(f"Data range: {self.datas[0].datetime.datetime(0)} to end")
        logger.info(f"Portfolio size: {self.config.portfolio_size}")
        logger.info(
            f"First few rebalance dates: {[d.date() for d in self.rebalance_dates[:3]]}"
        )

    def next(self):
        """Called for each bar/date."""
        current_datetime = self.datetime.datetime()
        current_date = current_datetime.date()

        # Check if today is a rebalancing date
        if self.rebalance_idx < len(self.rebalance_dates):
            rebalance_date = self.rebalance_dates[self.rebalance_idx].date()

            # Check if current date is on or after the target rebalancing date
            # This handles weekends and holidays by triggering on the next trading day
            if current_date >= rebalance_date:
                # Validate market conditions before rebalancing
                if self.is_market_suitable_for_rebalancing():
                    logger.debug(
                        f"🔄 Triggering rebalance on {current_date} (target was {rebalance_date})"
                    )
                    self.rebalance_portfolio()
                    self.rebalance_idx += 1

                    # Handle threshold-based rebalancing
                    if self.config.use_threshold_rebalancing:
                        self.check_threshold_rebalancing()
                else:
                    logger.warning(
                        f"Market conditions not suitable for rebalancing on {current_date}"
                    )

        # Always record daily portfolio state for tracking
        self.record_portfolio_state(current_datetime, [])

    def is_market_suitable_for_rebalancing(self) -> bool:
        """Check if market conditions are suitable for rebalancing."""
        current_datetime = self.datetime.datetime()

        # Check if it's a trading day (basic check)
        if current_datetime.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check data quality - need at least 80% of ETFs with valid data
        valid_data_count = 0
        for ticker in self.etf_universe:
            if ticker in self.data_feeds:
                data_feed = self.data_feeds[ticker]
                if data_feed.close[0] > 0:  # Valid price data
                    valid_data_count += 1

        return valid_data_count >= 0.8 * len(self.etf_universe)

    def validate_liquidity(
        self, ticker: str, shares: float, daily_volume: float
    ) -> Tuple[bool, float]:
        """Validate and adjust order size based on liquidity constraints."""
        if daily_volume < self.min_daily_volume:
            logger.warning(
                f"Insufficient liquidity for {ticker}: daily volume {daily_volume:,.0f} < {self.min_daily_volume:,.0f}"
            )
            return False, 0  # Insufficient liquidity

        max_shares = daily_volume * self.max_volume_participation

        if shares <= max_shares:
            return True, shares
        else:
            # Adjust to maximum allowable size
            logger.debug(
                f"Adjusting order size for {ticker}: {shares:.2f} -> {max_shares:.2f} shares"
            )
            return True, max_shares

    def rebalance_portfolio(self):
        """Rebalance portfolio using the same logic as original implementation."""
        current_datetime = self.datetime.datetime()
        current_date = current_datetime.date()

        logger.debug(f"Rebalancing portfolio on {current_date}")

        # Get current portfolio value
        current_portfolio_value = self.broker.getvalue()

        # Get data up to current date (avoiding look-ahead bias)
        current_data_prices = self.all_data_prices[
            self.all_data_prices.index <= current_datetime
        ]
        current_data_volumes = (
            self.all_data_volumes[self.all_data_volumes.index <= current_datetime]
            if self.all_data_volumes is not None
            else None
        )

        if current_data_prices.empty:
            logger.warning(
                f"No data available up to {current_date}. Skipping rebalance."
            )
            return

        # Apply filters to find eligible ETFs
        eligible_tickers = self.momentum_calculator.apply_filters(
            current_data_prices, current_data_volumes
        )

        if not eligible_tickers:
            logger.warning(f"No eligible ETFs found for {current_date}")
            self.record_portfolio_state(current_datetime, [])
            return

        # Calculate momentum scores
        eligible_data = current_data_prices[eligible_tickers]
        momentum_scores = self.momentum_calculator.calculate_momentum_scores(
            eligible_data
        )

        if momentum_scores.empty:
            logger.warning(f"Could not calculate momentum scores for {current_date}")
            self.record_portfolio_state(current_datetime, [])
            return

        # Rank ETFs by momentum score
        ranked_etfs = sorted(
            [(ticker, score) for ticker, score in momentum_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Get current prices for rebalancing
        current_prices = current_data_prices.iloc[-1]

        # Execute rebalancing logic
        self.execute_rebalancing(ranked_etfs, current_prices, current_datetime)

        # Record portfolio state
        self.record_portfolio_state(current_datetime, ranked_etfs[:10])

        logger.debug(
            f"Rebalanced on {current_date}. Portfolio value: ₹{current_portfolio_value:,.2f}"
        )

    def execute_rebalancing(
        self,
        ranked_etfs: List[Tuple[str, float]],
        current_prices: pd.Series,
        rebalance_date: datetime,
    ):
        """Execute rebalancing with realistic market constraints and costs."""

        # Get current portfolio value
        current_portfolio_value = self.broker.getvalue()

        # Get current volumes and calculate volatilities for market impact modeling
        current_volumes = (
            self.all_data_volumes.iloc[-1]
            if self.all_data_volumes is not None
            else pd.Series()
        )

        # Calculate 30-day volatility for each ETF
        volatility_window = min(30, len(self.all_data_prices))
        volatilities = {}
        for ticker, _ in ranked_etfs:
            if ticker in self.all_data_prices.columns:
                returns = (
                    self.all_data_prices[ticker].pct_change().tail(volatility_window)
                )
                volatilities[ticker] = returns.std() * np.sqrt(
                    252
                )  # Annualized volatility

        # Determine target portfolio (top N ETFs)
        target_tickers = [
            ticker for ticker, _ in ranked_etfs[: self.config.portfolio_size]
        ]

        # Calculate exit threshold
        exit_rank_threshold = int(
            self.config.portfolio_size * self.config.exit_rank_buffer_multiplier
        )

        # Get current holdings
        current_holdings = {}
        for ticker in self.etf_universe:
            if ticker in self.data_feeds:
                data_feed = self.data_feeds[ticker]
                position = self.getposition(data_feed)
                if position.size != 0:
                    current_holdings[ticker] = position.size

        # Phase 1: Exit positions that fall below threshold
        tickers_to_exit = []
        for ticker in current_holdings.keys():
            current_rank = next(
                (i for i, (t, _) in enumerate(ranked_etfs) if t == ticker), float("inf")
            )
            if current_rank >= exit_rank_threshold:
                tickers_to_exit.append(ticker)

        # Execute exits with realistic constraints
        for ticker in tickers_to_exit:
            if ticker in self.data_feeds and ticker in current_prices.index:
                data_feed = self.data_feeds[ticker]
                position = self.getposition(data_feed)
                if position.size > 0:
                    self.execute_realistic_sell(
                        ticker,
                        position.size,
                        current_prices,
                        current_volumes,
                        volatilities,
                        rebalance_date,
                    )

        # Phase 2: Calculate target allocations with position size limits
        remaining_tickers = [
            t
            for t in target_tickers
            if t in current_prices.index and t in self.data_feeds
        ]

        if remaining_tickers:
            # Equal weight allocation with max position size constraint
            base_allocation = current_portfolio_value / len(remaining_tickers)
            max_position_value = current_portfolio_value * self.max_position_size
            allocation_per_etf = min(base_allocation, max_position_value)

            # Phase 3: Rebalance positions with realistic constraints
            for ticker in remaining_tickers:
                if ticker in current_prices.index and ticker in self.data_feeds:
                    data_feed = self.data_feeds[ticker]
                    current_position = self.getposition(data_feed)
                    current_shares = current_position.size

                    # Calculate target shares
                    target_shares = allocation_per_etf / current_prices[ticker]
                    shares_diff = target_shares - current_shares

                    if abs(shares_diff) > 0.01:  # Minimum trade threshold
                        if shares_diff > 0:
                            # Buy additional shares
                            self.execute_realistic_buy(
                                ticker,
                                shares_diff,
                                current_prices,
                                current_volumes,
                                volatilities,
                                rebalance_date,
                            )
                        else:
                            # Sell excess shares
                            self.execute_realistic_sell(
                                ticker,
                                abs(shares_diff),
                                current_prices,
                                current_volumes,
                                volatilities,
                                rebalance_date,
                            )

    def _execute_realistic_trade(
        self,
        ticker: str,
        shares: float,
        prices: pd.Series,
        volumes: pd.Series,
        volatilities: Dict[str, float],
        date: datetime,
        action: str,
    ):
        """Executes a trade with realistic market constraints."""
        current_price = prices.get(ticker, 0)
        daily_volume = volumes.get(ticker, 0)
        if len(volumes) == 0:
            daily_volume = 100000  # Default to 100k if no volume data
            logger.warning(
                f"No volume data available for {ticker}. Using default volume."
            )

        volatility = volatilities.get(ticker, 0.2)  # Default 20% volatility

        if current_price <= 0:
            logger.warning(f"Invalid price for {ticker}: {current_price}")
            return

        # Validate liquidity and adjust order size
        is_valid, adjusted_shares = self.validate_liquidity(
            ticker, shares, daily_volume
        )
        if not is_valid:
            logger.warning(f"Order rejected for {ticker}: insufficient liquidity")
            return

        # Estimate slippage and market impact
        slippage = self.market_impact_model.estimate_slippage(
            ticker, adjusted_shares, daily_volume, volatility, action.upper()
        )

        # Calculate execution price with slippage
        if action.upper() == "BUY":
            execution_price = current_price * (1 + slippage)
        else:
            execution_price = current_price * (1 - slippage)

        # Calculate comprehensive trading costs
        cost_breakdown = self.cost_calculator.calculate_costs(
            ticker, adjusted_shares, execution_price, action.upper()
        )

        # Execute in backtrader
        if ticker in self.data_feeds:
            data_feed = self.data_feeds[ticker]
            if action.upper() == "BUY":
                self.buy(data=data_feed, size=adjusted_shares)
            else:
                self.sell(data=data_feed, size=adjusted_shares)

            # Record trade with comprehensive details
            self.record_realistic_trade(
                date,
                ticker,
                action,
                adjusted_shares,
                execution_price,
                cost_breakdown["total_cost"],
                slippage,
            )

            logger.debug(
                f"{action.capitalize()} {adjusted_shares:.2f} shares of {ticker} at ₹{execution_price:.2f} "
                f"(slippage: {slippage:.2%}, costs: ₹{cost_breakdown['total_cost']:.2f})"
            )

    def execute_realistic_buy(
        self,
        ticker: str,
        shares: float,
        prices: pd.Series,
        volumes: pd.Series,
        volatilities: Dict[str, float],
        date: datetime,
    ):
        """Execute a buy order with realistic market constraints."""
        self._execute_realistic_trade(
            ticker, shares, prices, volumes, volatilities, date, "buy"
        )

    def execute_realistic_sell(
        self,
        ticker: str,
        shares: float,
        prices: pd.Series,
        volumes: pd.Series,
        volatilities: Dict[str, float],
        date: datetime,
    ):
        """Execute a sell order with realistic market constraints."""
        self._execute_realistic_trade(
            ticker, shares, prices, volumes, volatilities, date, "sell"
        )

    def record_realistic_trade(
        self,
        date: datetime,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        total_costs: float,
        slippage: float,
    ):
        """Record trade with comprehensive cost breakdown."""
        cost_or_proceeds = shares * price
        slippage_cost = (
            shares * price * slippage if action == "buy" else shares * price * slippage
        )

        self.transaction_costs += total_costs
        self.total_slippage_costs += slippage_cost

        self.trade_log.append(
            {
                "date": date,
                "ticker": ticker,
                "action": action,
                "shares": shares,
                "price": price,
                "value": cost_or_proceeds,
                "total_costs": total_costs,
                "slippage_pct": slippage,
                "slippage_cost": slippage_cost,
                "cost" if action == "buy" else "proceeds": cost_or_proceeds,
            }
        )

    def record_portfolio_state(self, date: datetime, top_etfs: List[Tuple[str, float]]):
        """Record portfolio state for analysis."""
        current_value = self.broker.getvalue()
        current_cash = self.broker.getcash()

        # Get current holdings
        holdings = {}
        for ticker in self.etf_universe:
            if ticker in self.data_feeds:
                data_feed = self.data_feeds[ticker]
                position = self.getposition(data_feed)
                if position.size != 0:
                    holdings[ticker] = position.size

        self.portfolio_history.append(
            {
                "date": date,
                "total_value": current_value,
                "cash": current_cash,
                "holdings": holdings,
                "top_etfs": top_etfs,
            }
        )

    def check_threshold_rebalancing(self):
        """Check if threshold-based rebalancing is needed."""
        if not self.config.use_threshold_rebalancing:
            return

        current_value = self.broker.getvalue()
        period_return = (current_value / self.last_rebalance_value - 1) * 100

        threshold_triggered = False
        if period_return >= self.config.profit_threshold_pct:
            threshold_triggered = True
            logger.debug(
                f"Profit threshold reached: {period_return:.2f}% >= {self.config.profit_threshold_pct}%"
            )
        elif period_return <= self.config.loss_threshold_pct:
            threshold_triggered = True
            logger.debug(
                f"Loss threshold reached: {period_return:.2f}% <= {self.config.loss_threshold_pct}%"
            )

        if threshold_triggered:
            # Trigger immediate rebalancing
            self.rebalance_portfolio()
            self.last_rebalance_value = current_value

    def stop(self):
        """Called when strategy ends."""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.config.initial_capital - 1) * 100
        cost_impact = self.transaction_costs / self.config.initial_capital * 100
        slippage_impact = self.total_slippage_costs / self.config.initial_capital * 100

        logger.info(f"Real-world strategy completed:")
        logger.info(f"  Final portfolio value: ₹{final_value:,.2f}")
        logger.info(f"  Total return: {total_return:.2f}%")
        logger.info(f"  Total transaction costs: ₹{self.transaction_costs:,.2f}")
        logger.info(f"  Total slippage costs: ₹{self.total_slippage_costs:,.2f}")
        logger.info(f"  Cost impact: {cost_impact:.2f}% of initial capital")
        logger.info(f"  Slippage impact: {slippage_impact:.2f}% of initial capital")
        logger.debug(f"  Total trades executed: {len(self.trade_log)}")
        logger.info(
            f"  Average cost per trade: ₹{self.transaction_costs/len(self.trade_log) if self.trade_log else 0:,.2f}"
        )


class BacktraderMomentumStrategy:
    """
    Backtrader wrapper that replicates the exact same strategy logic
    as the original ETFMomentumStrategy implementation.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_provider = DataProvider(config)
        self.momentum_calculator = MomentumCalculator(config)

    def run_backtest(
        self, start_date: datetime, end_date: datetime, show_trade_plot: bool = False
    ) -> Dict:
        """
        Run backtest using backtrader with same logic as original implementation.

        Returns:
            Dictionary containing performance metrics and trade history
        """
        logger.info(
            f"Starting backtrader backtest from {start_date.date()} to {end_date.date()}"
        )

        # Generate rebalancing dates (same as original)
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date)

        # Fetch all historical data (same as original)
        max_lookback = max(
            self.config.long_term_period_days, self.config.moving_average_period
        )
        data_fetch_start = start_date - timedelta(days=max_lookback + 50)

        all_data = self.data_provider.fetch_etf_data(
            self.config.etf_universe, data_fetch_start, end_date
        )

        prices_df_full = self.data_provider.get_prices(all_data)
        if prices_df_full.index.tz is not None:
            prices_df_full.index = prices_df_full.index.tz_localize(None)

        volume_df_full = self.data_provider.get_volumes(all_data)
        if volume_df_full is not None and volume_df_full.index.tz is not None:
            volume_df_full.index = volume_df_full.index.tz_localize(None)

        # Initialize backtrader cerebro
        cerebro = bt.Cerebro()

        # Set initial cash and commission
        cerebro.broker.setcash(self.config.initial_capital)
        cerebro.broker.setcommission(commission=self.config.commission_rate)

        # Add data feeds for each ETF
        etf_feeds = []

        # Use the test date range instead of finding common dates
        test_start_date = start_date
        test_end_date = end_date

        # Filter prices data to test period
        test_prices = prices_df_full[
            (prices_df_full.index >= test_start_date)
            & (prices_df_full.index <= test_end_date)
        ]

        if test_prices.empty:
            raise ValueError("No data available for test period")

        logger.info(
            f"Test data range: {test_prices.index[0].date()} to {test_prices.index[-1].date()}"
        )

        # Create feeds for ETFs with sufficient data for the FULL test period
        for ticker in self.config.etf_universe:
            if ticker in test_prices.columns:
                etf_data = test_prices[ticker].dropna()

                # Check if ETF has data for most of the test period (at least 80%)
                data_coverage = len(etf_data) / len(test_prices)
                min_coverage = 0.8  # Require at least 80% data coverage

                if (
                    len(etf_data) > self.config.min_data_points
                    and data_coverage >= min_coverage
                ):
                    # Create backtrader data feed for the full test period
                    # Fill missing data with forward fill for consistency
                    aligned_data = (
                        test_prices[ticker].reindex(test_prices.index).ffill()
                    )
                    data_feed = self._create_data_feed(
                        aligned_data, ticker, volume_df_full
                    )
                    cerebro.adddata(data_feed, name=ticker)
                    etf_feeds.append(ticker)
                    logger.info(
                        f"Added data feed for {ticker} with {len(etf_data)} data points "
                        f"({data_coverage:.1%} coverage)"
                    )
                else:
                    logger.warning(
                        f"Skipping {ticker} - insufficient data coverage: "
                        f"{len(etf_data)} points ({data_coverage:.1%} coverage)"
                    )

        if not etf_feeds:
            raise ValueError("No valid ETF data feeds created")

        # Add strategy with parameters
        cerebro.addstrategy(
            MomentumPortfolioStrategy,
            config=self.config,
            rebalance_dates=rebalance_dates,
            etf_universe=etf_feeds,
            all_data_prices=prices_df_full,
            all_data_volumes=volume_df_full,
        )

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        # Run backtest
        logger.info("Running backtrader backtest...")
        results = cerebro.run()

        # Generate trade plot if requested
        if show_trade_plot:
            print(f"\n📊 Generating trade plot...")
            cerebro.plot(style="candlestick", barup="green", bardown="red")
            print(f"📁 Trade plot generated.")

        # Extract results
        strategy_result = results[0]

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            strategy_result, cerebro.broker.getvalue()
        )

        return {
            "portfolio_history": strategy_result.portfolio_history,
            "trade_log": strategy_result.trade_log,
            "performance_metrics": performance_metrics,
            "final_value": cerebro.broker.getvalue(),
            "total_return": (
                cerebro.broker.getvalue() / self.config.initial_capital - 1
            )
            * 100,
        }

    def _create_data_feed(
        self,
        price_series: pd.Series,
        ticker: str,
        volume_df: Optional[pd.DataFrame] = None,
    ):
        """Create backtrader data feed from price series."""
        # Ensure we have a proper DatetimeIndex
        if not isinstance(price_series.index, pd.DatetimeIndex):
            price_series.index = pd.to_datetime(price_series.index)

        # Create OHLCV data (using close price for all OHLC)
        df = pd.DataFrame(index=price_series.index)
        df["open"] = price_series.values
        df["high"] = price_series.values
        df["low"] = price_series.values
        df["close"] = price_series.values

        # Use real volume data if available, otherwise use dummy volume
        if volume_df is not None and ticker in volume_df.columns:
            # Get volume data for this ticker and align with price data
            volume_data = volume_df[ticker].reindex(price_series.index)
            # Fill missing volume with forward fill, then with a reasonable default
            volume_data = volume_data.fillna(method="ffill").fillna(1000000)
            df["volume"] = volume_data.values
            logger.info(f"Using real volume data for {ticker}")
        else:
            df["volume"] = 1000000  # Fallback to dummy volume
            logger.warning(
                f"Using dummy volume data for {ticker} - real volume not available"
            )

        # Remove any NaN values - this is important for backtrader
        df = df.dropna()

        if len(df) == 0:
            raise ValueError(f"No valid data for {ticker}")

        logger.info(
            f"Creating data feed for {ticker}: {len(df)} rows, "
            f"from {df.index[0].date()} to {df.index[-1].date()}"
        )

        # Create backtrader data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index as datetime
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )

        return data

    def _generate_rebalance_dates(
        self, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """Generate list of rebalancing dates (same as original)."""
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

        while current <= end_date:
            dates.append(current)
            # Move to next month
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

    def _calculate_performance_metrics(
        self, strategy_result, final_value: float
    ) -> Dict:
        """Calculate performance metrics from backtrader results."""
        portfolio_history = strategy_result.portfolio_history
        trade_log = strategy_result.trade_log

        if not portfolio_history:
            return {}

        # Sort by date
        portfolio_history = sorted(portfolio_history, key=lambda x: x["date"])

        values = [p["total_value"] for p in portfolio_history]
        dates = [p["date"] for p in portfolio_history]

        # Convert to pandas Series
        value_series = pd.Series(values, index=dates)
        returns = value_series.pct_change().dropna()

        # Basic metrics
        total_return = (final_value / self.config.initial_capital - 1) * 100

        if (dates[-1] - dates[0]).days > 0:
            annualized_return = (
                (final_value / self.config.initial_capital)
                ** (365.25 / (dates[-1] - dates[0]).days)
                - 1
            ) * 100
        else:
            annualized_return = 0.0

        # Risk metrics
        if len(returns) > 1:
            avg_days_between_rebalances = (
                (dates[-1] - dates[0]).days / (len(dates) - 1) if len(dates) > 1 else 30
            )
            annualization_factor = (
                365.25 / avg_days_between_rebalances
                if avg_days_between_rebalances > 0
                else 12
            )
            volatility = returns.std() * np.sqrt(annualization_factor) * 100
        else:
            volatility = 0.0

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

        # Sharpe ratio
        risk_free_rate = 0.03
        if len(returns) > 0:
            avg_days_between_rebalances = (
                (dates[-1] - dates[0]).days / (len(dates) - 1) if len(dates) > 1 else 30
            )
            risk_free_rate_per_period = (
                (1 + risk_free_rate) ** (avg_days_between_rebalances / 365.25) - 1
                if avg_days_between_rebalances > 0
                else risk_free_rate / 12
            )
            excess_returns = returns - risk_free_rate_per_period

            annualization_factor = (
                365.25 / avg_days_between_rebalances
                if avg_days_between_rebalances > 0
                else 12
            )
            sharpe_ratio = (
                excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)
                if returns.std() > 0
                else 0
            )
        else:
            sharpe_ratio = 0.0

        # Win ratio and streak calculation
        buy_prices = {}
        win_trades = 0
        total_trades = 0
        trade_outcomes = []

        for trade in trade_log:
            if trade["action"] == "buy":
                ticker = trade["ticker"]
                if ticker not in buy_prices:
                    buy_prices[ticker] = []
                buy_prices[ticker].append((trade["price"], trade["shares"]))

        for trade in trade_log:
            if trade["action"] == "sell":
                ticker = trade["ticker"]
                sell_price = trade["price"]
                sell_shares = trade["shares"]

                if ticker in buy_prices and buy_prices[ticker]:
                    avg_buy_price = 0
                    total_shares_matched = 0
                    shares_to_match = sell_shares

                    used_shares_list = []
                    i = 0
                    while shares_to_match > 0 and i < len(buy_prices[ticker]):
                        buy_price, buy_shares = buy_prices[ticker][i]
                        shares_from_this_buy = min(buy_shares, shares_to_match)

                        avg_buy_price += buy_price * shares_from_this_buy
                        total_shares_matched += shares_from_this_buy
                        shares_to_match -= shares_from_this_buy

                        used_shares_list.append((i, shares_from_this_buy))
                        i += 1

                    # Update buy_prices
                    for idx, used_shares in reversed(used_shares_list):
                        buy_price, buy_shares = buy_prices[ticker][idx]
                        remaining_shares = buy_shares - used_shares
                        if remaining_shares <= 0.01:
                            buy_prices[ticker].pop(idx)
                        else:
                            buy_prices[ticker][idx] = (buy_price, remaining_shares)

                    if total_shares_matched > 0:
                        avg_buy_price /= total_shares_matched
                        if sell_price > avg_buy_price:
                            win_trades += 1
                            trade_outcomes.append(1)
                        else:
                            trade_outcomes.append(-1)
                        total_trades += 1

        win_ratio = (win_trades / total_trades * 100) if total_trades > 0 else 0

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

        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "volatility_pct": volatility,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(trade_log),
            "transaction_costs": strategy_result.transaction_costs,
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
        }


def run_backtest(
    config: StrategyConfig = None,
    start_date: datetime = None,
    end_date: datetime = None,
    create_charts: bool = True,
    show_trade_plot: bool = False,
) -> Dict:
    """
    Run the ETF Momentum Strategy backtest using Backtrader.

    Args:
        config: Strategy configuration
        start_date: Start date for backtest
        end_date: End date for backtest
        create_charts: Whether to create performance charts

    Returns:
        Dictionary containing backtest results and performance metrics
    """
    # Use default parameters if not provided
    if config is None:
        config = StrategyConfig(
            portfolio_size=7,
            rebalance_day_of_month=5,
            exit_rank_buffer_multiplier=2.0,
            long_term_period_days=252,
            short_term_period_days=60,
            initial_capital=1000000,
            use_threshold_rebalancing=False,
        )

    if start_date is None:
        start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime(2024, 12, 31)

    print(f"\n{'='*80}")
    print("ETF MOMENTUM STRATEGY - BACKTRADER IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial Capital: ₹{config.initial_capital:,.0f}")
    print(f"📊 Portfolio Size: {config.portfolio_size}")
    print(f"🗓️ Rebalance Day: {config.rebalance_day_of_month}")
    print(f"{'='*80}")

    # Run backtrader strategy
    strategy = BacktraderMomentumStrategy(config)
    results = strategy.run_backtest(
        start_date, end_date, show_trade_plot=show_trade_plot
    )

    # Display results
    print(f"\n🎯 BACKTEST RESULTS:")
    print(f"{'='*50}")
    print(f"💰 Final Value: ₹{results['final_value']:,.2f}")
    print(f"📈 Total Return: {results['total_return']:.2f}%")
    print(
        f"📊 Annualized Return: {results['performance_metrics']['annualized_return_pct']:.2f}%"
    )
    print(f"🎲 Volatility: {results['performance_metrics']['volatility_pct']:.2f}%")
    print(f"📉 Max Drawdown: {results['performance_metrics']['max_drawdown_pct']:.2f}%")
    print(f"⚡ Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"🔄 Total Trades: {results['performance_metrics']['total_trades']}")
    print(f"🏆 Win Ratio: {results['performance_metrics']['win_ratio_pct']:.1f}%")
    print(
        f"💸 Transaction Costs: ₹{results['performance_metrics']['transaction_costs']:,.2f}"
    )
    recovery_days = results["performance_metrics"]["days_to_recovery"]
    if isinstance(recovery_days, int):
        print(f"🔄 Recovery Days: {recovery_days}")
    else:
        print(f"🔄 Recovery Days: {recovery_days}")
    print(f"{'='*50}")

    # Generate performance charts if requested
    if create_charts:
        print(f"\n📊 Generating performance charts...")
        create_performance_charts(results, config, start_date, end_date)
        print(f"📁 Chart files saved in current directory")

    print(f"\n✅ Backtest completed successfully!")
    return results


def run_backtrader_experiments(
    portfolio_sizes: List[int] = None,
    rebalance_days: List[int] = None,
    exit_rank_buffers: List[float] = None,
    long_term_periods: List[int] = None,
    short_term_periods: List[int] = None,
    initial_capitals: List[float] = None,
    use_threshold_rebalancing_values: List[bool] = None,
    profit_threshold_pct: float = 10.0,
    loss_threshold_pct: float = -5.0,
    start_date: datetime = None,
    end_date: datetime = None,
    max_workers: int = 4,  # For parallel execution
) -> List[Dict]:
    """
    Run Backtrader backtest with multiple parameter permutations in parallel.
    Generates a summary report of performance metrics for each permutation.
    """
    # Default values for experiments if not provided
    portfolio_sizes = portfolio_sizes or [5]
    rebalance_days = rebalance_days or [5]
    exit_rank_buffers = exit_rank_buffers or [2.0]
    long_term_periods = long_term_periods or [180]
    short_term_periods = short_term_periods or [60]
    initial_capitals = initial_capitals or [100000]
    use_threshold_rebalancing_values = use_threshold_rebalancing_values or [False]

    if start_date is None:
        end_date_default = datetime.now()
        start_date = datetime(end_date_default.year - 5, 1, 1)
    if end_date is None:
        end_date = datetime.now()

    all_results = []

    # Create all combinations of parameters
    param_combinations = list(
        itertools.product(
            portfolio_sizes,
            rebalance_days,
            exit_rank_buffers,
            long_term_periods,
            short_term_periods,
            initial_capitals,
            use_threshold_rebalancing_values,
        )
    )

    print(
        f"\nRunning {len(param_combinations)} Backtrader backtest permutations in parallel..."
    )
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*120}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {}
        for (
            p_size,
            r_day,
            exit_buffer,
            lt_period,
            st_period,
            init_capital,
            use_threshold,
        ) in param_combinations:
            config = StrategyConfig(
                portfolio_size=p_size,
                rebalance_day_of_month=r_day,
                exit_rank_buffer_multiplier=exit_buffer,
                long_term_period_days=lt_period,
                short_term_period_days=st_period,
                initial_capital=init_capital,
                use_threshold_rebalancing=use_threshold,
                profit_threshold_pct=profit_threshold_pct,
                loss_threshold_pct=loss_threshold_pct,
            )
            # Pass create_charts=False and show_trade_plot=False for experiments
            future = executor.submit(
                run_backtest,
                config,
                start_date,
                end_date,
                create_charts=False,
                show_trade_plot=False,
            )
            future_to_params[future] = (
                p_size,
                r_day,
                exit_buffer,
                lt_period,
                st_period,
                init_capital,
                use_threshold,
            )

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result:
                    all_results.append(
                        {
                            "portfolio_size": params[0],
                            "rebalance_day": params[1],
                            "exit_buffer": params[2],
                            "long_term_period_days": params[3],
                            "short_term_period_days": params[4],
                            "initial_capital": params[5],
                            "use_threshold_rebalancing": params[6],
                            "final_value": result["final_value"],
                            "total_return_pct": result["total_return"],
                            "annualized_return_pct": result["performance_metrics"].get(
                                "annualized_return_pct", 0.0
                            ),
                            "sharpe_ratio": result["performance_metrics"].get(
                                "sharpe_ratio", 0.0
                            ),
                            "max_drawdown_pct": result["performance_metrics"].get(
                                "max_drawdown_pct", 0.0
                            ),
                            "total_trades": result["performance_metrics"].get(
                                "total_trades", 0
                            ),
                            "transaction_costs": result["performance_metrics"].get(
                                "transaction_costs", 0.0
                            ),
                            "win_ratio_pct": result["performance_metrics"].get(
                                "win_ratio_pct", 0.0
                            ),
                            "days_to_recovery": result["performance_metrics"].get(
                                "days_to_recovery", "Not Recovered"
                            ),
                        }
                    )
            except Exception as e:
                logger.error(f"Backtest failed for params {params}: {e}", exc_info=True)

    # Print overall comparison summary for all permutations
    print(f"\n{'='*120}")
    print("BACKTRADER PERMUTATION COMPARISON SUMMARY")
    print(f"{'='*120}")

    # Prepare data for tabulate
    summary_table_data = []
    headers = [
        "Port. Size",
        "Rebal. Day",
        "Exit Buffer",
        "LT/ST Period",
        "Init. Capital",
        "Threshold Rebal",
        "Ann. Return",
        "Sharpe",
        "Max DD",
        "Win Ratio %",
        "Total Trades",
        "Txn Costs",
        "Recovery Days",
    ]
    for result in all_results:
        summary_table_data.append(
            [
                result["portfolio_size"],
                result["rebalance_day"],
                result["exit_buffer"],
                f"{result['long_term_period_days']}d/{result['short_term_period_days']}d",
                f"₹{result['initial_capital']:,.0f}",
                result["use_threshold_rebalancing"],
                f"{result['annualized_return_pct']:.1f}%",
                f"{result['sharpe_ratio']:.2f}",
                f"{result['max_drawdown_pct']:.1f}%",
                f"{result['win_ratio_pct']:.1f}%",
                result["total_trades"],
                f"₹{result['transaction_costs']:,.0f}",
                (
                    result["days_to_recovery"]
                    if isinstance(result["days_to_recovery"], int)
                    else str(result["days_to_recovery"])
                ),
            ]
        )

    # Sort results by annualized return (descending)
    summary_table_data_sorted = sorted(
        summary_table_data, key=lambda x: float(x[6].replace("%", "")), reverse=True
    )

    print(tabulate(summary_table_data_sorted, headers=headers, tablefmt="grid"))

    return all_results


def unified_cli():
    """
    Smart CLI that automatically detects single vs multiple parameter experiments.
    """
    parser = argparse.ArgumentParser(
        description="Run Backtrader ETF Momentum Strategy backtest - automatically detects single run vs experiments based on parameters provided."
    )

    # Strategy parameters - accept lists using nargs='*' (0 or more) or nargs='+' (1 or more)
    parser.add_argument(
        "--portfolio-sizes",
        nargs="+",
        type=int,
        default=[5],
        help="Portfolio size(s) to test. Single value = single run, multiple values = experiments (default: [5])",
    )
    parser.add_argument(
        "--rebalance-days",
        nargs="+",
        type=int,
        default=[5],
        help="Rebalance day(s) to test (default: [5])",
    )
    parser.add_argument(
        "--exit-rank-buffers",
        nargs="+",
        type=float,
        default=[2.0],
        help="Exit rank buffer(s) to test (default: [2.0])",
    )
    parser.add_argument(
        "--long-term-periods",
        nargs="+",
        type=int,
        default=[180],
        help="Long-term momentum period(s) in days (default: [180])",
    )
    parser.add_argument(
        "--short-term-periods",
        nargs="+",
        type=int,
        default=[60],
        help="Short-term momentum period(s) in days (default: [60])",
    )
    parser.add_argument(
        "--initial-capitals",
        nargs="+",
        type=float,
        default=[100000],
        help="Initial capital(s) to test (default: [100000])",
    )
    parser.add_argument(
        "--use-threshold-rebalancing-values",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False],
        help="Threshold rebalancing value(s) (True/False) (default: [False])",
    )
    parser.add_argument(
        "--profit-threshold",
        type=float,
        default=10.0,
        help="Profit threshold percent for rebalancing (default: 10.0)",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=-5.0,
        help="Loss threshold percent for rebalancing (default: -5.0)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now().year - 5, 1, 1),
        help="Backtest start date in YYYY-MM-DD format (default: 5 years ago)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Backtest end date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Do not generate performance charts",
    )
    parser.add_argument(
        "--show-trade-plot",
        action="store_true",
        help="Show interactive trade plot (requires matplotlib backend)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for experiments (default: 4)",
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = (
            datetime.strptime(args.start_date, "%Y-%m-%d")
            if isinstance(args.start_date, str)
            else datetime(*args.start_date)
        )
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        return

    # Auto-detect if this is a single run or experiments
    total_combinations = (
        len(args.portfolio_sizes)
        * len(args.rebalance_days)
        * len(args.exit_rank_buffers)
        * len(args.long_term_periods)
        * len(args.short_term_periods)
        * len(args.initial_capitals)
        * len(args.use_threshold_rebalancing_values)
    )

    if total_combinations == 1:
        # Single backtest - extract single values
        print("🎯 Detected single backtest configuration")
        config = StrategyConfig(
            portfolio_size=args.portfolio_sizes[0],
            rebalance_day_of_month=args.rebalance_days[0],
            exit_rank_buffer_multiplier=args.exit_rank_buffers[0],
            long_term_period_days=args.long_term_periods[0],
            short_term_period_days=args.short_term_periods[0],
            initial_capital=args.initial_capitals[0],
            use_threshold_rebalancing=args.use_threshold_rebalancing_values[0],
            profit_threshold_pct=args.profit_threshold,
            loss_threshold_pct=args.loss_threshold,
        )

        try:
            run_backtest(
                config=config,
                start_date=start_date,
                end_date=end_date,
                create_charts=not args.no_charts,
                show_trade_plot=args.show_trade_plot,
            )
            print("\n🎉 Backtest execution completed successfully!")
        except Exception as e:
            print(f"\n❌ Error running backtest: {str(e)}")
            logger.error(f"Backtest execution failed: {str(e)}", exc_info=True)

    else:
        # Multiple combinations - run experiments
        print(
            f"🧪 Detected experiments configuration ({total_combinations} combinations)"
        )
        try:
            run_backtrader_experiments(
                portfolio_sizes=args.portfolio_sizes,
                rebalance_days=args.rebalance_days,
                exit_rank_buffers=args.exit_rank_buffers,
                long_term_periods=args.long_term_periods,
                short_term_periods=args.short_term_periods,
                initial_capitals=args.initial_capitals,
                use_threshold_rebalancing_values=args.use_threshold_rebalancing_values,
                profit_threshold_pct=args.profit_threshold,
                loss_threshold_pct=args.loss_threshold,
                start_date=start_date,
                end_date=end_date,
                max_workers=args.max_workers,
            )
            print("\n🎉 Backtrader experiments completed successfully!")
        except Exception as e:
            print(f"\n❌ Error running Backtrader experiments: {str(e)}")
            logger.error(f"Backtrader experiments failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    unified_cli()
