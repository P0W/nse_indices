"""
Backtrader implementation of ETF Momentum Strategy.

"""

import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from tabulate import tabulate
import warnings

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Import utilities from the original implementation
from etf_momentum_strategy import (
    StrategyConfig,
    DataProvider,
    MomentumCalculator,
    logger,
)

warnings.filterwarnings("ignore")


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
                    logger.info(
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
            logger.info(
                f"Adjusting order size for {ticker}: {shares:.2f} -> {max_shares:.2f} shares"
            )
            return True, max_shares

    def rebalance_portfolio(self):
        """Rebalance portfolio using the same logic as original implementation."""
        current_datetime = self.datetime.datetime()
        current_date = current_datetime.date()

        logger.info(f"Rebalancing portfolio on {current_date}")

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

        logger.info(
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

            logger.info(
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
            logger.info(
                f"Profit threshold reached: {period_return:.2f}% >= {self.config.profit_threshold_pct}%"
            )
        elif period_return <= self.config.loss_threshold_pct:
            threshold_triggered = True
            logger.info(
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
        logger.info(f"  Total trades executed: {len(self.trade_log)}")
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
    print(f"{'='*50}")

    # Generate performance charts if requested
    if create_charts:
        print(f"\n📊 Generating performance charts...")
        create_performance_charts(results, config, start_date, end_date)
        print(f"📁 Chart files saved in current directory")

    print(f"\n✅ Backtest completed successfully!")
    return results


def create_performance_charts(
    results: Dict, config: StrategyConfig, start_date: datetime, end_date: datetime
):
    """Create and save performance visualization charts."""
    portfolio_history = results["portfolio_history"]
    trade_log = results["trade_log"]

    if not portfolio_history:
        print("❌ No portfolio history data available for charting")
        return

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Sort portfolio history by date
    portfolio_history = sorted(portfolio_history, key=lambda x: x["date"])

    # Extract data for plotting
    dates = [p["date"] for p in portfolio_history]
    values = [p["total_value"] for p in portfolio_history]
    cash = [p["cash"] for p in portfolio_history]

    # Convert to pandas for easier manipulation
    df = pd.DataFrame({"date": dates, "total_value": values, "cash": cash})
    df["invested"] = df["total_value"] - df["cash"]
    df["returns"] = df["total_value"].pct_change() * 100
    df["cumulative_returns"] = ((df["total_value"] / config.initial_capital) - 1) * 100

    # Calculate rolling drawdown
    df["peak"] = df["total_value"].expanding().max()
    df["drawdown"] = ((df["total_value"] / df["peak"]) - 1) * 100

    # Calculate additional metrics
    metrics = results["performance_metrics"]

    # Create main performance dashboard
    create_main_dashboard(df, config, results, start_date, end_date)

    # Create detailed analysis charts
    create_detailed_analysis(df, trade_log, config, results, start_date, end_date)


def create_main_dashboard(df, config, results, start_date, end_date):
    """Create the main performance dashboard with 6 key charts."""
    # Create a figure with better spacing
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")

    # Add main title with proper spacing
    fig.suptitle(
        f"ETF Momentum Strategy - Performance Dashboard\n"
        f'Period: {start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")} | '
        f"Strategy: Backtrader Implementation\n"
        f'🎯 Final Value: ₹{results["final_value"]:,.0f} | '
        f'📈 Total Return: {results["total_return"]:.1f}% | '
        f'🔥 Annualized: {results["performance_metrics"].get("annualized_return_pct", 0):.1f}%',
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create subplots with proper spacing - increased top margin to prevent overlap
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.83, bottom=0.08)

    # Subplot 1: Portfolio Value Over Time (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(
        df["date"],
        df["total_value"],
        linewidth=3,
        color="#2E8B57",
        label="Portfolio Value",
        alpha=0.9,
    )
    ax1.axhline(
        y=config.initial_capital,
        color="#FF4444",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Initial Capital",
    )

    # Add milestone markers
    max_value = df["total_value"].max()
    max_date = df.loc[df["total_value"].idxmax(), "date"]
    ax1.scatter(
        max_date,
        max_value,
        color="gold",
        s=100,
        zorder=5,
        label=f"Peak: ₹{max_value:,.0f}",
    )

    ax1.set_title("📈 Portfolio Growth Journey", fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle="-")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x/100000:.1f}L"))

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Key Performance Metrics (right side)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    metrics_text = [
        f"🎯 Total Return: {results['total_return']:.1f}%",
        f"📊 Annualized Return: {results['performance_metrics'].get('annualized_return_pct', 0):.1f}%",
        f"📈 Sharpe Ratio: {results['performance_metrics'].get('sharpe_ratio', 0):.2f}",
        f"📉 Max Drawdown: {results['performance_metrics'].get('max_drawdown_pct', 0):.1f}%",
        f"🎲 Volatility: {results['performance_metrics'].get('volatility_pct', 0):.1f}%",
        f"🔄 Total Trades: {results['performance_metrics'].get('total_trades', 0)}",
        f"🏆 Win Ratio: {results['performance_metrics'].get('win_ratio_pct', 0):.1f}%",
        f"💰 Transaction Costs: ₹{results['performance_metrics'].get('transaction_costs', 0):,.0f}",
    ]

    # Create a nice metrics box
    metrics_box = "\n".join(metrics_text)
    ax2.text(
        0.1,
        0.95,
        "📊 Key Metrics",
        fontsize=16,
        fontweight="bold",
        transform=ax2.transAxes,
        verticalalignment="top",
    )
    ax2.text(
        0.1,
        0.85,
        metrics_box,
        fontsize=12,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
    )

    # 3. Cumulative Returns
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(
        df["date"],
        df["cumulative_returns"],
        alpha=0.6,
        color="#4CAF50",
        label="Cumulative Returns",
    )
    ax3.plot(df["date"], df["cumulative_returns"], linewidth=2, color="#2E8B57")
    ax3.set_title("📈 Cumulative Returns", fontsize=14, fontweight="bold", pad=15)
    ax3.set_ylabel("Returns (%)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(
        df["date"], df["drawdown"], alpha=0.6, color="#FF6B6B", label="Drawdown"
    )
    ax4.plot(df["date"], df["drawdown"], linewidth=2, color="#D32F2F")
    ax4.set_title("📉 Drawdown Analysis", fontsize=14, fontweight="bold", pad=15)
    ax4.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Rolling Volatility (30-day)
    ax5 = fig.add_subplot(gs[1, 2])
    rolling_vol = df["returns"].rolling(window=30).std() * np.sqrt(252)  # Annualized
    ax5.plot(df["date"], rolling_vol, linewidth=2, color="#FF9800", alpha=0.8)
    ax5.set_title("📊 Rolling Volatility (30D)", fontsize=14, fontweight="bold", pad=15)
    ax5.set_ylabel("Volatility (%)", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Monthly Returns Heatmap
    ax6 = fig.add_subplot(gs[2, :2])
    # Create monthly returns
    df_monthly = df.set_index("date").resample("M")["returns"].sum()
    df_monthly.index = pd.to_datetime(df_monthly.index)

    # Create pivot table for heatmap
    monthly_pivot = df_monthly.to_frame()
    monthly_pivot["year"] = monthly_pivot.index.year
    monthly_pivot["month"] = monthly_pivot.index.month
    heatmap_data = monthly_pivot.pivot_table(
        values="returns", index="year", columns="month", aggfunc="first"
    )

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Monthly Returns (%)"},
        ax=ax6,
    )
    ax6.set_title("🔥 Monthly Returns Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax6.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Year", fontsize=11, fontweight="bold")

    # 7. Performance Summary (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    # Performance summary text
    metrics = results["performance_metrics"]
    summary_text = [
        f"🏁 Strategy Performance Summary",
        f"",
        f"Initial Capital: ₹{config.initial_capital:,.0f}",
        f"Final Value: ₹{results['final_value']:,.0f}",
        f"Profit/Loss: ₹{results['final_value'] - config.initial_capital:,.0f}",
        f"",
        f"📊 Risk & Trade Metrics:",
        f"• Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}% (₹{metrics.get('max_drawdown_amount', 0):,.0f})",
        f"  - Trough Value: ₹{metrics.get('trough_value_at_mdd', 0):,.0f} on {metrics.get('mdd_end_date', datetime.now()).strftime('%Y-%m-%d')}",
        f"• Recovery Days: {metrics.get('days_to_recovery', 'N/A')}",
        f"  - Recovery Date: {metrics.get('recovery_date', 'N/A').strftime('%Y-%m-%d') if isinstance(metrics.get('recovery_date'), datetime) else metrics.get('recovery_date', 'N/A')}",
        f"• Best Month: {df_monthly.max():.1f}%",
        f"• Worst Month: {df_monthly.min():.1f}%",
        f"• Positive Months: {(df_monthly > 0).sum()}/{len(df_monthly)}",
        f"• Max Win Streak: {metrics.get('max_win_streak', 0)} trades",
        f"• Max Loss Streak: {metrics.get('max_loss_streak', 0)} trades",
        f"",
        f"⚙️ Strategy Config:",
        f"• Portfolio Size: {config.portfolio_size}",
        f"• Rebalance Day: {config.rebalance_day_of_month}",
        f"• Long Period: {config.long_term_period_days}d",
        f"• Short Period: {config.short_term_period_days}d",
    ]

    summary_box = "\n".join(summary_text)
    ax7.text(
        0.05,
        0.95,
        summary_box,
        fontsize=11,
        transform=ax7.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
    )

    # Save the main dashboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtrader_dashboard_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"📊 Main dashboard saved as: {filename}")


def create_detailed_analysis(df, trade_log, config, results, start_date, end_date):
    """Create detailed analysis charts with additional insights."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")

    # Add title with proper spacing
    fig.suptitle(
        f"ETF Momentum Strategy - Detailed Analysis\n"
        f'Period: {start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")} | '
        f"Advanced Risk & Performance Analytics",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    # Create subplots with proper spacing - increased top margin to prevent overlap
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, top=0.84, bottom=0.08)

    # 1. Risk-Return Scatter (Rolling analysis)
    ax1 = fig.add_subplot(gs[0, 0])

    # Adjust rolling window size based on available data
    data_length = len(df)
    window_size = min(60, max(20, data_length // 3))  # Use smaller window if needed

    rolling_returns = (
        df["returns"].rolling(window=window_size, min_periods=10).mean() * 252
    )  # Annualized
    rolling_vol = df["returns"].rolling(
        window=window_size, min_periods=10
    ).std() * np.sqrt(
        252
    )  # Annualized

    # Remove NaN values for plotting
    valid_data = pd.DataFrame(
        {"returns": rolling_returns, "volatility": rolling_vol}
    ).dropna()

    if len(valid_data) > 0:
        # Create scatter plot with color based on time
        scatter = ax1.scatter(
            valid_data["volatility"],
            valid_data["returns"],
            c=range(len(valid_data)),
            cmap="viridis",
            alpha=0.7,
            s=30,
        )

        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Time Period", fontsize=10)

        # Set appropriate axis limits
        ax1.set_xlim(
            valid_data["volatility"].min() * 0.95, valid_data["volatility"].max() * 1.05
        )
        ax1.set_ylim(
            valid_data["returns"].min() * 0.95, valid_data["returns"].max() * 1.05
        )
    else:
        # If no valid data, show a message
        ax1.text(
            0.5,
            0.5,
            f"Insufficient data for\n{window_size}-day rolling analysis\n({data_length} data points available)",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
            style="italic",
        )

    ax1.set_xlabel("Volatility (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Returns (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"🎯 Risk-Return Profile ({window_size}D Rolling)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(True, alpha=0.3)

    # 2. Rolling Sharpe Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    rolling_sharpe = rolling_returns / rolling_vol

    # Remove NaN values for plotting
    valid_sharpe_data = pd.DataFrame(
        {"date": df["date"], "sharpe": rolling_sharpe}
    ).dropna()

    if len(valid_sharpe_data) > 0:
        ax2.plot(
            valid_sharpe_data["date"],
            valid_sharpe_data["sharpe"],
            linewidth=2,
            color="#9C27B0",
            alpha=0.8,
        )
        ax2.axhline(
            y=1.0, color="#FF5722", linestyle="--", alpha=0.8, label="Good (1.0)"
        )
        ax2.axhline(
            y=2.0, color="#4CAF50", linestyle="--", alpha=0.8, label="Excellent (2.0)"
        )
        ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(
            0.5,
            0.5,
            f"Insufficient data for\n{window_size}-day rolling analysis\n({data_length} data points available)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
            style="italic",
        )

    ax2.set_title(
        f"📈 Rolling Sharpe Ratio ({window_size}D)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Trade Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    if trade_log:
        # Analyze trade sizes and timing
        trade_df = pd.DataFrame(trade_log)
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["value"] = trade_df["shares"] * trade_df["price"]

        # Group by month
        monthly_trades = trade_df.groupby(trade_df["date"].dt.to_period("M"))[
            "value"
        ].sum()

        # Create bar plot
        bars = ax3.bar(
            range(len(monthly_trades)),
            monthly_trades.values,
            color="#607D8B",
            alpha=0.7,
        )
        ax3.set_title("💱 Monthly Trade Volume", fontsize=14, fontweight="bold", pad=15)
        ax3.set_ylabel("Trade Value (₹)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Month", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        # Format y-axis
        ax3.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"₹{x/100000:.1f}L")
        )

        # Set x-axis labels
        ax3.set_xticks(range(0, len(monthly_trades), max(1, len(monthly_trades) // 6)))
        ax3.set_xticklabels(
            [
                str(monthly_trades.index[i])
                for i in range(0, len(monthly_trades), max(1, len(monthly_trades) // 6))
            ],
            rotation=45,
        )
    else:
        ax3.text(
            0.5,
            0.5,
            "No trade data available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("💱 Monthly Trade Volume", fontsize=14, fontweight="bold", pad=15)

    # 4. Returns Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    returns_clean = df["returns"].dropna()
    ax4.hist(returns_clean, bins=30, alpha=0.7, color="#3F51B5", edgecolor="black")
    ax4.axvline(
        returns_clean.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {returns_clean.mean():.2f}%",
    )
    ax4.axvline(
        returns_clean.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {returns_clean.median():.2f}%",
    )
    ax4.set_title(
        "📊 Daily Returns Distribution", fontsize=14, fontweight="bold", pad=15
    )
    ax4.set_xlabel("Daily Returns (%)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)

    # 5. Underwater Curve (Drawdown from Peak)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(
        df["date"],
        df["drawdown"],
        alpha=0.6,
        color="#FF5722",
        label="Drawdown from Peak",
    )
    ax5.plot(df["date"], df["drawdown"], linewidth=2, color="#D32F2F")
    ax5.set_title("🌊 Underwater Curve", fontsize=14, fontweight="bold", pad=15)
    ax5.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Portfolio Allocation Over Time
    ax6 = fig.add_subplot(gs[1, 2])
    # Calculate allocation percentages
    allocation_pct = (df["invested"] / df["total_value"]) * 100
    cash_pct = (df["cash"] / df["total_value"]) * 100

    ax6.fill_between(
        df["date"], 0, allocation_pct, alpha=0.6, color="#4CAF50", label="Invested"
    )
    ax6.fill_between(
        df["date"], allocation_pct, 100, alpha=0.6, color="#FFC107", label="Cash"
    )
    ax6.set_title("🥧 Portfolio Allocation", fontsize=14, fontweight="bold", pad=15)
    ax6.set_ylabel("Allocation (%)", fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 100)
    ax6.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    # Save the detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtrader_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"📈 Detailed analysis saved as: {filename}")


if __name__ == "__main__":
    """
    Main execution block for running the ETF Momentum Strategy.

    Example usage:
        python backtrader_momentum_strategy.py
    """
    # Default configuration
    config = StrategyConfig(
        portfolio_size=5,
        rebalance_day_of_month=5,
        exit_rank_buffer_multiplier=2.0,
        long_term_period_days=180,
        short_term_period_days=60,
        initial_capital=100000,
        use_threshold_rebalancing=False,
    )

    # Default date range - last 5 years
    end_date = datetime.now()
    start_date = datetime(end_date.year - 5, 1, 1)

    # Run the backtest
    try:
        results = run_backtest(
            config, start_date, end_date, create_charts=True, show_trade_plot=False
        )
        print("\n🎉 Strategy execution completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running strategy: {str(e)}")
        logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
