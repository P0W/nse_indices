"""
Nifty Shop Strategy

A simplified momentum-based buy-and-hold strategy that:
1. Buys stocks when they fall below 20-day moving average
2. Sells when profit target is reached (3%)
3. Uses fixed position sizing with averaging down capability
4. Implements smart position management with loser exit rules
"""

import backtrader as bt
import logging
from datetime import datetime, time
from collections import defaultdict
from typing import Dict, Any, List

from .base_strategy import BaseStrategy, StrategyConfig


class NiftyShopStrategy(BaseStrategy):
    """
    Simplified Nifty Shop Strategy focusing on core trading logic
    """

    params = (
        ("sell_threshold", 0.03),  # 3% sell target
        ("averaging_threshold", 0.03),  # 3% averaging threshold
        ("ma_period", 20),  # 20-day moving average
        ("fixed_size", 15000),  # Fixed ₹15,000 per trade
        ("max_new_buys", 2),  # Max 2 fresh buys per day
        ("max_averaging_attempts", 2),  # Max 2 averaging attempts per stock
        ("loser_exit_threshold", 0.25),  # Exit if down 25% from avg price
        ("loser_exit_days", 60),  # Exit losers held for 60+ days
        ("momentum_lookback", 5),  # 5-day momentum check for averaging
    )

    def __init__(self):
        super().__init__()
        self.portfolio = (
            {}
        )  # {symbol: {'shares': int, 'avg_price': float, 'averaging_count': int, 'entry_date': datetime}}

        # Moving average indicators
        self.ma = {
            d: bt.indicators.SMA(d, period=self.params.ma_period) for d in self.datas
        }

        # Momentum indicators for averaging filter
        self.momentum = {
            d: bt.indicators.ROC(d, period=self.params.momentum_lookback)
            for d in self.datas
        }

        self.daily_executed = False

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            symbol = order.data._name
            current_date = self.datas[0].datetime.datetime(0)

            if order.isbuy():
                if symbol not in self.portfolio:
                    self.portfolio[symbol] = {
                        "shares": 0,
                        "avg_price": 0.0,
                        "averaging_count": 0,
                        "entry_date": current_date,
                    }

                # Update portfolio data
                prev_shares = self.portfolio[symbol]["shares"]
                prev_avg = self.portfolio[symbol]["avg_price"]
                new_shares = order.executed.size
                new_price = order.executed.price

                total_shares = prev_shares + new_shares
                total_cost = (prev_shares * prev_avg) + (new_shares * new_price)
                self.portfolio[symbol]["shares"] = total_shares
                self.portfolio[symbol]["avg_price"] = total_cost / total_shares

                if order.info and order.info.get("averaging", False):
                    self.portfolio[symbol]["averaging_count"] += 1

                self.log(
                    f"BUY {symbol}: {new_shares} shares @ ₹{new_price:.2f}, "
                    f"Avg Price: ₹{self.portfolio[symbol]['avg_price']:.2f}"
                )

            elif order.issell():
                if symbol in self.portfolio:
                    self.log(
                        f"SELL {symbol}: {order.executed.size} shares @ ₹{order.executed.price:.2f}"
                    )
                    del self.portfolio[symbol]

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.status != order.Margin:  # Don't log margin errors
                symbol = order.data._name if hasattr(order.data, "_name") else "Unknown"
                self.log(f"Order for {symbol} failed: {order.status}")

    def get_position_size(self, price):
        """Calculate position size based on fixed amount"""
        available_cash = self.broker.getcash()
        max_size = self.params.fixed_size // price
        affordable_size = (available_cash * 0.9) // price  # Keep 10% cash buffer
        return min(max_size, affordable_size, 500)  # Cap at 500 shares max

    def get_data_feed(self, symbol):
        """Helper method to get data feed for a symbol"""
        for d in self.datas:
            if d._name == symbol:
                return d
        return None

    def is_chronic_loser(self, symbol, data):
        """Check if a position should be exited as chronic loser"""
        if data["shares"] <= 0:
            return False

        data_feed = self.get_data_feed(symbol)
        if not data_feed:
            return False

        try:
            current_price = data_feed.close[0]
            current_date = self.datas[0].datetime.datetime(0)

            # Check loss percentage
            loss_pct = (data["avg_price"] - current_price) / data["avg_price"]
            if loss_pct >= self.params.loser_exit_threshold:
                return True

            # Check if held too long while losing
            entry_date = data.get("entry_date", current_date)
            days_held = (current_date - entry_date).days
            if days_held >= self.params.loser_exit_days and loss_pct > 0.05:
                return True

        except (IndexError, AttributeError, KeyError):
            pass

        return False

    def has_recent_momentum(self, symbol):
        """Check if stock has recent positive momentum for averaging"""
        data_feed = self.get_data_feed(symbol)
        if not data_feed or data_feed not in self.momentum:
            return False

        try:
            momentum_value = self.momentum[data_feed][0]
            return (
                momentum_value > -2.0
            )  # Allow averaging if not falling more than 2% in 5 days
        except (IndexError, AttributeError):
            return False

    def get_buy_signal_strength(self, data_feed):
        """Simple MA-based buy signal - returns 1 if price below 20-day MA"""
        if data_feed._name == "^NSEI":  # Skip index
            return 0

        try:
            if len(self.ma[data_feed]) > 1:
                ma_value = self.ma[data_feed][0]
                current_price = data_feed.close[0]
                return 1 if current_price < ma_value else 0
            return 0
        except (IndexError, AttributeError):
            return 0

    def get_sell_signal_strength(self, data_feed, current_gain):
        """Simple profit target sell signal"""
        try:
            return current_gain >= self.params.sell_threshold
        except (IndexError, AttributeError, KeyError):
            return True  # Default to selling if calculation fails

    def execute_strategy(self):
        """Main strategy execution logic"""
        # Initialize portfolio entries for all data feeds
        for d in self.datas:
            if d._name not in self.portfolio:
                self.portfolio[d._name] = {
                    "shares": 0,
                    "avg_price": 0.0,
                    "averaging_count": 0,
                    "entry_date": self.datas[0].datetime.datetime(0),
                }

        # STEP 1: Exit chronic losers first
        chronic_losers = []
        for symbol, data in self.portfolio.items():
            if data["shares"] > 0 and self.is_chronic_loser(symbol, data):
                chronic_losers.append(symbol)

        if chronic_losers:
            symbol = chronic_losers[0]  # Exit first chronic loser
            shares = self.portfolio[symbol]["shares"]
            data_feed = self.get_data_feed(symbol)
            if data_feed and shares > 0:
                self.sell(data=data_feed, size=shares)
                self.log(f"LOSER EXIT: {symbol} - {shares} shares")
                return  # Exit early

        # STEP 2: Sell profitable positions
        sell_candidates = []
        for symbol, data in self.portfolio.items():
            if data["shares"] > 0:
                data_feed = self.get_data_feed(symbol)
                if data_feed is None:
                    continue

                try:
                    current_price = data_feed.close[0]
                    gain = (current_price - data["avg_price"]) / data["avg_price"]

                    if self.get_sell_signal_strength(data_feed, gain):
                        sell_candidates.append((symbol, gain))

                except (IndexError, AttributeError):
                    continue

        # Sell the stock with highest profit
        if sell_candidates:
            sell_candidates.sort(key=lambda x: x[1], reverse=True)
            symbol, gain_pct = sell_candidates[0]
            shares = self.portfolio[symbol]["shares"]
            data_feed = self.get_data_feed(symbol)
            if data_feed and shares > 0:
                self.sell(data=data_feed, size=shares)
                self.log(f"SELL: {symbol} - {shares} shares (Gain: {gain_pct:.2%})")

        # STEP 3: Buy new positions (stocks below MA)
        buy_candidates = []
        for d in self.datas:
            if d._name == "^NSEI":  # Skip Nifty index
                continue

            try:
                buy_strength = self.get_buy_signal_strength(d)
                if buy_strength >= 1:
                    current_price = d.close[0]
                    buy_candidates.append((d._name, buy_strength, current_price))
            except (IndexError, AttributeError):
                continue

        # Sort by signal strength and take top candidates
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = buy_candidates[:5]

        # Buy max 2 new positions
        new_buys = 0
        for symbol, signal_strength, current_price in top_candidates:
            if symbol not in self.portfolio or self.portfolio[symbol]["shares"] == 0:
                data_feed = self.get_data_feed(symbol)
                if data_feed:
                    try:
                        size = self.get_position_size(current_price)
                        if size > 0 and self.broker.getcash() > size * current_price:
                            self.buy(data=data_feed, size=size)
                            self.log(
                                f"NEW BUY: {symbol} - {size} shares @ ₹{current_price:.2f}"
                            )
                            new_buys += 1
                        if new_buys >= self.params.max_new_buys:
                            break
                    except (IndexError, AttributeError):
                        continue

        # STEP 4: Averaging down (only if no new buys)
        if new_buys == 0:
            averaging_candidates = []
            for symbol, data in self.portfolio.items():
                if data["shares"] > 0:
                    # Check averaging limits
                    if data["averaging_count"] >= self.params.max_averaging_attempts:
                        continue

                    # Skip chronic losers
                    if self.is_chronic_loser(symbol, data):
                        continue

                    # Check momentum filter
                    if not self.has_recent_momentum(symbol):
                        continue

                    data_feed = self.get_data_feed(symbol)
                    if data_feed:
                        try:
                            current_price = data_feed.close[0]
                            drop = (data["avg_price"] - current_price) / data[
                                "avg_price"
                            ]

                            # Require both price drop AND MA signal
                            if drop >= self.params.averaging_threshold:
                                buy_strength = self.get_buy_signal_strength(data_feed)
                                if buy_strength >= 1:
                                    averaging_candidates.append(
                                        (symbol, drop, buy_strength)
                                    )

                        except (IndexError, AttributeError):
                            continue

            # Average down the best candidate
            if averaging_candidates:
                averaging_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                symbol, drop_pct, signal_strength = averaging_candidates[0]
                data_feed = self.get_data_feed(symbol)
                if data_feed:
                    try:
                        size = self.get_position_size(data_feed.close[0])
                        if (
                            size > 0
                            and self.broker.getcash() > size * data_feed.close[0]
                        ):
                            order = self.buy(data=data_feed, size=size)
                            order.addinfo(averaging=True)
                            avg_count = self.portfolio[symbol]["averaging_count"]
                            self.log(
                                f"AVERAGING: {symbol} - {size} shares, Drop: {drop_pct:.2%} "
                                f"(Attempt {avg_count + 1}/{self.params.max_averaging_attempts})"
                            )
                    except (IndexError, AttributeError):
                        pass

    def next(self):
        """Called on each bar - execute strategy once per day"""
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()

        # Reset daily execution flag at market open
        if current_time <= time(9, 15) and self.daily_executed:
            self.daily_executed = False

        # Execute strategy once per day (for daily data)
        if not self.daily_executed:
            self.daily_executed = True
            # Log periodically to avoid spam
            if len(self.datas[0]) % 100 == 0:
                self.log(f"Executing strategy for date: {dt.date()}")
            self.execute_strategy()

    def stop(self):
        """Called when strategy finishes"""
        self.log(f"Final Portfolio Value: ₹{self.broker.getvalue():.2f}")


class NiftyShopConfig(StrategyConfig):
    """
    Configuration for Nifty Shop Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define parameter grid for experiments
        """
        return {
            "sell_threshold": [0.02, 0.03, 0.04, 0.05],
            "averaging_threshold": [0.02, 0.03, 0.04, 0.05],
            "ma_period": [15, 20, 25, 30],
            "fixed_size": [10000, 15000, 20000],
            "max_new_buys": [1, 2, 3],
            "max_averaging_attempts": [1, 2, 3],
            "loser_exit_threshold": [0.20, 0.25, 0.30],
            "loser_exit_days": [45, 60, 75, 90],
            "momentum_lookback": [3, 5, 7, 10],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters
        """
        return {
            "sell_threshold": 0.03,
            "averaging_threshold": 0.03,
            "ma_period": 20,
            "fixed_size": 15000,
            "max_new_buys": 2,
            "max_averaging_attempts": 2,
            "loser_exit_threshold": 0.25,
            "loser_exit_days": 60,
            "momentum_lookback": 5,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters
        """
        # All thresholds should be positive and reasonable
        if (
            params.get("sell_threshold", 0) <= 0
            or params.get("sell_threshold", 0) > 0.2
        ):
            return False

        if (
            params.get("averaging_threshold", 0) <= 0
            or params.get("averaging_threshold", 0) > 0.2
        ):
            return False

        if (
            params.get("loser_exit_threshold", 0) <= 0
            or params.get("loser_exit_threshold", 0) > 0.5
        ):
            return False

        # MA period should be reasonable
        if params.get("ma_period", 0) < 5 or params.get("ma_period", 0) > 100:
            return False

        # Fixed size should be positive
        if params.get("fixed_size", 0) <= 0:
            return False

        # Counts should be positive and reasonable
        if params.get("max_new_buys", 0) <= 0 or params.get("max_new_buys", 0) > 10:
            return False

        if (
            params.get("max_averaging_attempts", 0) <= 0
            or params.get("max_averaging_attempts", 0) > 5
        ):
            return False

        # Days should be positive
        if (
            params.get("loser_exit_days", 0) <= 0
            or params.get("loser_exit_days", 0) > 365
        ):
            return False

        # Momentum lookback should be reasonable
        if (
            params.get("momentum_lookback", 0) <= 0
            or params.get("momentum_lookback", 0) > 30
        ):
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the strategy class
        """
        return NiftyShopStrategy

    def get_required_data_feeds(self) -> int:
        """
        Strategy works with multiple stocks
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights for composite score calculation
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.3}
