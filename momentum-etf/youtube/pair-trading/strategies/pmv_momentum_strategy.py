"""
P=MV Momentum Strategy

A momentum-based strategy that calculates momentum as 'p=mv' where:
- m is determined using composite scoring based on RSI, VWAP and returns
- v is based on volume dynamics
- Positions are exited after a week of holding or based on stop-loss conditions

This strategy identifies momentum using a combination of price action, volume,
technical indicators like RSI, and comparative metrics like VWAP.
"""

import backtrader as bt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .base_strategy import BaseStrategy, StrategyConfig


class PMVMomentumStrategy(BaseStrategy):
    """
    P=MV Momentum Strategy Implementation
    """

    params = (
        ("rsi_period", 14),
        ("top_n_stocks", 15),
        ("position_size", 0.2),
        ("weekly_exit", True),
        ("stop_loss", 5.0),
        ("take_profit", 10.0),
        ("printlog", False),
    )

    # --------------------- init ------------------------------
    def __init__(self):
        super().__init__()
        # keep original attribute names
        self.entry_dates: Dict[str, int] = {}  # bar index of entry
        self.peak_prices: Dict[str, float] = {}

        # lightweight indicators
        self.inds = {}
        for d in self.datas:
            self.inds[d._name] = dict(
                rsi=bt.ind.RSI(d.close, period=self.p.rsi_period),
                roc_5=bt.ind.ROC(d.close, period=5),
                roc_20=bt.ind.ROC(d.close, period=20),
                roc_252=bt.ind.ROC(d.close, period=252),
                vol_ratio=d.volume / bt.ind.SMA(d.volume, period=20),
            )

        self.log(f"PMV Momentum Strategy initialized with {len(self.datas)} stocks")
        self.log(
            f"Parameters: RSI={self.p.rsi_period}, Top N={self.p.top_n_stocks}, "
            f"Position Size={self.p.position_size}, Stop Loss={self.p.stop_loss}%, "
            f"Take Profit={self.p.take_profit}%"
        )

    def notify_trade(self, trade):
        """Log completed trades"""
        super().notify_trade(trade)  # Call parent to track statistics
        if trade.isclosed:
            # Calculate PnL percentage more safely
            if trade.value != 0:
                pnl_pct = (trade.pnl / abs(trade.value)) * 100
            else:
                pnl_pct = 0.0

            # Calculate exit price more safely
            if trade.size != 0:
                exit_price = trade.price + (trade.pnl / trade.size)
            else:
                exit_price = trade.price

            self.log(
                f"TRADE CLOSED: {trade.data._name} - "
                f"PnL: {trade.pnl:.2f} ({pnl_pct:.2f}%), "
                f"Size: {trade.size}, Entry: {trade.price:.2f}, "
                f"Exit: {exit_price:.2f}"
            )

    def notify_order(self, order):
        """Log order notifications"""
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"ORDER {action} {order.data._name}: "
                f"Size={order.executed.size}, Price={order.executed.price:.2f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED {order.data._name}: {order.getstatusname()}")

    # --------------------- helpers ---------------------------
    def _score(self, d) -> float:
        """Momentum score (higher = better)."""
        try:
            i = self.inds[d._name]
            score = (
                0.50 * i["roc_5"][0]
                + 0.30 * i["roc_20"][0]
                + 0.15 * i["roc_252"][0]
                + 0.05 * (50 - i["rsi"][0])
            ) * i["vol_ratio"][0]

            # Only log scores for top candidates to reduce noise
            if score > 0:
                self.log(
                    f"Score for {d._name}: {score:.4f} "
                    f"(ROC5:{i['roc_5'][0]:.2f}, ROC20:{i['roc_20'][0]:.2f}, "
                    f"ROC252:{i['roc_252'][0]:.2f}, RSI:{i['rsi'][0]:.1f}, "
                    f"VolRatio:{i['vol_ratio'][0]:.2f})"
                )
            return score
        except (IndexError, TypeError) as e:
            self.log(f"Score error {d._name}: {e}")
            return -np.inf

    def _should_exit(self, d) -> bool:
        """Exit rules: time, stop-loss, take-profit."""
        pos = self.getposition(d)
        if not pos:
            return False

        # Check if we have entry date recorded
        if d._name not in self.entry_dates:
            self.log(
                f"Warning: No entry date for {d._name}, using current bar as entry"
            )
            self.entry_dates[d._name] = len(d)

        # time-based: 5 trading bars
        bars_held = len(d) - self.entry_dates[d._name]
        if bars_held >= 5:
            self.log(f"Exit {d._name}: Time-based exit (held {bars_held} bars)")
            return True

        price = d.close[0]
        entry_price = pos.price
        pnl_pct = (price - entry_price) / entry_price * 100

        if price <= entry_price * (1 - self.p.stop_loss / 100):
            self.log(
                f"Exit {d._name}: Stop-loss triggered at {price:.2f} "
                f"(entry: {entry_price:.2f}, loss: {pnl_pct:.2f}%)"
            )
            return True
        if price >= entry_price * (1 + self.p.take_profit / 100):
            self.log(
                f"Exit {d._name}: Take-profit triggered at {price:.2f} "
                f"(entry: {entry_price:.2f}, profit: {pnl_pct:.2f}%)"
            )
            return True
        return False

    def _cash_reserved(self) -> float:
        """Cash already committed to pending buy orders."""
        return sum(
            o.size * o.price
            for o in self.broker.orders
            if o.isbuy() and o.status in [o.Accepted, o.Submitted]
        )

    def _cleanup_tracking_dicts(self):
        """Clean up tracking dictionaries to ensure consistency with actual positions"""
        # Remove entries for stocks we no longer have positions in
        stocks_with_positions = {d._name for d in self.datas if self.getposition(d)}

        # Clean entry_dates
        to_remove = []
        for stock_name in self.entry_dates:
            if stock_name not in stocks_with_positions:
                to_remove.append(stock_name)

        for stock_name in to_remove:
            self.log(f"Cleaning up orphaned entry_date for {stock_name}")
            del self.entry_dates[stock_name]

        # Clean peak_prices
        to_remove = []
        for stock_name in self.peak_prices:
            if stock_name not in stocks_with_positions:
                to_remove.append(stock_name)

        for stock_name in to_remove:
            self.log(f"Cleaning up orphaned peak_price for {stock_name}")
            del self.peak_prices[stock_name]

    # --------------------- core logic ------------------------
    def execute_strategy(self):
        if len(self.data) < 252:  # warm-up guard
            if len(self.data) == 251:
                self.log("Warm-up period complete, strategy will start next bar")
            return

        # Clean up any inconsistencies
        self._cleanup_tracking_dicts()

        current_positions = sum(bool(self.getposition(d)) for d in self.datas)
        equity = self.broker.getvalue()
        cash = self.broker.getcash()

        # Log portfolio status every 20 bars to avoid spam
        if len(self.data) % 20 == 0:
            self.log(
                f"Portfolio Status: Positions={current_positions}/{self.p.top_n_stocks}, "
                f"Equity={equity:.2f}, Cash={cash:.2f}, Bar={len(self.data)}"
            )

        # Detailed logging for strategy execution
        total_position_value = sum(
            self.getposition(d).size * d.close[0]
            for d in self.datas
            if self.getposition(d)
        )

        self.log(
            f"Strategy execution: Positions={current_positions}/{self.p.top_n_stocks}, "
            f"Equity={equity:.2f}, Cash={cash:.2f}, "
            f"Position Value={total_position_value:.2f}"
        )

        # ---------- exits ----------
        exits_count = 0
        for d in self.datas:
            if self.getposition(d) and self._should_exit(d):
                pos_value = self.getposition(d).size * d.close[0]
                self.log(f"Closing position {d._name}: value={pos_value:.2f}")
                self.close(d)

                # Safely remove from tracking dictionaries
                if d._name in self.entry_dates:
                    self.entry_dates.pop(d._name, None)
                else:
                    self.log(
                        f"Warning: {d._name} not found in entry_dates when exiting"
                    )

                if d._name in self.peak_prices:
                    self.peak_prices.pop(d._name, None)
                else:
                    self.log(
                        f"Warning: {d._name} not found in peak_prices when exiting"
                    )

                exits_count += 1

        if exits_count > 0:
            self.log(f"Executed {exits_count} exits")

        # ---------- entries ----------
        candidates = [
            (d, self._score(d)) for d in self.datas if not self.getposition(d)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        open_slots = max(
            0, self.p.top_n_stocks - sum(bool(self.getposition(d)) for d in self.datas)
        )

        cash_reserved = self._cash_reserved()
        available_cash = cash - cash_reserved

        # Show top 5 candidates for debugging
        top_candidates = candidates[:5]
        self.log(
            f"Top 5 candidates: {[(d._name, f'{score:.4f}') for d, score in top_candidates]}"
        )

        self.log(
            f"Entry analysis: {len(candidates)} candidates, {open_slots} open slots, "
            f"Available cash: {available_cash:.2f} (reserved: {cash_reserved:.2f})"
        )

        entries_count = 0
        for d, score in candidates[:open_slots]:
            if score <= 0:
                self.log(f"Skipping {d._name}: negative score {score:.4f}")
                continue

            # Check if we already have a position (shouldn't happen but let's be safe)
            if self.getposition(d):
                self.log(f"Warning: Already have position in {d._name}, skipping")
                continue

            # More conservative position sizing
            # Use the smaller of: target position size OR available cash divided by remaining slots
            remaining_slots = open_slots - entries_count
            if remaining_slots <= 0:
                break

            target_position_value = equity * self.p.position_size
            max_cash_per_position = available_cash / remaining_slots

            # Use the smaller value for position sizing
            position_value = min(target_position_value, max_cash_per_position)

            # Ensure we can afford at least 1 share
            if position_value < d.close[0]:
                self.log(
                    f"Cannot afford {d._name}: min cost {d.close[0]:.2f}, "
                    f"max position value {position_value:.2f}"
                )
                continue

            size = max(1, int(position_value / d.close[0]))
            cost = size * d.close[0]

            if cost <= available_cash:
                # Check if we already have an entry date (shouldn't happen)
                if d._name in self.entry_dates:
                    self.log(f"Warning: Overwriting existing entry date for {d._name}")

                self.buy(d, size=size)
                self.entry_dates[d._name] = len(d)
                self.peak_prices[d._name] = d.close[0]
                self.log(
                    f"BUY {d._name}: size={size}, price={d.close[0]:.2f}, "
                    f"cost={cost:.2f}, score={score:.4f}, "
                    f"target={target_position_value:.2f}, max_cash={max_cash_per_position:.2f}"
                )
                available_cash -= cost
                entries_count += 1
            else:
                self.log(
                    f"Insufficient cash for {d._name}: need {cost:.2f}, have {available_cash:.2f}"
                )
                # If we can't afford this position, we likely can't afford any more
                break

        if entries_count > 0:
            self.log(f"Executed {entries_count} entries")
        elif open_slots > 0:
            self.log(
                f"No entries executed despite {open_slots} open slots - likely insufficient cash"
            )


class PMVMomentumConfig(StrategyConfig):
    """
    Configuration for P=MV Momentum Strategy
    """

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for P=MV momentum strategy experiments
        """
        return {
            "rsi_period": [9, 14, 21],
            "top_n_stocks": [5, 10, 15],
            "position_size": [0.1, 0.15, 0.2],
            "weekly_exit": [True],  # Always exit after a week
            "stop_loss": [3.0, 5.0, 7.0],
            "take_profit": [7.0, 10.0, 15.0],
            "printlog": [False, True],  # Include both for testing
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for P=MV momentum strategy
        """
        return {
            "rsi_period": 14,
            "top_n_stocks": 15,
            "position_size": 0.2,
            "weekly_exit": True,
            "stop_loss": 5.0,
            "take_profit": 10.0,
            "printlog": True,  # Enable logging by default for debugging
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate P=MV momentum strategy parameters
        """
        # All numeric parameters must be positive
        for key in [
            "rsi_period",
            "top_n_stocks",
            "stop_loss",
            "take_profit",
        ]:
            if params.get(key, 0) <= 0:
                return False

        # Position size must be between 0 and 1
        if not 0 < params.get("position_size", 0) <= 1.0:
            return False

        # Top N stocks should be reasonable
        if params.get("top_n_stocks", 0) > 20:
            return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the P=MV momentum strategy class
        """
        return PMVMomentumStrategy

    def get_required_data_feeds(self) -> int:
        """
        P=MV Momentum strategy works with multiple stocks
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights for the composite score
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.3}
