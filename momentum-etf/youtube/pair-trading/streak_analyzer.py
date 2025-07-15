"""
Custom Backtrader Analyzer for Trade Streaks and Enhanced Metrics

This analyzer tracks winning and losing streaks, consecutive wins/losses,
and other detailed trading statistics.
"""

import backtrader as bt
from collections import deque


class StreakAnalyzer(bt.Analyzer):
    """
    Custom analyzer to track winning and losing streaks and other detailed trade metrics
    """

    def create_analysis(self):
        """Initialize analysis dictionary"""
        self.rets = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "even_trades": 0,
            "current_streak": 0,
            "current_streak_type": None,  # 'win', 'loss', or None
            "max_winning_streak": 0,
            "max_losing_streak": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "trade_results": [],  # List of individual trade P&L
            "streak_changes": 0,  # Number of times streak type changed
        }

    def notify_trade(self, trade):
        """Called when a trade is closed"""
        if trade.isclosed:
            # Get trade P&L
            pnl = trade.pnl

            # Update totals
            self.rets["total_trades"] += 1
            self.rets["trade_results"].append(pnl)

            # Classify trade result
            if pnl > 0:
                self.rets["winning_trades"] += 1
                self.rets["gross_profit"] += pnl
                self.rets["max_win"] = max(self.rets["max_win"], pnl)

                # Update streak tracking
                if self.rets["current_streak_type"] != "win":
                    if self.rets["current_streak_type"] is not None:
                        self.rets["streak_changes"] += 1
                    self.rets["current_streak_type"] = "win"
                    self.rets["current_streak"] = 1
                    self.rets["consecutive_losses"] = 0
                else:
                    self.rets["current_streak"] += 1

                self.rets["consecutive_wins"] += 1
                self.rets["max_winning_streak"] = max(
                    self.rets["max_winning_streak"], self.rets["current_streak"]
                )

            elif pnl < 0:
                self.rets["losing_trades"] += 1
                self.rets["gross_loss"] += abs(pnl)
                self.rets["max_loss"] = min(self.rets["max_loss"], pnl)

                # Update streak tracking
                if self.rets["current_streak_type"] != "loss":
                    if self.rets["current_streak_type"] is not None:
                        self.rets["streak_changes"] += 1
                    self.rets["current_streak_type"] = "loss"
                    self.rets["current_streak"] = 1
                    self.rets["consecutive_wins"] = 0
                else:
                    self.rets["current_streak"] += 1

                self.rets["consecutive_losses"] += 1
                self.rets["max_losing_streak"] = max(
                    self.rets["max_losing_streak"], self.rets["current_streak"]
                )

            else:  # pnl == 0
                self.rets["even_trades"] += 1
                # Even trades break streaks but don't count as wins/losses
                if self.rets["current_streak_type"] is not None:
                    self.rets["streak_changes"] += 1
                self.rets["current_streak_type"] = None
                self.rets["current_streak"] = 0
                self.rets["consecutive_wins"] = 0
                self.rets["consecutive_losses"] = 0

    def stop(self):
        """Called at the end of the strategy to finalize calculations"""
        # Calculate averages and ratios
        if self.rets["winning_trades"] > 0:
            self.rets["avg_win"] = (
                self.rets["gross_profit"] / self.rets["winning_trades"]
            )

        if self.rets["losing_trades"] > 0:
            self.rets["avg_loss"] = self.rets["gross_loss"] / self.rets["losing_trades"]

        if self.rets["total_trades"] > 0:
            self.rets["win_rate"] = (
                self.rets["winning_trades"] / self.rets["total_trades"]
            ) * 100

        if self.rets["gross_loss"] > 0:
            self.rets["profit_factor"] = (
                self.rets["gross_profit"] / self.rets["gross_loss"]
            )

    def get_analysis(self):
        """Return the analysis results"""
        return self.rets


class DetailedTradeAnalyzer(bt.Analyzer):
    """
    Enhanced trade analyzer with additional statistical measures
    """

    def create_analysis(self):
        """Initialize analysis dictionary"""
        self.rets = {
            "trades": {
                "total": 0,
                "open": 0,
                "closed": 0,
                "won": 0,
                "lost": 0,
                "even": 0,
            },
            "pnl": {
                "gross": 0.0,
                "net": 0.0,
                "total_won": 0.0,
                "total_lost": 0.0,
                "avg_won": 0.0,
                "avg_lost": 0.0,
                "max_won": 0.0,
                "max_lost": 0.0,
            },
            "streaks": {
                "max_winning": 0,
                "max_losing": 0,
                "current_winning": 0,
                "current_losing": 0,
            },
            "len": {
                "total": 0,
                "avg": 0.0,
                "max": 0,
                "min": float("inf"),
                "won": {"total": 0, "avg": 0.0, "max": 0, "min": float("inf")},
                "lost": {"total": 0, "avg": 0.0, "max": 0, "min": float("inf")},
            },
        }

        self._current_trades = []
        self._closed_trades = []

    def notify_trade(self, trade):
        """Track trade notifications"""
        if trade.isopen:
            self.rets["trades"]["open"] += 1
            self._current_trades.append(trade)

        elif trade.isclosed:
            self.rets["trades"]["closed"] += 1
            if trade in self._current_trades:
                self._current_trades.remove(trade)
            self._closed_trades.append(trade)

            # Track P&L
            pnl = trade.pnl
            self.rets["pnl"]["gross"] += pnl
            self.rets["pnl"]["net"] += trade.pnlcomm

            # Track trade length
            trade_len = trade.barclose - trade.baropen + 1
            self.rets["len"]["total"] += trade_len
            self.rets["len"]["max"] = max(self.rets["len"]["max"], trade_len)
            self.rets["len"]["min"] = min(self.rets["len"]["min"], trade_len)

            # Classify trade result
            if pnl > 0:
                self.rets["trades"]["won"] += 1
                self.rets["pnl"]["total_won"] += pnl
                self.rets["pnl"]["max_won"] = max(self.rets["pnl"]["max_won"], pnl)

                # Track winning trade length
                self.rets["len"]["won"]["total"] += trade_len
                self.rets["len"]["won"]["max"] = max(
                    self.rets["len"]["won"]["max"], trade_len
                )
                self.rets["len"]["won"]["min"] = min(
                    self.rets["len"]["won"]["min"], trade_len
                )

                # Update streaks
                self.rets["streaks"]["current_winning"] += 1
                self.rets["streaks"]["current_losing"] = 0
                self.rets["streaks"]["max_winning"] = max(
                    self.rets["streaks"]["max_winning"],
                    self.rets["streaks"]["current_winning"],
                )

            elif pnl < 0:
                self.rets["trades"]["lost"] += 1
                self.rets["pnl"]["total_lost"] += abs(pnl)
                self.rets["pnl"]["max_lost"] = min(self.rets["pnl"]["max_lost"], pnl)

                # Track losing trade length
                self.rets["len"]["lost"]["total"] += trade_len
                self.rets["len"]["lost"]["max"] = max(
                    self.rets["len"]["lost"]["max"], trade_len
                )
                self.rets["len"]["lost"]["min"] = min(
                    self.rets["len"]["lost"]["min"], trade_len
                )

                # Update streaks
                self.rets["streaks"]["current_losing"] += 1
                self.rets["streaks"]["current_winning"] = 0
                self.rets["streaks"]["max_losing"] = max(
                    self.rets["streaks"]["max_losing"],
                    self.rets["streaks"]["current_losing"],
                )

            else:  # pnl == 0
                self.rets["trades"]["even"] += 1
                # Even trades break streaks
                self.rets["streaks"]["current_winning"] = 0
                self.rets["streaks"]["current_losing"] = 0

    def stop(self):
        """Finalize calculations"""
        # Calculate total trades
        self.rets["trades"]["total"] = self.rets["trades"]["closed"]

        # Calculate averages
        if self.rets["trades"]["won"] > 0:
            self.rets["pnl"]["avg_won"] = (
                self.rets["pnl"]["total_won"] / self.rets["trades"]["won"]
            )
            self.rets["len"]["won"]["avg"] = (
                self.rets["len"]["won"]["total"] / self.rets["trades"]["won"]
            )

        if self.rets["trades"]["lost"] > 0:
            self.rets["pnl"]["avg_lost"] = (
                self.rets["pnl"]["total_lost"] / self.rets["trades"]["lost"]
            )
            self.rets["len"]["lost"]["avg"] = (
                self.rets["len"]["lost"]["total"] / self.rets["trades"]["lost"]
            )

        if self.rets["trades"]["total"] > 0:
            self.rets["len"]["avg"] = (
                self.rets["len"]["total"] / self.rets["trades"]["total"]
            )

        # Handle cases where no trades occurred
        if self.rets["len"]["min"] == float("inf"):
            self.rets["len"]["min"] = 0
        if self.rets["len"]["won"]["min"] == float("inf"):
            self.rets["len"]["won"]["min"] = 0
        if self.rets["len"]["lost"]["min"] == float("inf"):
            self.rets["len"]["lost"]["min"] = 0
