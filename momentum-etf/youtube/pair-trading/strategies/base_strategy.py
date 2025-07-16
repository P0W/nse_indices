"""
Base Strategy Classes

This module provides the abstract base classes that all trading strategies must inherit from.
It defines the interface for strategy configuration and ensures consistency across all strategies.
"""

import backtrader as bt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ExperimentResult:
    """
    Data class to store experiment results
    """

    params: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    final_value: float
    num_data_feeds: int = 0
    strategy_name: str = ""
    experiment_duration: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    composite_score: float = 0.0
    portfolio_values: Optional[List[float]] = field(default=None)
    dates: Optional[List[Any]] = field(default=None)

    # Enhanced trade statistics
    max_winning_streak: int = 0
    max_losing_streak: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    even_trades: int = 0
    avg_trade_length: float = 0.0
    annualized_return: float = 0.0
    expectancy: float = 0.0


class StrategyConfig(ABC):
    """
    Abstract base class for strategy configurations

    All strategy configuration classes must inherit from this class and implement
    the required methods to define parameters, validation, and other strategy-specific
    behavior.
    """

    @abstractmethod
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the parameter grid for strategy experiments

        Returns:
            Dict[str, List[Any]]: Dictionary mapping parameter names to lists of values to test
        """
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for the strategy

        Returns:
            Dict[str, Any]: Dictionary of default parameter values
        """
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters

        Args:
            params: Dictionary of parameter values to validate

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        pass

    @abstractmethod
    def get_strategy_class(self) -> type:
        """
        Get the strategy class for this configuration

        Returns:
            type: The strategy class that implements bt.Strategy
        """
        pass

    @abstractmethod
    def get_required_data_feeds(self) -> int:
        """
        Get the number of required data feeds for this strategy

        Returns:
            int: Number of required data feeds, or -1 for variable number
        """
        pass

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Get weights for composite score calculation

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their weights
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.3}

    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a composite score for ranking experiments

        Args:
            metrics: Dictionary containing 'total_return', 'sharpe_ratio', 'max_drawdown'

        Returns:
            float: Composite score (higher is better)
        """
        weights = self.get_composite_score_weights()

        # Normalize metrics (simple approach)
        normalized_return = min(metrics["total_return"] / 100, 2.0)  # Cap at 200%
        normalized_sharpe = min(metrics["sharpe_ratio"] / 2.0, 1.0)  # Cap at 2.0
        normalized_drawdown = max(
            1 - (abs(metrics["max_drawdown"]) / 50), 0
        )  # Cap at 50%

        composite = (
            weights["total_return"] * normalized_return
            + weights["sharpe_ratio"] * normalized_sharpe
            + weights["max_drawdown"] * normalized_drawdown
        )

        return composite


class BaseStrategy(bt.Strategy):
    """
    Base strategy class that all strategies should inherit from

    This class provides common functionality and structure for all trading strategies.
    """

    params = (("printlog", False),)  # Whether to print log messages

    def log(self, txt, dt=None):
        """Logging function for all strategies

        Args:
            txt: The text message to log
            dt: Optional date/time to include in the log (defaults to current bar date)
        """
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}: {txt}")

    def __init__(self):
        super().__init__()
        self.portfolio_values = []
        self.dates = []
        self.trades_count = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

    def notify_trade(self, trade):
        """Track trade statistics"""
        if trade.isclosed:
            self.trades_count += 1
            pnl = trade.pnl
            if pnl > 0:
                self.winning_trades += 1
                self.total_profit += pnl
            else:
                self.total_loss += abs(pnl)

    def next(self):
        """Default next method that tracks portfolio performance"""
        current_value = self.broker.getvalue()
        current_date = self.datas[0].datetime.date(0)

        self.portfolio_values.append(current_value)
        self.dates.append(current_date)

        # Call strategy-specific logic
        self.execute_strategy()

    @abstractmethod
    def execute_strategy(self):
        """
        Execute strategy-specific logic

        This method must be implemented by each strategy to define its trading logic.
        """
        pass

    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.trades_count == 0:
            return 0.0
        return (self.winning_trades / self.trades_count) * 100

    def get_profit_factor(self) -> float:
        """Calculate profit factor"""
        if self.total_loss == 0:
            return float("inf") if self.total_profit > 0 else 0.0
        return self.total_profit / self.total_loss
