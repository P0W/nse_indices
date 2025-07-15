"""
Unified Strategy Experiment Framework

This module provides a unified interface for running experiments across different trading strategies.
It supports any strategy that implements the StrategyConfig interface.
"""

import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import time
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
from typing import Dict, Any, List, Optional, Union, Type

import backtrader as bt
from utils import MarketDataLoader, IndianBrokerageCommission
from strategies.base_strategy import StrategyConfig, ExperimentResult

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class UnifiedExperimentFramework:
    """
    Unified framework for running experiments across different trading strategies
    """

    def __init__(
        self, strategy_config: StrategyConfig, results_dir: str = "experiment_results"
    ):
        """
        Initialize the experiment framework

        Args:
            strategy_config: Strategy configuration implementing StrategyConfig interface
            results_dir: Directory to store experiment results
        """
        self.strategy_config = strategy_config
        self.results_dir = results_dir
        self.strategy_name = strategy_config.__class__.__name__.replace("Config", "")

        # Create strategy-specific results directory
        self.strategy_results_dir = os.path.join(
            results_dir, self.strategy_name.lower()
        )
        if not os.path.exists(self.strategy_results_dir):
            os.makedirs(self.strategy_results_dir)

        self.results: List[ExperimentResult] = []
        self.best_result: Optional[ExperimentResult] = None
        self.best_score = -float("inf")

    def generate_parameter_combinations(
        self, max_combinations: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for testing

        Args:
            max_combinations: Maximum number of combinations to test

        Returns:
            list: List of parameter dictionaries
        """
        param_grid = self.strategy_config.get_parameter_grid()

        # Generate all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        all_combinations = list(itertools.product(*values))

        # Filter out invalid combinations using strategy validation
        valid_combinations = []
        for combo in all_combinations:
            params = dict(zip(keys, combo))
            if self.strategy_config.validate_params(params):
                valid_combinations.append(params)

        # If too many combinations, sample randomly
        if len(valid_combinations) > max_combinations:
            import random

            random.seed(42)  # For reproducibility
            valid_combinations = random.sample(valid_combinations, max_combinations)

        print(
            f"🧪 Generated {len(valid_combinations)} valid parameter combinations for {self.strategy_name}"
        )
        return valid_combinations

    def prepare_data_feeds(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[bt.feeds.PandasData]:
        """
        Prepare data feeds for the strategy

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            list: List of backtrader data feeds
        """
        loader = MarketDataLoader()

        required_feeds = self.strategy_config.get_required_data_feeds()

        if required_feeds == -1:  # Variable number (like momentum strategy)
            data_feeds = loader.load_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                force_refresh=False,
                use_parallel=False,  # Disable parallel for experiments
            )
        elif required_feeds == 2:  # Pairs trading
            # Use first two symbols for pairs trading
            if len(symbols) < 2:
                print(
                    f"❌ Pairs trading requires at least 2 symbols, got {len(symbols)}"
                )
                return []

            data_feeds = loader.load_market_data(
                symbols=symbols[:2],
                start_date=start_date,
                end_date=end_date,
                force_refresh=False,
                use_parallel=False,
            )
        else:
            # Fixed number of feeds
            data_feeds = loader.load_market_data(
                symbols=symbols[:required_feeds],
                start_date=start_date,
                end_date=end_date,
                force_refresh=False,
                use_parallel=False,
            )

        return data_feeds

    def run_single_experiment(
        self,
        params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_cash: float = 1000000,
    ) -> Optional[ExperimentResult]:
        """
        Run a single experiment with given parameters

        Args:
            params: Parameter dictionary
            symbols: List of stock symbols
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_cash: Initial cash for backtest

        Returns:
            ExperimentResult or None if experiment failed
        """
        experiment_start_time = time.time()

        try:
            # Suppress output for experiments
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            # Create cerebro instance
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.set_coc(True)

            # Apply Indian brokerage commission scheme
            commission_scheme = IndianBrokerageCommission()
            cerebro.broker.addcommissioninfo(commission_scheme)

            # Prepare data feeds
            data_feeds = self.prepare_data_feeds(symbols, start_date, end_date)

            if not data_feeds:
                sys.stdout = old_stdout
                return None

            # Add data feeds
            for data_feed in data_feeds:
                cerebro.adddata(data_feed)

            # Get strategy class and add with parameters
            strategy_class = self.strategy_config.get_strategy_class()
            if strategy_class is None:
                sys.stdout = old_stdout
                return None

            # Merge default params with experiment params
            default_params = self.strategy_config.get_default_params()
            final_params = {**default_params, **params}

            cerebro.addstrategy(strategy_class, **final_params)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

            # Run backtest
            result = cerebro.run()[0]

            # Restore stdout
            sys.stdout = old_stdout

            # Extract metrics
            final_value = cerebro.broker.getvalue()
            total_return = ((final_value - initial_cash) / initial_cash) * 100

            sharpe_analysis = result.analyzers.sharpe.get_analysis()
            drawdown_analysis = result.analyzers.drawdown.get_analysis()

            sharpe_ratio = sharpe_analysis.get("sharperatio", 0) or 0
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0

            # Calculate metrics dict for composite score
            metrics = {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            }

            # Calculate composite score using strategy-specific method
            composite_score = self.strategy_config.calculate_composite_score(metrics)

            experiment_duration = time.time() - experiment_start_time

            return ExperimentResult(
                params=params.copy(),
                final_value=final_value,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                composite_score=composite_score,
                num_data_feeds=len(data_feeds),
                strategy_name=self.strategy_name,
                experiment_duration=experiment_duration,
            )

        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            print(f"❌ Experiment failed: {str(e)}")
            return None

    def run_experiments(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_combinations: int = 50,
        initial_cash: float = 1000000,
        use_parallel: bool = True,
        max_workers: int = 4,
    ):
        """
        Run multiple experiments with different parameter combinations

        Args:
            symbols: List of stock symbols to use
            start_date: Start date for backtest (defaults to 2 years ago)
            end_date: End date for backtest (defaults to now)
            max_combinations: Maximum number of parameter combinations to test
            initial_cash: Initial cash for backtest
            use_parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 2)  # 2 years ago

        print(f"🚀 Starting {self.strategy_name} Strategy Experiments")
        print("=" * 60)
        print(
            f"📊 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"💰 Initial Cash: ₹{initial_cash:,.0f}")
        print(f"📈 Symbols: {len(symbols)} stocks")

        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(max_combinations)

        # Run experiments
        self.results = []

        if use_parallel and max_workers > 1:
            print(f"🔄 Running {len(param_combinations)} experiments in parallel...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_params = {
                    executor.submit(
                        self.run_single_experiment,
                        params,
                        symbols,
                        start_date,
                        end_date,
                        initial_cash,
                    ): params
                    for params in param_combinations
                }

                # Collect results with progress bar
                with tqdm(
                    total=len(param_combinations), desc="🧪 Running experiments"
                ) as pbar:
                    for future in as_completed(future_to_params):
                        result = future.result()
                        if result:
                            self.results.append(result)

                            # Update best result
                            if result.composite_score > self.best_score:
                                self.best_score = result.composite_score
                                self.best_result = result

                        pbar.update(1)
        else:
            print(f"🔄 Running {len(param_combinations)} experiments sequentially...")
            for params in tqdm(param_combinations, desc="🧪 Running experiments"):
                result = self.run_single_experiment(
                    params, symbols, start_date, end_date, initial_cash
                )
                if result:
                    self.results.append(result)

                    # Update best result
                    if result.composite_score > self.best_score:
                        self.best_score = result.composite_score
                        self.best_result = result

        print(f"\n✅ Completed {len(self.results)} successful experiments")

        # Save results
        self.save_results()

        # Display summary
        self.display_results_summary()

    def save_results(self):
        """Save experiment results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_file = os.path.join(
            self.strategy_results_dir,
            f"{self.strategy_name.lower()}_results_{timestamp}.json",
        )

        # Convert results to JSON-serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                "params": result.params,
                "final_value": result.final_value,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "composite_score": result.composite_score,
                "num_data_feeds": result.num_data_feeds,
                "strategy_name": result.strategy_name,
                "experiment_duration": result.experiment_duration,
            }
            results_data.append(result_dict)

        with open(json_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save as CSV
        csv_file = os.path.join(
            self.strategy_results_dir,
            f"{self.strategy_name.lower()}_results_{timestamp}.csv",
        )
        df = self.results_to_dataframe()
        df.to_csv(csv_file, index=False)

        print(f"💾 Results saved to {json_file} and {csv_file}")

    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        flat_results = []
        for result in self.results:
            flat_result = result.params.copy()
            flat_result.update(
                {
                    "final_value": result.final_value,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "composite_score": result.composite_score,
                    "num_data_feeds": result.num_data_feeds,
                    "strategy_name": result.strategy_name,
                    "experiment_duration": result.experiment_duration,
                }
            )
            flat_results.append(flat_result)

        return pd.DataFrame(flat_results)

    def display_results_summary(self):
        """Display summary of experiment results"""
        if not self.results:
            print("❌ No results to display")
            return

        df = self.results_to_dataframe()

        print(f"\n📊 {self.strategy_name.upper()} STRATEGY EXPERIMENT RESULTS")
        print("=" * 60)

        # Best parameters
        if self.best_result:
            print(f"\n🏆 BEST PARAMETERS (Score: {self.best_score:.2f})")
            best_data = [["Parameter", "Value"]]
            for param, value in self.best_result.params.items():
                best_data.append([param.replace("_", " ").title(), value])
            print(tabulate(best_data, headers="firstrow", tablefmt="grid"))

        # Top 10 results
        top_10 = df.nlargest(10, "composite_score")
        print(f"\n📈 TOP 10 RESULTS")
        display_cols = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "composite_score",
        ]
        print(
            tabulate(
                top_10[display_cols].round(2),
                headers=display_cols,
                tablefmt="grid",
                showindex=False,
            )
        )

        # Statistics
        print(f"\n📊 PERFORMANCE STATISTICS")
        stats_data = [
            ["Metric", "Mean", "Std", "Min", "Max"],
            [
                "Total Return (%)",
                f"{df['total_return'].mean():.2f}",
                f"{df['total_return'].std():.2f}",
                f"{df['total_return'].min():.2f}",
                f"{df['total_return'].max():.2f}",
            ],
            [
                "Sharpe Ratio",
                f"{df['sharpe_ratio'].mean():.2f}",
                f"{df['sharpe_ratio'].std():.2f}",
                f"{df['sharpe_ratio'].min():.2f}",
                f"{df['sharpe_ratio'].max():.2f}",
            ],
            [
                "Max Drawdown (%)",
                f"{df['max_drawdown'].mean():.2f}",
                f"{df['max_drawdown'].std():.2f}",
                f"{df['max_drawdown'].min():.2f}",
                f"{df['max_drawdown'].max():.2f}",
            ],
        ]
        print(tabulate(stats_data, headers="firstrow", tablefmt="grid"))

    def create_visualizations(self):
        """Create visualizations of experiment results"""
        if not self.results:
            print("❌ No results to visualize")
            return

        df = self.results_to_dataframe()

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            f"{self.strategy_name} Strategy Experiments - Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Composite Score Distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(
            df["composite_score"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax1.set_title("Composite Score Distribution")
        ax1.set_xlabel("Composite Score")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # 2. Return vs Sharpe Ratio
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(
            df["sharpe_ratio"],
            df["total_return"],
            c=df["composite_score"],
            cmap="viridis",
            alpha=0.6,
        )
        ax2.set_title("Return vs Sharpe Ratio")
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_ylabel("Total Return (%)")
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label="Composite Score")

        # 3. Drawdown vs Return
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(df["max_drawdown"], df["total_return"], alpha=0.6, color="coral")
        ax3.set_title("Drawdown vs Return")
        ax3.set_xlabel("Max Drawdown (%)")
        ax3.set_ylabel("Total Return (%)")
        ax3.grid(True, alpha=0.3)

        # 4. Parameter Importance (correlation with score)
        ax4 = plt.subplot(2, 3, 4)
        param_importance = {}
        numeric_params = []

        for param in df.columns:
            if param not in [
                "final_value",
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "composite_score",
                "num_data_feeds",
                "strategy_name",
                "experiment_duration",
            ]:
                if df[param].dtype in ["int64", "float64"]:
                    correlation = df[param].corr(df["composite_score"])
                    if not pd.isna(correlation):
                        param_importance[param] = abs(correlation)
                        numeric_params.append(param)

        if param_importance:
            params = list(param_importance.keys())
            importance = list(param_importance.values())
            ax4.barh(params, importance, color="lightgreen")
            ax4.set_title("Parameter Importance")
            ax4.set_xlabel("Absolute Correlation with Score")

        # 5. Performance over time (experiment duration)
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(
            df["experiment_duration"], df["composite_score"], alpha=0.6, color="orange"
        )
        ax5.set_title("Performance vs Experiment Duration")
        ax5.set_xlabel("Experiment Duration (seconds)")
        ax5.set_ylabel("Composite Score")
        ax5.grid(True, alpha=0.3)

        # 6. Best parameters visualization (if available)
        ax6 = plt.subplot(2, 3, 6)
        if len(numeric_params) >= 2:
            # Create a correlation matrix of numeric parameters
            param_corr = df[numeric_params + ["composite_score"]].corr()
            sns.heatmap(param_corr, annot=True, cmap="coolwarm", center=0, ax=ax6)
            ax6.set_title("Parameter Correlation Matrix")
        else:
            ax6.text(
                0.5,
                0.5,
                "Insufficient numeric\nparameters for\ncorrelation analysis",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title("Parameter Analysis")

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.strategy_results_dir,
            f"{self.strategy_name.lower()}_analysis_{timestamp}.png",
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"📊 Visualizations saved to {filename}")

        plt.show()

    def get_optimal_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the optimal parameters from experiments"""
        if self.best_result:
            return self.best_result.params
        else:
            print("❌ No experiments run yet. Run experiments first.")
            return None
