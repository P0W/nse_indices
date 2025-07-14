"""
Strategy Parameter Experiments

This module allows you to run systematic experiments with different parameter combinations
to find the optimal strategy configuration for the Adaptive Momentum Strategy.
"""

import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings

# Import the main strategy components
from adaptive_momentum import (
    run_backtest,
    filter_stocks_with_data,
    ALL_NIFTY_200_STOCKS,
    INITIAL_CASH,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class StrategyExperiments:
    """
    Run systematic experiments with different strategy parameters
    """

    def __init__(self, results_dir="experiment_results"):
        """
        Initialize the experiments framework

        Args:
            results_dir: Directory to store experiment results
        """
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.results = []
        self.best_params = None
        self.best_score = -float("inf")

    def define_parameter_grid(self):
        """
        Define the parameter grid for experiments

        Returns:
            dict: Dictionary of parameter ranges to test
        """
        return {
            "sma_fast": [20, 30, 50, 60],
            "sma_slow": [100, 150, 200, 250],
            "atr_period": [10, 14, 20],
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
            "momentum_period": [10, 15, 20, 25],
            "top_n_stocks": [3, 5, 7, 10],
            "rebalance_days": [5, 10, 15, 20, 30],
        }

    def generate_parameter_combinations(self, max_combinations=100):
        """
        Generate parameter combinations for testing

        Args:
            max_combinations: Maximum number of combinations to test

        Returns:
            list: List of parameter dictionaries
        """
        param_grid = self.define_parameter_grid()

        # Generate all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        all_combinations = list(itertools.product(*values))

        # Filter out invalid combinations (sma_fast >= sma_slow)
        valid_combinations = []
        for combo in all_combinations:
            params = dict(zip(keys, combo))
            if params["sma_fast"] < params["sma_slow"]:
                valid_combinations.append(params)

        # If too many combinations, sample randomly
        if len(valid_combinations) > max_combinations:
            import random

            random.seed(42)  # For reproducibility
            valid_combinations = random.sample(valid_combinations, max_combinations)

        print(f"🧪 Generated {len(valid_combinations)} valid parameter combinations")
        return valid_combinations

    def run_single_experiment(self, params, symbols, start_date, end_date):
        """
        Run a single experiment with given parameters

        Args:
            params: Parameter dictionary
            symbols: List of stock symbols
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            dict: Experiment results
        """
        try:
            # Temporarily modify the global STRATEGY_PARAMS
            import adaptive_momentum

            original_params = adaptive_momentum.STRATEGY_PARAMS.copy()
            adaptive_momentum.STRATEGY_PARAMS.update(params)

            # Suppress output for experiments
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            # Create a minimal cerebro instance for testing
            import backtrader as bt
            from utils import MarketDataLoader, IndianBrokerageCommission
            import adaptive_momentum

            cerebro = bt.Cerebro()
            cerebro.broker.setcash(INITIAL_CASH)
            cerebro.broker.set_coc(True)

            # Apply Indian brokerage commission scheme
            commission_scheme = IndianBrokerageCommission()
            cerebro.broker.addcommissioninfo(commission_scheme)

            # Load data
            loader = MarketDataLoader()
            data_feeds = loader.load_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                force_refresh=False,
                use_parallel=False,  # Disable parallel for experiments
            )

            if not data_feeds:
                sys.stdout = old_stdout
                adaptive_momentum.STRATEGY_PARAMS = original_params
                return None

            # Add data feeds
            for data_feed in data_feeds:
                cerebro.adddata(data_feed)

            # Add strategy with current parameters
            cerebro.addstrategy(adaptive_momentum.MomentumTrendStrategy, **params)
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

            # Run backtest
            result = cerebro.run()[0]

            # Restore stdout
            sys.stdout = old_stdout

            # Extract metrics
            final_value = cerebro.broker.getvalue()
            total_return = ((final_value - INITIAL_CASH) / INITIAL_CASH) * 100

            sharpe_analysis = result.analyzers.sharpe.get_analysis()
            drawdown_analysis = result.analyzers.drawdown.get_analysis()

            sharpe_ratio = sharpe_analysis.get("sharperatio", 0)
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0)

            # Calculate composite score (you can adjust this formula)
            composite_score = (
                total_return * 0.4
                + sharpe_ratio * 20 * 0.3
                + (100 - max_drawdown) * 0.3
            )

            result_dict = {
                "params": params.copy(),
                "final_value": final_value,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "composite_score": composite_score,
                "num_stocks": len(data_feeds),
            }

            # Restore original parameters
            adaptive_momentum.STRATEGY_PARAMS = original_params

            return result_dict

        except Exception as e:
            # Restore stdout and original params in case of error
            sys.stdout = old_stdout
            adaptive_momentum.STRATEGY_PARAMS = original_params
            return None

    def run_experiments(
        self, max_combinations=50, max_stocks=30, use_parallel=True, max_workers=4
    ):
        """
        Run multiple experiments with different parameter combinations

        Args:
            max_combinations: Maximum number of parameter combinations to test
            max_stocks: Maximum number of stocks to use for experiments
            use_parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
        """
        print("🚀 Starting Strategy Parameter Experiments")
        print("=" * 60)

        # Get stock symbols
        print("📊 Preparing stock data...")
        start_date = datetime(2020, 1, 1)  # Shorter period for experiments
        end_date = datetime.now()

        available_stocks = filter_stocks_with_data(
            ALL_NIFTY_200_STOCKS,
            start_date,
            end_date,
            max_stocks_to_test=max_stocks * 2,  # Test more to get enough good ones
            max_workers=4,
        )

        # Use subset of stocks for faster experiments
        test_symbols = available_stocks[:max_stocks]
        print(f"✅ Using {len(test_symbols)} stocks for experiments")

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
                        test_symbols,
                        start_date,
                        end_date,
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
                            if result["composite_score"] > self.best_score:
                                self.best_score = result["composite_score"]
                                self.best_params = result["params"]

                        pbar.update(1)
        else:
            print(f"🔄 Running {len(param_combinations)} experiments sequentially...")
            for params in tqdm(param_combinations, desc="🧪 Running experiments"):
                result = self.run_single_experiment(
                    params, test_symbols, start_date, end_date
                )
                if result:
                    self.results.append(result)

                    # Update best result
                    if result["composite_score"] > self.best_score:
                        self.best_score = result["composite_score"]
                        self.best_params = result["params"]

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
            self.results_dir, f"experiment_results_{timestamp}.json"
        )
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save as CSV
        csv_file = os.path.join(self.results_dir, f"experiment_results_{timestamp}.csv")
        df = self.results_to_dataframe()
        df.to_csv(csv_file, index=False)

        print(f"💾 Results saved to {json_file} and {csv_file}")

    def results_to_dataframe(self):
        """Convert results to pandas DataFrame"""
        flat_results = []
        for result in self.results:
            flat_result = result["params"].copy()
            flat_result.update(
                {
                    "final_value": result["final_value"],
                    "total_return": result["total_return"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                    "composite_score": result["composite_score"],
                    "num_stocks": result["num_stocks"],
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

        print("\n📊 EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)

        # Best parameters
        print(f"\n🏆 BEST PARAMETERS (Score: {self.best_score:.2f})")
        best_data = [
            ["Parameter", "Value"],
            ["SMA Fast", self.best_params["sma_fast"]],
            ["SMA Slow", self.best_params["sma_slow"]],
            ["ATR Period", self.best_params["atr_period"]],
            ["ATR Multiplier", self.best_params["atr_multiplier"]],
            ["Momentum Period", self.best_params["momentum_period"]],
            ["Top N Stocks", self.best_params["top_n_stocks"]],
            ["Rebalance Days", self.best_params["rebalance_days"]],
        ]
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
            "Strategy Parameter Experiments - Analysis", fontsize=16, fontweight="bold"
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

        # 4. Parameter Impact on Performance
        ax4 = plt.subplot(2, 3, 4)
        param_importance = {}
        for param in [
            "sma_fast",
            "sma_slow",
            "atr_period",
            "momentum_period",
            "top_n_stocks",
            "rebalance_days",
        ]:
            correlation = df[param].corr(df["composite_score"])
            param_importance[param] = abs(correlation)

        params = list(param_importance.keys())
        importance = list(param_importance.values())
        ax4.barh(params, importance, color="lightgreen")
        ax4.set_title("Parameter Importance (Correlation with Score)")
        ax4.set_xlabel("Absolute Correlation")

        # 5. Top N Stocks Impact
        ax5 = plt.subplot(2, 3, 5)
        top_n_performance = df.groupby("top_n_stocks")["composite_score"].mean()
        ax5.bar(
            top_n_performance.index, top_n_performance.values, color="orange", alpha=0.7
        )
        ax5.set_title("Performance by Top N Stocks")
        ax5.set_xlabel("Top N Stocks")
        ax5.set_ylabel("Average Composite Score")
        ax5.grid(True, alpha=0.3)

        # 6. Rebalance Frequency Impact
        ax6 = plt.subplot(2, 3, 6)
        rebalance_performance = df.groupby("rebalance_days")["composite_score"].mean()
        ax6.bar(
            rebalance_performance.index,
            rebalance_performance.values,
            color="purple",
            alpha=0.7,
        )
        ax6.set_title("Performance by Rebalance Frequency")
        ax6.set_xlabel("Rebalance Days")
        ax6.set_ylabel("Average Composite Score")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.results_dir, f"experiment_analysis_{timestamp}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"📊 Visualizations saved to {filename}")

        plt.show()

    def get_optimal_parameters(self):
        """Get the optimal parameters from experiments"""
        if self.best_params:
            return self.best_params
        else:
            print("❌ No experiments run yet. Run experiments first.")
            return None


def run_quick_experiments():
    """Run a quick set of experiments with limited parameters"""
    experiments = StrategyExperiments()
    experiments.run_experiments(
        max_combinations=20,  # Quick test with 20 combinations
        max_stocks=15,  # Use 15 stocks for speed
        use_parallel=True,
        max_workers=4,
    )
    experiments.create_visualizations()
    return experiments.get_optimal_parameters()


def run_comprehensive_experiments():
    """Run comprehensive experiments with full parameter space"""
    experiments = StrategyExperiments()
    experiments.run_experiments(
        max_combinations=100,  # More thorough testing
        max_stocks=30,  # More stocks for robustness
        use_parallel=True,
        max_workers=6,
    )
    experiments.create_visualizations()
    return experiments.get_optimal_parameters()


if __name__ == "__main__":
    print("🧪 Strategy Parameter Experiments")
    print("=" * 50)
    print("Choose experiment type:")
    print("1. Quick experiments (20 combinations, ~10 minutes)")
    print("2. Comprehensive experiments (100 combinations, ~30 minutes)")
    print("3. Custom experiments")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        print("🚀 Running quick experiments...")
        optimal_params = run_quick_experiments()
        print(f"\n🎯 Optimal parameters found: {optimal_params}")

    elif choice == "2":
        print("🚀 Running comprehensive experiments...")
        optimal_params = run_comprehensive_experiments()
        print(f"\n🎯 Optimal parameters found: {optimal_params}")

    elif choice == "3":
        print("🛠️ Custom experiments:")
        max_combinations = int(input("Max combinations to test: "))
        max_stocks = int(input("Max stocks to use: "))
        max_workers = int(input("Max parallel workers: "))

        experiments = StrategyExperiments()
        experiments.run_experiments(
            max_combinations=max_combinations,
            max_stocks=max_stocks,
            use_parallel=True,
            max_workers=max_workers,
        )
        experiments.create_visualizations()
        optimal_params = experiments.get_optimal_parameters()
        print(f"\n🎯 Optimal parameters found: {optimal_params}")

    else:
        print("❌ Invalid choice. Exiting.")
