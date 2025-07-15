"""
Unified Trading Strategies Runner

This script provides a unified interface to run any trading strategy from the strategies folder.
Users can easily add new strategies by inheriting from BaseStrategy and S    # Set default dates
    if start_date is None:
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            # Intraday data: Yahoo Finance limits to last 60 days for minute data
            start_date = datetime.now() - timedelta(days=55)  # Use 55 days to be safe
            print(f"⚠️ Using last 55 days for {interval} interval (Yahoo Finance limitation)")
        elif interval in ["1h"]:
            # Hourly data: Available for ~730 days
            start_date = datetime.now() - timedelta(days=700)  # Use 700 days to be safe
            print(f"⚠️ Using last 700 days for {interval} interval (Yahoo Finance limitation)")
        else:
            start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime.now()gyConfig classes.
"""

import sys
import os
from datetime import datetime, timedelta
import argparse

# Add strategies folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))

from strategies import (
    MomentumTrendStrategy,
    AdaptiveMomentumConfig,
    PairsStrategy,
    PairsConfig,
    PortfolioMeanReversionStrategy,
    PortfolioMeanReversionConfig,
    StatisticalTrendStrategy,
    StatisticalTrendConfig,
)
from experiment_framework import UnifiedExperimentFramework
from utils import MarketDataLoader, setup_logger, IndianBrokerageCommission


# Registry of available strategies
STRATEGY_REGISTRY = {
    "momentum": {
        "name": "Adaptive Momentum Strategy",
        "description": "Multi-stock momentum-based trend following strategy",
        "config_class": AdaptiveMomentumConfig,
        "requires_multiple_stocks": True,
    },
    "pairs": {
        "name": "Pairs Trading Strategy",
        "description": "Statistical arbitrage strategy for correlated pairs",
        "config_class": PairsConfig,
        "requires_multiple_stocks": False,
        "required_stocks": 2,
    },
    "mean_reversion": {
        "name": "Portfolio Mean Reversion Strategy",
        "description": "Multi-stock mean reversion strategy",
        "config_class": PortfolioMeanReversionConfig,
        "requires_multiple_stocks": True,
    },
    "statistical_trend": {
        "name": "Pure-equity Stat-Trend Hybrid Strategy",
        "description": "Z-score mean reversion with EMA and ADX momentum filters",
        "config_class": StatisticalTrendConfig,
        "requires_multiple_stocks": True,
    },
}


def get_default_stocks():
    """Get default stock list for testing"""
    return [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "KOTAKBANK.NS",
        "ITC.NS",
        "LT.NS",
        "ASIANPAINT.NS",
        "AXISBANK.NS",
        "MARUTI.NS",
        "TITAN.NS",
    ]


def get_pairs_stocks():
    """Get correlated pairs for pairs trading"""
    return [
        "HDFCBANK.NS",
        "ICICIBANK.NS",  # Banking pair
        # "RELIANCE.NS", "ONGC.NS",      # Oil & Gas pair
        # "TCS.NS", "INFY.NS",           # IT pair
    ]


def list_strategies():
    """List all available strategies"""
    print("\n🎯 Available Trading Strategies:")
    print("=" * 50)

    for key, info in STRATEGY_REGISTRY.items():
        print(f"\n📊 {key.upper()}")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        if "required_stocks" in info:
            print(f"   Required stocks: {info['required_stocks']}")
        elif info["requires_multiple_stocks"]:
            print(f"   Required stocks: Multiple (5+ recommended)")
        else:
            print(f"   Required stocks: 1")


def run_strategy_backtest(
    strategy_key,
    symbols=None,
    start_date=None,
    end_date=None,
    initial_cash=1000000,
    interval="1d",
):
    """Run a single strategy backtest"""
    if strategy_key not in STRATEGY_REGISTRY:
        print(
            f"❌ Strategy '{strategy_key}' not found. Use --list to see available strategies."
        )
        return

    strategy_info = STRATEGY_REGISTRY[strategy_key]
    config_class = strategy_info["config_class"]

    # Set default dates
    if start_date is None:
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            # Intraday data: Yahoo Finance limits to last 60 days for minute data
            start_date = datetime.now() - timedelta(days=55)  # Use 55 days to be safe
            print(
                f"⚠️ Using last 55 days for {interval} interval (Yahoo Finance limitation)"
            )
        elif interval in ["1h"]:
            # Hourly data: Available for ~730 days
            start_date = datetime.now() - timedelta(days=700)  # Use 700 days to be safe
            print(
                f"⚠️ Using last 700 days for {interval} interval (Yahoo Finance limitation)"
            )
        else:
            start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime.now()

    # Set default symbols based on strategy
    if symbols is None:
        if strategy_key == "pairs":
            symbols = get_pairs_stocks()
        else:
            symbols = get_default_stocks()

    # Validate symbol count
    if "required_stocks" in strategy_info:
        if len(symbols) != strategy_info["required_stocks"]:
            print(
                f"❌ {strategy_info['name']} requires exactly {strategy_info['required_stocks']} stocks"
            )
            return
    elif strategy_info["requires_multiple_stocks"] and len(symbols) < 3:
        print(
            f"❌ {strategy_info['name']} requires at least 3 stocks for proper diversification"
        )
        return

    print(f"\n🚀 Running {strategy_info['name']}")
    print(
        f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"💰 Initial Cash: ₹{initial_cash:,.2f}")
    print("=" * 60)

    # Create and run experiment
    config = config_class()
    framework = UnifiedExperimentFramework(config)

    # Run single experiment with default parameters
    result = framework.run_single_experiment(
        params=config.get_default_params(),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        interval=interval,
    )

    if result:
        print(f"\n✅ Backtest completed successfully!")
        print(f"📈 Total Return: {result.total_return:.2f}%")
        print(f"⚖️ Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"📉 Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"💰 Final Value: ₹{result.final_value:,.2f}")

        # Generate comprehensive visualization dashboard
        try:
            framework.create_portfolio_dashboard(result, symbols, start_date, end_date)
            print(f"📊 Portfolio performance dashboard generated successfully!")
        except Exception as e:
            print(f"⚠️ Could not generate portfolio dashboard: {e}")
    else:
        print("❌ Backtest failed")


def run_strategy_optimization(
    strategy_key,
    symbols=None,
    start_date=None,
    end_date=None,
    max_experiments=50,
    initial_cash=1000000,
    interval="1d",
):
    """Run strategy parameter optimization"""
    if strategy_key not in STRATEGY_REGISTRY:
        print(
            f"❌ Strategy '{strategy_key}' not found. Use --list to see available strategies."
        )
        return

    strategy_info = STRATEGY_REGISTRY[strategy_key]
    config_class = strategy_info["config_class"]

    # Set defaults
    if start_date is None:
        start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime.now()

    if symbols is None:
        if strategy_key == "pairs":
            symbols = get_pairs_stocks()
        else:
            symbols = get_default_stocks()[:10]  # Limit for optimization

    print(f"\n🔬 Optimizing {strategy_info['name']}")
    print(
        f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"💰 Initial Cash: ₹{initial_cash:,.2f}")
    print(f"🧪 Max experiments: {max_experiments}")
    print("=" * 60)

    # Create and run optimization
    config = config_class()
    framework = UnifiedExperimentFramework(config)

    framework.run_experiments(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_combinations=max_experiments,
        initial_cash=initial_cash,
        use_parallel=True,
        max_workers=4,
        interval=interval,
    )

    # Get sorted results by composite score
    results = sorted(framework.results, key=lambda x: x.composite_score, reverse=True)

    if results:
        print(f"\n✅ Optimization completed! Found {len(results)} valid results.")

        # Show top 5 results
        print("\n🏆 Top 5 Parameter Combinations:")
        print("-" * 80)
        for i, result in enumerate(results[:5], 1):
            print(f"\n#{i} - Composite Score: {result.composite_score:.4f}")
            print(
                f"   📈 Return: {result.total_return:.2f}% | ⚖️ Sharpe: {result.sharpe_ratio:.3f} | 📉 Drawdown: {result.max_drawdown:.2f}%"
            )
            params_str = ", ".join([f"{k}={v}" for k, v in result.params.items()])
            print(f"   🎛️ Parameters: {params_str}")

        # Generate comprehensive portfolio dashboard for best result
        try:
            best_result = results[0]
            # Create detailed portfolio dashboard for best result
            framework.create_portfolio_dashboard(
                best_result, symbols, start_date, end_date
            )
            print(f"📊 Best result portfolio dashboard generated!")
        except Exception as e:
            print(f"⚠️ Could not generate portfolio dashboard: {e}")
    else:
        print("❌ Optimization failed")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Unified Trading Strategies Runner")
    parser.add_argument("--list", action="store_true", help="List available strategies")
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy to run (momentum, pairs, mean_reversion, statistical_trend)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run parameter optimization instead of single backtest",
    )
    parser.add_argument(
        "--symbols", nargs="+", help="List of stock symbols (e.g., RELIANCE.NS TCS.NS)"
    )
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1000000,
        help="Initial cash amount (default: 1000000)",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=50,
        help="Maximum experiments for optimization",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (1d, 5m, 15m, 1h, etc.) - default: 1d. Note: Intraday data (5m, 15m) limited to last 60 days",
    )

    args = parser.parse_args()

    if args.list:
        list_strategies()
        return

    if not args.strategy:
        print("❌ Please specify a strategy. Use --list to see available strategies.")
        return

    # Validate and warn about interval limitations
    if args.interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        print(
            f"⚠️ Warning: {args.interval} interval data is limited to the last 60 days by Yahoo Finance"
        )
        if args.start_date and datetime.strptime(
            args.start_date, "%Y-%m-%d"
        ) < datetime.now() - timedelta(days=60):
            print(
                f"❌ Error: Start date too old for {args.interval} interval. Maximum 60 days back from today."
            )
            return
    elif args.interval in ["1h"]:
        print(
            f"⚠️ Warning: {args.interval} interval data is limited to approximately 2 years by Yahoo Finance"
        )
        if args.start_date and datetime.strptime(
            args.start_date, "%Y-%m-%d"
        ) < datetime.now() - timedelta(days=730):
            print(
                f"❌ Error: Start date too old for {args.interval} interval. Maximum ~2 years back from today."
            )
            return

    # Parse dates
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid start date format. Use YYYY-MM-DD")
            return

    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid end date format. Use YYYY-MM-DD")
            return

    # Run strategy
    if args.optimize:
        run_strategy_optimization(
            strategy_key=args.strategy,
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            max_experiments=args.max_experiments,
            initial_cash=args.initial_cash,
            interval=args.interval,
        )
    else:
        run_strategy_backtest(
            strategy_key=args.strategy,
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=args.initial_cash,
            interval=args.interval,
        )


if __name__ == "__main__":
    print("🚀 Unified Trading Strategies Runner")
    print("=" * 50)

    # If no command line arguments, show interactive menu
    if len(sys.argv) == 1:
        print("\n📋 Interactive Mode")
        list_strategies()

        print("\n💡 Example commands:")
        print("   python unified_runner.py --list")
        print("   python unified_runner.py --strategy momentum")
        print("   python unified_runner.py --strategy momentum --interval 5m")
        print(
            "   python unified_runner.py --strategy pairs --symbols HDFCBANK.NS ICICIBANK.NS"
        )
        print(
            "   python unified_runner.py --strategy momentum --optimize --max-experiments 20"
        )
        print("   python unified_runner.py --strategy momentum --initial-cash 500000")
        print(
            "   python unified_runner.py --strategy pairs --initial-cash 2000000 --interval 15m"
        )

        # Simple interactive mode
        strategy_key = (
            input(
                "\n🎯 Enter strategy name (momentum/pairs/mean_reversion/statistical_trend): "
            )
            .strip()
            .lower()
        )

        if strategy_key in STRATEGY_REGISTRY:
            optimize = input("🔬 Run optimization? (y/N): ").strip().lower() == "y"

            if optimize:
                max_exp = input("🧪 Max experiments (50): ").strip()
                max_exp = int(max_exp) if max_exp.isdigit() else 50
                run_strategy_optimization(strategy_key, max_experiments=max_exp)
            else:
                run_strategy_backtest(strategy_key)
        else:
            print("❌ Invalid strategy name")
    else:
        main()
