"""
Unified Trading Strategies Runner

This script provides a unified interface to run any trading strategy from the strategies folder.
Users can easily add new strategies by inheriting from BaseStrategy and StrategyConfig classes.
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
    PMVMomentumStrategy,
    PMVMomentumConfig,
    NiftyShopStrategy,
    NiftyShopConfig,
)
from experiment_framework import UnifiedExperimentFramework
from utils import MarketDataLoader, setup_logger, IndianBrokerageCommission
from nifty_universe import (
    get_nifty_universe,
    get_available_universes,
    get_universe_info,
    get_sector_stocks,
    get_available_sectors,
    get_sector_info,
)


# Registry of available strategies
STRATEGY_REGISTRY = {
    "momentum": {
        "name": "Adaptive Momentum Strategy",
        "description": "Multi-stock momentum-based trend following strategy",
        "config_class": AdaptiveMomentumConfig,
    },
    "pmv": {
        "name": "P=MV Momentum Strategy",
        "description": "Momentum strategy with weekly exits using RSI, VWAP and returns",
        "config_class": PMVMomentumConfig,
    },
    "pairs": {
        "name": "Pairs Trading Strategy",
        "description": "Statistical arbitrage strategy for correlated pairs",
        "config_class": PairsConfig,
    },
    "mean_reversion": {
        "name": "Portfolio Mean Reversion Strategy",
        "description": "Multi-stock mean reversion strategy",
        "config_class": PortfolioMeanReversionConfig,
    },
    "statistical_trend": {
        "name": "Pure-equity Stat-Trend Hybrid Strategy",
        "description": "Z-score mean reversion with EMA and ADX momentum filters",
        "config_class": StatisticalTrendConfig,
    },
    "niftyshop": {
        "name": "Nifty Shop Strategy",
        "description": "Simple buy-below-MA, sell-on-target strategy with averaging down",
        "config_class": NiftyShopConfig,
    },
}


def get_default_stocks(universe="nifty50", sector=None):
    """Get default stock list based on universe or sector"""
    if sector:
        try:
            return get_sector_stocks(sector)
        except ValueError:
            print(f"⚠️ Unknown sector '{sector}', falling back to universe '{universe}'")

    try:
        return get_nifty_universe(universe)
    except ValueError:
        # Fallback to original list if universe not found
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


def list_strategies():
    """List all available strategies"""
    print("\n🎯 Available Trading Strategies:")
    print("=" * 50)

    for key, info in STRATEGY_REGISTRY.items():
        print(f"\n📊 {key.upper()}")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")


def list_universes():
    """List all available Nifty universes and sectors"""
    print("\n🌌 Available Nifty Universes:")
    print("=" * 50)

    universe_info = get_universe_info()
    for universe, count in universe_info.items():
        print(f"📈 {universe.upper()}: {count} stocks")

    print(f"\n🏢 Available Sectors:")
    print("=" * 50)

    sector_info = get_sector_info()
    for sector, count in sector_info.items():
        print(f"🔍 {sector.upper()}: {count} stocks")

    print(f"\n💡 Usage examples:")
    print(f"   --universe nifty50     (Use Nifty 50 stocks)")
    print(f"   --universe nifty100    (Use Nifty 100 stocks)")
    print(f"   --universe nifty200    (Use Nifty 200 stocks)")
    print(f"   --sector banking       (Use banking sector stocks)")
    print(f"   --sector it            (Use IT sector stocks)")
    print(f"   --symbols HDFCBANK.NS ICICIBANK.NS  (Custom stock selection)")


def run_strategy_backtest(
    strategy_key,
    symbols=None,
    start_date=None,
    end_date=None,
    initial_cash=1000000,
    interval="1d",
    universe="nifty50",
    sector=None,
    printlog=False,
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

    # Set default symbols if not provided
    if symbols is None:
        # Use sector or universe selection for symbol list
        symbols = get_default_stocks(universe, sector)
        if sector:
            print(f"📊 Using {sector.upper()} sector: {len(symbols)} stocks")
        else:
            print(f"📊 Using {universe.upper()} universe: {len(symbols)} stocks")

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

    # Get default parameters and add printlog
    default_params = config.get_default_params()
    default_params["printlog"] = printlog

    # Run single experiment with default parameters
    result = framework.run_single_experiment(
        params=default_params,
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
    universe="nifty50",
    sector=None,
    max_stocks=None,
    printlog=False,
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
        # Get full symbol list
        full_symbols = get_default_stocks(universe, sector)

        # Apply stock limit if specified, otherwise use reasonable defaults
        if max_stocks is not None:
            # User explicitly set max_stocks
            if max_stocks <= 0:
                symbols = full_symbols  # Use all stocks if max_stocks is 0 or negative
            else:
                symbols = full_symbols[:max_stocks]
        else:
            # Default behavior: limit to 15 for performance unless it's a small universe/sector
            if len(full_symbols) > 15:
                symbols = full_symbols[
                    :15
                ]  # Default limit for optimization performance
            else:
                symbols = full_symbols

        # Print information about stock selection
        if len(symbols) == len(full_symbols):
            if sector:
                print(
                    f"📊 Using all {len(symbols)} stocks from {sector.upper()} sector"
                )
            else:
                print(
                    f"📊 Using all {len(symbols)} stocks from {universe.upper()} universe"
                )
        else:
            if sector:
                print(
                    f"📊 Using {len(symbols)} out of {len(full_symbols)} stocks from {sector.upper()} sector for optimization"
                )
            else:
                print(
                    f"📊 Using {len(symbols)} out of {len(full_symbols)} stocks from {universe.upper()} universe for optimization"
                )

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
        "--list-universes",
        action="store_true",
        help="List available Nifty universes and sector pairs",
    )
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
    parser.add_argument(
        "--universe",
        type=str,
        choices=["nifty50", "nifty100", "nifty200"],
        default="nifty50",
        help="Nifty universe to use for stock selection (default: nifty50)",
    )
    parser.add_argument(
        "--sector",
        type=str,
        choices=get_available_sectors(),
        help="Sector to use for stock selection (overrides universe if specified)",
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
        "--max-stocks",
        type=int,
        default=0,
        help="Maximum number of stocks to use for optimization (0 or negative = use all stocks). Default: 0 all stocks",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (1d, 5m, 15m, 1h, etc.) - default: 1d. Note: Intraday data (5m, 15m) limited to last 60 days",
    )
    parser.add_argument(
        "--printlog",
        action="store_true",
        help="Enable debug logging for strategy execution (default: False)",
    )

    args = parser.parse_args()

    if args.list:
        list_strategies()
        return

    if args.list_universes:
        list_universes()
        return

    if not args.strategy:
        print("❌ Please specify a strategy. Use --list to see available strategies.")
        return

    # Validate universe
    if args.universe not in get_available_universes():
        print(
            f"❌ Invalid universe: {args.universe}. Available: {get_available_universes()}"
        )
        return

    # Validate sector if provided
    if args.sector and args.sector not in get_available_sectors():
        print(f"❌ Invalid sector: {args.sector}. Available: {get_available_sectors()}")
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
            universe=args.universe,
            sector=args.sector,
            max_stocks=args.max_stocks,
            printlog=args.printlog,
        )
    else:
        run_strategy_backtest(
            strategy_key=args.strategy,
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=args.initial_cash,
            interval=args.interval,
            universe=args.universe,
            sector=args.sector,
            printlog=args.printlog,
        )


if __name__ == "__main__":
    print("🚀 Unified Trading Strategies Runner")
    print("=" * 50)

    # If no command line arguments, show interactive menu
    if len(sys.argv) == 1:
        print("\n📋 Interactive Mode")
        list_strategies()
        print()
        list_universes()

        print("\n💡 Example commands:")
        print("   python unified_runner.py --list")
        print("   python unified_runner.py --list-universes")
        print("   python unified_runner.py --strategy momentum")
        print("   python unified_runner.py --strategy momentum --universe nifty100")
        print("   python unified_runner.py --strategy momentum --universe nifty200")
        print("   python unified_runner.py --strategy momentum --sector banking")
        print("   python unified_runner.py --strategy momentum --sector it")
        print(
            "   python unified_runner.py --strategy pairs --symbols HDFCBANK.NS ICICIBANK.NS"
        )
        print("   python unified_runner.py --strategy pairs --sector banking")
        print(
            "   python unified_runner.py --strategy momentum --optimize --universe nifty100"
        )
        print(
            "   python unified_runner.py --strategy momentum --optimize --sector pharma"
        )
        print(
            "   python unified_runner.py --strategy momentum --optimize --universe nifty100 --max-stocks 0  # Use all stocks"
        )
        print(
            "   python unified_runner.py --strategy momentum --optimize --universe nifty200 --max-stocks 50"
        )
        print("   python unified_runner.py --strategy momentum --initial-cash 500000")

        # Simple interactive mode
        strategy_key = (
            input(
                "\n🎯 Enter strategy name (momentum/pairs/mean_reversion/statistical_trend): "
            )
            .strip()
            .lower()
        )

        if strategy_key in STRATEGY_REGISTRY:
            # Ask for selection preference
            selection_type = (
                input("🔍 Select by (universe/sector/skip) [universe]: ")
                .strip()
                .lower()
            )
            if not selection_type:
                selection_type = "universe"

            universe = "nifty50"
            sector = None

            if selection_type == "sector":
                sector = (
                    input(
                        f"� Enter sector ({'/'.join(get_available_sectors())}) [banking]: "
                    )
                    .strip()
                    .lower()
                )
                if not sector:
                    sector = "banking"
                if sector not in get_available_sectors():
                    print(f"❌ Invalid sector. Using default: banking")
                    sector = "banking"
            elif selection_type == "universe":
                universe = (
                    input("�🌌 Enter universe (nifty50/nifty100/nifty200) [nifty50]: ")
                    .strip()
                    .lower()
                )
                if not universe:
                    universe = "nifty50"
                if universe not in get_available_universes():
                    print(f"❌ Invalid universe. Using default: nifty50")
                    universe = "nifty50"

            optimize = input("🔬 Run optimization? (y/N): ").strip().lower() == "y"

            if optimize:
                max_exp = input("🧪 Max experiments (50): ").strip()
                max_exp = int(max_exp) if max_exp.isdigit() else 50
                max_stocks_input = input(
                    "📊 Max stocks for optimization (15, or 0 for all): "
                ).strip()
                max_stocks = int(max_stocks_input) if max_stocks_input.isdigit() else 15
                if max_stocks == 0:
                    max_stocks = None  # Use all stocks
                run_strategy_optimization(
                    strategy_key,
                    max_experiments=max_exp,
                    universe=universe,
                    sector=sector,
                    max_stocks=max_stocks,
                    printlog=False,  # Default to False for interactive mode
                )
            else:
                run_strategy_backtest(
                    strategy_key, universe=universe, sector=sector, printlog=False
                )
        else:
            print("❌ Invalid strategy name")
    else:
        main()
