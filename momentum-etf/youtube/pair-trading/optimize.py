"""
Strategy Optimization Runner

Simple interface to run experiments and optimize strategy parameters.
"""

import sys
import os
from datetime import datetime


def print_header():
    """Print application header"""
    print("🚀 ADAPTIVE MOMENTUM STRATEGY OPTIMIZER")
    print("=" * 50)
    print("Find optimal parameters for your trading strategy!")
    print()


def print_menu():
    """Print main menu"""
    print("📋 MAIN MENU")
    print("-" * 20)
    print("1. 🧪 Run Quick Experiments (20 combinations, ~10 min)")
    print("2. 🔬 Run Comprehensive Experiments (100 combinations, ~30 min)")
    print("3. ⚙️ Custom Experiments")
    print("4. 🛠️ Parameter Manager")
    print("5. 📊 View Latest Results")
    print("6. 🎯 Apply Optimal Parameters")
    print("7. 📈 Run Strategy with Current Parameters")
    print("8. 📚 Help & Instructions")
    print("9. 🚪 Exit")
    print()


def run_experiments(experiment_type="quick"):
    """Run experiments based on type"""
    from experiments import (
        StrategyExperiments,
        run_quick_experiments,
        run_comprehensive_experiments,
    )

    print(f"🚀 Starting {experiment_type} experiments...")
    print("⏱️ This may take some time. Please be patient.")
    print()

    try:
        if experiment_type == "quick":
            optimal_params = run_quick_experiments()
        elif experiment_type == "comprehensive":
            optimal_params = run_comprehensive_experiments()
        else:  # custom
            print("🛠️ Custom Experiment Configuration:")
            max_combinations = int(
                input("Max parameter combinations to test (default 50): ") or "50"
            )
            max_stocks = int(input("Max stocks to use (default 25): ") or "25")
            max_workers = int(input("Max parallel workers (default 4): ") or "4")

            experiments = StrategyExperiments()
            experiments.run_experiments(
                max_combinations=max_combinations,
                max_stocks=max_stocks,
                use_parallel=True,
                max_workers=max_workers,
            )
            experiments.create_visualizations()
            optimal_params = experiments.get_optimal_parameters()

        if optimal_params:
            print("\n🎉 EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print(f"🎯 Optimal parameters: {optimal_params}")

            apply_now = (
                input("\n🤔 Apply optimal parameters now? (y/n): ").strip().lower()
            )
            if apply_now == "y":
                apply_optimal_parameters(optimal_params)

    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        print("💡 Try reducing the number of combinations or stocks")


def apply_optimal_parameters(optimal_params):
    """Apply optimal parameters to the strategy"""
    from parameter_manager import ParameterManager

    manager = ParameterManager()
    manager.apply_parameters(optimal_params, "Optimal from latest experiments")

    print("\n✅ Optimal parameters applied!")
    print("💡 You can now run the main strategy with these optimized settings.")


def run_parameter_manager():
    """Run the interactive parameter manager"""
    from parameter_manager import interactive_parameter_manager

    interactive_parameter_manager()


def view_latest_results():
    """View results from the latest experiments"""
    results_dir = "experiment_results"

    if not os.path.exists(results_dir):
        print("📁 No experiment results found.")
        print("💡 Run experiments first to generate results.")
        return

    # Find the latest results file
    json_files = [
        f
        for f in os.listdir(results_dir)
        if f.startswith("experiment_results_") and f.endswith(".json")
    ]

    if not json_files:
        print("📁 No experiment results found.")
        return

    latest_file = max(
        json_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f))
    )
    latest_path = os.path.join(results_dir, latest_file)

    print(f"📊 Latest Results: {latest_file}")
    print("=" * 50)

    import json

    with open(latest_path, "r") as f:
        results = json.load(f)

    if not results:
        print("❌ No results found in the file")
        return

    # Show summary
    print(f"📈 Total Experiments: {len(results)}")

    # Find best result
    best_result = max(results, key=lambda x: x.get("composite_score", -float("inf")))

    print(f"\n🏆 BEST RESULT:")
    print(f"   Composite Score: {best_result['composite_score']:.2f}")
    print(f"   Total Return: {best_result['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {best_result['max_drawdown']:.2f}%")

    print(f"\n⚙️ Best Parameters:")
    for param, value in best_result["params"].items():
        print(f"   {param}: {value}")

    # Option to apply
    apply_best = input("\n🤔 Apply these optimal parameters? (y/n): ").strip().lower()
    if apply_best == "y":
        apply_optimal_parameters(best_result["params"])


def run_main_strategy():
    """Run the main strategy with current parameters"""
    print("🚀 Running main strategy with current parameters...")
    print("⏱️ This will run the full backtest and generate charts.")
    print()

    proceed = input("Continue? (y/n): ").strip().lower()
    if proceed == "y":
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "adaptive_momentum.py"], capture_output=False
            )
            if result.returncode == 0:
                print("\n✅ Strategy execution completed!")
            else:
                print("\n❌ Strategy execution failed.")
        except Exception as e:
            print(f"❌ Error running strategy: {e}")


def show_help():
    """Show help and instructions"""
    help_text = """
📚 HELP & INSTRUCTIONS
======================

🎯 PURPOSE:
This tool helps you find optimal parameters for the Adaptive Momentum Strategy
by running systematic experiments with different parameter combinations.

🔄 WORKFLOW:
1. Run experiments to test different parameter combinations
2. Review results and identify optimal parameters
3. Apply optimal parameters to your strategy
4. Run the main strategy with optimized settings

🧪 EXPERIMENT TYPES:
• Quick: Tests 20 combinations with 15 stocks (~10 minutes)
• Comprehensive: Tests 100 combinations with 30 stocks (~30 minutes)
• Custom: You choose the number of combinations and stocks

📊 WHAT GETS OPTIMIZED:
• SMA Fast/Slow periods
• ATR period and multiplier
• Momentum period
• Number of top stocks to select
• Rebalancing frequency

🏆 SCORING:
Results are ranked by a composite score that considers:
• Total return (40% weight)
• Sharpe ratio (30% weight)
• Drawdown protection (30% weight)

💡 TIPS:
• Start with quick experiments to get a feel for the process
• Use comprehensive experiments for final optimization
• Save configurations before making changes
• View results visualization charts to understand patterns

📁 FILES GENERATED:
• experiment_results/: JSON and CSV files with detailed results
• config_backups/: Backup of parameter configurations
• Visualization charts showing parameter impact

🚨 IMPORTANT:
• Experiments use a shorter time period (2022-present) for speed
• Results may vary with different market conditions
• Always validate optimized parameters with recent data
• Consider transaction costs in real trading

❓ NEED HELP?
Check the generated visualization charts to understand which
parameters have the most impact on performance.
"""
    print(help_text)


def main():
    """Main application loop"""
    print_header()

    while True:
        print_menu()
        choice = input("Enter your choice (1-9): ").strip()

        try:
            if choice == "1":
                run_experiments("quick")

            elif choice == "2":
                run_experiments("comprehensive")

            elif choice == "3":
                run_experiments("custom")

            elif choice == "4":
                run_parameter_manager()

            elif choice == "5":
                view_latest_results()

            elif choice == "6":
                view_latest_results()  # This includes option to apply

            elif choice == "7":
                run_main_strategy()

            elif choice == "8":
                show_help()

            elif choice == "9":
                print("👋 Thank you for using the Strategy Optimizer!")
                print("🚀 Happy trading!")
                break

            else:
                print("❌ Invalid choice. Please enter a number between 1-9.")

        except KeyboardInterrupt:
            print("\n\n⏹️ Operation interrupted by user.")
            continue
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("💡 Please try again or choose a different option.")

        input("\nPress Enter to continue...")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
