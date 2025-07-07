#!/usr/bin/env python3
"""
ETF Portfolio CLI - Simple 4-command interface for ETF momentum portfolio management

Usage with uv:
    uv run cli.py portfolio               # Show current optimal portfolio with allocations
    uv run cli.py rebalance               # Show rebalancing needed for existing portfolio
    uv run cli.py historical              # Show portfolio changes between dates
    uv run cli.py backtest                # Run historical backtest
"""

import argparse
import sys
from datetime import datetime, timedelta
from tabulate import tabulate
from etf_momentum_strategy import (
    StrategyConfig,
    ETFMomentumStrategy,
    run_multi_investment_backtest,
)


def show_current_portfolio(investment_amount=1000000, portfolio_size=5):
    """Show current optimal portfolio with exact allocations."""

    print(f"💼 CURRENT OPTIMAL ETF PORTFOLIO")
    print(f"💰 Investment Amount: ₹{investment_amount:,.2f}")
    print("=" * 70)

    # Get current momentum rankings
    config = StrategyConfig(portfolio_size=portfolio_size)
    strategy = ETFMomentumStrategy(config)

    try:
        status = strategy.get_current_portfolio_status()

        if status["status"] == "error":
            print(f"❌ Error: {status['message']}")
            return

        current_prices = status["current_prices"]
        top_etfs = status["top_10_momentum_scores"][:portfolio_size]

        print(f"📅 Data Date: {status['market_data_date']}")
        print(f"🎯 Portfolio Size: {portfolio_size} ETFs")
        print(f"📊 Equal Weight Allocation: {100/portfolio_size:.1f}% per ETF")

        print(f"\n📈 OPTIMAL PORTFOLIO ALLOCATION:")

        # Prepare data for table
        table_data = []
        total_investment = 0

        for i, (ticker, score) in enumerate(top_etfs, 1):
            price = current_prices.get(ticker, 0)
            allocation = investment_amount / portfolio_size
            units = int(allocation / price) if price > 0 else 0
            actual_investment = units * price
            weight = (
                (actual_investment / investment_amount) * 100
                if investment_amount > 0
                else 0
            )
            total_investment += actual_investment

            table_data.append(
                [
                    i,
                    ticker,
                    f"₹{price:.2f}",
                    f"{units:,}",
                    f"₹{actual_investment:,.0f}",
                    f"{weight:.1f}%",
                ]
            )

        # Add total row
        cash_remaining = investment_amount - total_investment
        table_data.append(
            [
                "",
                "TOTAL",
                "",
                "",
                f"₹{total_investment:,.0f}",
                f"{(total_investment/investment_amount*100):.1f}%",
            ]
        )

        headers = ["Rank", "ETF Name", "Price", "Units", "Investment", "Weight"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print(
            f"\n💰 Cash Remaining: ₹{cash_remaining:,.0f} ({(cash_remaining/investment_amount*100):.1f}%)"
        )

        print(f"\n📊 MOMENTUM SCORES:")

        momentum_data = []
        for i, (ticker, score) in enumerate(top_etfs, 1):
            momentum_data.append([i, ticker, f"{score:.4f}"])

        print(
            tabulate(momentum_data, headers=["Rank", "ETF", "Score"], tablefmt="simple")
        )

        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Buy the above ETFs in specified quantities")
        print(f"   2. Monitor monthly for rebalancing needs")
        print(f"   3. Use 'rebalance' command if you already have a portfolio")

    except Exception as e:
        print(f"❌ Error: {e}")


def show_rebalancing_needs():
    """Show what rebalancing is needed for an existing portfolio."""

    print(f"🔄 PORTFOLIO REBALANCING ANALYSIS")
    print("=" * 70)
    print(
        "💡 To use this feature, please edit the function to input your current holdings"
    )
    print("📝 Example format needed:")
    print("   current_holdings = {")
    print("       'NIFTYBEES.NS': 350,    # number of units you own")
    print("       'BANKBEES.NS': 171,")
    print("       'GOLDBEES.NS': 1240")
    print("   }")

    # For now, show what the optimal portfolio should be
    print(f"\n🎯 CURRENT OPTIMAL ALLOCATION:")
    show_current_portfolio(1000000, 5)

    print(f"\n⚠️  TO IMPLEMENT:")
    print(f"   • Add your current holdings in the code")
    print(f"   • System will calculate what to buy/sell")
    print(f"   • Shows exact rebalancing actions needed")


def show_historical_portfolio(
    from_date_str, to_date_str=None, investment_amount=1000000, portfolio_size=5
):
    """Show what the optimal portfolio was on specific dates."""

    # Parse dates
    try:
        from_date = datetime.strptime(from_date_str, "%Y-%m-%d")
        if to_date_str:
            to_date = datetime.strptime(to_date_str, "%Y-%m-%d")
        else:
            to_date = datetime.now()

        # Validate dates
        today = datetime.now()
        if to_date > today:
            to_date = today
            print(f"⚠️  To-date adjusted to today: {to_date.strftime('%Y-%m-%d')}")

        if from_date > to_date:
            print(
                f"❌ Error: From-date ({from_date_str}) cannot be after To-date ({to_date.strftime('%Y-%m-%d')})"
            )
            return

    except ValueError as e:
        print(f"❌ Error: Invalid date format. Use YYYY-MM-DD format. {e}")
        return

    print(f"📅 HISTORICAL PORTFOLIO ANALYSIS")
    print(
        f"📊 From: {from_date.strftime('%Y-%m-%d')} To: {to_date.strftime('%Y-%m-%d')}"
    )
    print(f"💰 Investment Amount: ₹{investment_amount:,.2f}")
    print("=" * 80)

    config = StrategyConfig(portfolio_size=portfolio_size)
    strategy = ETFMomentumStrategy(config)

    try:
        # Fetch historical data with extra buffer for momentum calculation
        data_start = from_date - timedelta(days=config.min_data_points + 100)

        print(
            f"📡 Fetching historical data from {data_start.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}..."
        )

        all_data = strategy.data_provider.fetch_etf_data(
            config.etf_universe, data_start, to_date
        )

        prices_df = strategy.data_provider.get_prices(all_data)

        # Convert timezone-aware index to timezone-naive
        if prices_df.index.tz is not None:
            prices_df.index = prices_df.index.tz_localize(None)

        # Get data up to from_date for initial portfolio
        from_data = prices_df[prices_df.index <= from_date]
        if from_data.empty:
            print(f"❌ No data available for from-date: {from_date_str}")
            return

        print(f"\n📈 PORTFOLIO ON {from_date.strftime('%Y-%m-%d')}:")
        print("-" * 60)

        # Calculate momentum scores for from_date
        eligible_tickers = strategy.momentum_calculator.apply_filters(from_data, None)
        if eligible_tickers:
            eligible_from_data = from_data[eligible_tickers]
            from_momentum_scores = (
                strategy.momentum_calculator.calculate_momentum_scores(
                    eligible_from_data
                )
            )
            from_top_etfs = [
                (ticker, score) for ticker, score in from_momentum_scores.items()
            ][:portfolio_size]
            from_prices = from_data.iloc[-1]

            # Prepare table data
            from_table_data = []
            from_total = 0
            for i, (ticker, score) in enumerate(from_top_etfs, 1):
                price = from_prices.get(ticker, 0)
                allocation = investment_amount / portfolio_size
                units = int(allocation / price) if price > 0 else 0
                actual_investment = units * price
                from_total += actual_investment

                from_table_data.append(
                    [
                        i,
                        ticker,
                        f"₹{price:.2f}",
                        f"{units:,}",
                        f"₹{actual_investment:,.0f}",
                        f"{score:.4f}",
                    ]
                )

            headers = ["Rank", "ETF Name", "Price", "Units", "Investment", "Score"]
            print(tabulate(from_table_data, headers=headers, tablefmt="grid"))
        else:
            print("❌ No eligible ETFs found for from-date")
            return

        # Get data up to to_date for final portfolio
        to_data = prices_df[prices_df.index <= to_date]
        if to_data.empty:
            print(f"❌ No data available for to-date: {to_date_str}")
            return

        print(f"\n📈 PORTFOLIO ON {to_date.strftime('%Y-%m-%d')}:")
        print("-" * 60)

        # Calculate momentum scores for to_date
        eligible_tickers = strategy.momentum_calculator.apply_filters(to_data, None)
        if eligible_tickers:
            eligible_to_data = to_data[eligible_tickers]
            to_momentum_scores = strategy.momentum_calculator.calculate_momentum_scores(
                eligible_to_data
            )
            to_top_etfs = [
                (ticker, score) for ticker, score in to_momentum_scores.items()
            ][:portfolio_size]
            to_prices = to_data.iloc[-1]

            # Prepare table data
            to_table_data = []
            to_total = 0
            for i, (ticker, score) in enumerate(to_top_etfs, 1):
                price = to_prices.get(ticker, 0)
                allocation = investment_amount / portfolio_size
                units = int(allocation / price) if price > 0 else 0
                actual_investment = units * price
                to_total += actual_investment

                to_table_data.append(
                    [
                        i,
                        ticker,
                        f"₹{price:.2f}",
                        f"{units:,}",
                        f"₹{actual_investment:,.0f}",
                        f"{score:.4f}",
                    ]
                )

            headers = ["Rank", "ETF Name", "Price", "Units", "Investment", "Score"]
            print(tabulate(to_table_data, headers=headers, tablefmt="grid"))
        else:
            print("❌ No eligible ETFs found for to-date")
            return

        # Show changes needed
        print(f"\n🔄 REBALANCING CHANGES NEEDED:")
        print("-" * 60)

        from_tickers = set([ticker for ticker, _ in from_top_etfs])
        to_tickers = set([ticker for ticker, _ in to_top_etfs])

        # ETFs to sell (in from but not in to)
        to_sell = from_tickers - to_tickers
        if to_sell:
            print(f"❌ SELL (no longer in top {portfolio_size}):")
            for ticker in to_sell:
                from_units = int(
                    (investment_amount / portfolio_size) / from_prices[ticker]
                )
                print(f"   • {ticker}: {from_units} units")

        # ETFs to buy (in to but not in from)
        to_buy = to_tickers - from_tickers
        if to_buy:
            print(f"✅ BUY (new entries to top {portfolio_size}):")
            for ticker in to_buy:
                to_units = int((investment_amount / portfolio_size) / to_prices[ticker])
                print(f"   • {ticker}: {to_units} units")

        # ETFs that remained (adjust quantities)
        remained = from_tickers & to_tickers
        if remained:
            print(f"🔄 ADJUST (remained in portfolio):")
            for ticker in remained:
                from_units = int(
                    (investment_amount / portfolio_size) / from_prices[ticker]
                )
                to_units = int((investment_amount / portfolio_size) / to_prices[ticker])
                diff = to_units - from_units
                if diff > 0:
                    print(f"   • {ticker}: BUY {diff} more units")
                elif diff < 0:
                    print(f"   • {ticker}: SELL {abs(diff)} units")
                else:
                    print(f"   • {ticker}: No change needed")

        if not to_sell and not to_buy and not remained:
            print("✅ No changes needed - portfolio remained optimal")

        # Calculate period performance if we have price data
        print(f"\n📊 PERIOD PERFORMANCE SUMMARY:")
        print("-" * 40)
        days_diff = (to_date - from_date).days
        print(f"📅 Period: {days_diff} days")

        # Calculate overall portfolio performance
        from_portfolio_value = 0
        to_portfolio_value = 0

        for ticker, _ in from_top_etfs:
            from_price = from_prices.get(ticker, 0)
            allocation = investment_amount / portfolio_size
            units = int(allocation / from_price) if from_price > 0 else 0
            from_portfolio_value += units * from_price

            # Calculate value of same units at to_date
            if ticker in to_prices.index:
                to_price = to_prices[ticker]
                to_portfolio_value += units * to_price

        # Overall portfolio performance
        if from_portfolio_value > 0:
            portfolio_return = ((to_portfolio_value / from_portfolio_value) - 1) * 100
            portfolio_gain_loss = to_portfolio_value - from_portfolio_value

            print(f"\n💰 OVERALL PORTFOLIO PERFORMANCE:")
            print(f"   From Portfolio Value: ₹{from_portfolio_value:,.0f}")
            print(f"   To Portfolio Value:   ₹{to_portfolio_value:,.0f}")
            print(f"   Absolute Gain/Loss:   ₹{portfolio_gain_loss:+,.0f}")
            print(f"   Percentage Return:    {portfolio_return:+.2f}%")

            # Annualized return if period is long enough
            if days_diff >= 30:
                annualized_return = (
                    (to_portfolio_value / from_portfolio_value) ** (365 / days_diff) - 1
                ) * 100
                print(f"   Annualized Return:    {annualized_return:+.2f}%")

        # Show how each ETF performed during the holding period
        print(f"\n📈 ETF PERFORMANCE DURING PERIOD:")

        # ETFs held for entire period
        held_entire_period = from_tickers & to_tickers
        performance_data = []

        if held_entire_period:
            print(f"🔄 HELD FOR ENTIRE PERIOD:")
            for ticker in held_entire_period:
                if ticker in from_prices.index and ticker in to_prices.index:
                    from_price = from_prices[ticker]
                    to_price = to_prices[ticker]
                    return_pct = ((to_price / from_price) - 1) * 100
                    performance_data.append(
                        [
                            ticker,
                            f"₹{from_price:.2f}",
                            f"₹{to_price:.2f}",
                            f"{return_pct:+.1f}%",
                        ]
                    )

            if performance_data:
                headers = ["ETF", "From Price", "To Price", "Return"]
                print(tabulate(performance_data, headers=headers, tablefmt="grid"))

        # ETFs that were sold (price movement after we sold them)
        sold_etfs = from_tickers - to_tickers
        if sold_etfs:
            print(f"\n❌ SOLD DURING PERIOD (price movement after exit):")
            sold_data = []
            for ticker in sold_etfs:
                if ticker in from_prices.index and ticker in to_prices.index:
                    from_price = from_prices[ticker]
                    to_price = to_prices[ticker]
                    return_pct = ((to_price / from_price) - 1) * 100
                    sold_data.append(
                        [
                            ticker,
                            f"₹{from_price:.2f}",
                            f"₹{to_price:.2f}",
                            f"{return_pct:+.1f}%",
                        ]
                    )

            if sold_data:
                headers = ["ETF", "Exit Price", "Current Price", "Missed Return"]
                print(tabulate(sold_data, headers=headers, tablefmt="grid"))

        # ETFs that were bought (we didn't own them initially)
        bought_etfs = to_tickers - from_tickers
        if bought_etfs:
            print(f"\n✅ NEWLY ADDED (not held during period):")
            bought_data = []
            for ticker in bought_etfs:
                if ticker in from_prices.index and ticker in to_prices.index:
                    from_price = from_prices[ticker]
                    to_price = to_prices[ticker]
                    return_pct = ((to_price / from_price) - 1) * 100
                    bought_data.append(
                        [
                            ticker,
                            f"₹{from_price:.2f}",
                            f"₹{to_price:.2f}",
                            f"{return_pct:+.1f}% (not realized)",
                        ]
                    )

            if bought_data:
                headers = [
                    "ETF",
                    "Period Start Price",
                    "Entry Price",
                    "Period Performance",
                ]
                print(tabulate(bought_data, headers=headers, tablefmt="grid"))

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ETF Momentum Portfolio Management - Simple 4-Command Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run cli.py portfolio                # Show current optimal portfolio with allocations
  uv run cli.py portfolio --amount 500000 --size 7  # Custom amount and portfolio size
  uv run cli.py rebalance               # Show rebalancing needed for existing portfolio
  uv run cli.py historical --from-date 2024-01-01  # Portfolio changes from Jan 1 to today
  uv run cli.py historical --from-date 2024-01-01 --to-date 2024-06-30  # Portfolio changes between dates
  uv run cli.py backtest                # Run historical backtest
  uv run cli.py backtest --amounts 1000000 5000000  # Test specific amounts
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Portfolio command - Show current optimal portfolio
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Show current optimal portfolio with allocations"
    )
    portfolio_parser.add_argument(
        "--amount",
        type=float,
        default=1000000,
        help="Investment amount in INR (default: 1000000)",
    )
    portfolio_parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Portfolio size (number of ETFs to hold, default: 5)",
    )

    # Rebalance command - Show rebalancing needs
    rebalance_parser = subparsers.add_parser(
        "rebalance", help="Show rebalancing needed for existing portfolio"
    )

    # Historical command - Show portfolio between two dates
    historical_parser = subparsers.add_parser(
        "historical", help="Show portfolio changes between two dates"
    )
    historical_parser.add_argument(
        "--from-date", required=True, help="Start date in YYYY-MM-DD format"
    )
    historical_parser.add_argument(
        "--to-date", help="End date in YYYY-MM-DD format (default: today)"
    )
    historical_parser.add_argument(
        "--amount",
        type=float,
        default=1000000,
        help="Investment amount in INR (default: 1000000)",
    )
    historical_parser.add_argument(
        "--size", type=int, default=5, help="Portfolio size (default: 5)"
    )

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    backtest_parser.add_argument(
        "--amounts",
        nargs="+",
        type=float,
        default=[1000000, 2000000, 5000000],
        help="Initial investment amounts to test",
    )
    # New: threshold-based rebalancing options
    backtest_parser.add_argument(
        "--use-threshold",
        action="store_true",
        help="Enable profit/loss threshold-based rebalancing",
    )
    backtest_parser.add_argument(
        "--profit-threshold",
        type=float,
        default=10.0,
        help="Profit threshold percent for rebalancing (default: 10.0)",
    )
    backtest_parser.add_argument(
        "--loss-threshold",
        type=float,
        default=-5.0,
        help="Loss threshold percent for rebalancing (default: -5.0)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "portfolio":
            show_current_portfolio(args.amount, args.size)

        elif args.command == "rebalance":
            show_rebalancing_needs()

        elif args.command == "historical":
            show_historical_portfolio(
                getattr(args, "from_date"),
                getattr(args, "to_date"),
                args.amount,
                args.size,
            )

        elif args.command == "backtest":
            print(f"📊 Running backtest with amounts: {args.amounts}")
            # Pass threshold options to run_multi_investment_backtest
            run_multi_investment_backtest(
                args.amounts,
                use_threshold_rebalancing=getattr(args, "use_threshold", False),
                profit_threshold_pct=getattr(args, "profit_threshold", 10.0),
                loss_threshold_pct=getattr(args, "loss_threshold", -5.0),
            )

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection for data fetching.")
        print("💡 Run with: uv run cli.py <command>")
        sys.exit(1)


if __name__ == "__main__":
    main()
