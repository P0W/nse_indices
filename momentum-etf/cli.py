#!/usr/bin/env python3
"""
ETF Portfolio CLI Functions - Individual entry points for ETF momentum portfolio management
Author : Prashant Srivastava
"""

import argparse
import sys
from datetime import datetime, timedelta
from tabulate import tabulate
from etf_momentum_strategy import (
    ETFMomentumStrategy,
    run_parameter_experiments,
)
from core import StrategyConfig


# Shared parser factories to eliminate duplication


def _create_portfolio_parser(parser):
    """Add portfolio command arguments to a parser."""
    parser.add_argument(
        "--amount",
        type=float,
        default=1000000,
        help="Investment amount (default: 1000000)",
    )
    parser.add_argument(
        "--size", type=int, default=5, help="Portfolio size (default: 5)"
    )
    return parser


def _create_rebalance_parser(parser):
    """Add rebalance command arguments to a parser."""
    parser.add_argument(
        "--holdings-file",
        type=str,
        required=True,
        help="Path to JSON, CSV, or Smallcase export file containing current holdings",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        required=False,
        help="Purchase date in YYYY-MM-DD format (used for price lookup when price is -1, not needed for Smallcase exports)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=None,
        help="Target investment amount for rebalancing (default: use current portfolio value)",
    )
    parser.add_argument(
        "--size", type=int, default=5, help="Portfolio size (default: 5)"
    )
    return parser


def _create_historical_parser(parser):
    """Add historical command arguments to a parser."""
    parser.add_argument(
        "--from-date", required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--to-date", help="End date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=1000000,
        help="Investment amount in INR (default: 1000000)",
    )
    parser.add_argument(
        "--size", type=int, default=5, help="Portfolio size (default: 5)"
    )
    return parser


def _create_backtest_parser(parser):
    """Add backtest command arguments to a parser."""
    parser.add_argument(
        "--amounts",
        nargs="+",
        type=float,
        default=[1000000],
        help="Initial investment amounts to test (default: 1000000)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5],
        help="Portfolio size (number of ETFs to hold, default: 5)",
    )
    parser.add_argument(
        "--use-threshold",
        action="store_true",
        help="Enable profit/loss threshold-based rebalancing",
    )
    parser.add_argument(
        "--profit-threshold",
        type=float,
        default=10.0,
        help="Profit threshold percent for rebalancing (default: 10.0)",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=-5.0,
        help="Loss threshold percent for rebalancing (default: -5.0)",
    )
    return parser


# Individual CLI entry points using shared parsers


def portfolio_cli():
    """Entry point for portfolio command with CLI arguments."""
    parser = argparse.ArgumentParser(description="Show current optimal ETF portfolio")
    parser = _create_portfolio_parser(parser)
    args = parser.parse_args()
    show_current_portfolio(investment_amount=args.amount, portfolio_size=args.size)


def rebalance_cli():
    """Entry point for rebalance command with CLI arguments."""
    parser = argparse.ArgumentParser(description="Show portfolio rebalancing analysis")
    parser = _create_rebalance_parser(parser)
    args = parser.parse_args()
    show_rebalancing_needs(
        holdings_file=args.holdings_file,
        from_date=args.from_date,
        target_amount=args.amount,
        portfolio_size=args.size,
    )


def history_cli():
    """Entry point for historical command with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Show portfolio changes between two dates"
    )
    parser = _create_historical_parser(parser)
    args = parser.parse_args()
    show_historical_portfolio(
        from_date_str=args.from_date,
        to_date_str=args.to_date,
        investment_amount=args.amount,
        portfolio_size=args.size,
    )


def backtest_cli():
    """Entry point for backtest command with CLI arguments."""
    parser = argparse.ArgumentParser(description="Quick ETF Momentum Strategy Backtest")
    parser = _create_backtest_parser(parser)
    args = parser.parse_args()

    print(f"🚀 Quick Backtest - Portfolio Size: {args.sizes}, Amounts: {args.amounts}")
    if args.use_threshold:
        print(
            f"📊 Threshold Rebalancing: Profit {args.profit_threshold}%, Loss {args.loss_threshold}%"
        )

    # Run the backtest
    run_parameter_experiments(
        investment_amounts=args.amounts,
        portfolio_sizes=args.sizes,
        use_threshold_rebalancing_values=[args.use_threshold],
        profit_threshold_pct=args.profit_threshold,
        loss_threshold_pct=args.loss_threshold,
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

        # Filter to only include ETFs with positive momentum scores
        top_etfs = [(t, s) for t, s in top_etfs if s > 0]
        actual_size = len(top_etfs)

        if actual_size == 0:
            print("❌ No ETFs have positive momentum. Consider holding cash.")
            return

        print(f"📅 Data Date: {status['market_data_date']}")

        if actual_size < portfolio_size:
            print(f"⚠️  Only {actual_size} ETFs have positive momentum (requested {portfolio_size})")
            print(f"   Allocating to {actual_size} ETFs to avoid investing in declining assets")

        print(f"🎯 Portfolio Size: {actual_size} ETFs")
        print(f"📊 Equal Weight Allocation: {100/actual_size:.1f}% per ETF")

        print(f"\n📈 OPTIMAL PORTFOLIO ALLOCATION:")

        # Prepare data for table
        table_data = []
        total_investment = 0

        for i, (ticker, score) in enumerate(top_etfs, 1):
            price = current_prices.get(ticker, 0)
            allocation = investment_amount / actual_size
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


def show_rebalancing_needs(holdings_file, from_date, target_amount=None, portfolio_size=5):
    """Show what rebalancing is needed for an existing portfolio."""
    import json
    import csv
    import pandas as pd
    from pathlib import Path

    print(f"🔄 PORTFOLIO REBALANCING ANALYSIS")
    print("=" * 70)

    # Parse from_date (optional for Smallcase exports)
    purchase_date = None
    if from_date:
        try:
            purchase_date = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Error: Invalid from-date format. Use YYYY-MM-DD format.")
            return

    current_date = datetime.now()
    if purchase_date:
        print(f"📅 Purchase Date: {purchase_date.strftime('%Y-%m-%d')}")
    print(f"📅 Current Date: {current_date.strftime('%Y-%m-%d')}")
    print(f"🎯 Target Portfolio Size: {portfolio_size} ETFs")

    # Load holdings from file
    try:
        holdings_path = Path(holdings_file)
        if not holdings_path.exists():
            print(f"❌ Error: Holdings file not found: {holdings_file}")
            return

        current_holdings = {}

        if holdings_path.suffix.lower() == ".json":
            # Load from JSON
            with open(holdings_path, "r") as f:
                data = json.load(f)

            # Expected format: [{"symbol": "NIFTYBEES.NS", "units": 350, "price": 120.50}, ...]
            # or {"NIFTYBEES.NS": {"units": 350, "price": 120.50}, ...}
            if isinstance(data, list):
                for item in data:
                    symbol = item.get("symbol", "").strip()
                    units = item.get("units", 0)
                    price = item.get("price", -1)
                    if symbol and units > 0:
                        current_holdings[symbol] = {
                            "units": units,
                            "purchase_price": price,
                        }
            elif isinstance(data, dict):
                for symbol, details in data.items():
                    if isinstance(details, dict):
                        units = details.get("units", 0)
                        price = details.get("price", -1)
                        if units > 0:
                            current_holdings[symbol] = {
                                "units": units,
                                "purchase_price": price,
                            }
                    else:
                        # Simple format: {"NIFTYBEES.NS": 350, ...}
                        current_holdings[symbol] = {
                            "units": details,
                            "purchase_price": -1,
                        }

        elif holdings_path.suffix.lower() == ".csv":
            # Load from CSV
            # First, try to detect if this is a Smallcase export by checking for the data structure
            try:
                # Read the first few lines to detect format
                with open(holdings_path, "r") as f:
                    lines = f.readlines()

                # Look for the header line with Ticker, Shares, etc.
                # Smallcase exports may have different column name formats
                header_line_idx = None
                for i, line in enumerate(lines):
                    if (
                        "Ticker" in line
                        and "Shares" in line
                        and ("Avg Buy Price" in line or "Avg. Buy Price" in line)
                    ):
                        header_line_idx = i
                        break

                if header_line_idx is not None:
                    # This is a Smallcase export - read from the header line
                    df = pd.read_csv(holdings_path, skiprows=header_line_idx)
                    print(
                        f"📊 Detected Smallcase export format (skipped {header_line_idx} header rows)"
                    )

                    # Find the Avg Buy Price column (handles different formats)
                    avg_buy_price_col = None
                    for col in df.columns:
                        if "Avg" in col and "Buy" in col and "Price" in col:
                            avg_buy_price_col = col
                            break

                    # Find the Current Price column (handles different formats)
                    current_price_col = None
                    for col in df.columns:
                        if "Current" in col and "Price" in col:
                            current_price_col = col
                            break

                    # Check if this is a Smallcase export format
                    if (
                        "Ticker" in df.columns
                        and "Shares" in df.columns
                        and avg_buy_price_col is not None
                    ):
                        # Smallcase export format: Ticker, Shares, Avg Buy Price, Current Price
                        for _, row in df.iterrows():
                            ticker = str(row["Ticker"]).strip()
                            shares = row["Shares"]
                            avg_buy_price = row[avg_buy_price_col]
                            # Get current price from CSV if available
                            csv_current_price = row[current_price_col] if current_price_col else None

                            if ticker and shares > 0:
                                # Add .NS suffix if not present for Yahoo Finance compatibility
                                if not ticker.endswith(".NS") and not ticker.endswith(
                                    ".BO"
                                ):
                                    ticker = ticker + ".NS"

                                current_holdings[ticker] = {
                                    "units": shares,
                                    "purchase_price": avg_buy_price,
                                    "csv_current_price": csv_current_price,
                                }
                        
                        if current_price_col:
                            print(f"   Using current prices from Smallcase export (not Yahoo Finance)")
                    else:
                        raise ValueError(
                            "Smallcase format detected but required columns not found"
                        )
                else:
                    # Not a Smallcase export - try standard CSV format
                    df = pd.read_csv(holdings_path)

                    # Standard CSV format - check required columns
                    required_columns = ["symbol", "units"]

                    if not all(col in df.columns for col in required_columns):
                        print(f"❌ Error: CSV must contain columns: {required_columns}")
                        print(
                            f"   Optional column: 'price' (use -1 to fetch from purchase date)"
                        )
                        print(
                            f"   OR use Smallcase export format with columns: Ticker, Shares, Avg Buy Price (Rs.)"
                        )
                        return

                    for _, row in df.iterrows():
                        symbol = str(row["symbol"]).strip()
                        units = row["units"]
                        price = row.get("price", -1)

                        if symbol and units > 0:
                            current_holdings[symbol] = {
                                "units": units,
                                "purchase_price": price,
                                "csv_current_price": None,
                            }

                    # Check if any holdings have price = -1 and from_date is not provided
                    has_missing_prices = any(
                        holding["purchase_price"] == -1
                        for holding in current_holdings.values()
                    )

                    if has_missing_prices and not from_date:
                        print(
                            f"❌ Error: --from-date is required when using standard CSV format with price = -1"
                        )
                        print(
                            f"   Found holdings with missing prices that need historical lookup"
                        )
                        print(
                            f"   Use --from-date YYYY-MM-DD to specify purchase date for price lookup"
                        )
                        print(
                            f"   OR use Smallcase export format which includes average buy prices"
                        )
                        return

            except Exception as csv_error:
                print(f"❌ Error reading CSV file: {csv_error}")
                return

        else:
            print(f"❌ Error: Unsupported file format. Use .json or .csv")
            return

        if not current_holdings:
            print(f"❌ Error: No valid holdings found in file")
            return

        print(f"\n📊 CURRENT HOLDINGS LOADED:")
        print(f"   Found {len(current_holdings)} ETFs in portfolio")

        # Show summary of holdings format and validation
        holdings_with_prices = sum(
            1 for h in current_holdings.values() if h["purchase_price"] != -1
        )
        holdings_without_prices = len(current_holdings) - holdings_with_prices

        if holdings_without_prices > 0:
            print(f"   Holdings with prices: {holdings_with_prices}")
            print(f"   Holdings needing historical lookup: {holdings_without_prices}")
        else:
            print(f"   All holdings have purchase prices ✅")

    except Exception as e:
        print(f"❌ Error loading holdings file: {e}")
        return

    # Initialize strategy
    try:
        config = StrategyConfig(portfolio_size=portfolio_size)
        strategy = ETFMomentumStrategy(config)

        # Get current market data and optimal portfolio
        print(f"\n📡 Fetching current market data...")
        status = strategy.get_current_portfolio_status()

        if status["status"] == "error":
            print(f"❌ Error: {status['message']}")
            return

        current_prices = status["current_prices"]
        top_etfs = status["top_10_momentum_scores"][:portfolio_size]

        # Filter to only include ETFs with positive momentum scores
        top_etfs = [(t, s) for t, s in top_etfs if s > 0]
        optimal_tickers = [ticker for ticker, _ in top_etfs]

        if not top_etfs:
            print("❌ No ETFs have positive momentum. Consider holding cash or staying put.")
            return

        print(f"📅 Market Data Date: {status['market_data_date']}")

        # Check if we need historical prices (any holdings with price = -1)
        need_historical_prices = any(
            holding["purchase_price"] == -1 for holding in current_holdings.values()
        )

        purchase_prices = {}
        if need_historical_prices:
            if not purchase_date:
                print(
                    f"❌ Error: --from-date is required when holdings have price = -1"
                )
                print(
                    f"   This should have been caught earlier - please report this as a bug"
                )
                return

            # Get historical prices for purchase date (for holdings with price = -1)
            print(f"\n📡 Fetching historical prices for purchase date...")

            # Fetch historical data
            data_start = purchase_date - timedelta(
                days=30
            )  # Buffer for weekends/holidays
            data_end = purchase_date + timedelta(days=5)

            all_tickers = list(current_holdings.keys())
            historical_data = strategy.data_provider.fetch_etf_data(
                all_tickers, data_start, data_end
            )
            historical_prices = strategy.data_provider.get_prices(historical_data)

            # Convert timezone-aware index to timezone-naive if needed
            if historical_prices.index.tz is not None:
                historical_prices.index = historical_prices.index.tz_localize(None)

            # Find closest available date to purchase_date
            available_dates = historical_prices.index
            closest_date = min(available_dates, key=lambda x: abs(x - purchase_date))
            purchase_prices = historical_prices.loc[closest_date]

            print(
                f"   Using prices from: {closest_date.strftime('%Y-%m-%d')} (closest to purchase date)"
            )
        else:
            print(f"\n✅ All holdings have purchase prices - no historical data needed")

        # Calculate current portfolio value and performance
        print(f"\n💼 CURRENT PORTFOLIO ANALYSIS:")
        print("-" * 60)

        current_portfolio_data = []
        total_current_value = 0
        total_invested = 0

        for symbol, holding in current_holdings.items():
            units = holding["units"]
            purchase_price = holding["purchase_price"]

            # Use historical price if purchase_price is -1
            if purchase_price == -1:
                if symbol in purchase_prices and len(purchase_prices) > 0:
                    purchase_price = purchase_prices[symbol]
                else:
                    print(
                        f"⚠️  Warning: No historical price found for {symbol}, skipping..."
                    )
                    continue

            # Get current price - prefer CSV price from Smallcase, fallback to Yahoo Finance
            csv_price = holding.get("csv_current_price")
            if csv_price is not None and csv_price > 0:
                current_price = csv_price
            else:
                current_price = current_prices.get(symbol, 0)
            if current_price == 0:
                print(f"⚠️  Warning: No current price found for {symbol}, skipping...")
                continue

            invested_amount = units * purchase_price
            current_value = units * current_price
            gain_loss = current_value - invested_amount
            gain_loss_pct = (
                (gain_loss / invested_amount) * 100 if invested_amount > 0 else 0
            )

            total_invested += invested_amount
            total_current_value += current_value

            current_portfolio_data.append(
                [
                    symbol,
                    f"{units:,}",
                    f"₹{purchase_price:.2f}",
                    f"₹{current_price:.2f}",
                    f"₹{invested_amount:,.0f}",
                    f"₹{current_value:,.0f}",
                    f"₹{gain_loss:+,.0f}",
                    f"{gain_loss_pct:+.1f}%",
                ]
            )

        headers = [
            "ETF",
            "Units",
            "Buy Price",
            "Current Price",
            "Invested",
            "Current Value",
            "Gain/Loss",
            "Return %",
        ]
        print(tabulate(current_portfolio_data, headers=headers, tablefmt="grid"))

        total_gain_loss = total_current_value - total_invested
        total_return_pct = (
            (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0
        )

        print(f"\n💰 PORTFOLIO SUMMARY:")
        print(f"   Total Invested:    ₹{total_invested:,.0f}")
        print(f"   Current Value:     ₹{total_current_value:,.0f}")
        print(f"   Total Gain/Loss:   ₹{total_gain_loss:+,.0f}")
        print(f"   Total Return:      {total_return_pct:+.1f}%")

        # Calculate days and annualized return
        if purchase_date:
            days_held = (current_date - purchase_date).days
            if days_held >= 30:
                annualized_return = (
                    (total_current_value / total_invested) ** (365 / days_held) - 1
                ) * 100
                print(f"   Days Held:         {days_held} days")
                print(f"   Annualized Return: {annualized_return:+.1f}%")
        else:
            # For Smallcase exports, we don't have exact purchase dates
            # but we can show that returns are based on average buy prices
            print(f"   Days Held:         N/A (based on average buy prices)")
            print(f"   Annualized Return: N/A (requires specific purchase date)")

        # Show optimal portfolio for comparison
        print(f"\n🎯 CURRENT OPTIMAL PORTFOLIO:")
        print("-" * 60)

        # Use target_amount if specified, otherwise use current portfolio value
        rebalance_target_value = target_amount if target_amount else total_current_value
        
        # Check how many ETFs are actually available
        available_etfs = len(top_etfs)
        actual_portfolio_size = min(portfolio_size, available_etfs)
        
        if available_etfs < portfolio_size:
            print(f"⚠️  Only {available_etfs} ETFs have positive momentum (requested {portfolio_size})")
            print(f"   Adjusting portfolio to {available_etfs} ETFs with full allocation")
            print()
        
        if target_amount:
            print(f"📊 Target Investment Amount: ₹{target_amount:,.0f}")
        else:
            print(f"📊 Using Current Portfolio Value: ₹{total_current_value:,.0f}")
        
        print(f"📊 Actual Portfolio Size: {actual_portfolio_size} ETFs")

        # Build a merged price dictionary: prefer CSV prices, fallback to Yahoo Finance for new ETFs
        # This ensures consistent pricing across all calculations
        prices_for_calc = {}
        for ticker in set(optimal_tickers) | set(current_holdings.keys()):
            # First check if we have a CSV price from Smallcase
            if ticker in current_holdings:
                csv_price = current_holdings[ticker].get("csv_current_price")
                if csv_price is not None and csv_price > 0:
                    prices_for_calc[ticker] = csv_price
                    continue
            # Fallback to Yahoo Finance price
            prices_for_calc[ticker] = current_prices.get(ticker, 0)
        
        # Check if we're using mixed price sources
        csv_price_count = sum(1 for t in optimal_tickers if t in current_holdings and current_holdings[t].get("csv_current_price"))
        yf_price_count = len(optimal_tickers) - csv_price_count
        if csv_price_count > 0 and yf_price_count > 0:
            print(f"📊 Prices: {csv_price_count} from CSV, {yf_price_count} from Yahoo Finance (for new ETFs)")

        optimal_data = []
        # Allocate across actual available ETFs, not requested size
        target_allocation = rebalance_target_value / actual_portfolio_size

        for i, (ticker, score) in enumerate(top_etfs, 1):
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            target_value = target_units * price

            optimal_data.append(
                [
                    i,
                    ticker,
                    f"₹{price:.2f}",
                    f"{target_units:,}",
                    f"₹{target_value:,.0f}",
                    f"{score:.4f}",
                ]
            )

        headers = [
            "Rank",
            "ETF",
            "Price",
            "Target Units",
            "Target Value",
            "Momentum Score",
        ]
        print(tabulate(optimal_data, headers=headers, tablefmt="grid"))

        # Rebalancing recommendations
        print(f"\n🔄 REBALANCING RECOMMENDATIONS:")
        print("-" * 60)

        current_tickers = set(current_holdings.keys())
        optimal_tickers_set = set(optimal_tickers)

        # ETFs to sell (not in optimal portfolio)
        to_sell = current_tickers - optimal_tickers_set
        # ETFs to buy (in optimal portfolio but not held)
        to_buy = optimal_tickers_set - current_tickers
        # ETFs to adjust (in both portfolios)
        to_adjust = current_tickers & optimal_tickers_set

        if to_sell:
            print(f"\n❌ SELL (no longer in optimal portfolio):")
            sell_value = 0
            for ticker in to_sell:
                units = current_holdings[ticker]["units"]
                price = prices_for_calc.get(ticker, 0)
                value = units * price
                sell_value += value
                print(f"   • {ticker}: SELL ALL {units:,} units → ₹{value:,.0f}")
            print(f"   Total proceeds from sales: ₹{sell_value:,.0f}")

        if to_buy:
            print(f"\n✅ BUY (new entries to optimal portfolio):")
            for ticker in to_buy:
                price = prices_for_calc.get(ticker, 0)
                target_units = int(target_allocation / price) if price > 0 else 0
                target_value = target_units * price
                print(
                    f"   • {ticker}: BUY {target_units:,} units → ₹{target_value:,.0f}"
                )

        if to_adjust:
            print(f"\n🔄 ADJUST (rebalance existing holdings):")
            for ticker in to_adjust:
                current_units = current_holdings[ticker]["units"]
                price = prices_for_calc.get(ticker, 0)
                target_units = int(target_allocation / price) if price > 0 else 0

                current_value = current_units * price
                target_value = target_units * price

                diff_units = target_units - current_units
                diff_value = diff_units * price

                if diff_units > 0:
                    print(
                        f"   • {ticker}: BUY {diff_units:,} more units → ₹{diff_value:+,.0f}"
                    )
                elif diff_units < 0:
                    print(
                        f"   • {ticker}: SELL {abs(diff_units):,} units → ₹{diff_value:+,.0f}"
                    )
                else:
                    print(
                        f"   • {ticker}: HOLD {current_units:,} units (no change needed)"
                    )

        if not to_sell and not to_buy and not to_adjust:
            print(f"✅ Your portfolio is already optimal! No rebalancing needed.")

        # Calculate detailed cash flow summary
        total_sell_value = 0
        total_buy_value = 0
        total_adjust_buy = 0
        total_adjust_sell = 0

        # Calculate sell value
        for ticker in to_sell:
            units = current_holdings[ticker]["units"]
            price = prices_for_calc.get(ticker, 0)
            total_sell_value += units * price

        # Calculate buy value for new ETFs
        for ticker in to_buy:
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            total_buy_value += target_units * price

        # Calculate adjustments
        for ticker in to_adjust:
            current_units = current_holdings[ticker]["units"]
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            diff_units = target_units - current_units
            diff_value = diff_units * price
            if diff_value > 0:
                total_adjust_buy += diff_value
            else:
                total_adjust_sell += abs(diff_value)

        # Calculate total optimal portfolio value
        total_optimal_value = 0
        for ticker, _ in top_etfs:
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            total_optimal_value += target_units * price

        # Net cash flow
        total_proceeds = total_sell_value + total_adjust_sell
        total_purchases = total_buy_value + total_adjust_buy
        net_cash_flow = total_proceeds - total_purchases

        # Summary of actions needed
        print(f"\n📋 REBALANCING SUMMARY:")
        print("-" * 40)
        total_transactions = (
            len(to_sell)
            + len(to_buy)
            + len(
                [
                    t
                    for t in to_adjust
                    if int(target_allocation / prices_for_calc.get(t, 1))
                    != current_holdings[t]["units"]
                ]
            )
        )
        print(f"   ETFs to sell:     {len(to_sell)}")
        print(f"   ETFs to buy:      {len(to_buy)}")
        print(f"   ETFs to adjust:   {len(to_adjust)}")
        print(f"   Total transactions: {total_transactions}")

        # Cash flow summary
        print(f"\n💵 CASH FLOW SUMMARY:")
        print("-" * 40)
        print(f"   Current Portfolio Value:  ₹{total_current_value:,.0f}")
        print(f"   Target Investment:        ₹{rebalance_target_value:,.0f}")
        
        # Calculate how much new capital is needed to reach target
        new_capital_needed = rebalance_target_value - total_current_value
        if new_capital_needed > 0:
            print(f"   New Capital Needed:       ₹{new_capital_needed:,.0f} (to reach target)")
        elif new_capital_needed < 0:
            print(f"   Capital to Withdraw:      ₹{abs(new_capital_needed):,.0f} (target < current)")
        
        print(f"   Optimal Portfolio Value:  ₹{total_optimal_value:,.0f}")
        print()
        print(f"   📤 Proceeds from selling:  ₹{total_proceeds:,.0f}")
        print(f"   📥 Cost of buying:         ₹{total_purchases:,.0f}")
        
        # Net transaction cash flow
        transaction_net = total_proceeds - total_purchases
        print()
        if transaction_net >= 0:
            print(f"   Transaction Net:          ₹{transaction_net:,.0f} (cash left from trades)")
        else:
            print(f"   Transaction Net:          ₹{transaction_net:,.0f} (need cash for trades)")
        
        # Final summary - what user actually needs to do
        print()
        print("   " + "=" * 35)
        if new_capital_needed > 0:
            print(f"   💸 ADD ₹{new_capital_needed:,.0f} to reach ₹{rebalance_target_value:,.0f} target")
        elif new_capital_needed < 0:
            print(f"   ✅ WITHDRAW ₹{abs(new_capital_needed):,.0f} (portfolio exceeds target)")
        else:
            print(f"   ✅ No additional capital needed")

        # Show gap between target and actual optimal
        gap = rebalance_target_value - total_optimal_value
        if abs(gap) > 100:
            print()
            if gap > 0:
                print(f"   ⚠️  ₹{gap:,.0f} uninvested due to lot sizes")

        if total_transactions > 0:
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. Review the recommended changes above")
            print(f"   2. Execute sell orders first")
            print(f"   3. Use proceeds to buy new ETFs")
            print(f"   4. Adjust existing holdings as needed")
            print(f"   5. Monitor for next rebalancing opportunity")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


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


def unified_cli():
    """Unified CLI that combines all individual command CLIs."""
    parser = argparse.ArgumentParser(
        description="ETF Momentum Portfolio Management - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run cli.py portfolio                # Show current optimal portfolio
  uv run cli.py portfolio --amount 500000 --size 7  # Custom amount and portfolio size
  uv run cli.py rebalance --holdings-file holdings.json --from-date 2024-01-01  # Rebalance analysis
  uv run cli.py historical --from-date 2024-01-01  # Historical portfolio changes
  uv run cli.py backtest --amounts 1000000 --size 5  # Quick backtest

Holdings File Formats:
  JSON: [{"symbol": "NIFTYBEES.NS", "units": 350, "price": 120.50}, ...]
        OR {"NIFTYBEES.NS": {"units": 350, "price": 120.50}, ...}
        OR {"NIFTYBEES.NS": 350, ...} (price will be fetched from from-date)
  CSV:  symbol,units,price
        NIFTYBEES.NS,350,120.50
        BANKBEES.NS,171,-1    (price -1 means fetch from from-date)
  Smallcase Export: Direct export from Smallcase with columns: Ticker, Shares, Avg Buy Price (Rs.)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Portfolio command - use shared parser factory
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Show current optimal portfolio"
    )
    _create_portfolio_parser(portfolio_parser)

    # Rebalance command - use shared parser factory
    rebalance_parser = subparsers.add_parser(
        "rebalance", help="Show portfolio rebalancing analysis"
    )
    _create_rebalance_parser(rebalance_parser)

    # Historical command - use shared parser factory
    historical_parser = subparsers.add_parser(
        "historical", help="Show portfolio changes between dates"
    )
    _create_historical_parser(historical_parser)

    # Backtest command - use shared parser factory
    backtest_parser = subparsers.add_parser(
        "backtest", help="Quick ETF momentum strategy backtest"
    )
    _create_backtest_parser(backtest_parser)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "portfolio":
            show_current_portfolio(
                investment_amount=args.amount, portfolio_size=args.size
            )

        elif args.command == "rebalance":
            show_rebalancing_needs(
                holdings_file=args.holdings_file,
                from_date=args.from_date,
                target_amount=args.amount,
                portfolio_size=args.size,
            )

        elif args.command == "historical":
            show_historical_portfolio(
                from_date_str=args.from_date,
                to_date_str=args.to_date,
                investment_amount=args.amount,
                portfolio_size=args.size,
            )

        elif args.command == "backtest":
            print(
                f"🚀 Quick Backtest - Portfolio Size: {args.size}, Amounts: {args.amounts}"
            )
            if args.use_threshold:
                print(
                    f"📊 Threshold Rebalancing: Profit {args.profit_threshold}%, Loss {args.loss_threshold}%"
                )

            run_parameter_experiments(
                investment_amounts=args.amounts,
                portfolio_sizes=args.sizes,
                use_threshold_rebalancing_values=[args.use_threshold],
                profit_threshold_pct=args.profit_threshold,
                loss_threshold_pct=args.loss_threshold,
            )

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection for data fetching.")
        sys.exit(1)


if __name__ == "__main__":
    # If run directly, use the unified CLI
    unified_cli()
