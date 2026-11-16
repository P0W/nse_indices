#!/usr/bin/env python3
"""
ETF Portfolio CLI Functions - Individual entry points for ETF momentum portfolio management
Author : Prashant Srivastava
"""

import argparse
import sys
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate
from etf_momentum_strategy import (
    ETFMomentumStrategy,
    run_parameter_experiments,
)
from core import StrategyConfig, TradingCostCalculator

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _normalize_ticker(symbol: str, config: StrategyConfig) -> str:
    """Normalize ticker symbol to Yahoo Finance .NS format and resolve legacy NSE renames."""
    sym = str(symbol).strip().upper()
    if not sym:
        return ""
    if sym in config.ticker_aliases:
        return config.ticker_aliases[sym]
    if not sym.endswith(".NS") and not sym.endswith(".BO"):
        sym = f"{sym}.NS"
    return config.ticker_aliases.get(sym, sym)


def _merge_holding(
    holdings: dict,
    symbol: str,
    units: float,
    purchase_price: float,
    csv_current_price: float = None,
) -> None:
    """Insert or merge duplicate/aliased holdings using weighted-average buy price."""
    units = float(units)
    purchase_price = (
        float(purchase_price)
        if purchase_price is not None and pd.notna(purchase_price)
        else -1.0
    )
    if purchase_price < 0:
        purchase_price = -1.0
    csv_val = (
        float(csv_current_price)
        if csv_current_price is not None and str(csv_current_price).strip() != ""
        else None
    )
    if csv_val is not None and (pd.isna(csv_val) or csv_val <= 0):
        csv_val = None

    if symbol in holdings:
        prev = holdings[symbol]
        total_units = prev["units"] + units
        prev_buy = prev["purchase_price"]
        if (
            pd.notna(prev_buy)
            and prev_buy >= 0
            and pd.notna(purchase_price)
            and purchase_price >= 0
            and total_units > 0
        ):
            merged_buy = (
                prev["units"] * prev_buy + units * purchase_price
            ) / total_units
        else:
            merged_buy = purchase_price if purchase_price >= 0 else prev_buy
        holdings[symbol] = {
            "units": int(round(total_units))
            if float(total_units).is_integer()
            else total_units,
            "purchase_price": merged_buy,
            "csv_current_price": csv_val or prev.get("csv_current_price"),
        }
    else:
        holdings[symbol] = {
            "units": int(round(units)) if float(units).is_integer() else units,
            "purchase_price": purchase_price,
            "csv_current_price": csv_val,
        }


def _parse_smallcase_overview(lines: list, header_line_idx: int) -> dict:
    """Extract Investment Overview metrics from top rows of a Smallcase CSV export."""
    import csv

    meta = {}
    known_labels = {
        "Current Value": "current_value",
        "Total Returns": "total_returns",
        "Current Investment": "current_investment",
        "Money Put In": "money_put_in",
        "Current Returns": "current_returns",
        "Realized Returns": "realized_returns",
        "Dividends": "dividends",
    }
    try:
        rows = list(csv.reader(lines[:header_line_idx]))
        for r_idx in range(len(rows) - 1):
            label_row = rows[r_idx]
            val_row = rows[r_idx + 1]
            for c_idx, cell in enumerate(label_row):
                key = known_labels.get(cell.strip())
                if key and c_idx < len(val_row):
                    raw_val = val_row[c_idx].strip().replace(",", "")
                    if raw_val:
                        try:
                            meta[key] = float(raw_val)
                        except ValueError:
                            pass
    except Exception:
        pass
    return meta


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
            print(f"📅 Data Date: {status['market_data_date']}")
            print(
                "⚠️  0 ETFs currently pass trend/retracement/volume filters with positive momentum."
            )
            print("💡 Recommendation: Hold 100% Cash / LIQUIDBEES until regime improves.")
            return

        print(f"📅 Data Date: {status['market_data_date']}")

        slot_weight_pct = 100.0 / portfolio_size
        slot_allocation = investment_amount / portfolio_size

        if actual_size < portfolio_size:
            print(
                f"⚠️  Only {actual_size} of {portfolio_size} ETF slots qualify with positive momentum"
            )
            print(
                f"   Capping each ETF at {slot_weight_pct:.1f}% (₹{slot_allocation:,.0f}/slot) "
                f"and keeping {portfolio_size - actual_size} slot(s) in Cash / LIQUIDBEES"
            )

        print(f"🎯 Qualifying ETFs: {actual_size} / {portfolio_size} slots")
        print(f"📊 Per-Slot Allocation Cap: {slot_weight_pct:.1f}% per ETF")

        print(f"\n📈 OPTIMAL PORTFOLIO ALLOCATION:")

        # Prepare data for table
        table_data = []
        total_investment = 0

        for i, (ticker, score) in enumerate(top_etfs, 1):
            price = current_prices.get(ticker, 0)
            units = int(slot_allocation / price) if price > 0 else 0
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

    config = StrategyConfig(portfolio_size=portfolio_size)
    smallcase_meta = {}

    # Load holdings from file
    try:
        holdings_path = Path(holdings_file)
        if not holdings_path.exists():
            print(f"❌ Error: Holdings file not found: {holdings_file}")
            return

        current_holdings = {}

        if holdings_path.suffix.lower() == ".json":
            # Load from JSON
            with open(holdings_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            # Expected format: [{"symbol": "NIFTYBEES.NS", "units": 350, "price": 120.50}, ...]
            # or {"NIFTYBEES.NS": {"units": 350, "price": 120.50}, ...}
            if isinstance(data, list):
                for item in data:
                    symbol = _normalize_ticker(item.get("symbol", ""), config)
                    units = item.get("units", 0)
                    price = item.get("price", -1)
                    if symbol and units > 0:
                        _merge_holding(current_holdings, symbol, units, price)
            elif isinstance(data, dict):
                for raw_symbol, details in data.items():
                    symbol = _normalize_ticker(raw_symbol, config)
                    if not symbol:
                        continue
                    if isinstance(details, dict):
                        units = details.get("units", 0)
                        price = details.get("price", -1)
                        if units > 0:
                            _merge_holding(current_holdings, symbol, units, price)
                    elif details > 0:
                        _merge_holding(current_holdings, symbol, details, -1)

        elif holdings_path.suffix.lower() == ".csv":
            # Load from CSV
            try:
                with open(holdings_path, "r", encoding="utf-8-sig") as f:
                    lines = f.readlines()

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
                    smallcase_meta = _parse_smallcase_overview(lines, header_line_idx)
                    df = pd.read_csv(
                        holdings_path, skiprows=header_line_idx, encoding="utf-8-sig"
                    )
                    print(
                        f"📊 Detected Smallcase export format (skipped {header_line_idx} header rows)"
                    )

                    avg_buy_price_col = None
                    for col in df.columns:
                        if "Avg" in col and "Buy" in col and "Price" in col:
                            avg_buy_price_col = col
                            break

                    current_price_col = None
                    for col in df.columns:
                        if "Current" in col and "Price" in col:
                            current_price_col = col
                            break

                    if (
                        "Ticker" in df.columns
                        and "Shares" in df.columns
                        and avg_buy_price_col is not None
                    ):
                        for _, row in df.iterrows():
                            raw_ticker = str(row["Ticker"]).strip()
                            shares = row["Shares"]
                            avg_buy_price = row[avg_buy_price_col]
                            csv_current_price = (
                                row[current_price_col] if current_price_col else None
                            )

                            if (
                                raw_ticker
                                and raw_ticker.lower() != "nan"
                                and pd.notna(shares)
                                and float(shares) > 0
                            ):
                                ticker = _normalize_ticker(raw_ticker, config)
                                _merge_holding(
                                    current_holdings,
                                    ticker,
                                    shares,
                                    avg_buy_price,
                                    csv_current_price,
                                )

                        if current_price_col:
                            print(
                                f"   Using current prices from Smallcase export (Yahoo Finance for new ETFs)"
                            )
                    else:
                        raise ValueError(
                            "Smallcase format detected but required columns not found"
                        )
                else:
                    # Standard CSV format
                    df = pd.read_csv(holdings_path, encoding="utf-8-sig")
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
                        raw_symbol = str(row["symbol"]).strip()
                        units = row["units"]
                        price = row.get("price", -1)

                        if (
                            raw_symbol
                            and raw_symbol.lower() != "nan"
                            and pd.notna(units)
                            and float(units) > 0
                        ):
                            symbol = _normalize_ticker(raw_symbol, config)
                            _merge_holding(current_holdings, symbol, units, price)

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
        strategy = ETFMomentumStrategy(config)

        # Get current market data and optimal portfolio
        print(f"\n📡 Fetching current market data...")
        status = strategy.get_current_portfolio_status()

        if status["status"] == "error":
            print(f"❌ Error: {status['message']}")
            return

        current_prices = dict(status["current_prices"])

        # Fetch live prices for any legacy/out-of-universe held tickers missing CSV prices
        missing_live_tickers = [
            sym
            for sym, h in current_holdings.items()
            if sym not in current_prices
            and not (h.get("csv_current_price") and h["csv_current_price"] > 0)
        ]
        if missing_live_tickers:
            try:
                extra_data = strategy.data_provider.fetch_etf_data(
                    missing_live_tickers,
                    current_date - timedelta(days=15),
                    current_date,
                )
                extra_prices = strategy.data_provider.get_prices(extra_data)
                if not extra_prices.empty:
                    for sym in missing_live_tickers:
                        if sym in extra_prices.columns and pd.notna(
                            extra_prices[sym].iloc[-1]
                        ):
                            current_prices[sym] = float(extra_prices[sym].iloc[-1])
            except Exception as extra_err:
                print(
                    f"⚠️  Warning: Could not fetch live price for non-universe ticker(s) {missing_live_tickers}: {extra_err}"
                )

        top_etfs = status["top_10_momentum_scores"][:portfolio_size]
        top_etfs = [(t, s) for t, s in top_etfs if s > 0]
        optimal_tickers = [ticker for ticker, _ in top_etfs]

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
            if purchase_price == -1 or pd.isna(purchase_price):
                if (
                    symbol in purchase_prices
                    and len(purchase_prices) > 0
                    and pd.notna(purchase_prices[symbol])
                    and purchase_prices[symbol] > 0
                ):
                    purchase_price = float(purchase_prices[symbol])
                else:
                    print(
                        f"⚠️  Warning: No historical price found for {symbol}, skipping..."
                    )
                    continue

            # Get current price - prefer CSV price from Smallcase, fallback to Yahoo Finance
            csv_price = holding.get("csv_current_price")
            if csv_price is not None and pd.notna(csv_price) and csv_price > 0:
                current_price = float(csv_price)
            else:
                yf_px = current_prices.get(symbol, 0)
                current_price = float(yf_px) if pd.notna(yf_px) and yf_px > 0 else 0.0
            if current_price <= 0:
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
        print(f"   Cost Basis (Open Shares): ₹{total_invested:,.0f}")
        print(f"   Current Value:            ₹{total_current_value:,.0f}")
        print(f"   Unrealized Gain/Loss:     ₹{total_gain_loss:+,.0f}")
        print(f"   Unrealized Return:        {total_return_pct:+.1f}%")

        if (
            "current_value" in smallcase_meta
            and "total_returns" in smallcase_meta
        ):
            net_principal = (
                smallcase_meta["current_value"] - smallcase_meta["total_returns"]
            )
            realized_ret = smallcase_meta.get("realized_returns", 0.0)
            total_ret = smallcase_meta["total_returns"]
            print(f"\n🏦 SMALLCASE CAPITAL BREAKDOWN:")
            print(f"   Net Principal Retained:   ₹{net_principal:,.0f}")
            print(f"   Realized Gains Reinvested:₹{realized_ret:+,.0f}")
            print(f"   Total Net Strategy P&L:   ₹{total_ret:+,.0f}")

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

        # Cap each ETF slot at 1 / portfolio_size so unfilled slots remain in cash / LIQUIDBEES
        available_etfs = len(top_etfs)
        slot_allocation = (
            rebalance_target_value / portfolio_size if portfolio_size > 0 else 0.0
        )
        slot_weight_pct = 100.0 / portfolio_size if portfolio_size > 0 else 0.0

        if target_amount:
            print(f"📊 Target Investment Amount: ₹{target_amount:,.0f}")
        else:
            print(f"📊 Using Current Portfolio Value: ₹{total_current_value:,.0f}")

        print(
            f"📊 Qualifying Slots: {available_etfs} / {portfolio_size} ETFs "
            f"(max {slot_weight_pct:.1f}% = ₹{slot_allocation:,.0f} per slot)"
        )

        if available_etfs == 0:
            print(
                "⚠️  0 ETFs currently pass regime/volume filters with positive momentum!"
            )
            print(
                "   Recommending 100% exit to Cash / LIQUIDBEES until momentum turns positive."
            )
        elif available_etfs < portfolio_size:
            unfilled = portfolio_size - available_etfs
            print(
                f"⚠️  Only {available_etfs} ETFs qualify — keeping {unfilled} slot(s) "
                f"(~₹{unfilled * slot_allocation:,.0f}) in Cash / LIQUIDBEES"
            )

        # Build a merged price dictionary: prefer CSV prices, fallback to Yahoo Finance for new ETFs
        prices_for_calc = {}
        for ticker in set(optimal_tickers) | set(current_holdings.keys()):
            if ticker in current_holdings:
                csv_price = current_holdings[ticker].get("csv_current_price")
                if csv_price is not None and pd.notna(csv_price) and csv_price > 0:
                    prices_for_calc[ticker] = float(csv_price)
                    continue
            yf_px = current_prices.get(ticker, 0)
            prices_for_calc[ticker] = (
                float(yf_px) if pd.notna(yf_px) and yf_px > 0 else 0.0
            )

        csv_price_count = sum(
            1
            for t in optimal_tickers
            if t in current_holdings and current_holdings[t].get("csv_current_price")
        )
        yf_price_count = len(optimal_tickers) - csv_price_count
        if csv_price_count > 0 and yf_price_count > 0:
            print(
                f"📊 Prices: {csv_price_count} from CSV, {yf_price_count} from Yahoo Finance (for new ETFs)"
            )

        target_allocation = slot_allocation
        if top_etfs:
            optimal_data = []
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

        # Calculate detailed cash flow and Zerodha/NSE statutory + DP cost summary
        cost_calc = TradingCostCalculator(config)
        total_sell_value = 0
        total_buy_value = 0
        total_adjust_buy = 0
        total_adjust_sell = 0
        estimated_fees = 0.0
        adjusted_trade_count = 0

        for ticker in sorted(to_sell):
            units = current_holdings[ticker]["units"]
            price = prices_for_calc.get(ticker, 0)
            total_sell_value += units * price
            if price > 0 and units > 0:
                estimated_fees += cost_calc.calculate_costs(
                    ticker,
                    units,
                    price,
                    "SELL",
                    include_dp_charge=True,
                    zero_delivery_brokerage=True,
                )["total_cost"]

        for ticker in sorted(to_buy):
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            total_buy_value += target_units * price
            if price > 0 and target_units > 0:
                estimated_fees += cost_calc.calculate_costs(
                    ticker,
                    target_units,
                    price,
                    "BUY",
                    include_dp_charge=False,
                    zero_delivery_brokerage=True,
                )["total_cost"]

        for ticker in sorted(to_adjust):
            current_units = current_holdings[ticker]["units"]
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            diff_units = target_units - current_units
            diff_value = diff_units * price
            if diff_units != 0:
                adjusted_trade_count += 1
            if diff_value > 0:
                total_adjust_buy += diff_value
                estimated_fees += cost_calc.calculate_costs(
                    ticker,
                    diff_units,
                    price,
                    "BUY",
                    include_dp_charge=False,
                    zero_delivery_brokerage=True,
                )["total_cost"]
            elif diff_value < 0:
                total_adjust_sell += abs(diff_value)
                estimated_fees += cost_calc.calculate_costs(
                    ticker,
                    abs(diff_units),
                    price,
                    "SELL",
                    include_dp_charge=True,
                    zero_delivery_brokerage=True,
                )["total_cost"]

        # Calculate total optimal portfolio value
        total_optimal_value = 0
        for ticker, _ in top_etfs:
            price = prices_for_calc.get(ticker, 0)
            target_units = int(target_allocation / price) if price > 0 else 0
            total_optimal_value += target_units * price

        # Net cash flow
        total_proceeds = total_sell_value + total_adjust_sell
        total_purchases = total_buy_value + total_adjust_buy

        # Summary of actions needed
        print(f"\n📋 REBALANCING SUMMARY:")
        print("-" * 40)
        total_transactions = len(to_sell) + len(to_buy) + adjusted_trade_count
        print(f"   ETFs to sell:     {len(to_sell)}")
        print(f"   ETFs to buy:      {len(to_buy)}")
        print(f"   ETFs to adjust:   {adjusted_trade_count} (of {len(to_adjust)} retained)")
        print(f"   Total transactions: {total_transactions}")

        # Cash flow summary
        print(f"\n💵 CASH FLOW SUMMARY:")
        print("-" * 40)
        print(f"   Current Portfolio Value:  ₹{total_current_value:,.0f}")
        print(f"   Target Investment:        ₹{rebalance_target_value:,.0f}")

        new_capital_needed = rebalance_target_value - total_current_value
        if new_capital_needed > 0:
            print(
                f"   New Capital Needed:       ₹{new_capital_needed:,.0f} (to reach target)"
            )
        elif new_capital_needed < 0:
            print(
                f"   Capital to Withdraw:      ₹{abs(new_capital_needed):,.0f} (target < current)"
            )

        print(f"   Optimal Portfolio Value:  ₹{total_optimal_value:,.0f}")
        print()
        print(f"   📤 Proceeds from selling:  ₹{total_proceeds:,.0f}")
        print(f"   📥 Cost of buying:         ₹{total_purchases:,.0f}")
        print(
            f"   🧾 Est. Statutory + DP:    ₹{estimated_fees:,.0f} (STT/Stamp/Exch/GST/CDSL DP)"
        )

        transaction_net = total_proceeds - total_purchases - estimated_fees
        unfilled_slots = max(0, portfolio_size - available_etfs)
        unfilled_slot_reserve = unfilled_slots * slot_allocation
        # External cash delta relative to target capital (reserving unfilled slots in Cash/LIQUIDBEES)
        external_cash_delta = transaction_net - unfilled_slot_reserve

        print()
        if transaction_net >= 0:
            print(
                f"   ETF Trades Net Cash:      ₹{transaction_net:,.0f} (cash freed after ETF orders & charges)"
            )
        else:
            print(
                f"   ETF Trades Net Cash:      ₹{transaction_net:,.0f} (cash needed for ETF orders & charges)"
            )

        if unfilled_slots > 0:
            print(
                f"   Unfilled Slot Reserve:    ₹{unfilled_slot_reserve:,.0f} ({unfilled_slots} slot(s) kept in Cash / LIQUIDBEES)"
            )

        # Final summary - what user actually needs to do (consistent across filled & unfilled slot regimes)
        print()
        print("   " + "=" * 35)
        if external_cash_delta < -1.0:
            print(
                f"   💸 ADD / KEEP ₹{abs(external_cash_delta):,.0f} fresh cash in broker "
                f"(funds target + ~₹{estimated_fees:,.0f} charges"
                + (
                    f" + ₹{unfilled_slot_reserve:,.0f} cash-slot reserve)"
                    if unfilled_slots > 0
                    else ")"
                )
            )
        elif new_capital_needed < 0 and external_cash_delta > 1.0:
            print(
                f"   ✅ WITHDRAW up to ₹{external_cash_delta:,.0f} net cash "
                + (
                    f"(after keeping ₹{unfilled_slot_reserve:,.0f} in Cash / LIQUIDBEES reserve)"
                    if unfilled_slots > 0
                    else "after trades & charges"
                )
            )
        else:
            print(
                f"   ✅ Self-funding rebalance "
                + (
                    f"(keep ₹{transaction_net:,.0f} freed proceeds in Cash / LIQUIDBEES for {unfilled_slots} unfilled slot(s))"
                    if unfilled_slots > 0
                    else f"(₹{transaction_net:,.0f} rounding cash buffer left after charges)"
                )
            )

        # Show whole-share rounding residual separately from unfilled slot reserve
        invested_slot_target = available_etfs * slot_allocation
        rounding_gap = max(0.0, invested_slot_target - total_optimal_value)
        if rounding_gap > 100:
            print()
            print(f"   ⚠️  ₹{rounding_gap:,.0f} uninvested due to whole-share rounding")

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
        # Fetch historical data with generous calendar-day buffer for 200-trading-day lookback
        data_start = from_date - timedelta(days=strategy._calendar_lookback_days())

        print(
            f"📡 Fetching historical data from {data_start.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}..."
        )

        all_data = strategy.data_provider.fetch_etf_data(
            config.etf_universe, data_start, to_date
        )

        prices_df = strategy._ensure_ist_naive_index(
            strategy.data_provider.get_prices(all_data)
        )
        signal_prices_df = strategy._ensure_ist_naive_index(
            strategy.data_provider.get_signal_prices(all_data)
        )
        volumes_df = strategy._ensure_ist_naive_index(
            strategy.data_provider.get_volumes(all_data)
        )

        # Get data up to from_date for initial portfolio
        from_data = prices_df[prices_df.index <= from_date]
        from_signal = signal_prices_df[signal_prices_df.index <= from_date]
        from_vol = (
            volumes_df[volumes_df.index <= from_date]
            if volumes_df is not None
            else None
        )
        if from_data.empty or from_signal.empty:
            print(f"❌ No data available for from-date: {from_date_str}")
            return

        print(f"\n📈 PORTFOLIO ON {from_date.strftime('%Y-%m-%d')}:")
        print("-" * 60)

        # Calculate momentum scores for from_date
        eligible_tickers = strategy.momentum_calculator.apply_filters(
            from_signal, from_vol
        )
        from_top_etfs = []
        from_prices = from_data.iloc[-1]
        if eligible_tickers:
            eligible_from_data = from_signal[eligible_tickers]
            from_momentum_scores = (
                strategy.momentum_calculator.calculate_momentum_scores(
                    eligible_from_data, market_prices_df=from_signal
                )
            )
            from_top_etfs = [
                (ticker, score)
                for ticker, score in from_momentum_scores.items()
                if score > 0
            ][:portfolio_size]

        if from_top_etfs:
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
            print("⚠️  0 ETFs passed regime/momentum filters on from-date (100% Cash).")

        # Get data up to to_date for final portfolio
        to_data = prices_df[prices_df.index <= to_date]
        to_signal = signal_prices_df[signal_prices_df.index <= to_date]
        to_vol = (
            volumes_df[volumes_df.index <= to_date]
            if volumes_df is not None
            else None
        )
        if to_data.empty or to_signal.empty:
            print(f"❌ No data available for to-date: {to_date_str}")
            return

        print(f"\n📈 PORTFOLIO ON {to_date.strftime('%Y-%m-%d')}:")
        print("-" * 60)

        # Calculate momentum scores for to_date
        eligible_tickers = strategy.momentum_calculator.apply_filters(to_signal, to_vol)
        to_top_etfs = []
        to_prices = to_data.iloc[-1]
        if eligible_tickers:
            eligible_to_data = to_signal[eligible_tickers]
            to_momentum_scores = strategy.momentum_calculator.calculate_momentum_scores(
                eligible_to_data, market_prices_df=to_signal
            )
            to_top_etfs = [
                (ticker, score)
                for ticker, score in to_momentum_scores.items()
                if score > 0
            ][:portfolio_size]

        if to_top_etfs:
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
            print("⚠️  0 ETFs passed regime/momentum filters on to-date (100% Cash).")

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

        # Calculate overall portfolio performance (including uninvested cash slots)
        from_invested_value = 0
        to_invested_value = 0

        for ticker, _ in from_top_etfs:
            from_price = from_prices.get(ticker, 0)
            allocation = investment_amount / portfolio_size
            units = int(allocation / from_price) if from_price > 0 else 0
            from_invested_value += units * from_price

            if ticker in to_prices.index and pd.notna(to_prices[ticker]):
                to_invested_value += units * to_prices[ticker]
            else:
                to_invested_value += units * from_price

        cash_held = max(0.0, investment_amount - from_invested_value)
        from_portfolio_value = from_invested_value + cash_held
        to_portfolio_value = to_invested_value + cash_held

        if from_portfolio_value > 0:
            portfolio_return = ((to_portfolio_value / from_portfolio_value) - 1) * 100
            portfolio_gain_loss = to_portfolio_value - from_portfolio_value

            print(f"\n💰 OVERALL PORTFOLIO PERFORMANCE:")
            print(
                f"   Initial Deployed / Cash: ₹{from_invested_value:,.0f} invested + ₹{cash_held:,.0f} cash"
            )
            print(f"   From Portfolio Value:    ₹{from_portfolio_value:,.0f}")
            print(f"   To Portfolio Value:      ₹{to_portfolio_value:,.0f}")
            print(f"   Absolute Gain/Loss:      ₹{portfolio_gain_loss:+,.0f}")
            print(f"   Percentage Return:       {portfolio_return:+.2f}%")

            if days_diff >= 30:
                annualized_return = (
                    (to_portfolio_value / from_portfolio_value) ** (365 / days_diff) - 1
                ) * 100
                print(f"   Annualized Return:       {annualized_return:+.2f}%")

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
                f"🚀 Quick Backtest - Portfolio Sizes: {args.sizes}, Amounts: {args.amounts}"
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
