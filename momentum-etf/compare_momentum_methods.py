#!/usr/bin/env python3
"""
Compare OLD vs FIXED momentum calculation methods across different permutations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
import pytz

from core import StrategyConfig, DataProvider, MomentumCalculator
from core_fixed import MomentumCalculatorFixed


def compare_methods(investment_amount: float, portfolio_size: int):
    """
    Compare old vs fixed momentum calculation for a given configuration.

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*80}")
    print(f"COMPARING: Amount=₹{investment_amount:,.0f}, Size={portfolio_size}")
    print(f"{'='*80}")

    # Initialize configurations
    config = StrategyConfig(portfolio_size=portfolio_size, initial_capital=investment_amount)

    # Get current date in IST
    ist_tz = pytz.timezone('Asia/Kolkata')
    current_date = datetime.now(ist_tz).replace(tzinfo=None)
    data_start = current_date - timedelta(days=config.min_data_points + 100)

    # Fetch data
    print(f"📡 Fetching data from {data_start.date()} to {current_date.date()}...")
    data_provider = DataProvider(config)

    try:
        all_data = data_provider.fetch_etf_data(
            config.etf_universe, data_start, current_date
        )

        prices_df = data_provider.get_prices(all_data)

        # Convert timezone
        if prices_df.index.tz is not None:
            ist_tz = pytz.timezone('Asia/Kolkata')
            prices_df.index = prices_df.index.tz_convert(ist_tz).tz_localize(None)
        else:
            prices_df.index = pd.to_datetime(prices_df.index).tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

        prices_df = prices_df.fillna(method="ffill")

        volume_df = data_provider.get_volumes(all_data)
        if volume_df is not None:
            if volume_df.index.tz is not None:
                volume_df.index = volume_df.index.tz_convert(ist_tz).tz_localize(None)
            else:
                volume_df.index = pd.to_datetime(volume_df.index).tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

        latest_date = prices_df.index[-1]
        current_prices = prices_df.iloc[-1]

        # OLD METHOD
        print(f"\n🔴 OLD METHOD (Index-based lookback + dropna)")
        print("-" * 60)

        old_calculator = MomentumCalculator(config)
        old_eligible = old_calculator.apply_filters(prices_df, volume_df)
        old_data = prices_df[old_eligible]
        old_scores = old_calculator.calculate_momentum_scores(old_data)

        old_ranked = sorted(
            [(ticker, score) for ticker, score in old_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        old_top = old_ranked[:portfolio_size]

        print(f"Eligible ETFs: {len(old_eligible)}")
        print(f"Top {portfolio_size} ETFs:")
        for i, (ticker, score) in enumerate(old_top, 1):
            print(f"  {i}. {ticker:15s} Score: {score:.4f}")

        # FIXED METHOD
        print(f"\n🟢 FIXED METHOD (Date-based lookback + forward-fill)")
        print("-" * 60)

        fixed_calculator = MomentumCalculatorFixed(config)
        fixed_eligible = fixed_calculator.apply_filters(prices_df, volume_df)
        fixed_data = prices_df[fixed_eligible]
        fixed_scores = fixed_calculator.calculate_momentum_scores(fixed_data)

        fixed_ranked = sorted(
            [(ticker, score) for ticker, score in fixed_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        fixed_top = fixed_ranked[:portfolio_size]

        print(f"Eligible ETFs: {len(fixed_eligible)}")
        print(f"Top {portfolio_size} ETFs:")
        for i, (ticker, score) in enumerate(fixed_top, 1):
            print(f"  {i}. {ticker:15s} Score: {score:.4f}")

        # COMPARISON
        print(f"\n📊 COMPARISON")
        print("-" * 60)

        old_tickers = set([t for t, _ in old_top])
        fixed_tickers = set([t for t, _ in fixed_top])

        same_etfs = old_tickers & fixed_tickers
        only_in_old = old_tickers - fixed_tickers
        only_in_fixed = fixed_tickers - old_tickers

        match_percentage = (len(same_etfs) / portfolio_size) * 100

        print(f"ETFs in both: {len(same_etfs)}/{portfolio_size} ({match_percentage:.1f}%)")

        if same_etfs:
            print(f"\n✅ Common ETFs: {sorted(same_etfs)}")

        if only_in_old:
            print(f"\n❌ Only in OLD: {sorted(only_in_old)}")

        if only_in_fixed:
            print(f"\n✅ Only in FIXED: {sorted(only_in_fixed)}")

        # Score comparison for common ETFs
        if same_etfs:
            print(f"\n📈 Score Comparison (Common ETFs):")
            score_comparison = []
            for ticker in sorted(same_etfs):
                old_score = old_scores.get(ticker, 0)
                fixed_score = fixed_scores.get(ticker, 0)
                diff = fixed_score - old_score
                diff_pct = (diff / old_score * 100) if old_score != 0 else 0

                score_comparison.append([
                    ticker,
                    f"{old_score:.4f}",
                    f"{fixed_score:.4f}",
                    f"{diff:+.4f}",
                    f"{diff_pct:+.1f}%"
                ])

            headers = ["ETF", "Old Score", "Fixed Score", "Diff", "Diff %"]
            print(tabulate(score_comparison, headers=headers, tablefmt="grid"))

        # Return results for aggregation
        return {
            'amount': investment_amount,
            'size': portfolio_size,
            'match_percentage': match_percentage,
            'same_count': len(same_etfs),
            'only_old': list(only_in_old),
            'only_fixed': list(only_in_fixed),
            'old_top': old_top,
            'fixed_top': fixed_top,
            'date': latest_date
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_comparisons():
    """Run comparisons across all permutations."""
    amounts = [100000, 500000, 1000000]
    sizes = [3, 5, 10]

    print("=" * 80)
    print("MOMENTUM CALCULATION COMPARISON: OLD vs FIXED")
    print("=" * 80)
    print(f"Testing {len(amounts)} amounts × {len(sizes)} sizes = {len(amounts) * len(sizes)} permutations")
    print(f"Amounts: {amounts}")
    print(f"Sizes: {sizes}")

    results = []

    for amount in amounts:
        for size in sizes:
            result = compare_methods(amount, size)
            if result:
                results.append(result)

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL COMPARISONS")
    print(f"{'='*80}\n")

    summary_data = []
    for r in results:
        summary_data.append([
            f"₹{r['amount']:,.0f}",
            r['size'],
            f"{r['match_percentage']:.1f}%",
            f"{r['same_count']}/{r['size']}",
            len(r['only_old']),
            len(r['only_fixed'])
        ])

    headers = ["Amount", "Size", "Match %", "Same", "Only OLD", "Only FIXED"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Overall statistics
    avg_match = sum(r['match_percentage'] for r in results) / len(results)
    min_match = min(r['match_percentage'] for r in results)
    max_match = max(r['match_percentage'] for r in results)

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Average Match: {avg_match:.1f}%")
    print(f"   Min Match: {min_match:.1f}%")
    print(f"   Max Match: {max_match:.1f}%")

    # ETFs that appear frequently in differences
    all_only_old = []
    all_only_fixed = []

    for r in results:
        all_only_old.extend(r['only_old'])
        all_only_fixed.extend(r['only_fixed'])

    if all_only_old:
        from collections import Counter
        old_freq = Counter(all_only_old)
        print(f"\n❌ ETFs frequently appearing ONLY in OLD method:")
        for etf, count in old_freq.most_common(5):
            print(f"   {etf}: {count}/{len(results)} times ({count/len(results)*100:.1f}%)")

    if all_only_fixed:
        from collections import Counter
        fixed_freq = Counter(all_only_fixed)
        print(f"\n✅ ETFs frequently appearing ONLY in FIXED method:")
        for etf, count in fixed_freq.most_common(5):
            print(f"   {etf}: {count}/{len(results)} times ({count/len(results)*100:.1f}%)")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    if avg_match >= 80:
        print(f"✓ Methods are SIMILAR (avg {avg_match:.1f}% match)")
        print(f"  → Fixed method provides minor improvements in accuracy")
    elif avg_match >= 60:
        print(f"⚠ Methods show MODERATE differences (avg {avg_match:.1f}% match)")
        print(f"  → Fixed method may provide better ETF selection")
    else:
        print(f"❌ Methods are SIGNIFICANTLY DIFFERENT (avg {avg_match:.1f}% match)")
        print(f"  → Fixed method provides substantially different rankings")
        print(f"  → Current portfolios may be based on incorrect momentum scores")

    print(f"\n💡 RECOMMENDATION:")
    if avg_match < 100:
        print(f"   Consider adopting the FIXED method for more accurate momentum calculations")
        print(f"   The date-based lookback ensures consistent time periods across all ETFs")
    else:
        print(f"   Both methods produce identical results - no changes needed")


if __name__ == "__main__":
    run_all_comparisons()
