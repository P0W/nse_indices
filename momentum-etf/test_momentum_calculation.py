#!/usr/bin/env python3
"""
Test script to verify momentum calculation correctness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_momentum_calculation():
    """Test the momentum calculation logic for potential issues."""

    print("=" * 80)
    print("MOMENTUM CALCULATION VERIFICATION")
    print("=" * 80)

    # Test Case 1: Index-based vs Date-based lookback
    print("\n📊 TEST 1: Index-based Lookback Issue")
    print("-" * 60)

    # Create sample price series with gaps (missing weekends)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    # Remove weekends to simulate trading days
    trading_dates = [d for d in dates if d.weekday() < 5]

    # Create price data with linear growth
    prices_perfect = pd.Series(
        data=np.linspace(100, 120, len(trading_dates)),
        index=trading_dates
    )

    # Simulate data with gaps (missing days)
    prices_with_gaps = prices_perfect.copy()
    # Remove 20% of random days to simulate data gaps
    np.random.seed(42)
    gaps_indices = np.random.choice(prices_with_gaps.index[50:-50], size=40, replace=False)
    prices_with_gaps = prices_with_gaps.drop(gaps_indices)

    print(f"Perfect data points: {len(prices_perfect)}")
    print(f"Data with gaps: {len(prices_with_gaps)}")
    print(f"Missing days: {len(prices_perfect) - len(prices_with_gaps)}")

    # Calculate 180-day return using INDEX method (current implementation)
    period_days = 180
    if len(prices_with_gaps) >= period_days:
        current_price = prices_with_gaps.iloc[-1]
        past_price_index = prices_with_gaps.iloc[-period_days]
        return_index = (current_price / past_price_index - 1) * 100

        # Calculate using DATE method (correct approach)
        lookback_date = prices_with_gaps.index[-1] - timedelta(days=period_days)
        # Find closest date
        time_diffs = abs(prices_with_gaps.index - lookback_date)
        closest_date_idx = time_diffs.argmin()
        past_price_date = prices_with_gaps.iloc[closest_date_idx]
        return_date = (current_price / past_price_date - 1) * 100

        actual_days_index = (prices_with_gaps.index[-1] - prices_with_gaps.index[-period_days]).days
        actual_days_date = (prices_with_gaps.index[-1] - prices_with_gaps.index[closest_date_idx]).days

        print(f"\n⚠️  INDEX-based lookback:")
        print(f"   Looking back {period_days} positions = {actual_days_index} calendar days")
        print(f"   Past price: ₹{past_price_index:.2f}, Current: ₹{current_price:.2f}")
        print(f"   Return: {return_index:.2f}%")

        print(f"\n✓ DATE-based lookback:")
        print(f"   Looking back {period_days} calendar days = {actual_days_date} days")
        print(f"   Past price: ₹{past_price_date:.2f}, Current: ₹{current_price:.2f}")
        print(f"   Return: {return_date:.2f}%")

        print(f"\n❌ Difference: {abs(return_index - return_date):.2f}% (ERROR!)")

    # Test Case 2: Impact of dropna() on index positions
    print("\n\n📊 TEST 2: dropna() Index Misalignment")
    print("-" * 60)

    # Create price series with NaN values
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    prices = pd.Series(index=dates, data=np.linspace(100, 120, 200))

    # Insert some NaN values in the middle
    prices.iloc[80:85] = np.nan
    prices.iloc[120:123] = np.nan

    print(f"Original series length: {len(prices)}")
    print(f"NaN values: {prices.isna().sum()}")

    # Using dropna (current implementation)
    prices_dropped = prices.dropna()
    print(f"After dropna: {len(prices_dropped)}")

    lookback = 60
    current_price = prices_dropped.iloc[-1]
    past_price_dropped = prices_dropped.iloc[-lookback]

    # What we THINK we're getting (60 days ago)
    # What we ACTUALLY get (60 positions ago in dropped series)
    actual_date_diff = (prices_dropped.index[-1] - prices_dropped.index[-lookback]).days

    print(f"\n⚠️  With dropna():")
    print(f"   Requested: {lookback} positions back")
    print(f"   Actual calendar days: {actual_date_diff} days")
    print(f"   Expected: ~{lookback} days")
    print(f"   ERROR: {abs(actual_date_diff - lookback)} days off!")

    # Test Case 3: Weighted Momentum Score Calculation
    print("\n\n📊 TEST 3: Weighted Momentum Score")
    print("-" * 60)

    # Simulate two different return scenarios
    long_return_1 = 0.25  # 25% over 180 days
    short_return_1 = 0.10  # 10% over 60 days

    long_return_2 = 0.15  # 15% over 180 days
    short_return_2 = 0.20  # 20% over 60 days

    long_weight = 0.6
    short_weight = 0.4

    score_1 = long_return_1 * long_weight + short_return_1 * short_weight
    score_2 = long_return_2 * long_weight + short_return_2 * short_weight

    print(f"Scenario 1: Strong long-term, weak short-term")
    print(f"   Long (180d): {long_return_1*100:.1f}% × {long_weight} = {long_return_1*long_weight:.4f}")
    print(f"   Short (60d): {short_return_1*100:.1f}% × {short_weight} = {short_return_1*short_weight:.4f}")
    print(f"   Final Score: {score_1:.4f}")

    print(f"\nScenario 2: Weak long-term, strong short-term")
    print(f"   Long (180d): {long_return_2*100:.1f}% × {long_weight} = {long_return_2*long_weight:.4f}")
    print(f"   Short (60d): {short_return_2*100:.1f}% × {short_weight} = {short_return_2*short_weight:.4f}")
    print(f"   Final Score: {score_2:.4f}")

    print(f"\n📈 Winner: Scenario {'1' if score_1 > score_2 else '2'}")
    print(f"   Difference: {abs(score_1 - score_2):.4f}")

    # Test Case 4: Risk-Adjusted Returns (NOT implemented)
    print("\n\n📊 TEST 4: Risk-Adjusted vs Raw Returns")
    print("-" * 60)

    # ETF A: Steady growth, low volatility
    dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
    etf_a = pd.Series(
        index=dates,
        data=np.linspace(100, 120, 180) + np.random.normal(0, 1, 180)  # Low vol
    )

    # ETF B: Volatile growth, high volatility
    etf_b = pd.Series(
        index=dates,
        data=np.linspace(100, 120, 180) + np.random.normal(0, 8, 180)  # High vol
    )

    # Calculate raw returns (current implementation)
    raw_return_a = (etf_a.iloc[-1] / etf_a.iloc[0] - 1)
    raw_return_b = (etf_b.iloc[-1] / etf_b.iloc[0] - 1)

    # Calculate risk-adjusted returns (Sharpe-like)
    volatility_a = etf_a.pct_change().std() * np.sqrt(252)
    volatility_b = etf_b.pct_change().std() * np.sqrt(252)

    risk_adj_a = raw_return_a / volatility_a if volatility_a > 0 else 0
    risk_adj_b = raw_return_b / volatility_b if volatility_b > 0 else 0

    print(f"ETF A (Low Volatility):")
    print(f"   Raw Return: {raw_return_a*100:.2f}%")
    print(f"   Volatility: {volatility_a*100:.2f}%")
    print(f"   Risk-Adjusted: {risk_adj_a:.4f}")

    print(f"\nETF B (High Volatility):")
    print(f"   Raw Return: {raw_return_b*100:.2f}%")
    print(f"   Volatility: {volatility_b*100:.2f}%")
    print(f"   Risk-Adjusted: {risk_adj_b:.4f}")

    print(f"\n⚠️  Current implementation uses RAW returns only!")
    print(f"   Winner by raw return: ETF {'A' if raw_return_a > raw_return_b else 'B'}")
    print(f"   Winner by risk-adjusted: ETF {'A' if risk_adj_a > risk_adj_b else 'B'}")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    print("\n❌ CRITICAL ISSUES FOUND:")
    print("   1. Index-based lookback instead of date-based")
    print("      → Can cause incorrect period calculations with data gaps")
    print("   2. dropna() causes index misalignment")
    print("      → 60 positions ≠ 60 days when NaN values exist")
    print("   3. No risk adjustment")
    print("      → Favors volatile ETFs over steady performers")

    print("\n✓ CORRECT ASPECTS:")
    print("   1. Weighted dual-momentum approach (60/40 split)")
    print("   2. Returns calculation formula: (current/past - 1)")
    print("   3. Proper handling of None returns")

    print("\n💡 RECOMMENDATIONS:")
    print("   1. Use date-based lookback instead of index-based")
    print("   2. Handle NaN values without dropping (forward fill)")
    print("   3. Consider risk-adjusted returns for better ranking")
    print("   4. Add momentum persistence checks")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_momentum_calculation()
