import backtrader as bt
import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from utils import MarketDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
)


class PairsStrategy(bt.Strategy):
    params = (
        ("beta_window", 60),  # Longer window for stable beta
        ("std_period", 20),  # Longer period for spread volatility
        ("devfactor_entry", 1.5),  # Much lower entry threshold
        ("devfactor_exit", 0.5),  # Tighter exit
        ("capital", 100000),
        ("base_size", 100),
        ("printlog", True),
        ("min_corr", 0.85),  # Slightly lower correlation requirement
        ("stop_dev", 4),  # Tighter stop loss
        ("max_hold_days", 15),  # Shorter holding period
        ("vol_factor", True),
        ("min_profit_target", 0.002),  # Lower minimum profit target (0.2%)
        ("use_returns", True),  # Use returns for beta calculation
        ("atr_period", 14),  # For volatility-based sizing
        ("risk_per_trade", 0.01),  # 1% risk per trade
    )

    def log(self, txt):
        if self.p.printlog:
            logging.info(f"{self.datetime.date(0)}: {txt}")

    def __init__(self):
        self.data1 = self.datas[0]  # e.g., Nifty
        self.data2 = self.datas[1]  # e.g., Bank Nifty
        self.order = None
        self.loglist = []  # For charting data
        self.position_start_len = None  # Track when position started

        # Calculate returns for beta calculation using backtrader indicators
        self.returns1 = bt.indicators.PctChange(self.data1.close, period=1)
        self.returns2 = bt.indicators.PctChange(self.data2.close, period=1)

        # ATR for volatility-based sizing
        self.atr1 = bt.indicators.ATR(self.data1, period=self.p.atr_period)
        self.atr2 = bt.indicators.ATR(self.data2, period=self.p.atr_period)

    def calculate_beta_on_returns(self):
        """Calculate beta using returns instead of price levels"""
        if len(self.returns1) < self.p.beta_window:
            return None, None

        returns1_data = []
        returns2_data = []

        for i in range(1, self.p.beta_window + 1):
            if len(self.returns1) >= i and len(self.returns2) >= i:
                ret1 = self.returns1[-i]
                ret2 = self.returns2[-i]
                if not (np.isnan(ret1) or np.isnan(ret2)):
                    returns1_data.append(ret1)
                    returns2_data.append(ret2)

        if len(returns1_data) < self.p.beta_window * 0.8:  # Need at least 80% of data
            return None, None

        returns1_data.reverse()
        returns2_data.reverse()

        result = linregress(returns2_data, returns1_data)
        return result.slope, result.intercept

    def calculate_hedge_ratio(self):
        """Calculate hedge ratio using price levels for actual trading"""
        if len(self.data1) < self.p.beta_window:
            return None

        p1_data = []
        p2_data = []

        for i in range(1, self.p.beta_window + 1):
            if len(self.data1) >= i and len(self.data2) >= i:
                p1_data.append(self.data1[-i])
                p2_data.append(self.data2[-i])

        if len(p1_data) < self.p.beta_window:
            return None

        p1_data.reverse()
        p2_data.reverse()

        result = linregress(p2_data, p1_data)
        return result.slope

    def calculate_volatility_adjusted_size(self, base_size, data_feed, atr):
        """Calculate position size based on volatility"""
        if atr[0] <= 0:
            return base_size

        # Calculate position size based on risk per trade
        price = data_feed[0]
        portfolio_value = self.broker.getvalue()
        risk_amount = portfolio_value * self.p.risk_per_trade

        # Use ATR as proxy for potential loss per share
        atr_value = atr[0]
        position_size = min(base_size, int(risk_amount / atr_value))

        return max(1, position_size)  # Minimum 1 share

    def next(self):
        if len(self.data1) < max(self.p.beta_window, self.p.std_period):
            return

        # Calculate beta on returns for regime detection
        beta_returns, _ = self.calculate_beta_on_returns()
        if beta_returns is None:
            return

        # Calculate hedge ratio for actual trading
        hedge_ratio = self.calculate_hedge_ratio()
        if hedge_ratio is None:
            return

        # Calculate spread using hedge ratio
        spread_data = []
        for i in range(1, self.p.std_period + 1):
            if len(self.data1) >= i and len(self.data2) >= i:
                spread = self.data1[-i] - hedge_ratio * self.data2[-i]
                spread_data.append(spread)

        if len(spread_data) < self.p.std_period:
            return

        mean_spread = np.mean(spread_data)
        std_spread = np.std(spread_data)
        if std_spread == 0:
            return

        # Current spread and z-score
        current_spread = self.data1[0] - hedge_ratio * self.data2[0]
        zscore = (current_spread - mean_spread) / std_spread

        # Rolling correlation using returns
        returns1_corr = []
        returns2_corr = []

        for i in range(1, self.p.beta_window + 1):
            if len(self.returns1) >= i and len(self.returns2) >= i:
                ret1 = self.returns1[-i]
                ret2 = self.returns2[-i]
                if not (np.isnan(ret1) or np.isnan(ret2)):
                    returns1_corr.append(ret1)
                    returns2_corr.append(ret2)

        if len(returns1_corr) > 10:
            corr = np.corrcoef(returns1_corr, returns2_corr)[0, 1]
        else:
            corr = 0

        # Log data
        self.loglist.append(
            {
                "date": self.datetime.date(),
                "zscore": zscore,
                "spread": current_spread,
                "equity": self.broker.getvalue(),
                "hedge_ratio": hedge_ratio,
                "beta_returns": beta_returns,
                "corr": corr,
                "std_spread": std_spread,
            }
        )

        position1 = self.getposition(self.data1).size

        if position1 != 0:  # In position
            if self.position_start_len is None:
                self.position_start_len = len(self) - 1
            hold_bars = len(self) - self.position_start_len

            # Exit conditions
            if abs(zscore) < self.p.devfactor_exit:
                self.close(data=self.data1)
                self.close(data=self.data2)
                self.log(f"Exit: Z-score converged to {zscore:.2f}")
                self.position_start_len = None
            elif abs(zscore) > self.p.stop_dev:
                self.close(data=self.data1)
                self.close(data=self.data2)
                self.log(f"Exit: Stop-loss triggered at {zscore:.2f}")
                self.position_start_len = None
            elif hold_bars > self.p.max_hold_days:
                self.close(data=self.data1)
                self.close(data=self.data2)
                self.log(f"Exit: Max hold period reached ({hold_bars} days)")
                self.position_start_len = None
        else:  # No position
            if corr < self.p.min_corr:
                return

            # Calculate expected profit
            expected_reversion = abs(zscore) * std_spread
            expected_profit_pct = expected_reversion / (
                (self.data1[0] + self.data2[0]) / 2
            )

            if expected_profit_pct < self.p.min_profit_target:
                return

            # Volatility-adjusted position sizing
            if self.p.vol_factor:
                size1 = self.calculate_volatility_adjusted_size(
                    self.p.base_size, self.data1, self.atr1
                )
                size2 = self.calculate_volatility_adjusted_size(
                    int(self.p.base_size * abs(hedge_ratio)), self.data2, self.atr2
                )
            else:
                size1 = self.p.base_size
                size2 = int(self.p.base_size * abs(hedge_ratio))

            if size1 == 0 or size2 == 0:
                return

            # Entry logic
            if zscore > self.p.devfactor_entry:  # Short spread
                if size1 > 0 and size2 > 0:
                    self.sell(data=self.data1, size=size1)
                    self.buy(data=self.data2, size=size2)
                    self.log(
                        f"Entry Short: Z={zscore:.2f}, Exp={expected_profit_pct:.3f}, HR={hedge_ratio:.3f}"
                    )
                    self.position_start_len = len(self)
            elif zscore < -self.p.devfactor_entry:  # Long spread
                if size1 > 0 and size2 > 0:
                    self.buy(data=self.data1, size=size1)
                    self.sell(data=self.data2, size=size2)
                    self.log(
                        f"Entry Long: Z={zscore:.2f}, Exp={expected_profit_pct:.3f}, HR={hedge_ratio:.3f}"
                    )
                    self.position_start_len = len(self)


# Test multiple pairs for better performance
def test_multiple_pairs():
    """Test different pairs and return the best performing one"""

    # Better Indian index pairs - more similar characteristics
    pairs_to_test = [
        # Large Cap indices (more stable, better correlation)
        ("^NSEI", "^CNXIT"),  # Nifty vs IT Index - similar market caps
        ("^NSEI", "^CNXPHARMA"),  # Nifty vs Pharma - defensive sectors
        ("^NSEI", "^CNXFMCG"),  # Nifty vs FMCG - stable sectors
        # Sector pairs (similar volatility)
        ("^CNXIT", "^CNXPHARMA"),  # IT vs Pharma - both large cap
        ("^CNXFMCG", "^CNXPHARMA"),  # FMCG vs Pharma - defensive
        # Size-based pairs (should have better correlation)
        ("^NSEI", "NIFTYNXT50.NS"),  # Nifty vs Next 50 - similar companies
        # The problematic original pair
        ("^NSEI", "^NSEBANK"),  # Original: Nifty vs Bank Nifty (for comparison)
    ]

    print("🔍 Testing Multiple Pairs for Best Performance...")
    print("=" * 70)

    data_loader = MarketDataLoader(cache_dir="data_cache")
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 7, 13)

    best_pair = None
    best_score = -999
    pair_results = []

    for i, (symbol1, symbol2) in enumerate(pairs_to_test):
        try:
            print(
                f"\n📊 Testing Pair {i+1}/{len(pairs_to_test)}: {symbol1} vs {symbol2}"
            )

            # Load data for this pair
            symbols = [symbol1, symbol2]
            try:
                data_feeds = data_loader.load_market_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=False,
                )

                if len(data_feeds) < 2:
                    print(f"   ❌ Could not load data for both symbols")
                    continue

            except Exception as e:
                print(f"   ❌ Data loading failed: {e}")
                continue

            # Extract dataframes
            try:
                feed1 = next(feed for feed in data_feeds if feed._name == symbol1)
                feed2 = next(feed for feed in data_feeds if feed._name == symbol2)

                df1 = feed1._dataname["close"]
                df2 = feed2._dataname["close"]

                # Align data
                df_aligned = pd.concat([df1, df2], axis=1).dropna()
                df_aligned.columns = ["Asset1", "Asset2"]

                if len(df_aligned) < 100:
                    print(f"   ❌ Insufficient data ({len(df_aligned)} rows)")
                    continue

            except Exception as e:
                print(f"   ❌ Data processing failed: {e}")
                continue

            # Calculate pair statistics
            try:
                # Correlation
                correlation = df_aligned["Asset1"].corr(df_aligned["Asset2"])

                # Volatility ratio (should be close to 1 for good pairs)
                vol1 = df_aligned["Asset1"].pct_change().std() * np.sqrt(252)
                vol2 = df_aligned["Asset2"].pct_change().std() * np.sqrt(252)
                vol_ratio = max(vol1, vol2) / min(vol1, vol2)

                # Cointegration test
                model = sm.OLS(
                    df_aligned["Asset1"], sm.add_constant(df_aligned["Asset2"])
                ).fit()
                residuals = model.resid
                adf_result = adfuller(residuals)
                p_value = adf_result[1]
                is_cointegrated = p_value < 0.05

                # Calculate a composite score
                corr_score = correlation * 100  # Higher is better
                coint_score = (
                    (1 - p_value) * 100 if p_value < 1 else 0
                )  # Higher is better
                vol_score = max(
                    0, 100 - (vol_ratio - 1) * 50
                )  # Penalize high vol ratio

                composite_score = (
                    (corr_score * 0.4) + (coint_score * 0.4) + (vol_score * 0.2)
                )

                result = {
                    "pair": f"{symbol1} vs {symbol2}",
                    "symbols": (symbol1, symbol2),
                    "correlation": correlation,
                    "vol_ratio": vol_ratio,
                    "p_value": p_value,
                    "cointegrated": is_cointegrated,
                    "composite_score": composite_score,
                    "data_feeds": data_feeds,
                }

                pair_results.append(result)

                print(f"   📈 Correlation: {correlation:.3f}")
                print(f"   🎯 Volatility Ratio: {vol_ratio:.2f}")
                print(
                    f"   🔗 Cointegrated: {'✅ Yes' if is_cointegrated else '❌ No'} (p={p_value:.4f})"
                )
                print(f"   🏆 Composite Score: {composite_score:.1f}")

                if composite_score > best_score:
                    best_score = composite_score
                    best_pair = result

            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
                continue

        except Exception as e:
            print(f"   ❌ Overall test failed: {e}")
            continue

    # Print summary
    print("\n" + "=" * 70)
    print("📊 PAIR TESTING SUMMARY")
    print("=" * 70)

    if pair_results:
        # Sort by composite score
        pair_results.sort(key=lambda x: x["composite_score"], reverse=True)

        print("\n🏆 Top 3 Pairs:")
        for i, result in enumerate(pair_results[:3]):
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            coint_emoji = "✅" if result["cointegrated"] else "❌"
            print(f"{rank_emoji} {result['pair']}")
            print(
                f"    Score: {result['composite_score']:.1f} | Corr: {result['correlation']:.3f} | Vol Ratio: {result['vol_ratio']:.2f} | Coint: {coint_emoji}"
            )

        if best_pair and best_pair["composite_score"] > 50:
            print(f"\n🎯 Best Pair Selected: {best_pair['pair']}")
            print(f"   Score: {best_pair['composite_score']:.1f}")
            return best_pair
        else:
            print("\n⚠️ No pair scored above 50. All pairs are problematic.")
            print("💡 Consider:")
            print("   - Using individual stocks instead of indices")
            print("   - Looking at international markets")
            print("   - Using ETFs instead of indices")
            return None
    else:
        print("❌ No pairs could be tested successfully")
        return None


# Enhanced cointegration check
def enhanced_cointegration_check(df1, df2, window=252):
    """Rolling cointegration test"""
    results = []

    for i in range(window, len(df1)):
        subset1 = df1.iloc[i - window : i]
        subset2 = df2.iloc[i - window : i]

        try:
            model = sm.OLS(subset1, sm.add_constant(subset2)).fit()
            residuals = model.resid
            adf = adfuller(residuals)
            results.append(
                {
                    "date": df1.index[i],
                    "p_value": adf[1],
                    "is_cointegrated": adf[1] < 0.05,
                }
            )
        except:
            results.append(
                {"date": df1.index[i], "p_value": 1.0, "is_cointegrated": False}
            )

    results_df = pd.DataFrame(results)
    cointegration_pct = results_df["is_cointegrated"].mean() * 100

    print(f"Rolling cointegration analysis:")
    print(f"Cointegrated {cointegration_pct:.1f}% of the time")
    print(f"Average p-value: {results_df['p_value'].mean():.4f}")

    return results_df


# Pre-check cointegration for accuracy
def check_cointegration(df1, df2):
    model = sm.OLS(df1, sm.add_constant(df2)).fit()
    residuals = model.resid
    adf = adfuller(residuals)
    if adf[1] < 0.05:
        logging.info("Pair is cointegrated (p-value: %.4f)" % adf[1])
        return True
    else:
        logging.warning("Pair not cointegrated (p-value: %.4f)" % adf[1])
        return False


# Main backtest function
if __name__ == "__main__":
    print("🚀 PAIRS TRADING STRATEGY - MULTI-PAIR TESTING")
    print("=" * 70)

    # First, test multiple pairs to find the best one
    best_pair_result = test_multiple_pairs()

    if best_pair_result is None:
        print("\n❌ No suitable pairs found. Exiting.")
        exit()

    # Use the best pair for backtesting
    symbol1, symbol2 = best_pair_result["symbols"]
    data_feeds = best_pair_result["data_feeds"]

    print(f"\n🎯 Running backtest with: {symbol1} vs {symbol2}")
    print("=" * 70)

    # Extract dataframes for final cointegration check
    feed1 = next(feed for feed in data_feeds if feed._name == symbol1)
    feed2 = next(feed for feed in data_feeds if feed._name == symbol2)

    # Get the underlying dataframes
    df1 = feed1._dataname["close"]
    df2 = feed2._dataname["close"]

    # Align data
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = ["Asset1", "Asset2"]

    # Final cointegration check
    if not check_cointegration(df["Asset1"], df["Asset2"]):
        print("⚠️ Warning: Selected pair failed final cointegration test.")
        print("Proceeding anyway for demonstration...")

    # Backtrader setup
    cerebro = bt.Cerebro()

    # Add data feeds directly (already in backtrader format)
    cerebro.adddata(feed1, name=symbol1)
    cerebro.adddata(feed2, name=symbol2)

    cerebro.addstrategy(PairsStrategy)
    cerebro.broker.setcash(200000)  # Total cash (for both legs)
    cerebro.broker.setcommission(
        0.00005
    )  # 0.005% commission (more realistic for Indian brokers)

    # Built-in analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade")

    # Run backtest
    strats = cerebro.run()

    # Print results
    strat = strats[0]

    print("\n" + "=" * 60)
    print(f"     PAIRS TRADING RESULTS: {symbol1} vs {symbol2}")
    print("=" * 60)

    # Sharpe Ratio Analysis
    sharpe_data = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_data.get("sharperatio", 0)
    print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")
    if sharpe_ratio > 1.0:
        print("   🟢 EXCELLENT - Very strong risk-adjusted returns")
    elif sharpe_ratio > 0.5:
        print("   🟡 GOOD - Decent risk-adjusted returns")
    elif sharpe_ratio > 0:
        print("   🟠 POOR - Low risk-adjusted returns")
    else:
        print("   🔴 BAD - Negative risk-adjusted returns (strategy loses money)")

    # Drawdown Analysis
    dd_data = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_data.get("max", {}).get("drawdown", 0)
    max_dd_money = dd_data.get("max", {}).get("moneydown", 0)
    print(f"\n📉 Maximum Drawdown: {max_dd:.2f}% (₹{max_dd_money:,.2f})")
    if max_dd < 5:
        print("   🟢 EXCELLENT - Very low drawdown")
    elif max_dd < 10:
        print("   🟡 GOOD - Manageable drawdown")
    elif max_dd < 20:
        print("   🟠 CONCERNING - High drawdown")
    else:
        print("   🔴 DANGEROUS - Very high drawdown")

    # Trade Analysis
    trade_data = strat.analyzers.trade.get_analysis()
    total_trades = trade_data.get("total", {}).get("closed", 0)
    won_trades = trade_data.get("won", {}).get("total", 0)
    lost_trades = trade_data.get("lost", {}).get("total", 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    gross_pnl = trade_data.get("pnl", {}).get("gross", {}).get("total", 0)
    net_pnl = trade_data.get("pnl", {}).get("net", {}).get("total", 0)

    print(f"\n📈 Trading Performance:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1f}% ({won_trades} wins, {lost_trades} losses)")
    print(f"   Gross P&L: ₹{gross_pnl:,.2f}")
    print(f"   Net P&L: ₹{net_pnl:,.2f}")
    print(f"   Trading Costs: ₹{gross_pnl - net_pnl:,.2f}")

    # Portfolio Performance
    initial_cash = 200000
    final_value = strat.broker.getvalue()
    total_return = ((final_value - initial_cash) / initial_cash) * 100

    print(f"\n💰 Portfolio Performance:")
    print(f"   Initial Capital: ₹{initial_cash:,.2f}")
    print(f"   Final Value: ₹{final_value:,.2f}")
    print(f"   Total Return: {total_return:.2f}%")

    # Performance Analysis
    print(f"\n🔍 Strategy Analysis:")
    if sharpe_ratio < 0:
        print("   ❌ Strategy is losing money consistently")
        print("   💡 Possible issues:")
        print("      - Entry/exit thresholds may be too tight or too loose")
        print("      - Transaction costs are eating into profits")
        print("      - Market regime may not be suitable for mean reversion")
        print("      - Beta calculation window might be suboptimal")

    if win_rate < 50:
        print(f"   ⚠️  Low win rate ({win_rate:.1f}%) suggests:")
        print("      - Entry signals may be premature")
        print("      - Exit strategy might be cutting winners short")

    if max_dd > 15:
        print(f"   ⚠️  High drawdown ({max_dd:.1f}%) suggests:")
        print("      - Position sizing may be too aggressive")
        print("      - Stop-loss mechanisms needed")

    # Indian Market Specific Issues
    print(f"\n🇮🇳 Indian Market Specific Analysis:")
    print("   📊 Nifty vs Bank Nifty Pair Issues:")
    print("      - Bank Nifty is much more volatile than Nifty")
    print("      - Different sector exposures cause divergence")
    print("      - RBI policy affects banking sector disproportionately")
    print("      - COVID period (2020-2022) disrupted normal correlations")

    # Calculate some basic stats for better understanding
    vol1 = df1.pct_change().std() * np.sqrt(252) * 100
    vol2 = df2.pct_change().std() * np.sqrt(252) * 100
    correlation = df1.corr(df2)

    print(f"\n📈 Market Statistics for {symbol1} vs {symbol2} (2020-2025):")
    print(f"   {symbol1} Annualized Volatility: {vol1:.1f}%")
    print(f"   {symbol2} Annualized Volatility: {vol2:.1f}%")
    print(f"   Correlation: {correlation:.3f}")
    print(f"   Volatility Ratio: {max(vol1,vol2)/min(vol1,vol2):.2f}x")

    if max(vol1, vol2) / min(vol1, vol2) > 1.5:
        print("   ⚠️  High volatility difference between assets!")
        print("      - This makes beta calculation unstable")
        print("      - Position sizing becomes critical")

    if correlation < 0.85:
        print(f"   ⚠️  Low correlation ({correlation:.3f}) suggests:")
        print("      - Pair may not be suitable for mean reversion")
        print("      - Consider testing other pairs")

    print("=" * 60)

    # Strategy Parameters Used
    print(f"\n⚙️  Strategy Parameters:")
    print(f"   Beta Window: {strat.p.beta_window} days")
    print(f"   Std Period: {strat.p.std_period} days")
    print(f"   Entry Threshold: ±{strat.p.devfactor_entry} std deviations")
    print(f"   Exit Threshold: ±{strat.p.devfactor_exit} std deviations")
    print(f"   Base Position Size: {strat.p.base_size}")

    # Improvement Suggestions
    print(f"\n💡 Potential Improvements for Indian Markets:")
    print("   1. Better Pair Selection:")
    print("      - Try Nifty vs Nifty Next 50 (more similar)")
    print("      - Consider Nifty vs Nifty Midcap 100")
    print("      - Use sector-specific pairs (IT vs Pharma)")
    print("   2. Indian Market Specific Parameters:")
    print("      - Use wider entry thresholds (2.0-2.5σ) due to high volatility")
    print("      - Shorter lookback periods (50-75 days) for faster adaptation")
    print("      - Account for RBI policy announcement dates")
    print("   3. Risk Management for High Volatility:")
    print("      - Implement volatility-based position sizing")
    print("      - Add stop-loss at 3σ to prevent large losses")
    print("      - Consider maximum holding period (10-15 days)")
    print("   4. Transaction Cost Optimization:")
    print("      - Indian markets have lower brokerage (₹20 per trade)")
    print("      - But impact cost can be high for large sizes")
    print("      - Consider using index futures instead of spot")
    print("   5. Market Regime Awareness:")
    print("      - Avoid trading during earnings seasons")
    print("      - Be cautious around budget announcements")
    print("      - Monitor FII/DII flow patterns")

    # Charting with Seaborn
    df_log = pd.DataFrame(strat.loglist)
    if len(df_log) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Z-Score plot with entry/exit thresholds
        sns.lineplot(
            data=df_log, x="date", y="zscore", ax=axes[0], color="blue", alpha=0.7
        )
        axes[0].axhline(
            y=strat.p.devfactor_entry,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Entry (+{strat.p.devfactor_entry})",
        )
        axes[0].axhline(
            y=-strat.p.devfactor_entry,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Entry (-{strat.p.devfactor_entry})",
        )
        axes[0].axhline(
            y=strat.p.devfactor_exit,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Exit (+{strat.p.devfactor_exit})",
        )
        axes[0].axhline(
            y=-strat.p.devfactor_exit,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Exit (-{strat.p.devfactor_exit})",
        )
        axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[0].set_title("Z-Score Over Time with Entry/Exit Thresholds")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Equity curve with buy & hold comparison
        sns.lineplot(
            data=df_log,
            x="date",
            y="equity",
            ax=axes[1],
            color="green",
            label="Pairs Strategy",
        )
        # Calculate buy & hold for comparison
        initial_value = df_log["equity"].iloc[0]
        asset1_returns = (df1 / df1.iloc[0]) * initial_value
        asset2_returns = (df2 / df2.iloc[0]) * initial_value

        # Align dates for buy & hold comparison
        if len(asset1_returns) > 0:
            buy_hold_dates = asset1_returns.index[: len(df_log)]
            axes[1].plot(
                buy_hold_dates,
                asset1_returns.iloc[: len(df_log)],
                color="blue",
                alpha=0.7,
                label=f"{symbol1} Buy & Hold",
            )
            axes[1].plot(
                buy_hold_dates,
                asset2_returns.iloc[: len(df_log)],
                color="orange",
                alpha=0.7,
                label=f"{symbol2} Buy & Hold",
            )

        axes[1].set_title("Strategy Performance vs Buy & Hold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Spread plot
        sns.lineplot(
            data=df_log, x="date", y="spread", ax=axes[2], color="purple", alpha=0.7
        )
        axes[2].set_title(f"Spread ({symbol1} - β × {symbol2}) Over Time")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️  No trading data to plot - strategy may not have executed any trades")
