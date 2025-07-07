import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class StrategyAnalyzer:
    """Advanced analysis and visualization tools for ETF momentum strategy."""

    def __init__(self, backtest_results: Dict):
        self.results = backtest_results
        self.portfolio_history = pd.DataFrame(backtest_results["portfolio_history"])
        self.trade_log = pd.DataFrame(backtest_results["trade_log"])
        self.performance_metrics = backtest_results["performance_metrics"]

    def create_performance_dashboard(self) -> go.Figure:
        """Create an interactive performance dashboard using Plotly."""

        # Prepare data
        portfolio_df = self.portfolio_history.copy()
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        portfolio_df["returns"] = portfolio_df["total_value"].pct_change()
        portfolio_df["cumulative_return"] = (
            portfolio_df["total_value"] / portfolio_df["total_value"].iloc[0] - 1
        ) * 100
        portfolio_df["drawdown"] = (
            (
                portfolio_df["total_value"]
                / portfolio_df["total_value"].expanding().max()
            )
            - 1
        ) * 100

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Portfolio Value Over Time",
                "Cumulative Returns",
                "Monthly Returns Distribution",
                "Drawdown Analysis",
                "Cash vs Holdings Allocation",
                "Rolling Sharpe Ratio",
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Portfolio value over time
        fig.add_trace(
            go.Scatter(
                x=portfolio_df["date"],
                y=portfolio_df["total_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Monthly returns histogram
        monthly_returns = portfolio_df["returns"].dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=monthly_returns,
                nbinsx=20,
                name="Monthly Returns (%)",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )

        # Drawdown analysis
        fig.add_trace(
            go.Scatter(
                x=portfolio_df["date"],
                y=portfolio_df["drawdown"],
                mode="lines",
                name="Drawdown (%)",
                fill="tonexty",
                line=dict(color="red"),
            ),
            row=2,
            col=2,
        )

        # Cash vs Holdings
        holdings_value = portfolio_df["total_value"] - portfolio_df["cash"]
        fig.add_trace(
            go.Bar(
                x=portfolio_df["date"],
                y=portfolio_df["cash"],
                name="Cash",
                marker_color="green",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=portfolio_df["date"],
                y=holdings_value,
                name="Holdings",
                marker_color="blue",
            ),
            row=3,
            col=1,
        )

        # Rolling Sharpe ratio (6-month window)
        if len(portfolio_df) >= 6:
            rolling_sharpe = (
                portfolio_df["returns"]
                .rolling(6)
                .apply(lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0)
            )
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df["date"],
                    y=rolling_sharpe,
                    mode="lines",
                    name="6M Rolling Sharpe",
                    line=dict(color="purple"),
                ),
                row=3,
                col=2,
            )

        fig.update_layout(
            height=900,
            title="ETF Momentum Strategy Performance Dashboard",
            showlegend=True,
        )

        return fig

    def analyze_etf_selection_frequency(self) -> pd.DataFrame:
        """Analyze how frequently each ETF is selected in the portfolio."""

        etf_counts = {}
        total_periods = len(self.portfolio_history)

        for _, row in self.portfolio_history.iterrows():
            holdings = row["holdings"]
            if isinstance(holdings, dict):
                for etf in holdings.keys():
                    etf_counts[etf] = etf_counts.get(etf, 0) + 1

        frequency_df = pd.DataFrame(
            [
                {
                    "ETF": etf,
                    "Selection_Count": count,
                    "Selection_Frequency_%": (count / total_periods) * 100,
                }
                for etf, count in etf_counts.items()
            ]
        ).sort_values("Selection_Count", ascending=False)

        return frequency_df

    def analyze_trade_patterns(self) -> Dict[str, Any]:
        """Analyze trading patterns and costs."""

        if self.trade_log.empty:
            return {}

        trade_df = self.trade_log.copy()
        trade_df["date"] = pd.to_datetime(trade_df["date"])

        # Trading frequency analysis
        trades_per_month = trade_df.groupby(trade_df["date"].dt.to_period("M")).size()

        # ETF trading frequency
        etf_trade_counts = trade_df["ticker"].value_counts()

        # Buy vs Sell analysis
        action_counts = trade_df["action"].value_counts()

        # Average trade size
        avg_trade_size = (
            trade_df.groupby("action")
            .agg({"shares": "mean", "cost": "mean", "proceeds": "mean"})
            .fillna(0)
        )

        return {
            "trades_per_month": trades_per_month,
            "etf_trade_frequency": etf_trade_counts,
            "action_distribution": action_counts,
            "average_trade_metrics": avg_trade_size,
            "total_transaction_costs": self.performance_metrics.get(
                "transaction_costs", 0
            ),
        }

    def compare_with_benchmark(
        self,
        benchmark_ticker: str = "SPY",
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Dict:
        """Compare strategy performance with a benchmark ETF."""

        try:
            import yfinance as yf

            # Get benchmark data
            if start_date is None:
                start_date = self.portfolio_history["date"].min()
            if end_date is None:
                end_date = self.portfolio_history["date"].max()

            benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date)[
                "Adj Close"
            ]

            # Align dates
            portfolio_df = self.portfolio_history.copy()
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
            portfolio_df.set_index("date", inplace=True)

            # Resample to match benchmark frequency
            benchmark_resampled = benchmark.resample("M").last()
            portfolio_resampled = portfolio_df["total_value"].resample("M").last()

            # Calculate benchmark returns
            benchmark_initial = benchmark_resampled.iloc[0]
            benchmark_returns = (benchmark_resampled / benchmark_initial - 1) * 100

            # Calculate strategy returns
            strategy_initial = portfolio_resampled.iloc[0]
            strategy_returns = (portfolio_resampled / strategy_initial - 1) * 100

            # Performance comparison
            comparison_df = pd.DataFrame(
                {
                    "Strategy_Returns_%": strategy_returns,
                    "Benchmark_Returns_%": benchmark_returns,
                }
            ).dropna()

            # Calculate metrics
            strategy_final_return = strategy_returns.iloc[-1]
            benchmark_final_return = benchmark_returns.iloc[-1]

            outperformance = strategy_final_return - benchmark_final_return

            return {
                "comparison_data": comparison_df,
                "strategy_return": strategy_final_return,
                "benchmark_return": benchmark_final_return,
                "outperformance": outperformance,
                "benchmark_ticker": benchmark_ticker,
            }

        except Exception as e:
            print(f"Benchmark comparison failed: {e}")
            return {}

    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap of selected ETFs over time."""

        # Extract all unique ETFs that were ever held
        all_etfs = set()
        for _, row in self.portfolio_history.iterrows():
            holdings = row["holdings"]
            if isinstance(holdings, dict):
                all_etfs.update(holdings.keys())

        if len(all_etfs) < 2:
            return go.Figure().add_annotation(
                text="Insufficient data for correlation analysis"
            )

        # For demonstration, create a synthetic correlation matrix
        # In practice, you'd fetch actual price data for these ETFs
        etf_list = list(all_etfs)
        correlation_matrix = np.random.rand(len(etf_list), len(etf_list))
        correlation_matrix = (
            correlation_matrix + correlation_matrix.T
        ) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1)  # Diagonal should be 1

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=etf_list,
                y=etf_list,
                colorscale="RdBu",
                zmid=0,
                text=np.round(correlation_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(
            title="ETF Correlation Heatmap", xaxis_title="ETFs", yaxis_title="ETFs"
        )

        return fig

    def generate_performance_report(self) -> str:
        """Generate a comprehensive text report of strategy performance."""

        metrics = self.performance_metrics

        report = f"""
ETF MOMENTUM STRATEGY - PERFORMANCE REPORT
{'='*50}

OVERALL PERFORMANCE:
- Total Return: {metrics.get('total_return_pct', 0):.2f}%
- Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%
- Volatility: {metrics.get('volatility_pct', 0):.2f}%
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Maximum Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%

TRADING ACTIVITY:
- Total Trades: {metrics.get('total_trades', 0)}
- Transaction Costs: ₹{metrics.get('transaction_costs', 0):,.2f}

PORTFOLIO ANALYSIS:
- Final Portfolio Value: ₹{self.results['final_value']:,.2f}
- Cash Position: ₹{self.portfolio_history.iloc[-1]['cash']:,.2f} ({(self.portfolio_history.iloc[-1]['cash']/self.results['final_value']*100):.1f}%)

RISK METRICS:
- Best Month: {(self.portfolio_history['total_value'].pct_change().max()*100):.2f}%
- Worst Month: {(self.portfolio_history['total_value'].pct_change().min()*100):.2f}%
- Win Rate: {((self.portfolio_history['total_value'].pct_change() > 0).sum() / len(self.portfolio_history)*100):.1f}%

TOP PERFORMING ETFS:
"""

        # Add ETF frequency analysis
        etf_freq = self.analyze_etf_selection_frequency()
        if not etf_freq.empty:
            report += "\nMost Frequently Selected ETFs:\n"
            for _, row in etf_freq.head(5).iterrows():
                report += (
                    f"- {row['ETF']}: {row['Selection_Frequency_%']:.1f}% of periods\n"
                )

        return report

    def export_results(self, filename_prefix: str = "etf_momentum_strategy"):
        """Export results to Excel file with multiple sheets."""

        with pd.ExcelWriter(
            f"{filename_prefix}_results.xlsx", engine="openpyxl"
        ) as writer:
            # Portfolio history
            self.portfolio_history.to_excel(
                writer, sheet_name="Portfolio_History", index=False
            )

            # Trade log
            if not self.trade_log.empty:
                self.trade_log.to_excel(writer, sheet_name="Trade_Log", index=False)

            # Performance metrics
            metrics_df = pd.DataFrame([self.performance_metrics]).T
            metrics_df.columns = ["Value"]
            metrics_df.to_excel(writer, sheet_name="Performance_Metrics")

            # ETF selection frequency
            etf_freq = self.analyze_etf_selection_frequency()
            if not etf_freq.empty:
                etf_freq.to_excel(
                    writer, sheet_name="ETF_Selection_Frequency", index=False
                )

        print(f"Results exported to {filename_prefix}_results.xlsx")


# Risk Management Add-on
class RiskManager:
    """Advanced risk management tools for the ETF momentum strategy."""

    def __init__(self, config):
        self.config = config
        self.risk_metrics = {}

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk (VaR)."""
        if returns.empty:
            return 0
        return np.percentile(returns, confidence_level * 100)

    def calculate_expected_shortfall(
        self, returns: pd.Series, confidence_level: float = 0.05
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if returns.empty:
            return 0
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def position_sizing_kelly(
        self, expected_return: float, return_variance: float
    ) -> float:
        """Calculate optimal position size using Kelly Criterion."""
        if return_variance <= 0:
            return 0
        kelly_fraction = expected_return / return_variance
        return max(0, min(kelly_fraction, self.config.max_position_size))

    def assess_portfolio_risk(self, portfolio_history: pd.DataFrame) -> Dict:
        """Comprehensive portfolio risk assessment."""

        returns = portfolio_history["total_value"].pct_change().dropna()

        risk_metrics = {
            "var_5%": self.calculate_var(returns, 0.05),
            "var_1%": self.calculate_var(returns, 0.01),
            "expected_shortfall_5%": self.calculate_expected_shortfall(returns, 0.05),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "worst_month": returns.min(),
            "best_month": returns.max(),
            "volatility": returns.std(),
            "downside_deviation": returns[returns < 0].std(),
        }

        return risk_metrics


# Example usage with enhanced analysis
def run_complete_analysis():
    """Example of running complete strategy analysis."""

    from etf_momentum_strategy import StrategyConfig, ETFMomentumStrategy
    from datetime import datetime

    # Configure strategy
    config = StrategyConfig(
        portfolio_size=5,
        initial_capital=1000000.0,  # 10 Lakh INR
        use_retracement_filter=True,
        use_moving_average_filter=True,
    )

    # Run backtest
    strategy = ETFMomentumStrategy(config)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    results = strategy.run_backtest(start_date, end_date)

    # Analyze results
    analyzer = StrategyAnalyzer(results)

    # Generate performance dashboard
    dashboard = analyzer.create_performance_dashboard()
    dashboard.show()

    # Generate text report
    report = analyzer.generate_performance_report()
    print(report)

    # Export results
    analyzer.export_results("momentum_strategy_2020_2024")

    # Risk analysis
    risk_manager = RiskManager(config)
    risk_metrics = risk_manager.assess_portfolio_risk(analyzer.portfolio_history)

    print("\nRISK ANALYSIS:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_complete_analysis()
