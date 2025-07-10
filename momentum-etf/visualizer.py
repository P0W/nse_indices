# Performance Visualization Module for ETF Momentum Strategy.
# This module provides functions to generate comprehensive performance charts
# and dashboards for analyzing backtest results.
# Author: Prashant Srivastava

from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from core import StrategyConfig


def create_performance_charts(
    results: Dict, config: StrategyConfig, start_date: datetime, end_date: datetime
):
    """Create and save performance visualization charts."""
    portfolio_history = results["portfolio_history"]
    trade_log = results["trade_log"]

    if not portfolio_history:
        print("❌ No portfolio history data available for charting")
        return

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Sort portfolio history by date
    portfolio_history = sorted(portfolio_history, key=lambda x: x["date"])

    # Extract data for plotting
    dates = [p["date"] for p in portfolio_history]
    values = [p["total_value"] for p in portfolio_history]
    cash = [p["cash"] for p in portfolio_history]

    # Convert to pandas for easier manipulation
    df = pd.DataFrame({"date": dates, "total_value": values, "cash": cash})
    df["invested"] = df["total_value"] - df["cash"]
    df["returns"] = df["total_value"].pct_change() * 100
    df["cumulative_returns"] = ((df["total_value"] / config.initial_capital) - 1) * 100

    # Calculate rolling drawdown
    df["peak"] = df["total_value"].expanding().max()
    df["drawdown"] = ((df["total_value"] / df["peak"]) - 1) * 100

    # Create main performance dashboard
    create_main_dashboard(df, config, results, start_date, end_date)

    # Create detailed analysis charts
    create_detailed_analysis(df, trade_log, config, results, start_date, end_date)


def create_main_dashboard(df, config, results, start_date, end_date):
    """Create the main performance dashboard with 6 key charts."""
    # Create a figure with better spacing
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")

    # Add main title with proper spacing
    fig.suptitle(
        f"ETF Momentum Strategy - Performance Dashboard\n"
        f'Period: {start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")} | '
        f"Strategy: Backtrader Implementation\n"
        f'🎯 Final Value: ₹{results["final_value"]:,.0f} | '
        f'📈 Total Return: {results["total_return"]:.1f}% | '
        f'🔥 Annualized: {results["performance_metrics"].get("annualized_return_pct", 0):.1f}%',
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create subplots with proper spacing - increased top margin to prevent overlap
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.83, bottom=0.08)

    # Subplot 1: Portfolio Value Over Time (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(
        df["date"],
        df["total_value"],
        linewidth=3,
        color="#2E8B57",
        label="Portfolio Value",
        alpha=0.9,
    )
    ax1.axhline(
        y=config.initial_capital,
        color="#FF4444",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Initial Capital",
    )

    # Add milestone markers
    max_value = df["total_value"].max()
    max_date = df.loc[df["total_value"].idxmax(), "date"]
    ax1.scatter(
        max_date,
        max_value,
        color="gold",
        s=100,
        zorder=5,
        label=f"Peak: ₹{max_value:,.0f}",
    )

    ax1.set_title("📈 Portfolio Growth Journey", fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle="-")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x/100000:.1f}L"))

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Key Performance Metrics (right side)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    metrics_text = [
        f"🎯 Total Return: {results["total_return"]:.1f}%",
        f"📊 Annualized Return: {results["performance_metrics"].get("annualized_return_pct", 0):.1f}%",
        f"📈 Sharpe Ratio: {results["performance_metrics"].get("sharpe_ratio", 0):.2f}",
        f"📉 Max Drawdown: {results["performance_metrics"].get("max_drawdown_pct", 0):.1f}%",
        f"🎲 Volatility: {results["performance_metrics"].get("volatility_pct", 0):.1f}%",
        f"🔄 Total Trades: {results["performance_metrics"].get("total_trades", 0)}",
        f"🏆 Win Ratio: {results["performance_metrics"].get("win_ratio_pct", 0):.1f}%",
        f"💰 Transaction Costs: ₹{results["performance_metrics"].get("transaction_costs", 0):,.0f}",
    ]

    # Create a nice metrics box
    metrics_box = "\n".join(metrics_text)
    ax2.text(
        0.1,
        0.95,
        "📊 Key Metrics",
        fontsize=16,
        fontweight="bold",
        transform=ax2.transAxes,
        verticalalignment="top",
    )
    ax2.text(
        0.1,
        0.85,
        metrics_box,
        fontsize=12,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
    )

    # 3. Cumulative Returns
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(
        df["date"],
        df["cumulative_returns"],
        alpha=0.6,
        color="#4CAF50",
        label="Cumulative Returns",
    )
    ax3.plot(df["date"], df["cumulative_returns"], linewidth=2, color="#2E8B57")
    ax3.set_title("📈 Cumulative Returns", fontsize=14, fontweight="bold", pad=15)
    ax3.set_ylabel("Returns (%)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(
        df["date"], df["drawdown"], alpha=0.6, color="#FF6B6B", label="Drawdown"
    )
    ax4.plot(df["date"], df["drawdown"], linewidth=2, color="#D32F2F")
    ax4.set_title("📉 Drawdown Analysis", fontsize=14, fontweight="bold", pad=15)
    ax4.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Rolling Volatility (30-day)
    ax5 = fig.add_subplot(gs[1, 2])
    rolling_vol = df["returns"].rolling(window=30).std() * np.sqrt(252)  # Annualized
    ax5.plot(df["date"], rolling_vol, linewidth=2, color="#FF9800", alpha=0.8)
    ax5.set_title("📊 Rolling Volatility (30D)", fontsize=14, fontweight="bold", pad=15)
    ax5.set_ylabel("Volatility (%)", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Monthly Returns Heatmap
    ax6 = fig.add_subplot(gs[2, :2])
    # Create monthly returns
    df_monthly = df.set_index("date").resample("M")["returns"].sum()
    df_monthly.index = pd.to_datetime(df_monthly.index)

    # Create pivot table for heatmap
    monthly_pivot = df_monthly.to_frame()
    monthly_pivot["year"] = monthly_pivot.index.year
    monthly_pivot["month"] = monthly_pivot.index.month
    heatmap_data = monthly_pivot.pivot_table(
        values="returns", index="year", columns="month", aggfunc="first"
    )

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Monthly Returns (%)"},
        ax=ax6,
    )
    ax6.set_title("🔥 Monthly Returns Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax6.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Year", fontsize=11, fontweight="bold")

    # 7. Performance Summary (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    # Performance summary text
    metrics = results["performance_metrics"]
    summary_text = [
        f"🏁 Strategy Performance Summary",
        f"",
        f"Initial Capital: ₹{config.initial_capital:,.0f}",
        f"Final Value: ₹{results["final_value"]:,.0f}",
        f"Profit/Loss: ₹{results["final_value"] - config.initial_capital:,.0f}",
        f"",
        f"📊 Risk & Trade Metrics:",
        f"• Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}% (₹{metrics.get('max_drawdown_amount', 0):,.0f})",
        f"  - Trough Value: ₹{metrics.get('trough_value_at_mdd', 0):,.0f} on {metrics.get('mdd_end_date', datetime.now()).strftime('%Y-%m-%d')}",
        f"• Recovery Days: {metrics.get('days_to_recovery', 'N/A')}",
        f"  - Recovery Date: {metrics.get('recovery_date', 'N/A').strftime('%Y-%m-%d') if isinstance(metrics.get('recovery_date'), datetime) else metrics.get('recovery_date', 'N/A')}",
        f"• Best Month: {df_monthly.max():.1f}%",
        f"• Worst Month: {df_monthly.min():.1f}%",
        f"• Positive Months: {(df_monthly > 0).sum()}/{len(df_monthly)}",
        f"• Max Win Streak: {metrics.get('max_win_streak', 0)} trades",
        f"• Max Loss Streak: {metrics.get('max_loss_streak', 0)} trades",
        f"",
        f"⚙️ Strategy Config:",
        f"• Portfolio Size: {config.portfolio_size}",
        f"• Rebalance Day: {config.rebalance_day_of_month}",
        f"• Long Period: {config.long_term_period_days}d",
        f"• Short Period: {config.short_term_period_days}d",
    ]

    summary_box = "\n".join(summary_text)
    ax7.text(
        0.05,
        0.95,
        summary_box,
        fontsize=11,
        transform=ax7.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
    )

    # Save the main dashboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtrader_dashboard_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"📊 Main dashboard saved as: {filename}")


def create_detailed_analysis(df, trade_log, config, results, start_date, end_date):
    """Create detailed analysis charts with additional insights."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")

    # Add title with proper spacing
    fig.suptitle(
        f"ETF Momentum Strategy - Detailed Analysis\n"
        f'Period: {start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")} | '
        f"Advanced Risk & Performance Analytics",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    # Create subplots with proper spacing - increased top margin to prevent overlap
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, top=0.84, bottom=0.08)

    # 1. Risk-Return Scatter (Rolling analysis)
    ax1 = fig.add_subplot(gs[0, 0])

    # Adjust rolling window size based on available data
    data_length = len(df)
    window_size = min(60, max(20, data_length // 3))  # Use smaller window if needed

    rolling_returns = (
        df["returns"].rolling(window=window_size, min_periods=10).mean() * 252
    )  # Annualized
    rolling_vol = df["returns"].rolling(
        window=window_size, min_periods=10
    ).std() * np.sqrt(
        252
    )  # Annualized

    # Remove NaN values for plotting
    valid_data = pd.DataFrame(
        {"returns": rolling_returns, "volatility": rolling_vol}
    ).dropna()

    if len(valid_data) > 0:
        # Create scatter plot with color based on time
        scatter = ax1.scatter(
            valid_data["volatility"],
            valid_data["returns"],
            c=range(len(valid_data)),
            cmap="viridis",
            alpha=0.7,
            s=30,
        )

        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Time Period", fontsize=10)

        # Set appropriate axis limits
        ax1.set_xlim(
            valid_data["volatility"].min() * 0.95, valid_data["volatility"].max() * 1.05
        )
        ax1.set_ylim(
            valid_data["returns"].min() * 0.95, valid_data["returns"].max() * 1.05
        )
    else:
        # If no valid data, show a message
        ax1.text(
            0.5,
            0.5,
            f"Insufficient data for\n{window_size}-day rolling analysis\n({data_length} data points available)",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
            style="italic",
        )

    ax1.set_xlabel("Volatility (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Returns (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"🎯 Risk-Return Profile ({window_size}D Rolling)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(True, alpha=0.3)

    # 2. Rolling Sharpe Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    rolling_sharpe = rolling_returns / rolling_vol

    # Remove NaN values for plotting
    valid_sharpe_data = pd.DataFrame(
        {"date": df["date"], "sharpe": rolling_sharpe}
    ).dropna()

    if len(valid_sharpe_data) > 0:
        ax2.plot(
            valid_sharpe_data["date"],
            valid_sharpe_data["sharpe"],
            linewidth=2,
            color="#9C27B0",
            alpha=0.8,
        )
        ax2.axhline(
            y=1.0, color="#FF5722", linestyle="--", alpha=0.8, label="Good (1.0)"
        )
        ax2.axhline(
            y=2.0, color="#4CAF50", linestyle="--", alpha=0.8, label="Excellent (2.0)"
        )
        ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(
            0.5,
            0.5,
            f"Insufficient data for\n{window_size}-day rolling analysis\n({data_length} data points available)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
            style="italic",
        )

    ax2.set_title(
        f"📈 Rolling Sharpe Ratio ({window_size}D)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Trade Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    if trade_log:
        # Analyze trade sizes and timing
        trade_df = pd.DataFrame(trade_log)
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["value"] = trade_df["shares"] * trade_df["price"]

        # Group by month
        monthly_trades = trade_df.groupby(trade_df["date"].dt.to_period("M"))[
            "value"
        ].sum()

        # Create bar plot
        bars = ax3.bar(
            range(len(monthly_trades)),
            monthly_trades.values,
            color="#607D8B",
            alpha=0.7,
        )
        ax3.set_title("💱 Monthly Trade Volume", fontsize=14, fontweight="bold", pad=15)
        ax3.set_ylabel("Trade Value (₹)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Month", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        # Format y-axis
        ax3.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"₹{x/100000:.1f}L")
        )

        # Set x-axis labels
        ax3.set_xticks(range(0, len(monthly_trades), max(1, len(monthly_trades) // 6)))
        ax3.set_xticklabels(
            [
                str(monthly_trades.index[i])
                for i in range(0, len(monthly_trades), max(1, len(monthly_trades) // 6))
            ],
            rotation=45,
        )
    else:
        ax3.text(
            0.5,
            0.5,
            "No trade data available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("💱 Monthly Trade Volume", fontsize=14, fontweight="bold", pad=15)

    # 4. Returns Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    returns_clean = df["returns"].dropna()
    ax4.hist(returns_clean, bins=30, alpha=0.7, color="#3F51B5", edgecolor="black")
    ax4.axvline(
        returns_clean.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {returns_clean.mean():.2f}%",
    )
    ax4.axvline(
        returns_clean.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {returns_clean.median():.2f}%",
    )
    ax4.set_title(
        "📊 Daily Returns Distribution", fontsize=14, fontweight="bold", pad=15
    )
    ax4.set_xlabel("Daily Returns (%)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)

    # 5. Underwater Curve (Drawdown from Peak)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(
        df["date"],
        df["drawdown"],
        alpha=0.6,
        color="#FF5722",
        label="Drawdown from Peak",
    )
    ax5.plot(df["date"], df["drawdown"], linewidth=2, color="#D32F2F")
    ax5.set_title("🌊 Underwater Curve", fontsize=14, fontweight="bold", pad=15)
    ax5.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Portfolio Allocation Over Time
    ax6 = fig.add_subplot(gs[1, 2])
    # Calculate allocation percentages
    allocation_pct = (df["invested"] / df["total_value"]) * 100
    cash_pct = (df["cash"] / df["total_value"]) * 100

    ax6.fill_between(
        df["date"], 0, allocation_pct, alpha=0.6, color="#4CAF50", label="Invested"
    )
    ax6.fill_between(
        df["date"], allocation_pct, 100, alpha=0.6, color="#FFC107", label="Cash"
    )
    ax6.set_title("🥧 Portfolio Allocation", fontsize=14, fontweight="bold", pad=15)
    ax6.set_ylabel("Allocation (%)", fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 100)
    ax6.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    # Save the detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtrader_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"📈 Detailed analysis saved as: {filename}")
