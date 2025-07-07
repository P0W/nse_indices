# ETF Momentum Strategy

A comprehensive 4-command ETF momentum trading strategy for Indian markets with real-time portfolio management, historical analysis, and backtesting capabilities.

## 📊 Strategy Overview

This system implements a **dual-timeframe momentum strategy** specifically designed for Indian ETF markets:

### Core Strategy Logic

**Momentum Calculation:**
- **Long-term Momentum**: 180 days (~6 months) - 60% weight
- **Short-term Momentum**: 60 days (~3 months) - 40% weight  
- **Combined Score**: `(Long-term Return × 0.6) + (Short-term Return × 0.4)`

**Portfolio Construction:**
- Equal-weighted portfolio of top 5 ETFs by momentum score
- Monthly rebalancing on the 5th of each month
- Exit buffer: Sell ETFs when they fall below rank #10 (2x portfolio size)

**Risk Filters Applied:**
1. **Moving Average Filter**: ETF price must be above 50-day moving average
2. **Retracement Filter**: Maximum 50% drawdown from recent highs allowed
3. **Data Quality Filter**: Minimum 200 days of historical data required

**Rebalancing Discipline:**
- **Scheduled**: 5th of every month (systematic discipline)
- **Emergency**: When holdings drop below rank #10
- **Opportunity**: When new high-momentum ETFs emerge in top 5

## Features

- **Current Portfolio**: Shows exactly what ETFs to buy and how many units
- **Historical Analysis**: Compare portfolios between any two dates with rebalancing actions
- **Rebalancing Analysis**: Compare your current holdings with optimal portfolio  
- **Historical Backtesting**: Test strategy performance across different periods
- **Real-time Data**: Fetches current market prices and momentum scores

## Installation

```bash
# Clone the repository and install dependencies
uv sync
```

## Quick Start - 4 Simple Commands

### 1. 💼 Current Optimal Portfolio

Shows exactly what to buy today with precise allocations:

```bash
# Default: ₹10 Lakh portfolio with 5 ETFs
uv run cli.py portfolio

# Custom amount and size
uv run cli.py portfolio --amount 500000 --size 3
uv run cli.py portfolio --amount 2000000 --size 5
```

**Output Example:**
```
💼 CURRENT OPTIMAL ETF PORTFOLIO
💰 Investment Amount: ₹1,000,000.00
📈 OPTIMAL PORTFOLIO ALLOCATION:
+--------+---------------+---------+---------+--------------+----------+
| Rank   | ETF Name      | Price   | Units   | Investment   | Weight   |
+========+===============+=========+=========+==============+==========+
| 1      | GOLDBEES.NS   | ₹80.53  | 2,483   | ₹199,956     | 20.0%    |
+--------+---------------+---------+---------+--------------+----------+
| 2      | SILVERBEES.NS | ₹103.79 | 1,926   | ₹199,900     | 20.0%    |
+--------+---------------+---------+---------+--------------+----------+
| 3      | PSUBNKBEES.NS | ₹79.50  | 2,515   | ₹199,942     | 20.0%    |
+--------+---------------+---------+---------+--------------+----------+
| 4      | INFRAIETF.NS  | ₹96.63  | 2,069   | ₹199,927     | 20.0%    |
+--------+---------------+---------+---------+--------------+----------+
| 5      | NIFTYBEES.NS  | ₹286.07 | 699     | ₹199,963     | 20.0%    |
+--------+---------------+---------+---------+--------------+----------+
|        | TOTAL         |         |         | ₹999,688     | 100.0%   |
+--------+---------------+---------+---------+--------------+----------+
💰 Cash Remaining: ₹312 (0.0%)
📊 MOMENTUM SCORES:
  Rank  ETF              Score
------  -------------  -------
     1  GOLDBEES.NS     0.1799
     2  SILVERBEES.NS   0.1696
     3  PSUBNKBEES.NS   0.1164
     4  INFRAIETF.NS    0.0804
     5  NIFTYBEES.NS    0.0721
💡 NEXT STEPS:
   1. Buy the above ETFs in specified quantities
   2. Monitor monthly for rebalancing needs
   3. Use 'rebalance' command if you already have a portfolio
```

### 2. 📅 Historical Portfolio Analysis

See how the optimal portfolio changed between any two dates and what rebalancing would be needed:

```bash
# Portfolio changes from June 1 to today
uv run cli.py historical --from-date 2025-06-01

# Portfolio changes between specific dates
uv run cli.py historical --from-date 2024-12-01 --to-date 2025-01-31

# Custom amount and portfolio size
uv run cli.py historical --from-date 2025-01-01 --amount 500000 --size 3
```

**Output Example:**
```
📅 HISTORICAL PORTFOLIO ANALYSIS
📊 From: 2025-01-01 To: 2025-07-07
💰 Investment Amount: ₹500,000.00
================================================================================
📡 Fetching historical data from 2024-03-07 to 2025-07-07...
📈 PORTFOLIO ON 2025-01-01:
------------------------------------------------------------
+--------+---------------+---------+---------+--------------+---------+
|   Rank | ETF Name      | Price   | Units   | Investment   |   Score |
+========+===============+=========+=========+==============+=========+
|      1 | PHARMABEES.NS | ₹23.87  | 6,982   | ₹166,660     |  0.1428 |
+--------+---------------+---------+---------+--------------+---------+
|      2 | GOLDBEES.NS   | ₹64.35  | 2,590   | ₹166,666     |  0.0403 |
+--------+---------------+---------+---------+--------------+---------+
📈 PORTFOLIO ON 2025-07-07:
------------------------------------------------------------
+--------+---------------+---------+---------+--------------+---------+
|   Rank | ETF Name      | Price   | Units   | Investment   |   Score |
+========+===============+=========+=========+==============+=========+
|      1 | GOLDBEES.NS   | ₹80.53  | 2,069   | ₹166,617     |  0.1799 |
+--------+---------------+---------+---------+--------------+---------+
|      2 | SILVERBEES.NS | ₹103.79 | 1,605   | ₹166,583     |  0.1696 |
+--------+---------------+---------+---------+--------------+---------+
|      3 | PSUBNKBEES.NS | ₹79.50  | 2,096   | ₹166,632     |  0.1164 |
+--------+---------------+---------+---------+--------------+---------+
🔄 REBALANCING CHANGES NEEDED:
------------------------------------------------------------
❌ SELL (no longer in top 3):
   • PHARMABEES.NS: 6982 units
✅ BUY (new entries to top 3):
   • SILVERBEES.NS: 1605 units
   • PSUBNKBEES.NS: 2096 units
🔄 ADJUST (remained in portfolio):
   • GOLDBEES.NS: SELL 521 units
📊 PERIOD PERFORMANCE SUMMARY:
----------------------------------------
📅 Period: 187 days
💰 OVERALL PORTFOLIO PERFORMANCE:
   From Portfolio Value: ₹333,327
   To Portfolio Value:   ₹367,972
   Absolute Gain/Loss:   ₹+34,645
   Percentage Return:    +10.39%
   Annualized Return:    +21.29%
📈 ETF PERFORMANCE DURING PERIOD:
🔄 HELD FOR ENTIRE PERIOD:
+-------------+--------------+------------+----------+
| ETF         | From Price   | To Price   | Return   |
+=============+==============+============+==========+
| GOLDBEES.NS | ₹64.35       | ₹80.53     | +25.1%   |
+-------------+--------------+------------+----------+
❌ SOLD DURING PERIOD (price movement after exit):
+---------------+--------------+-----------------+-----------------+
| ETF           | Exit Price   | Current Price   | Missed Return   |
+===============+==============+=================+=================+
| PHARMABEES.NS | ₹23.87       | ₹22.83          | -4.4%           |
+---------------+--------------+-----------------+-----------------+
✅ NEWLY ADDED (not held during period):
+---------------+----------------------+---------------+-----------------------+
| ETF           | Period Start Price   | Entry Price   | Period Performance    |
+===============+======================+===============+=======================+
| SILVERBEES.NS | ₹83.85               | ₹103.79       | +23.8% (not realized) |
+---------------+----------------------+---------------+-----------------------+
| PSUBNKBEES.NS | ₹72.83               | ₹79.50        | +9.2% (not realized)  |
+---------------+----------------------+---------------+-----------------------+
```

### 3. 🔄 Portfolio Rebalancing  

Compare your current holdings with optimal allocation:

```bash
uv run cli.py rebalance
```

*(Currently shows template - you need to input your current holdings in the code)*

### 4. 📊 Historical Backtest

Test strategy performance with different investment amounts:

```bash
# Default backtesting
uv run cli.py backtest

# Custom amounts
uv run cli.py backtest --amounts 1000000 5000000
```

**Output Example:**
```
📊 Running backtest with amounts: [1000000.0]
============================================================
TESTING WITH INITIAL CAPITAL: ₹1,000,000.00
============================================================
=== Backtest Results ===
Initial Capital: ₹1,000,000.00
Final Value: ₹2,770,287.60
Absolute Gain: ₹1,770,287.60
Total Return: 177.03%
Annualized Return: 25.41%
Max Drawdown: -7.01%
Sharpe Ratio: 1.89
Total Trades: 232
Transaction Costs: ₹26,088.80
Transaction Costs %: 0.94%
Win Ratio: 89.21%
================================================================================
INVESTMENT COMPARISON SUMMARY
================================================================================
+-------------------+---------------+----------------+---------------+----------+----------+--------------+---------------+
| Initial Capital   | Final Value   | Total Return   | Ann. Return   |   Sharpe | Max DD   | Tx Costs %   | Win Ratio %   |
+===================+===============+================+===============+==========+==========+==============+===============+
| ₹10L              | ₹27.7L        | 177.0%         | 25.4%         |     1.89 | -7.0%    | 0.94%        | 89.21%        |
+-------------------+---------------+----------------+---------------+----------+----------+--------------+---------------+
```

---


## Command Line Interface

```bash
# Show current optimal portfolio with allocations
uv run cli.py portfolio

# Custom amount and portfolio size
uv run cli.py portfolio --amount 500000 --size 5

# Show rebalancing needed for existing portfolio
uv run cli.py rebalance

# Run historical portfolio analysis
uv run cli.py historical --from-date 2024-12-01

# Run historical portfolio analysis between dates
uv run cli.py historical --from-date 2024-12-01 --to-date 2025-01-31

# Run backtest
uv run cli.py backtest

# Run backtest with specific amounts
uv run cli.py backtest --amounts 1000000 5000000
```

## Strategy Configuration

The strategy uses optimized defaults that work well for real-time usage:

- **Portfolio Size**: 5 ETFs (equal weighted)
- **Long-term Momentum**: 180 days (~6 months) - 60% weight
- **Short-term Momentum**: 60 days (~3 months) - 40% weight
- **Rebalancing**: Monthly on 5th (discipline-based)
- **Exit Buffer**: 2x portfolio size (exit when rank > 10)
- **Risk Filters**: Moving average and retracement filters enabled

## ETF Universe

The strategy analyzes these liquid Indian ETFs:
- **NIFTYBEES.NS** (Nifty 50 ETF)
- **SETFNN50.NS** (Nifty Next 50 ETF)
- **GOLDBEES.NS** (Gold ETF)
- **SILVERBEES.NS** (Silver ETF)
- **CPSEETF.NS** (CPSE ETF)
- **PSUBNKBEES.NS** (PSU Bank ETF)
- **PHARMABEES.NS** (Pharma ETF)
- **ITBEES.NS** (IT ETF)
- **AUTOBEES.NS** (Auto ETF)
- **INFRAIETF.NS** (Infra ETF)
- **SHARIABEES.NS** (Shariah ETF)
- **DIVOPPBEES.NS** (Dividend Opportunities ETF)
- **CONSUMBEES.NS** (Consumer Goods - Nifty India Consumption)

## Current Market Analysis (July 2025)

Based on the latest analysis, the top 5 momentum ETFs are:

1. **GOLDBEES.NS** (Gold ETF) - 17.99% momentum score
2. **PSUBNKBEES.NS** (PSU Banks) - 11.64% momentum score  
3. **NIFTYBEES.NS** (Nifty 50) - 7.21% momentum score
4. **ITBEES.NS** (IT) - 4.18% momentum score
5. **SILVERBEES.NS** (Silver ETF) - 3.85% momentum score

**Key Insights:**
- Gold and Silver ETFs showing strong momentum (safe haven demand)
- Banking sector (PSU) performing well
- Broad market participation through Nifty 50 ETF
- Technology sector showing recovery
- Infrastructure and Consumer sectors providing diversification

## Simple Investment Process

1. **Run portfolio command** to see current optimal allocation
2. **Buy the recommended ETFs** in specified quantities  
3. **Monitor monthly** - check on or before 5th of each month
4. **Use historical command** to analyze changes and rebalancing needs
5. **Maintain discipline** - stick to the systematic schedule

## Troubleshooting

### Common Issues

1. **Import Error:**
   ```bash
   # Make sure dependencies are installed
   uv sync
   ```

2. **Network Error:**
   - Check internet connection
   - Some data providers may have rate limits

3. **No Data Available:**
   - Market may be closed
   - Check if running on a trading day

### Getting Help

```bash
# Show all available commands
uv run cli.py --help

# Show help for specific command
uv run cli.py portfolio --help
uv run cli.py historical --help
uv run cli.py backtest --help
```

## Data Source & Limitations

- Market data from Yahoo Finance (`yfinance` library)
- Real-time pricing (market hours dependent)
- Designed for educational/research purposes
- Past performance doesn't guarantee future results

## Risk Management Features

- **Maximum Position Size**: 14.3% limit per ETF (safety constraint)
- **Data Quality Filter**: Minimum 200 days historical data required
- **Moving Average Filter**: Only ETFs above 50-day MA
- **Retracement Filter**: Excludes ETFs with >50% drawdown
- **Exit Discipline**: Systematic exit when momentum deteriorates

## Strategy Logic Deep Dive

### Why This Strategy Works:

1. **Momentum Persistence**: ETFs with strong recent performance tend to continue outperforming
2. **Dual Timeframe**: Combines long-term trend (6 months) with shorter-term acceleration (3 months)
3. **Risk Filtering**: Avoids ETFs in severe downtrends or with poor data quality
4. **Systematic Rebalancing**: Removes emotional decision-making with fixed schedule
5. **Equal Weighting**: Avoids concentration risk while maintaining simplicity

### When to Expect Performance:

- **Strong Trending Markets**: Strategy excels when clear momentum exists
- **Sector Rotation Periods**: Captures shifts between different market segments
- **Bull Market Phases**: Benefits from sustained upward momentum

### When Strategy May Struggle:

- **Highly Volatile/Choppy Markets**: Frequent false signals
- **Market Reversals**: Momentum can persist past optimal exit points
- **Low Volatility Periods**: Limited differentiation between ETFs

## Disclaimer

This tool is for educational and research purposes only. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.
