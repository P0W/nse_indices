# Cross-Sectional ETF Momentum Strategy (NSE)

Systematic dual-timeframe cross-sectional momentum engine for liquid Indian NSE-listed ETFs, supporting live 5-ETF portfolio rotation (deployed via Smallcase/broker baskets), stateful holdings rebalancing, and multi-parameter `Backtrader` simulation with Indian statutory transaction costs and square-root market-impact slippage modeling.

---

## Quantitative Methodology

### 1. Dual-Horizon Momentum Score (`core.py`)
For each ETF $i$ in the liquid universe with at least `min_data_points = 201` trading bars of history, the composite momentum score combines long-horizon trend persistence with medium-term acceleration:

$$\text{Score}_i = w_L \cdot R_{i, L} + w_S \cdot R_{i, S}$$

- **Long-term lookback ($L$):** `200` trading days (~10 months)
- **Short-term lookback ($S$):** `90` trading days (~4.5 months)
- **Default weights:** $(w_L, w_S) = (0.70, 0.30)$
- **Volatility-adaptive regime (`adaptive_weights=True`):** Shifts weights to $(0.50, 0.50)$ when 30-day annualized `NIFTYBEES.NS` volatility exceeds `20%`.

### 2. Regime Filters & Slot-Capped Allocation
Before ranking, candidate ETFs are screened through three filters:
- **Trend Filter (`20-EMA`):** Requires current adjusted close above its 20-day exponential moving average ($P_t > \text{EMA}_{20}$).
- **Retracement Filter:** Excludes ETFs whose drawdown from their rolling 200-bar close high exceeds `25%`.
- **Liquidity Filter:** Requires 30-day average completed daily volume $\ge 200,000$ shares (`JUNIORBEES.NS` acts as signal/liquidity proxy for `SBINEXT50.NS` following the NSE rename from `SETFNN50.NS`).
- **Rolling Top-$N$ Selection & Per-Slot Cap:** Rebalances on a rolling `30`-day cadence into the top $N$ strictly positive-momentum ETFs, capping each slot at `1 / N` (`20%` max per ETF for $N = 5$) and holding any unfilled slots in cash / `LIQUIDBEES`.

### 3. Statutory Transaction Costs & Market Impact (`TradingCostCalculator`, `MarketImpactModel`)
- **Indian NSE ETF Delivery Cost Structure:** Models discount brokerage (`₹0` on live delivery or `0.03%` capped at ₹20/order), equity-ETF sell-side STT (`0.001%`, `0%` on buys and `0%` on Gold/Silver/Liquid/International ETFs), NSE exchange turnover charges (`0.00297%`), SEBI regulatory fees (`₹10/crore`), buy-side stamp duty (`0.015%`), `18%` GST on brokerage + exchange + SEBI fees, and `₹15.34` CDSL DP charge per sell scrip.
- **Market-Impact Slippage Model:** Models execution slippage as a function of order volume participation ($Q / V_{\text{daily}}$) and realized volatility ($\sigma$):
  $$\text{Slippage} = \min\!\left(0.03,\; \left(c_{\text{lin}} \frac{Q}{V} + c_{\text{sqrt}} \sqrt{\frac{Q}{V}} + 0.02\sigma\right) \cdot \eta_{\text{dir}}\right)$$

---

## Backtest Evaluation (Jan 2020 – Jul 2025)

Evaluated over `2020-01-01` to `2025-07-09` (5.5 years) across the 13-ETF NSE universe using `Backtrader` (`backtrader_momentum_strategy.py`) and the native portfolio simulator (`etf_momentum_strategy.py`):

| Metric | Backtrader (`L=252, S=60, N=5`) | Parameter Sweep (`L=200, S=60, N=5`) |
| :--- | :--- | :--- |
| **Initial Capital** | ₹1,00,000 | ₹1,00,000 |
| **Final Portfolio Value** | ₹2,36,568 | ₹2,42,126 |
| **Cumulative Return** | `136.6%` | `142.1%` |
| **Annualized Return (CAGR)** | `19.5%` | `19.3%` |
| **Annualized Volatility** | `11.7%` | `11.7%` |
| **Sharpe Ratio** | `1.29` | `1.32` |
| **Maximum Drawdown** | `-16.1%` | `-12.1%` |
| **Total Rebalance Trades** | `265` | `265` |
| **Simulated Transaction Costs** | ₹2,067 | ₹2,005 |

### Strategy Performance Analysis

![Backtrader Analysis](backtrader_analysis_20200101_20250709.png)

*Portfolio equity curve, return distribution, and rolling performance metrics (`2020-01-01` to `2025-07-09`).*

### Portfolio Risk & Drawdown Dashboard

![Backtrader Dashboard](backtrader_dashboard_20200101_20250709.png)

*Cumulative returns, underwater drawdown profile, trade distribution, and risk diagnostics.*

---

## Installation & CLI Usage

```bash
cd momentum-etf
uv sync
```

### 1. Generate Target Top-5 Momentum Portfolio

```bash
uv run portfolio --amount 1000000
```

**Sample Output:**
```text
TODAY'S OPTIMAL ETF PORTFOLIO
Investment Amount: ₹1,000,000.00

TOP 5 MOMENTUM ETFS:
┌────────────────┬──────────┬─────────┬──────────────┬─────────────┐
│ 1. GOLDBEES    │ ₹80.53   │ 2,483   │ ₹199,956     │ 17.99%      │
│ 2. SILVERBEES  │ ₹103.79  │ 1,926   │ ₹199,900     │ 16.96%      │
│ 3. PSUBNKBEES  │ ₹79.50   │ 2,515   │ ₹199,942     │ 11.64%      │
│ 4. INFRAIETF   │ ₹96.63   │ 2,069   │ ₹199,927     │ 8.04%       │
│ 5. NIFTYBEES   │ ₹286.07  │ 699     │ ₹199,963     │ 7.21%       │
└────────────────┴──────────┴─────────┴──────────────┴─────────────┘
Cash Remaining: ₹312
```

### 2. Rebalance Existing Holdings (`JSON` or `CSV`)

```bash
uv run rebalance --holdings-file holdings.json --from-date 2024-01-01
```

**Holdings File Format (`holdings.json`):**
```json
[
  {"symbol": "NIFTYBEES.NS", "units": 350, "price": 120.50},
  {"symbol": "GOLDBEES.NS", "units": 200, "price": -1}
]
```
*(Setting `"price": -1` instructs the data provider to look up the historical adjusted close on `--from-date`.)*

### 3. Historical Point-in-Time Portfolio Reconstruction

```bash
uv run historical --from-date 2024-01-01 --to-date 2024-12-31 --amount 500000 --size 5
```

### 4. Multi-Parameter Backtest Sweep

```bash
uv run backtest --portfolio-sizes 5 --exit-rank-buffers 2 --long-term-periods 200 --short-term-periods 60 --initial-capitals 100000 --start-date 2020-01-01 --end-date 2025-07-09
```

---

## Liquid NSE ETF Universe (13 Instruments)

| Category | Ticker (`.NS`) | Underlying Benchmark / Exposure |
| :--- | :--- | :--- |
| **Broad Large-Cap** | `NIFTYBEES` | Nifty 50 Index |
| **Large/Mid Growth** | `SBINEXT50` | Nifty Next 50 Index (formerly `SETFNN50`, scored via `JUNIORBEES` proxy) |
| **PSU Banking** | `PSUBNKBEES` | Nifty PSU Bank Index |
| **Precious Metals (Gold)** | `GOLDBEES` | Domestic physical gold bullion |
| **Precious Metals (Silver)** | `SILVERBEES` | Domestic physical silver bullion |
| **Healthcare / Pharma** | `PHARMABEES` | Nifty Pharma Index |
| **Information Technology** | `ITBEES` | Nifty IT Index |
| **Automotive** | `AUTOBEES` | Nifty Auto Index |
| **Infrastructure** | `INFRAIETF` | Nifty Infrastructure Index |
| **Consumption** | `CONSUMBEES` | Nifty India Consumption Index |
| **Dividend Factor** | `DIVOPPBEES` | Nifty Dividend Opportunities 50 Index |
| **Rules-Based Screen** | `SHARIABEES` | Nifty50 Shariah Index |
| **Public Sector Enterprises** | `CPSEETF` | Nifty CPSE Index |

---

## Configuration Reference (`StrategyConfig` in `core.py`)

```python
portfolio_size = 5                  # Number of ETFs held (max 1/N per slot)
exit_rank_buffer_multiplier = 1.0   # Strict top-N rotation matching rolling strategy
rebalance_frequency = "rolling"     # Rolling N-day rebalance cadence
rebalance_period_days = 30          # Days between rolling rebalances
long_term_period_days = 200         # Long-horizon momentum window (trading bars)
short_term_period_days = 90         # Short-horizon momentum window (trading bars)
momentum_weights = (0.7, 0.3)       # (Long, Short) weights; (0.5, 0.5) if Nifty vol > 20%
moving_average_period = 20          # EMA regime filter period
max_retracement_percentage = 0.25   # Max 200-bar peak-to-close drawdown filter
use_volume_filter = True            # Require 30d avg completed volume >= 200,000
min_avg_volume = 200000
```

---

## Disclaimer

For quantitative research and educational purposes only. Historical backtest performance does not guarantee future live trading results.
