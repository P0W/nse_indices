[![Auto Run and Push](https://github.com/P0W/nse_indices/actions/workflows/main.yml/badge.svg)](https://github.com/P0W/nse_indices/actions/workflows/main.yml)

# NSE Indices & Cross-Sectional ETF Momentum Engine

Quantitative tooling for **Indian NSE Thematic/Strategy Index ingestion** and **Cross-Sectional ETF Momentum portfolio rotation** (running live as a 5-ETF portfolio on Smallcase).

---

## Repository Architecture

| Component | Path | Description |
| :--- | :--- | :--- |
| **Cross-Sectional ETF Momentum Engine** | [`momentum-etf/`](./momentum-etf/README.md) | Dual-horizon (`252D` / `60D`) momentum rotation engine across 13 liquid NSE ETFs with `50-EMA` and drawdown regime filters, `2×` rank-buffer exit rules, Indian statutory transaction-cost accounting, square-root ($\sqrt{Q/V}$) market-impact slippage modeling, `Backtrader` validation, FastAPI web interface, and Azure Container Apps Bicep deployment. |
| **Automated NSE Index Ingestion & Tracking** | `main.py`, `delivery.py`, `charter.py`, `data/` | Scheduled GitHub Actions pipeline (`.github/workflows/main.yml`) that ingests constituent lists, fact sheets, delivery statistics, and historical performance series across NSE Thematic and Strategy indices (`Nifty India Defence`, `Nifty SME Emerge`, `Nifty200 Momentum 30`). |

---

## 1. Cross-Sectional ETF Momentum Portfolio ([`momentum-etf/`](./momentum-etf/README.md))

Implements a systematic top-$N$ (live 5-ETF equal-weight) monthly rotation strategy across 13 liquid NSE-listed ETFs (`NIFTYBEES`, `SETFNN50`, `GOLDBEES`, `SILVERBEES`, `PSUBNKBEES`, `ITBEES`, `PHARMABEES`, `AUTOBEES`, `INFRAIETF`, `CONSUMBEES`, `DIVOPPBEES`, `SHARIABEES`, `CPSEETF`):

- **Composite Momentum Scoring (`core.py`):**
  $$\text{Score}_i = w_L \cdot R_{i, 252\text{D}} + w_S \cdot R_{i, 60\text{D}}, \quad (w_L, w_S) = (0.60, 0.40)$$
  with optional volatility-adaptive weighting ($w_L = w_S = 0.50$ when annualized cross-sectional market volatility exceeds $20\%$).
- **Regime & Turnover Controls:** `50-day EMA` trend filter ($P_t > \text{EMA}_{50}$), maximum peak-to-trough retracement filter ($\le 50\%$), `30-day` volume filter, and a `2×` rank exit buffer (hold existing constituents until rank drops below $2 \times N$) to minimize monthly rebalancing turnover.
- **Transaction Cost & Market-Impact Modeling (`TradingCostCalculator`, `MarketImpactModel`):**
  - Full Indian delivery cost structure: brokerage (capped), sell-side STT ($0.025\%$), NSE exchange turnover fees ($0.00325\%$), SEBI turnover fees, buy-side stamp duty ($0.003\%$), and $18\%$ GST.
  - Volume-participation market impact: combines linear participation ($c_1 \cdot Q/V$) and square-root market impact ($c_2 \cdot \sqrt{Q/V}$) scaled by realized volatility.
- **Backtest Summary (Jan 2020 – Jul 2025, `Backtrader`):** `19.5%` annualized return (`136.6%` cumulative), `11.7%` annualized volatility, `1.29–1.32` Sharpe ratio, and `-12.1%` to `-16.1%` maximum drawdown net of simulated brokerage costs.

```bash
cd momentum-etf
uv sync
uv run portfolio --amount 500000
uv run rebalance --holdings-file holdings.json --from-date 2024-01-01
uv run backtest --portfolio-sizes 5 --exit-rank-buffers 2 --long-term-periods 252 --short-term-periods 60
```

See [`momentum-etf/README.md`](./momentum-etf/README.md) for full methodology, CLI usage, and `Backtrader` analytics.

---

## 2. Automated NSE Thematic & Strategy Index Tracker

Tracks constituent shifts and multi-year performance across NSE [Thematic Indices](https://niftyindices.com/indices/equity/thematic-indices) and [Strategy Indices](https://niftyindices.com/indices/equity/strategy-indices), including:
- **[Nifty India Defence Index](https://niftyindices.com/indices/equity/thematic-indices/nifty-india-defence)** (rebalanced semi-annually)
- **[Nifty SME Emerge Index](https://niftyindices.com/indices/equity/thematic-indices/nifty-sme-emerge)** (rebalanced quarterly)
- **Nifty200 Momentum 30** (`NSE200_Momentum.ipynb`)

### Historical 1-Year Comparative Snapshot
![image](https://github.com/user-attachments/assets/eed4f21a-b090-45c1-8489-479d0290dc20)

### Setup & Local Execution

```bash
git clone https://github.com/P0W/nse_indices.git
cd nse_indices
python -m venv venv
# Windows: .\venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python main.py --data_dir data
```

### Automated Data Snapshots
Scheduled GitHub Actions runs commit updated constituent CSVs and JSON snapshots to [`data/`](https://github.com/P0W/nse_indices/tree/main/data).

<sub>_Data sources: [NSE Indices](https://niftyindices.com/) and [Finology Ticker](https://ticker.finology.in/)._</sub>
