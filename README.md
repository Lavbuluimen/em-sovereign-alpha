# EM Sovereign Alpha Project
A systematic research framework for constructing a global sovereign bond portfolio designed to outperform the J.P. Morgan Emerging Markets Bond Index (EMBI).

The project builds a data pipeline, signal engine, and portfolio construction model that allocates across Emerging Market and Developed Market sovereign bonds, including both hard-currency and local-currency exposures.

## Project Objective
The goal of this project is to develop a systematic sovereign bond allocation strategy that:
**Outperforms the J.P. Morgan EMBI index on an annualized basis**
**Maintains approximately 5% tracking error vs benchmark**
**Operates long-only**
**Uses publicly available macro and market data**

The model generates daily signals and produces weekly portfolio adjustments.

The system ultimately provides recommendations for:
**Countries to buy / hold / sell**
**Position sizing**
**Hard vs local currency allocation**
**Duration positioning**
**Spread duration adjustments**

## System Architecture
The project uses a modular research pipeline.
'''
        Market Data
            ↓
        Country Panel
            ↓
        Country Scoring
            ↓
        Portfolio Allocation
            ↓
        Weekly Trade Actions
'''
All stages can be executed with:

python run/update_all.py

## Repository Structure

em-sovereign-alpha
│
├─ src/em
│
│   ├─ country
│   │   ├─ data.py
│   │   ├─ universe.py
│   │   └─ score.py
│   │
│   ├─ portfolio
│   │   └─ allocator.py
│   │
│   ├─ ingestion
│   │   └─ market.py
│   │
│   ├─ models
│   │   └─ baseline.py
│   │
│   └─ utils
│
├─ run
│   ├─ ingest_all.py
│   ├─ build_country_panel.py
│   ├─ build_country_scores.py
│   ├─ build_portfolio.py
│   ├─ weekly_actions.py
│   └─ update_all.py
│
├─ dashboard
│   ├─ app.py
│   └─ coverage.py
│
├─ data
│   ├─ raw
│   └─ processed
│
├─ notebooks
└─ reports

# Model Architecture

The EM Sovereign Alpha framework follows a modular research architecture.

```
                ┌─────────────────────┐
                │  Market Data Layer  │
                │  FX / Yields / Macro│
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   Country Panel     │
                │  FX, Yields, Spread │
                │  Return Proxies     │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   Signal Engine     │
                │  Momentum           │
                │  Valuation          │
                │  Rate Trend         │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   Country Scores    │
                │  Cross-Section Rank │
                │  Normalized Alpha   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Portfolio Allocator │
                │ Long-Only Weights   │
                │ Hard vs Local       │
                │ Duration Tilt       │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Portfolio Outputs  │
                │ Weekly Actions      │
                │ Risk & Allocation   │
                └─────────────────────┘
```


## Data Pipeline
The system builds several intermediate datasets.

### Country Panel
data/processed/country_daily.parquet

Contains:
FX levels
FX returns
sovereign yields
yield changes
spread proxies
local bond return proxies

### Country Scores
data/processed/country_scores_daily.parquet

Contains:
cross-country signals
normalized alpha scores
ranking metrics

### Portfolio Allocation
data/processed/portfolio_daily.parquet

Contains:
country weights
benchmark weights
active weights
hard vs local allocation
duration tilt

### Weekly Portfolio Actions
data/processed/weekly_actions.parquet

Contains:
BUY / HOLD / SELL signals
weight changes
position adjustments

## Signals
The model currently uses a combination of momentum, valuation, and rate signals.

### Momentum
20-day local bond returns
60-day local bond returns
FX momentum

### Rate Signals
sovereign yield changes
yield curve movements

### Valuation
sovereign spread vs US Treasury

Signals are standardized cross-sectionally using z-scores.

## Country Alpha Score
The current composite score is:
Score =
  0.35 × spread valuation
+ 0.35 × bond momentum
+ 0.30 × rate trend

Scores are normalized to:

-1 → weakest
 0 → neutral
+1 → strongest opportunity

These scores drive country ranking and portfolio allocation.

## Portfolio Construction
The allocator converts country scores into long-only portfolio weights.

### Key Features
Benchmark Anchor
Portfolio begins from an equal-weight country benchmark.

Active Overlay
Active weights are derived from country scores.

Constraints
Long only
Active weight limits
Weight normalization

## Hard vs Local Currency Allocation
Local currency exposure is determined by:
country score ranking
FX momentum signals

**Higher conviction countries receive greater local exposure.**

### Duration Positioning
Portfolio duration tilts based on:
US Treasury yield trends.

## Dashboard
A Streamlit dashboard provides visualization of signals and portfolio results.

Launch with:
streamli

The dashboard visualizes:

- portfolio allocations
- signal rankings
- weekly trade recommendations
- data coverage diagnostics

### Dashboard Tabs
**Portfolio**
country weights
hard vs local exposure

**Scores**
country ranking
score history

**Weekly Actions**
buy / hold / sell signals

**Coverage**
missing data diagnostics
dataset completeness
source mapping

# Data Sources

The model relies on publicly available macro and market data.

| Data Type | Source | Example Variables |
|------|------|------|
| FX Rates | Yahoo Finance | BRL/USD, MXN/USD |
| US Treasury Yields | FRED | DGS10 |
| Sovereign Yields | FRED / OECD | 10Y Government Bond |
| Commodities | Yahoo Finance | Oil, Copper, Gold |
| Volatility | Yahoo Finance | VIX |
| Dollar Index | Yahoo Finance | DXY |
| EM Credit Proxy | ETF / Index Proxy | EMB |

These datasets are combined into a **daily panel** used for cross-country comparisons.

---

# Example Portfolio Output

A typical model output produces **country-level allocations**.

## Example Portfolio Snapshot

| Country | Weight | Hard | Local | Score |
|-------|------|------|------|------|
| Brazil | 10.2% | 5.1% | 5.1% | 0.69 |
| Romania | 8.9% | 5.4% | 3.4% | 0.32 |
| South Africa | 8.7% | 5.5% | 3.2% | 0.29 |
| Philippines | 8.6% | 4.7% | 3.9% | 0.24 |
| Mexico | 8.3% | 4.6% | 3.6% | 0.16 |

## Weekly Action Report

| Country | Action | Weight Change |
|------|------|------|
| Indonesia | BUY / ADD | +1.46% |
| Philippines | BUY / ADD | +0.92% |
| Chile | BUY / ADD | +0.68% |
| Poland | SELL / TRIM | −0.39% |
| Brazil | SELL / TRIM | −0.41% |

---

# Research Workflow

Typical research workflow:

## 1. Update Data

```
python run/update_all.py
```

This rebuilds:

- country panel
- signals
- portfolio weights
- weekly actions

---

## Current Limitations
The main limitation currently is data quality.
Some sovereign yield series have incomplete coverage when using free data sources. Improving the data layer is the most important next step.

Iterate Research

New ideas are tested through:

```
notebooks/
```

and then integrated into the production pipeline.

---

# Future Research Roadmap

The long-term research roadmap includes:

## Data Improvements

- sovereign CDS spreads
- EMBI country spreads
- local bond index returns

## Risk Model

- covariance estimation
- volatility estimation
- factor exposures
- tracking error calculation

## Portfolio Optimization

Move from heuristic allocation to optimization:

```
maximize alpha

subject to

tracking error ≤ 5%
weights ≥ 0
country caps
liquidity constraints
```

## Macro Overlay

Add regime signals:

- global liquidity
- dollar cycle
- commodity cycle
- volatility regime

---

# Why This Project Matters

Emerging market sovereign returns are driven by:

- global liquidity
- commodity cycles
- currency movements
- sovereign credit spreads

A systematic framework allows these signals to be combined into **consistent portfolio decisions** rather than discretionary views.

---

# Optional Enhancements

Future versions of the system could support:

- automated daily updates
- portfolio risk attribution
- scenario analysis
- macro regime classification
- live portfolio monitoring

---

# Next Steps

Near-term priorities:

1. Improve sovereign yield data coverage  
2. Add tracking-error based portfolio optimization  
3. Expand the dashboard with risk analytics

## Long-Term Vision
The final framework could support:

- Systematic EM sovereign strategy
- Macro sovereign allocation fund
- ETF or mutual fund implementation
- Institutional portfolio construction research.

## Running the Project
Install dependencies
pip install -r requirements.txt

## Run the full pipeline
python run/update_all.py
## Launch the dashboard
streamlit run dashboard/app.py

## License
This repository is intended for research and educational purposes.