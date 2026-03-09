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

Market Data
     ↓
Country Panel
     ↓
Country Scoring
     ↓
Portfolio Allocation
     ↓
Weekly Trade Actions

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
streamlit run dashboard/app.py

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

## Current Limitations
The main limitation currently is data quality.
Some sovereign yield series have incomplete coverage when using free data sources.
Improving the data layer is the most important next step.

## Remaining Development Tasks
### Improve Data Sources
Replace incomplete yield series with more reliable sources.

Possible improvements include:
FRED sovereign yield series
TradingEconomics API
sovereign CDS spreads
EMBI spread proxies
local bond index returns

### Add Risk Model
Introduce formal portfolio risk analytics:

covariance estimation
volatility estimation
factor exposures
tracking error calculation

### Implement Portfolio Optimizer
Replace heuristic allocation rules with optimization.

Target problem:
maximize alpha

subject to:
tracking error ≤ 5%
weights ≥ 0
country caps
liquidity constraints

### Add Macro Regime Overlay
Incorporate global macro signals such as:

global liquidity
US dollar regime
commodity cycle
risk sentiment

These signals should scale overall EM exposure.

### Expand Backtesting Framework
Develop a full historical backtest including:

portfolio returns
benchmark comparison
drawdowns
information ratio
turnover
transaction costs

## Long-Term Vision
The final framework could support:

systematic EM sovereign strategy
macro sovereign allocation fund
ETF or mutual fund implementation
institutional portfolio construction research.

## Running the Project
Install dependencies
pip install -r requirements.txt

## Run the full pipeline
python run/update_all.py
## Launch the dashboard
streamlit run dashboard/app.py

## License
This repository is intended for research and educational purposes.