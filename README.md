# EM Sovereign Alpha

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-research-green)
![License](https://img.shields.io/badge/license-research-lightgrey)
![Dashboard](https://img.shields.io/badge/dashboard-streamlit-red)

A systematic research framework for constructing a **global sovereign bond portfolio** designed to outperform the **J.P. Morgan Emerging Markets Bond Index (EMBI)**.

The project builds a **data pipeline, signal engine, and portfolio construction model** that allocates across **Emerging Market and Developed Market sovereign bonds**, including both **hard-currency and local-currencies**.

---

# Table of Contents

- [Strategy Overview](#strategy-overview)
- [Investment Philosophy](#investment-philosophy)
- [Why Systematic EM Sovereign Investing Can Work](#why-systematic-em-sovereign-investing-can-work)
- [Strategy Characteristics](#strategy-characteristics)
- [Model Architecture](#model-architecture)
  - [Scoring Signals](#scoring-signals)
  - [Credit Quality Guard](#credit-quality-guard)
  - [Signal Confidence](#signal-confidence)
  - [Risk Regime Overlay](#risk-regime-overlay)
  - [Portfolio Construction Rules](#portfolio-construction-rules)
- [Data Sources](#data-sources)
- [Example Portfolio Output](#example-portfolio-output)
- [Research Workflow](#research-workflow)
- [Current Research Roadmap](#current-research-roadmap)
- [Improvements](#improvements)
- [Why This Project ICYMI](#why-this-project-matters-icymi)
- [Running the Project](#running-the-project)
- [License](#license)

---

# Strategy Overview

**EM Sovereign Alpha** is a systematic portfolio construction framework designed to allocate across **emerging market sovereign bonds and currencies** in order to generate **consistent excess returns relative to the J.P. Morgan EMBI index**.

The strategy combines **macro signals, cross-country valuation metrics, and momentum indicators** to dynamically allocate capital across sovereign issuers, currencies, and duration exposures.

Unlike traditional discretionary approaches to emerging market debt, the framework applies a **rules-based investment process** that systematically identifies relative value opportunities across countries.

---

# Investment Philosophy

Emerging market sovereign returns are driven by several key forces:

- Global liquidity conditions
- Commodity cycles
- Currency dynamics
- Sovereign credit spreads
- Domestic rate environments

These drivers often lead to **mispriced assets** in sovereign bond markets.

The strategy seeks to take advantage of these mispricing opportunites through a **disciplined cross-sectional ranking process** that allocates capital toward countries offering the most attractive risk-adjusted opportunities.

---

# Why Systematic EM Sovereign Investing Can Work

Emerging market sovereign bond markets have several structural characteristics that make them well suited to **systematic investment strategies**.

Unlike DM sovereign bond markets, EM markets are often influenced by **macro cycles, capital flows, and commodity dynamics**, which can create on-going cross-country mispricing.

These characteristics provide opportunities for a systematic framework to identify relative value more accurately across countries.

---

## Structural Limitations

### Information Dispersion

Emerging markets often have **less consistent data availability and analyst coverage** than developed markets. As a result, macroeconomic developments and policy shifts may not be immediately reflected in sovereign bond prices. Systematic models can incorporate a wide range of macro signals to capture these dynamics sooner rather than later.

---

### Heterogeneous Macro Cycles

Emerging market economies frequently operate in **different stages of the economic cycle**.

For example:

- commodity exporters may benefit from rising commodity prices
- commodity importers may suffer from inflation shocks
- countries with different monetary policy regimes react differently to global rate changes

This dispersion creates **cross-country opportunities** for active allocation.

---

### Currency and Local Rate Dynamics

Returns in EM sovereign debt are driven by multiple components:

- sovereign credit spreads
- local interest rate movements
- currency fluctuations

The interaction between these drivers creates complex return patterns that may be challenging to evaluate consistently. Systematic models are well suited to **integrating multiple return drivers simultaneously**.

---

### Global Liquidity Sensitivity

Emerging market assets are highly sensitive to global liquidity conditions, including:

- US interest rate cycles
- dollar strength
- global risk sentiment

Systematic models can monitor these macro variables continuously and adjust portfolio exposures accordingly.

---

## Advantages of a Systematic Framework

### Consistency

Investment decisions follow **defined rules and signals**, reducing behavioral biases that can affect discretionary portfolio management.

### Breadth

Systematic models can evaluate **multiple countries and signals simultaneously**, allowing for a more comprehensive view of global sovereign opportunities.

### Discipline

Portfolio construction rules enforce:

- risk limits
- diversification
- consistent exposure management

### Adaptability

Because signals are updated regularly, the strategy can **respond dynamically to changes in macroeconomic conditions**.

---

# Strategy Characteristics

| Feature | Description |
|------|------|
| Asset Class | Emerging Market Sovereign Bonds |
| Instruments | Hard Currency (USD) and Local Currency Bonds |
| Strategy Type | Systematic Global Macro |
| Portfolio Structure | Long-only, benchmark-aware |
| Benchmark | J.P. Morgan EMBI Global Diversified (EMBI-GD) |
| Universe | 11 countries: Brazil, Mexico, Colombia, Chile, South Africa, Poland, Hungary, Romania, Indonesia, Malaysia, Philippines |
| Target Tracking Error | ~4% (L2-norm active weight constraint) |
| Rebalancing | Daily signals, weekly to monthly adjustments (50bp threshold) |
| Backtest Start | January 2015 |
| Data Sources | FRED, Stooq, Yahoo Finance, World Bank (all public, free) |

---

# Model Architecture

The EM Sovereign Alpha framework follows a modular research architecture. The composite score combines **nine signals** across two layers: seven cross-sectional signals z-scored relative to the 11-country universe, and two global adjustments applied after ranking.

## Scoring Signals

| Signal | Weight | Lookback | Notes |
|--------|--------|----------|-------|
| Spread value (blended TS + CS, quality-guarded) | 15% | 2yr rolling | 50% cross-sectional + 50% per-country time-series z-score; capped for distressed or low-quality credits |
| Spread momentum (20/60/120d blend) | 15% | 20/60/120d | Weighted blend: 25% short / 50% medium / 25% long horizon |
| FX carry | 15% | Spot monthly | Local short rate − US short rate; zero for 4 non-OECD countries (no FRED data) |
| Real yield | 10% | Spot monthly | 10Y nominal yield − trailing CPI; zero for 4 non-OECD countries |
| Local return (60d) | 10% | 60 days | Cumulative local bond + FX return in USD |
| FX momentum (60d) | 5% | 60 days | Cumulative FX return vs USD |
| Commodity terms-of-trade | 10% | 60d Brent return | Country sensitivity × Brent 60d return z-score |
| **US real rate global tilt** | **15%** | **60d change** | Post-ranking additive tilt — rising US real rates reduce all EM scores uniformly |
| **Risk regime (VIX / DXY / EM OAS)** | **5%** | **Spot thresholds** | Green: full active risk; Amber: halved; Red: near-zero |

### Credit Quality Guard

The spread value signal is **capped at zero** for any country where:
- sovereign spread > 700bp, **OR**
- credit quality composite score < 0.30

The credit quality composite blends four dimensions:
- **Sovereign rating** (30%) — S&P / Moody's numeric composite
- **Fiscal balance / GDP** (25%) — World Bank annual series
- **Debt / GDP** (25%) — World Bank annual series
- **Reserves / months of imports** (20%) — World Bank annual series

### Signal Confidence

Each country's final score is multiplied by a **signal confidence** factor (0–1) derived from rolling 60-day data coverage across all signals. The four countries with zero FRED CPI/short-rate coverage (Colombia, Romania, Philippines, Malaysia) are capped at ≤0.70, automatically reducing their active tilts.

## Risk Regime Overlay

The allocator classifies each date into one of three regimes using live macro thresholds:

| Regime | VIX | DXY 60d z-score | EM OAS | Max active weight | Duration | Local FX |
|--------|-----|-----------------|--------|-------------------|----------|----------|
| Green | < 20 | < 1.0 | < 400bp | 4% | Full | Allowed |
| Amber | 20–30 | 1.0–2.0 | 400–550bp | 2% | Zero | Disallowed |
| Red | > 30 | > 2.0 | > 550bp | 0.5% | Zero | Disallowed |

Active weights are also halved within ±3 calendar days of scheduled FOMC meetings.

## Portfolio Construction Rules

- **TE constraint**: active weights are L2-norm scaled so `√Σ(active²) ≤ 4%`
- **Country caps**: max `2.5× EMBI benchmark weight` or `5%`; floor `0.25× EMBI weight` or `0.5%`
- **Benchmark**: EMBI Global Diversified index weights (normalized to 11-country universe)
- **Local currency overlay**: base allocation is 0%; added only when FX carry > 0 AND FX momentum > 0 in Green regime (up to 40% local per country)
- **Minimum holding period**: 5-day block on active-weight sign reversals
- **Rebalancing threshold**: 50bp weight change triggers a BUY or SELL signal
- **Transaction cost model**: 15bp one-way (hard currency), 10bp one-way (local); trades where estimated cost exceeds 50% of weight change are flagged as `HOLD (cost)`

```
                ┌──────────────────────────┐
                │     Market Data Layer    │
                │  FX / Yields / Macro /   │
                │  Commodities / VIX / DXY │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │      Country Panel       │
                │  Yields, Spreads, FX     │
                │  Real Yield, FX Carry    │
                │  Fiscal Fundamentals     │  ← World Bank annual data
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  9-Signal Score Engine   │
                │  Spread Value (TS+CS)    │
                │  Spread Momentum (blend) │
                │  FX Carry + Real Yield   │
                │  Local Return + FX Mom   │
                │  Commodity ToT           │
                │  ↓ Credit Quality Guard  │
                │  ↓ Signal Confidence     │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │    Global Adjustments    │
                │  US Real Rate Tilt (15%) │
                │  Risk Regime Overlay (5%)│
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │   Portfolio Allocator    │
                │  EMBI Benchmark Weights  │
                │  TE Constraint (4%)      │
                │  Country Caps (2.5×)     │
                │  Local Overlay (active)  │
                │  FOMC Calendar Scaling   │
                │  Min Holding Period (5d) │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     Portfolio Outputs    │
                │  Weekly Actions (50bp)   │
                │  TC-Aware Signals        │
                │  Regime Classification   │
                └──────────────────────────┘
```

---

# Data Sources

The model relies on publicly available macro and market data.

| Data Type | Source | Variables |
|------|------|------|
| FX Rates | Yahoo Finance | BRL/USD, MXN/USD, COP/USD, CLP/USD, ZAR/USD, PLN/USD, HUF/USD, RON/USD, IDR/USD, MYR/USD, PHP/USD |
| US Treasury Yields | FRED | DGS10 (10Y), DGS2 (2Y) |
| 10Y Government Bond Yields | Stooq (daily) | Per-country tickers (e.g. 10YBRY.B for Brazil) |
| CPI Inflation (YoY) | FRED / OECD | CPALTT01{CC}M659N — 7 OECD countries + US |
| Short-Term Interest Rates | FRED / OECD | IRSTCI01{CC}M156N — 7 OECD countries + US |
| EM Credit Spreads | FRED (ICE BofA) | BAMLEMCBPIOAS (EM OAS), BAMLEMHBHYCRPIOAS (EM HY OAS), BAMLH0A0HYM2 (US HY OAS) |
| Commodities | Yahoo Finance | Brent (BZ=F), WTI (CL=F), Copper (HG=F), Gold (GC=F) |
| Volatility | Yahoo Finance | VIX (^VIX) |
| Dollar Index | FRED | DTWEXEMEGS (Fed broad trade-weighted USD index) |
| Fiscal Fundamentals | World Bank API (`wbdata`) | Fiscal balance/GDP, Debt/GDP, Reserves/months of imports — annual, forward-filled |

> **Coverage note:** CPI and short-rate series are unavailable on FRED for Colombia, Romania, Philippines, and Malaysia (non-OECD). Real yield and FX carry signals are set to zero for these countries; their `signal_confidence` is capped at ≤0.70.

These datasets are combined into a **daily country panel** and a **global macro panel** used for cross-country scoring and portfolio construction.

---

# Example Portfolio Output

## Example Portfolio Snapshot

| Country | Weight | Hard | Local | Score |
|-------|------|------|------|------|
| Brazil | 10.2% | 5.1% | 5.1% | 0.69 |
| Romania | 8.9% | 5.4% | 3.4% | 0.32 |
| South Africa | 8.7% | 5.5% | 3.2% | 0.29 |
| Philippines | 8.6% | 4.7% | 3.9% | 0.24 |
| Mexico | 8.3% | 4.6% | 3.6% | 0.16 |

---

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

## 2. Inspect Dashboard

Current version is still in beta mode, with updates made regularly.

```
streamlit run dashboard/app.py
```

The dashboard visualizes:

- portfolio allocations and weekly trade recommendations
- country score rankings with factor attribution
- top and bottom performer narratives with score drivers
- carry and value panel for OECD Countries (real yield vs FX carry scatter, ranked bar chart, decomposition table)
- week-over-week score and signal history
- global macro context (rates, spreads, commodities, VIX)
- data coverage diagnostics per country
- VAR-based macro forecasts with confidence bands, impulse response functions, and Granger causality tables
- macro data surprise analysis: CPI and rate surprises vs naive forecast, surprise-to-FX/spread predictive regressions, and country-by-month heatmap

---

## 3. Iterate Research

In most cases, new ideas are tested through:

```
notebooks/
```

and then integrated into the production pipeline.

---

# Current Research Roadmap

## Data Improvements

- local bond index total returns (replace proxy with actual index data)
- EMBI country-level spread data (replace yield-spread proxy)

## Risk Model Updates

- full covariance matrix estimation
- factor exposure decomposition (duration, spread, currency)
- formal ex-ante tracking error calculation

## Portfolio Optimization (Phase 4)

Replace heuristic allocation with a formal optimizer:

```
maximize alpha

subject to

tracking error ≤ 4%
weights ≥ 0
country caps
liquidity constraints
transaction cost penalties
```

## Signal Enhancements

- current account balance as a carry quality filter
- political risk / governance overlay
- local vs external debt ratio as a credit signal

---

# Improvements

*As of March 2026 — Phase 1–3 Model Update*

## Phase 1: Critical Fixes

**Benchmark**
- Replaced equal-weight benchmark with **EMBI Global Diversified index weights** — portfolio active tilts and country caps are now computed relative to the actual index

**Risk Regime Overlay**
- Added a three-regime framework (Green / Amber / Red) using VIX, DXY 60d z-score, and EM OAS thresholds
- Active weight limits automatically scale with market stress: 4% in Green, 2% in Amber, 0.5% in Red
- Duration tilts and local currency allocations are zeroed in Amber and Red regimes

**New Scoring Signals**
- Added **FX carry** (local short rate − US rate) at 15% weight — the primary income signal for EM sovereign bonds
- Added **real yield** (10Y nominal − CPI inflation) at 10% weight — a measure of true rate compensation after inflation
- Added **US real rate global tilt** (15%) — a post-ranking additive adjustment; rising US real rates uniformly reduce all EM scores, reflecting the global risk-off pressure this creates

**Data Coverage**
- Identified that Colombia, Romania, Philippines, and Malaysia have zero FRED coverage for CPI and short rates
- Implemented split coverage tracking (`real_yield_coverage_60d`, `fx_carry_coverage_60d`) so each missing signal degrades `signal_confidence` proportionally — these four countries are capped at ≤0.70

**Rebalancing Threshold**
- Raised weekly action threshold from 25bp to **50bp** to reduce unnecessary turnover and transaction costs

## Phase 2: Signal Enhancement

**Multi-Horizon Spread Momentum**
- Replaced single 20-day spread change with a **blended 20/60/120d momentum signal** (weights: 25/50/25%) — captures both short-term reversal and medium-term trend

**Time-Series Spread Z-Score**
- Added a per-country 2-year rolling z-score for the spread level — blended 50/50 with the cross-sectional z-score to distinguish "cheap vs peers" from "cheap vs own history"

**Credit Quality Guard**
- Added a **credit quality composite score** (0–1) using sovereign ratings (30%), fiscal balance/GDP (25%), debt/GDP (25%), and reserves/months (20%)
- Positive spread-value contribution is capped at zero when spread > 700bp OR credit quality < 0.30 — prevents value traps in distressed credits

**Commodity Terms-of-Trade**
- Added a **commodity ToT signal** (10%) using country-specific sensitivity coefficients × Brent 60d return z-score
- Commodity exporters (Brazil, Colombia, Chile, South Africa, Indonesia) benefit from rallies; importers (Poland, Hungary, Philippines) are penalised

**Extended Lookbacks**
- Local return and FX momentum extended from 20d to **60d** for a more stable, less noise-sensitive signal

**Winsorized Z-Score**
- Replaced percentile rank with a **winsorized z-score** (clip ±3, rescale to [−1, 1]) — preserves signal magnitude so a 3σ outlier scores more than a 1σ outlier

**FOMC Calendar Flag**
- Active weights are scaled by **0.5× within ±3 days of scheduled FOMC meetings** to reduce risk around policy announcements

## Phase 3: Portfolio Construction

**TE Constraint**
- Added an L2-norm tracking error constraint: active weights are proportionally scaled if `√Σ(active²) > 4%` before per-country clip limits are applied

**Local Currency as Active Overlay**
- Changed local currency base allocation from 35% → **0%**; local exposure is now added only when FX carry > 0 AND FX momentum > 0 in Green regime (up to 40% local per country)
- Aligns with EMBI benchmark (hard currency only) — any local exposure is explicitly an active decision

**Transaction Cost Model**
- Added estimated one-way transaction costs: 15bp hard currency, 10bp local currency
- Trades where estimated TC exceeds 50% of the weight change are flagged as **HOLD (cost)** in the weekly action report

**Minimum Holding Period**
- Active-weight sign reversals within 5 business days are blocked — prevents excessive flip-flopping from high-frequency noise

**Country Caps**
- Per-country weight ceiling: `max(2.5× EMBI weight, 5%)`
- Per-country floor: `max(0.25× EMBI weight, 0.5%)`
- Replaced flat long-only floor with benchmark-relative bounds

## Phase 4: Research Analytics
**Still Making Improvements to These Features**

**VAR Macro Forecasting (`src/em/models/macro_forecast.py`)**
- Added a per-country **Vector Autoregression (VAR)** model fitted on monthly CPI, short-term rate, and FX data
- Produces h-step-ahead forecasts with 68% and 95% confidence bands, impulse response functions (IRF), and a Granger causality p-value matrix
- Lag order selected by AIC/BIC; variables are automatically differenced if ADF unit-root test fails
- Outputs saved to `data/processed/macro_forecasts/` as parquet; run via `run/run_macro_forecast.py`

**Data Surprise Analysis (`src/em/features/surprise.py`)**
- Added a macro surprise engine computing monthly **CPI and rate surprises** (actual minus random-walk naive forecast) for each country
- Runs a panel regression of surprises against 5-day forward FX returns and spread changes to test predictive power
- Outputs a country-by-month surprise heatmap and regression coefficient table

**New Dashboard Tabs (8-tab layout)**
- **Macro Forecasts tab**: per-country VAR forecast charts, IRF grid, and Granger causality heatmap
- **Data Surprises tab**: CPI/rate surprise heatmap, regime breakdown, and predictive regression results
- **Research tab**: persistent markdown research notes panel

---

*Earlier improvements (pre-Phase 1)*

- Replaced CDS data (unavailable at low cost) with yield-spread proxy (`hard_spread_proxy` = local 10Y − US 10Y)
- Added signal confidence weighting — scores scaled by 60-day data coverage per signal
- Duration tilt made country-specific (derived from each country's own 60-day yield trend)
- Added carry & value panel to the dashboard: real yield vs FX carry scatter, ranked bar chart, decomposition table

---

# Why This Project Matters ICYMI

Emerging market sovereign returns are driven by:

- global liquidity
- commodity cycles
- currency movements
- sovereign credit spreads

A systematic framework allows these signals to be combined into **consistent portfolio decisions** rather than discretionary views.

---

# Running the Project

## Install dependencies

```
pip install -r requirements.txt
pip install -e .
```

---

## Run the full pipeline

```
python run/update_all.py
```

---

## Launch the dashboard

```
streamlit run dashboard/app.py
```

---

# License

This repository is intended for **research and educational purposes**.
