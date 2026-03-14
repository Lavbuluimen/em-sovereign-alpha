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
| Portfolio Structure | Long-only |
| Benchmark | J.P. Morgan EMBI |
| Universe | 11 countries: Brazil, Mexico, Colombia, Chile, South Africa, Poland, Hungary, Romania, Indonesia, Malaysia, Philippines |
| Target Tracking Error | ~5% |
| Rebalancing | Daily signals, weekly to monthly adjustments |
| Backtest Start | January 2015 |
| Data Sources | FRED, Stooq, Yahoo Finance (all public, free) |

---

# Model Architecture

The EM Sovereign Alpha framework follows a modular research architecture. The composite score combines five cross-sectional signals, all z-scored relative to the 11-country universe on each date:

| Signal | Weight | Lookback |
|------|------|------|------|
| Sovereign spread vs US 10Y | 25% | Spot |
| Spread change | 25% | 20 days |
| Local bond + FX return | 20% | 20 days |
| 10Y yield change | 15% | 60 days |
| FX return vs USD | 15% | 20 days |

Scores are scaled to [−1, +1] and adjusted by a **signal confidence** factor based on data coverage. Countries with signal confidence below 0.5 are flagged as unreliable.

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
                │  Credit Spread      │
                │  Spread Momentum    │
                │  Local Bond Return  │
                │  Rate Trend         │
                │  FX Momentum        │
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
                │ Hard vs Local Split │
                │ Per-Country         │
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

- local bond index total returns (replace proxy)
- EMBI country-level spread data

## Risk Model Updates

- covariance estimation
- volatility estimation
- factor exposures
- tracking error calculation

## Portfolio Optimization

Move from simplified allocation to optimization:

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

## Carry & Value Signals

- integrate real yield and FX carry into the allocator's local/hard currency split

---

# Improvements

*As of March 2026*

**Scoring & Signals**
- Replaced CDS data (unavailable at low cost) with an EMBI spread proxy: each country's 10Y yield minus US 10Y yield (`hard_spread_proxy`)
- Extended yield trend lookback from 20 to 60 days for a cleaner, less noisy rate momentum signal
- Added signal confidence weighting — scores are scaled by data coverage quality, countries with sparse data are downweighted

**Carry & Value Panel**
- Added real yield per country: 10Y nominal yield minus trailing CPI inflation (7 OECD countries with FRED data)
- Added FX carry per country: local short-term interbank rate minus US short-term rate
- Added US real yield as a benchmark (US 10Y minus US CPI)
- Cross-sectional ranks computed for both metrics at pipeline time
- Dashboard carry & value panel: scatter plot (real yield vs FX carry, bubble = weight, colour = alpha score), real yield bar chart vs US baseline, and full decomposition table

**Portfolio Construction**
- Duration tilt is now **country-specific** — each country's tilt is derived from its own 60-day local yield trend rather than a single global US 10Y proxy

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