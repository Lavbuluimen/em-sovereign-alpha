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
| Instruments | Hard Currency Bonds and Local Currency Bonds |
| Strategy Type | Systematic Global Macro |
| Portfolio Structure | Long-only |
| Benchmark | J.P. Morgan EMBI |
| Target Tracking Error | ~5% |
| Rebalancing | Daily signals, weekly to monthly adjustments |
| Data Sources | Public macro and market data |

---

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

---

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

Current version is still in beta mode, with updates being made regularly.

```
streamlit run dashboard/app.py
```

The dashboard visualizes:

- portfolio allocations
- signal rankings
- weekly trade recommendations
- data coverage diagnostics

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

- sovereign CDS spreads
- EMBI country spreads
- local bond index returns

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