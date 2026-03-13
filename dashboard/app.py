"""
EM Sovereign Alpha — Research Dashboard
========================================
Streamlit dashboard for the EM Sovereign Allocation Research Platform.

Run from the project root:
    cd em-sovereign-alpha-main
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config & styling
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# AI Chatbot
# ---------------------------------------------------------------------------
CHATBOT_SYSTEM_PROMPT = """You are an expert research assistant for the EM Sovereign Alpha platform — a systematic emerging market sovereign debt research tool. You help portfolio managers, analysts, and researchers understand the model, data, and dashboard outputs.

## Model Overview
The EM Sovereign Alpha model ranks 11 emerging market countries by their relative attractiveness for sovereign debt investment, covering both local-currency and hard-currency (USD) bonds.

## Active Country Universe (as of 2026-03-13)
Brazil, Mexico, Colombia, Chile, South Africa, Poland, Hungary, Romania, Indonesia, Malaysia, Philippines.
Peru and Thailand were removed in March 2026 because Stooq had no reliable daily 10Y yield data for them — their signal_confidence was 0.0 throughout the backtest.

## Composite Score Formula
The model produces a cross-sectional z-scored composite score for each country. The weights are:

| Signal | Weight | Direction | Meaning |
|---|---|---|---|
| hard_spread_proxy | 25% | Higher spread = worse | 10Y local yield minus US 10Y yield (sovereign credit risk proxy) |
| embi_spread_20d_chg | 25% | Spread widening = worse | 20-day change in hard spread proxy (momentum) |
| local_ret_20d | 20% | Higher return = better | 20-day local-currency total return proxy (duration + FX) |
| yield_60d_chg | 15% | Yield rise = worse | 60-day change in 10Y yield (rate momentum) |
| fx_ret_20d | 15% | FX appreciation = better | 20-day USD return on local currency |

Score formula:
  score_raw = 0.25 * hard_spread_proxy_z + 0.20 * local_ret_20d_z + 0.15 * (-yield_60d_chg_z) + 0.25 * (-embi_spread_20d_chg_z) + 0.15 * fx_ret_20d_z

All z-scores are cross-sectional (computed across the 11 countries on each date), not time-series z-scores.

## Signal Confidence
Each country gets a signal_confidence score (0–1) based on data availability:
  signal_confidence = 0.4 * yield_coverage_60d + 0.3 * spread_coverage_60d + 0.3 * embi_coverage_60d

Countries with signal_confidence < 0.5 are flagged as unreliable. The final adjusted score is: score_adj = score_raw * signal_confidence.

## Trade Signals
- BUY: score_adj > 0.4 (strong positive signal)
- SELL: score_adj < -0.4 (strong negative signal)
- HOLD: otherwise

## Data Sources

### 10Y Government Bond Yields (primary credit signal)
Source: Stooq (free, daily)
Tickers: 10YBRY.B (Brazil), 10YMXY.B (Mexico), 10YCOY.B (Colombia), 10YCLY.B (Chile), 10YZAY.B (South Africa), 10YPLY.B (Poland), 10YHUY.B (Hungary), 10YROY.B (Romania), 10YIDY.B (Indonesia), 10YMYY.B (Malaysia), 10YPHY.B (Philippines)

### FX Rates (local currency per USD)
Source: Yahoo Finance (daily close)
Tickers: BRL=X, MXN=X, COP=X, CLP=X, ZAR=X, PLN=X, HUF=X, RON=X, IDR=X, MYR=X, PHP=X

### US Rates
Source: FRED
- DGS10: US 10-year Treasury yield (daily)
- DGS2: US 2-year Treasury yield (daily)

### Global EM Credit Environment (ICE BofA indices via FRED)
- BAMLEMCBPIOAS (em_oas): ICE BofA EM Corporate Plus OAS — broad EM investment-grade credit spread
- BAMLEMHBHYCRPIOAS (em_hy_oas): ICE BofA EM HY Corporate Plus OAS
- BAMLH0A0HYM2 (us_hy_oas): ICE BofA US HY Master II OAS — global risk-off signal

### Derived Feature
- em_hy_ig_spread = em_hy_oas − em_oas: measures risk appetite within EM credit (HY premium over IG)

### Commodities & Risk
Source: Yahoo Finance
- ^VIX: CBOE Volatility Index (equity market fear gauge)
- BZ=F: Brent crude oil futures
- CL=F: WTI crude oil futures
- HG=F: Copper futures (global growth proxy)
- GC=F: Gold futures (safe-haven signal)

### USD Index
Source: FRED — DTWEXEMEGS (Federal Reserve broad trade-weighted USD index, replaces the broken Yahoo DX-Y.NYB ticker)

## Key Variables Explained

**hard_spread_proxy**: Local 10Y yield minus US 10Y yield. This is a proxy for the sovereign credit spread (EMBI spread). Higher values mean higher perceived credit risk for a country.

**embi_spread_proxy**: Same as hard_spread_proxy (direct copy). The name reflects its use as an EMBI spread substitute.

**embi_spread_20d_chg**: 20-day change in the embi_spread_proxy. Positive means spreads are widening (deteriorating credit). Negative means spreads tightening (improving credit).

**local_ret_proxy_usd**: Estimated total return in USD from holding a 5-year local-currency bond. Formula: (-5 * yield_change/100) + fx_usd_return. This approximates the P&L from duration and currency combined.

**local_ret_20d**: 20-day cumulative version of local_ret_proxy_usd.

**fx_usd_ret**: Daily USD return from holding the local currency (approximately -pct_change of local/USD rate).

**fx_ret_20d**: 20-day cumulative FX return in USD.

**yield_60d_chg**: 60-day change in 10Y local yield (in percentage points). Rising yields = capital losses for bond holders.

**signal_confidence**: Data quality score (0–1). Based on how often yield data, spread data, and EMBI data were available in the last 60 trading days.

**score_raw**: The raw composite score before confidence adjustment.

**score_adj**: score_raw * signal_confidence. This is the final score used for rankings and signals.

## Dashboard Tabs

1. **Executive Summary**: Top-line KPIs (active countries, BUY/SELL counts, latest date), top BUY and SELL countries, VIX and spread gauges, macro context table.

2. **Country Scores**: Full ranking table with scores, signals, and factor breakdown. Includes score bar chart and factor heatmap for cross-country comparison.

3. **Portfolio Detail**: Suggested portfolio weights (overweight BUY, underweight SELL), portfolio construction details with weight chart.

4. **Weekly History**: Week-over-week comparison of scores and signals. Use the slider to compare any two weeks. Shows trade signal changes, WoW score movements, and portfolio weight evolution. The "portfolio snapshot" shows the model portfolio for each selected week. The weekly comparison section at the top shows side-by-side views of the two selected weeks.

5. **Market Data & Coverage**: Global macro data (rates, spreads, commodities), data coverage heatmap showing signal availability per country.

## How to Interpret the Tables

**Country Scores table**: Countries are ranked from highest to lowest score_adj. Green scores are bullish, red are bearish. The signal column shows BUY/HOLD/SELL based on the ±0.4 threshold on score_adj.

**Factor heatmap**: Shows each factor's z-score per country. Green = positive (bullish contribution), red = negative (bearish contribution). This lets you see WHICH factors are driving each country's score.

**Coverage heatmap**: Shows data availability per country and variable. Green = data present, red = missing. Low coverage = lower signal_confidence.

**WoW Comparison**: The two columns show the same country ranking for two different weeks. Orange arrows (↑↓) show rank changes. Score differences are shown in the delta columns.

## Methodology Notes
- All z-scores are cross-sectional, not time-series. A z-score of +2 means 2 standard deviations above the current cross-sectional average.
- The model is relative, not absolute — it tells you which countries look better or worse vs. each other, not whether EM as an asset class is attractive.
- Forward-fill is applied to yields and macro data to handle weekends and holidays.
- The backtest starts from 2015-01-01.

Answer questions clearly and concisely. When explaining numbers from tables the user mentions, use the context above to provide meaningful interpretation. If asked about something outside the model's scope, say so clearly."""

# Colour palette — dark professional theme
COLORS = {
    "bg": "#0E1117",
    "card": "#1A1D24",
    "border": "#2D3139",
    "text": "#E6E9EF",
    "muted": "#8B929E",
    "accent": "#4DA3FF",
    "green": "#00D26A",
    "red": "#FF4B4B",
    "amber": "#FFB347",
    "buy": "#00D26A",
    "hold": "#8B929E",
    "sell": "#FF4B4B",
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, Source Sans Pro, sans-serif", color=COLORS["text"]),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#2D3139", zerolinecolor="#2D3139"),
    yaxis=dict(gridcolor="#2D3139", zerolinecolor="#2D3139"),
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1D24 0%, #22262E 100%);
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 500;
        color: #8B929E;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #E6E9EF;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 12px;
        color: #8B929E;
        margin-top: 2px;
    }

    /* Action badges */
    .badge-buy {
        display: inline-block;
        background: #00D26A20;
        color: #00D26A;
        border: 1px solid #00D26A40;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-sell {
        display: inline-block;
        background: #FF4B4B20;
        color: #FF4B4B;
        border: 1px solid #FF4B4B40;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-hold {
        display: inline-block;
        background: #8B929E20;
        color: #8B929E;
        border: 1px solid #8B929E40;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Country row */
    .country-row {
        background: #1A1D24;
        border: 1px solid #2D3139;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .country-name {
        font-size: 16px;
        font-weight: 600;
        color: #E6E9EF;
        min-width: 130px;
    }
    .country-stat {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        color: #E6E9EF;
        min-width: 80px;
        text-align: right;
    }
    .country-stat-label {
        font-size: 10px;
        color: #8B929E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section divider */
    .section-divider {
        border-top: 1px solid #2D3139;
        margin: 32px 0 24px 0;
    }

    /* Subtitle */
    .section-subtitle {
        font-size: 13px;
        color: #8B929E;
        font-weight: 400;
        margin-top: -12px;
        margin-bottom: 20px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 14px;
        padding: 10px 24px;
        border-radius: 8px;
    }

    /* Floating AI chat button */
    .chat-fab-wrapper {
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
    }
    .chat-fab-wrapper button {
        width: 56px !important;
        height: 56px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #4DA3FF 0%, #6B5FFF 100%) !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(77, 163, 255, 0.45) !important;
        font-size: 24px !important;
        padding: 0 !important;
        line-height: 56px !important;
        cursor: pointer !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .chat-fab-wrapper button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 6px 28px rgba(77, 163, 255, 0.6) !important;
    }
    .chat-fab-wrapper button p {
        margin: 0 !important;
        line-height: 1 !important;
    }

    /* Chat dialog messages */
    .chat-user-msg {
        background: #2D3139;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 14px;
        margin: 6px 0 6px 20%;
        font-size: 14px;
        color: #E6E9EF;
    }
    .chat-assistant-msg {
        background: linear-gradient(135deg, #1A2A3A 0%, #1E2D40 100%);
        border: 1px solid #2D4A6A;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 14px;
        margin: 6px 20% 6px 0;
        font-size: 14px;
        color: #E6E9EF;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_parquet(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    if path.exists():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None


# ---------------------------------------------------------------------------
# Helper components
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def action_badge(action: str) -> str:
    action_upper = str(action).strip().upper()
    if "BUY" in action_upper:
        return f'<span class="badge-buy">{action}</span>'
    elif "SELL" in action_upper:
        return f'<span class="badge-sell">{action}</span>'
    return f'<span class="badge-hold">{action}</span>'


def score_bar_color(score: float) -> str:
    if score > 0.3:
        return COLORS["green"]
    elif score < -0.3:
        return COLORS["red"]
    return COLORS["amber"]


def format_pct(val, decimals=1) -> str:
    if pd.isna(val):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def format_bp(val) -> str:
    if pd.isna(val):
        return "—"
    return f"{val * 100:.0f}bp"


def format_float(val, decimals=2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Tab 1: Executive Summary
# ---------------------------------------------------------------------------
def render_executive_summary():
    portfolio = load_parquet("portfolio_daily.parquet")
    actions = load_parquet("weekly_actions.parquet")

    if portfolio is None:
        st.info("No portfolio data found. Run the pipeline first.")
        return

    latest_date = portfolio["date"].max()
    latest_port = portfolio[portfolio["date"] == latest_date].sort_values("weight", ascending=False)

    st.markdown(f'<p class="section-subtitle">As of {latest_date.strftime("%B %d, %Y")}</p>',
                unsafe_allow_html=True)

    # --- Top metric cards ---
    n_countries = len(latest_port)
    top_country = latest_port.iloc[0]["country"]
    top_weight = latest_port.iloc[0]["weight"]
    duration_tilt = latest_port["duration_tilt_years"].iloc[0]
    hard_total = latest_port["hard_w"].sum()
    local_total = latest_port["local_w"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Countries", str(n_countries), "Active in universe")
    with c2:
        metric_card("Top Allocation", f"{top_weight:.1%}", top_country)
    with c3:
        metric_card("Duration Tilt", f"{duration_tilt:+.2f}yr",
                     "Short" if duration_tilt < 0 else "Long" if duration_tilt > 0 else "Neutral")
    with c4:
        metric_card("Hard / Local", f"{hard_total:.0%} / {local_total:.0%}", "Currency split")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Weekly actions summary ---
    if actions is not None and not actions.empty:
        st.subheader("Weekly Trade Signals")
        st.markdown('<p class="section-subtitle">Recommended portfolio rebalancing actions</p>',
                    unsafe_allow_html=True)

        buys = actions[actions["action"].str.contains("BUY", na=False)].sort_values("w_change", ascending=False)
        sells = actions[actions["action"].str.contains("SELL", na=False)].sort_values("w_change", ascending=True)
        holds = actions[actions["action"].str.contains("HOLD", na=False)]

        col_b, col_h, col_s = st.columns(3)

        with col_b:
            st.markdown(f"**🟢 BUY / ADD** ({len(buys)})")
            for _, row in buys.iterrows():
                chg = row.get("w_change", 0)
                st.markdown(
                    f'{action_badge(row["action"])} &nbsp; **{row["country"]}** '
                    f'<span style="color:{COLORS["green"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                    f'+{chg:.1%}</span>',
                    unsafe_allow_html=True
                )

        with col_h:
            st.markdown(f"**⚪ HOLD** ({len(holds)})")
            for _, row in holds.iterrows():
                st.markdown(
                    f'{action_badge("HOLD")} &nbsp; **{row["country"]}**',
                    unsafe_allow_html=True
                )

        with col_s:
            st.markdown(f"**🔴 SELL / TRIM** ({len(sells)})")
            for _, row in sells.iterrows():
                chg = row.get("w_change", 0)
                st.markdown(
                    f'{action_badge(row["action"])} &nbsp; **{row["country"]}** '
                    f'<span style="color:{COLORS["red"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                    f'{chg:.1%}</span>',
                    unsafe_allow_html=True
                )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Portfolio weights chart ---
    st.subheader("Country Allocations")

    # Stacked bar: hard + local
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=latest_port["country"],
        y=latest_port["hard_w"],
        name="Hard Currency",
        marker_color="#4DA3FF",
    ))
    fig.add_trace(go.Bar(
        x=latest_port["country"],
        y=latest_port["local_w"],
        name="Local Currency",
        marker_color="#00D26A",
    ))

    bench_w = 1.0 / n_countries
    fig.add_hline(y=bench_w, line_dash="dot", line_color=COLORS["muted"],
                  annotation_text=f"Benchmark ({bench_w:.1%})",
                  annotation_font_color=COLORS["muted"])

    fig.update_layout(
        barmode="stack",
        title="Portfolio Weights by Country (Hard + Local)",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: Country Scores
# ---------------------------------------------------------------------------
def render_scores():
    scores = load_parquet("country_scores_daily.parquet")
    if scores is None:
        st.info("No scores data found. Run the pipeline first.")
        return

    latest_date = scores["date"].max()
    latest = scores[scores["date"] == latest_date].sort_values("score", ascending=False)

    st.markdown(f'<p class="section-subtitle">Cross-sectional alpha scores as of {latest_date.strftime("%B %d, %Y")}</p>',
                unsafe_allow_html=True)

    # --- Score bar chart (horizontal, sorted) ---
    colors = [score_bar_color(s) for s in latest["score"]]

    fig = go.Figure(go.Bar(
        x=latest["score"],
        y=latest["country"],
        orientation="h",
        marker_color=colors,
        text=latest["score"].apply(lambda s: f"{s:+.3f}"),
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12),
    ))
    fig.update_layout(
        title="Sovereign Alpha Scores",
        xaxis_title="Score",
        height=max(400, len(latest) * 45),
        **PLOTLY_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    # --- Score decomposition table ---
    st.subheader("Score Components")
    st.markdown('<p class="section-subtitle">Signal breakdown by feature z-score</p>',
                unsafe_allow_html=True)

    decomp_cols = [c for c in [
        "country", "score", "signal_confidence",
        "hard_spread_proxy", "y10y", "fx_usd_ret",
        "credit_risk_proxy", "credit_risk_20d_chg",
        "embi_spread_proxy", "embi_spread_20d_chg",
        "local_ret_20d", "fx_ret_20d", "yield_60d_chg",
    ] if c in latest.columns]

    display = latest[decomp_cols].copy()
    st.dataframe(
        display.style.format({
            "score": "{:+.4f}",
            "signal_confidence": "{:.2f}",
            "hard_spread_proxy": "{:.2f}",
            "y10y": "{:.3f}",
            "fx_usd_ret": "{:+.4f}",
        }, na_rep="—").background_gradient(
            subset=["score"], cmap="RdYlGn", vmin=-1, vmax=1
        ),
        width="stretch",
        hide_index=True,
    )

    # --- Score history ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Score History")

    countries = sorted(scores["country"].dropna().unique().tolist())
    selected = st.multiselect(
        "Select countries to compare",
        countries,
        default=countries[:3] if len(countries) >= 3 else countries,
    )

    if selected:
        # Date range filter
        min_date = scores["date"].min().date()
        max_date = scores["date"].max().date()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start = st.date_input("From", value=max_date - pd.Timedelta(days=180),
                                  min_value=min_date, max_value=max_date)
        with col_d2:
            end = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

        hist = scores[
            (scores["country"].isin(selected)) &
            (scores["date"].dt.date >= start) &
            (scores["date"].dt.date <= end)
        ].sort_values("date")

        fig = px.line(
            hist, x="date", y="score", color="country",
            title="Score Evolution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
        fig.update_layout(
            yaxis_title="Score",
            legend_title="",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Tab 3: Portfolio Detail
# ---------------------------------------------------------------------------
def render_portfolio_detail():
    portfolio = load_parquet("portfolio_daily.parquet")
    if portfolio is None:
        st.info("No portfolio data found.")
        return

    latest_date = portfolio["date"].max()
    latest = portfolio[portfolio["date"] == latest_date].sort_values("weight", ascending=False)

    st.markdown(f'<p class="section-subtitle">Detailed allocation breakdown as of {latest_date.strftime("%B %d, %Y")}</p>',
                unsafe_allow_html=True)

    # --- Allocation detail table ---
    display_cols = ["country", "score", "weight", "hard_w", "local_w",
                    "local_share", "active_w", "bench_w", "duration_tilt_years"]
    display_cols = [c for c in display_cols if c in latest.columns]
    display = latest[display_cols].copy()

    st.dataframe(
        display.style.format({
            "score": "{:+.3f}",
            "weight": "{:.1%}",
            "hard_w": "{:.1%}",
            "local_w": "{:.1%}",
            "local_share": "{:.0%}",
            "active_w": "{:+.1%}",
            "bench_w": "{:.1%}",
            "duration_tilt_years": "{:+.2f}",
        }, na_rep="—").background_gradient(
            subset=["weight"], cmap="Blues", vmin=0
        ),
        width="stretch",
        hide_index=True,
    )

    # --- Hard vs Local pie ---
    col1, col2 = st.columns(2)

    with col1:
        hard_total = latest["hard_w"].sum()
        local_total = latest["local_w"].sum()
        fig = go.Figure(go.Pie(
            labels=["Hard Currency", "Local Currency"],
            values=[hard_total, local_total],
            marker_colors=["#4DA3FF", "#00D26A"],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(family="DM Sans", size=13),
        ))
        fig.update_layout(
            title="Aggregate Currency Split",
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Active weights waterfall
        active_data = latest[["country", "active_w"]].copy() if "active_w" in latest.columns else None
        if active_data is not None:
            active_data = active_data.sort_values("active_w", ascending=False)
            colors_active = [COLORS["green"] if v >= 0 else COLORS["red"] for v in active_data["active_w"]]

            fig = go.Figure(go.Bar(
                x=active_data["country"],
                y=active_data["active_w"],
                marker_color=colors_active,
                text=active_data["active_w"].apply(lambda v: f"{v:+.1%}"),
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig.update_layout(
                title="Active Tilts vs Benchmark",
                yaxis_title="Active Weight",
                yaxis_tickformat="+.1%",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, width="stretch")

    # --- Weight history ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Weight History")

    countries = sorted(portfolio["country"].dropna().unique().tolist())
    selected = st.multiselect(
        "Select countries",
        countries,
        default=countries[:4] if len(countries) >= 4 else countries,
        key="port_countries",
    )

    if selected:
        lookback = st.slider("Lookback (days)", 30, 365, 90, key="port_lookback")
        cutoff = latest_date - pd.Timedelta(days=lookback)
        hist = portfolio[
            (portfolio["country"].isin(selected)) &
            (portfolio["date"] >= cutoff)
        ].sort_values("date")

        fig = px.line(
            hist, x="date", y="weight", color="country",
            title="Portfolio Weight Evolution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            yaxis_title="Weight",
            yaxis_tickformat=".1%",
            legend_title="",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Tab 4: Market Data
# ---------------------------------------------------------------------------
def render_market_data():
    panel = load_parquet("country_daily.parquet")
    macro = load_parquet("global_macro_daily.parquet")

    if panel is None:
        st.info("No country data found.")
        return

    latest_date = panel["date"].max()
    st.markdown(f'<p class="section-subtitle">Market snapshot as of {latest_date.strftime("%B %d, %Y")}</p>',
                unsafe_allow_html=True)

    latest = panel[panel["date"] == latest_date].sort_values("country")

    # --- Country market snapshot ---
    snap_cols = [c for c in [
        "country", "y10y", "hard_spread_proxy", "fx_level_local_per_usd", "fx_usd_ret",
        "embi_spread_proxy", "embi_spread_20d_chg",
        "credit_risk_proxy", "fx_vol_20d", "fx_drawdown_60d",
    ] if c in latest.columns]

    st.subheader("Country Market Snapshot")
    st.dataframe(
        latest[snap_cols].style.format({
            "y10y": "{:.3f}",
            "hard_spread_proxy": "{:.2f}",
            "fx_level_local_per_usd": "{:.2f}",
            "fx_usd_ret": "{:+.4f}",
            "embi_spread_proxy": "{:.3f}",
            "embi_spread_20d_chg": "{:+.3f}",
            "credit_risk_proxy": "{:.3f}",
            "fx_vol_20d": "{:.4f}",
            "fx_drawdown_60d": "{:.3f}",
        }, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    # --- Yield comparison ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("10Y Government Bond Yields")

    if "y10y" in latest.columns:
        sorted_yields = latest.sort_values("y10y", ascending=False)
        fig = go.Figure(go.Bar(
            x=sorted_yields["country"],
            y=sorted_yields["y10y"],
            marker_color="#4DA3FF",
            text=sorted_yields["y10y"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "—"),
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))

        # Add US 10Y line
        us10y_val = latest["us10y"].dropna().iloc[0] if "us10y" in latest.columns and latest["us10y"].notna().any() else None
        if us10y_val:
            fig.add_hline(y=us10y_val, line_dash="dot", line_color=COLORS["amber"],
                          annotation_text=f"US 10Y ({us10y_val:.2f}%)",
                          annotation_font_color=COLORS["amber"])

        fig.update_layout(
            yaxis_title="Yield (%)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    # --- Spread comparison ---
    if "hard_spread_proxy" in latest.columns:
        st.subheader("Hard Spread Proxy (Local 10Y − US 10Y)")
        sorted_spreads = latest.sort_values("hard_spread_proxy", ascending=False)
        spread_colors = ["#FF4B4B" if v > 5 else "#FFB347" if v > 2 else "#00D26A"
                         for v in sorted_spreads["hard_spread_proxy"].fillna(0)]

        fig = go.Figure(go.Bar(
            x=sorted_spreads["country"],
            y=sorted_spreads["hard_spread_proxy"],
            marker_color=spread_colors,
            text=sorted_spreads["hard_spread_proxy"].apply(
                lambda v: f"{v:.2f}%" if pd.notna(v) else "—"),
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(
            yaxis_title="Spread (%)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    # --- Global macro ---
    if macro is not None and not macro.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Global Macro Environment")

        lookback = st.slider("Lookback (days)", 30, 365 * 3, 180, key="macro_lookback")
        cutoff = macro["date"].max() - pd.Timedelta(days=lookback)
        macro_recent = macro[macro["date"] >= cutoff].sort_values("date")

        available_macro = [c for c in macro.columns if c != "date" and macro[c].notna().any()]
        selected_macro = st.multiselect(
            "Select macro series",
            available_macro,
            default=available_macro[:3] if len(available_macro) >= 3 else available_macro,
            key="macro_series",
        )

        if selected_macro:
            for series_name in selected_macro:
                fig = px.line(
                    macro_recent, x="date", y=series_name,
                    title=series_name,
                    color_discrete_sequence=[COLORS["accent"]],
                )
                fig.update_layout(
                    yaxis_title=series_name,
                    height=300,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Tab 5: Weekly History
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def _build_weekly_snapshots() -> pd.DataFrame | None:
    """Reconstruct full weekly history from portfolio + scores daily data."""
    portfolio = load_parquet("portfolio_daily.parquet")
    scores = load_parquet("country_scores_daily.parquet")

    if portfolio is None or portfolio.empty:
        return None

    port = portfolio.sort_values(["country", "date"]).copy()
    port["week"] = port["date"].dt.to_period("W")

    # Take last trading day per country-week
    rows = []
    for (country, _), g in port.groupby(["country", "week"]):
        g = g.sort_values("date")
        rows.append(g.iloc[-1].to_dict())

    snap = pd.DataFrame(rows)
    snap["date"] = pd.to_datetime(snap["date"])
    snap = snap.sort_values(["country", "date"]).reset_index(drop=True)

    # Week-over-week weight change
    snap["w_change"] = snap.groupby("country")["weight"].diff()

    # Action labels
    threshold = 0.0025
    snap["action"] = "HOLD"
    snap.loc[snap["w_change"] >= threshold, "action"] = "BUY / ADD"
    snap.loc[snap["w_change"] <= -threshold, "action"] = "SELL / TRIM"

    # Merge score details if available
    if scores is not None and not scores.empty:
        score_cols = ["date", "country", "signal_confidence"]
        # Add any feature columns that exist
        for c in ["hard_spread_proxy", "y10y", "fx_ret_20d",
                   "embi_spread_proxy", "embi_spread_20d_chg",
                   "credit_risk_proxy", "credit_risk_20d_chg"]:
            if c in scores.columns:
                score_cols.append(c)

        score_cols = list(dict.fromkeys(score_cols))  # dedupe
        score_sub = scores[score_cols].copy()
        score_sub["date"] = pd.to_datetime(score_sub["date"])

        snap = snap.merge(score_sub, on=["date", "country"], how="left")

    # Week label for display
    snap["week_label"] = snap["date"].dt.strftime("%Y-%m-%d")

    return snap


def render_weekly_history():
    snap = _build_weekly_snapshots()

    if snap is None or snap.empty:
        st.info("No portfolio history found. Run the pipeline first.")
        return

    # --- Week selector ---
    all_weeks = snap.sort_values("date")["week_label"].unique().tolist()

    # Limit to ~52 weeks (1 year)
    max_weeks = min(len(all_weeks), 52)
    recent_weeks = all_weeks[-max_weeks:]

    st.markdown('<p class="section-subtitle">Browse weekly portfolio snapshots and trade signals from the past year</p>',
                unsafe_allow_html=True)

    # Week picker — default to most recent
    col_sel, col_nav = st.columns([3, 1])

    with col_sel:
        selected_week = st.select_slider(
            "Select week ending",
            options=recent_weeks,
            value=recent_weeks[-1],
            key="history_week_slider",
        )

    with col_nav:
        idx = recent_weeks.index(selected_week)
        st.markdown("<br>", unsafe_allow_html=True)
        c_prev, c_next = st.columns(2)
        with c_prev:
            if st.button("← Prev", disabled=(idx == 0), key="prev_week"):
                selected_week = recent_weeks[idx - 1]
                st.rerun()
        with c_next:
            if st.button("Next →", disabled=(idx == len(recent_weeks) - 1), key="next_week"):
                selected_week = recent_weeks[idx + 1]
                st.rerun()

    week_data = snap[snap["week_label"] == selected_week].copy()
    if week_data.empty:
        st.warning(f"No data for week ending {selected_week}")
        return

    st.markdown(f'### Week ending {selected_week}')

    # --- Trade signals for this week ---
    st.subheader("Trade Signals")

    buys = week_data[week_data["action"].str.contains("BUY", na=False)].sort_values("w_change", ascending=False)
    holds = week_data[week_data["action"].str.contains("HOLD", na=False)].sort_values("score", ascending=False)
    sells = week_data[week_data["action"].str.contains("SELL", na=False)].sort_values("w_change", ascending=True)

    col_b, col_h, col_s = st.columns(3)

    with col_b:
        st.markdown(f"**🟢 BUY / ADD** ({len(buys)})")
        if buys.empty:
            st.markdown(f'<span style="color:{COLORS["muted"]}">None</span>', unsafe_allow_html=True)
        for _, row in buys.iterrows():
            chg = row.get("w_change", 0)
            chg_str = f"+{chg:.1%}" if pd.notna(chg) else ""
            st.markdown(
                f'{action_badge(row["action"])} &nbsp; **{row["country"]}** '
                f'<span style="color:{COLORS["green"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                f'{chg_str}</span>',
                unsafe_allow_html=True
            )

    with col_h:
        st.markdown(f"**⚪ HOLD** ({len(holds)})")
        if holds.empty:
            st.markdown(f'<span style="color:{COLORS["muted"]}">None</span>', unsafe_allow_html=True)
        for _, row in holds.iterrows():
            st.markdown(
                f'{action_badge("HOLD")} &nbsp; **{row["country"]}**',
                unsafe_allow_html=True
            )

    with col_s:
        st.markdown(f"**🔴 SELL / TRIM** ({len(sells)})")
        if sells.empty:
            st.markdown(f'<span style="color:{COLORS["muted"]}">None</span>', unsafe_allow_html=True)
        for _, row in sells.iterrows():
            chg = row.get("w_change", 0)
            chg_str = f"{chg:.1%}" if pd.notna(chg) else ""
            st.markdown(
                f'{action_badge(row["action"])} &nbsp; **{row["country"]}** '
                f'<span style="color:{COLORS["red"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                f'{chg_str}</span>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Week-over-week comparison ---
    st.subheader("Week-over-Week Comparison")
    st.markdown('<p class="section-subtitle">Compare two weeks side by side</p>',
                unsafe_allow_html=True)

    col_w1, _ = st.columns(2)
    with col_w1:
        compare_week = st.selectbox(
            "Compare against",
            [w for w in recent_weeks if w != selected_week],
            index=max(0, recent_weeks.index(selected_week) - 1) if len(recent_weeks) > 1 else 0,
            key="compare_week",
        )

    if compare_week:
        compare_data = snap[snap["week_label"] == compare_week].copy()

        if not compare_data.empty:
            current = week_data[["country", "score", "weight"]].set_index("country")
            previous = compare_data[["country", "score", "weight"]].set_index("country")

            diff = current.join(previous, lsuffix="_current", rsuffix="_previous", how="outer")
            diff["score_change"] = diff["score_current"] - diff["score_previous"]
            diff["weight_change"] = diff["weight_current"] - diff["weight_previous"]
            diff = diff.sort_values("score_change", ascending=False).reset_index()

            # Score change bar chart
            colors_chg = [COLORS["green"] if v >= 0 else COLORS["red"]
                          for v in diff["score_change"].fillna(0)]

            fig = go.Figure(go.Bar(
                x=diff["country"],
                y=diff["score_change"],
                marker_color=colors_chg,
                text=diff["score_change"].apply(
                    lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig.update_layout(
                title=f"Score Change: {compare_week} → {selected_week}",
                yaxis_title="Score Δ",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, width="stretch")

            # Comparison table
            display_diff = diff[["country", "score_previous", "score_current",
                                 "score_change", "weight_previous", "weight_current",
                                 "weight_change"]].copy()
            display_diff.columns = [
                "Country", "Score (prev)", "Score (current)", "Score Δ",
                "Weight (prev)", "Weight (current)", "Weight Δ",
            ]

            st.dataframe(
                display_diff.style.format({
                    "Score (prev)": "{:+.3f}",
                    "Score (current)": "{:+.3f}",
                    "Score Δ": "{:+.3f}",
                    "Weight (prev)": "{:.1%}",
                    "Weight (current)": "{:.1%}",
                    "Weight Δ": "{:+.1%}",
                }, na_rep="—").background_gradient(
                    subset=["Score Δ"], cmap="RdYlGn", vmin=-0.5, vmax=0.5
                ),
                width="stretch",
                hide_index=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Portfolio snapshot table ---
    st.subheader("Portfolio Snapshot")

    week_sorted = week_data.sort_values("weight", ascending=False)
    table_cols = [c for c in [
        "country", "score", "weight", "w_change", "action",
        "hard_w", "local_w", "local_share", "duration_tilt_years",
    ] if c in week_sorted.columns]

    st.dataframe(
        week_sorted[table_cols].style.format({
            "score": "{:+.3f}",
            "weight": "{:.1%}",
            "w_change": "{:+.1%}",
            "hard_w": "{:.1%}",
            "local_w": "{:.1%}",
            "local_share": "{:.0%}",
            "duration_tilt_years": "{:+.2f}",
        }, na_rep="—").background_gradient(
            subset=["score"], cmap="RdYlGn", vmin=-1, vmax=1
        ),
        width="stretch",
        hide_index=True,
    )

    # --- Stacked weight chart for this week ---
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=week_sorted["country"],
        y=week_sorted["hard_w"],
        name="Hard Currency",
        marker_color="#4DA3FF",
    ))
    fig.add_trace(go.Bar(
        x=week_sorted["country"],
        y=week_sorted["local_w"],
        name="Local Currency",
        marker_color="#00D26A",
    ))

    n_countries = len(week_sorted)
    if n_countries > 0:
        bench = 1.0 / n_countries
        fig.add_hline(y=bench, line_dash="dot", line_color=COLORS["muted"],
                      annotation_text=f"Benchmark ({bench:.1%})",
                      annotation_font_color=COLORS["muted"])

    fig.update_layout(
        barmode="stack",
        title=f"Country Weights — Week ending {selected_week}",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Market context for this week ---
    if "hard_spread_proxy" in week_data.columns:
        st.subheader("Market Context")

        market_cols = [c for c in [
            "country", "y10y", "hard_spread_proxy", "fx_usd_ret",
            "embi_spread_proxy", "embi_spread_20d_chg",
            "credit_risk_proxy", "signal_confidence",
        ] if c in week_data.columns]

        market_display = week_data.sort_values("score", ascending=False)[market_cols]
        st.dataframe(
            market_display.style.format({
                "y10y": "{:.3f}",
                "hard_spread_proxy": "{:.2f}",
                "fx_usd_ret": "{:+.4f}",
                "embi_spread_proxy": "{:.3f}",
                "embi_spread_20d_chg": "{:+.3f}",
                "credit_risk_proxy": "{:.3f}",
                "signal_confidence": "{:.2f}",
            }, na_rep="—"),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 6: Data Coverage
# ---------------------------------------------------------------------------
def render_coverage():
    panel = load_parquet("country_daily.parquet")
    scores = load_parquet("country_scores_daily.parquet")
    portfolio = load_parquet("portfolio_daily.parquet")
    macro = load_parquet("global_macro_daily.parquet")

    st.markdown('<p class="section-subtitle">Data quality diagnostics and coverage analysis</p>',
                unsafe_allow_html=True)

    # --- File inventory ---
    files = {
        "country_daily.parquet": panel,
        "country_scores_daily.parquet": scores,
        "portfolio_daily.parquet": portfolio,
        "global_macro_daily.parquet": macro,
    }

    file_info = []
    for name, df in files.items():
        if df is not None:
            rows = len(df)
            cols = len(df.columns)
            date_range = ""
            if "date" in df.columns:
                date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
            file_info.append({"File": name, "Status": "✅", "Rows": f"{rows:,}", "Cols": cols, "Date Range": date_range})
        else:
            file_info.append({"File": name, "Status": "❌", "Rows": "—", "Cols": "—", "Date Range": "—"})

    st.dataframe(pd.DataFrame(file_info), width="stretch", hide_index=True)

    # --- Country-level coverage heatmap ---
    if panel is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Country Data Coverage")

        key_cols = [c for c in [
            "y10y", "fx_usd_ret", "hard_spread_proxy", "local_ret_proxy_usd",
            "embi_spread_proxy", "credit_risk_proxy",
        ] if c in panel.columns]

        if key_cols:
            cov = panel.groupby("country")[key_cols].apply(
                lambda x: x.notna().mean()
            ).reset_index()

            # Heatmap
            fig = go.Figure(go.Heatmap(
                z=cov[key_cols].values,
                x=key_cols,
                y=cov["country"],
                colorscale=[[0, "#FF4B4B"], [0.5, "#FFB347"], [1, "#00D26A"]],
                text=cov[key_cols].map(lambda v: f"{v:.0%}").values,
                texttemplate="%{text}",
                textfont=dict(size=12, family="JetBrains Mono"),
                zmin=0, zmax=1,
                colorbar=dict(title="Coverage", tickformat=".0%"),
            ))
            fig.update_layout(
                title="Data Coverage by Country & Feature",
                height=max(400, len(cov) * 40),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, width="stretch")

    # --- Signal confidence ---
    if scores is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Signal Confidence")

        latest_date = scores["date"].max()
        latest = scores[scores["date"] == latest_date].sort_values("signal_confidence", ascending=False)

        conf_cols = [c for c in [
            "country", "signal_confidence",
            "yield_coverage_60d", "spread_coverage_60d",
            "credit_proxy_coverage_60d", "fx_coverage_60d",
            "embi_coverage_60d",
        ] if c in latest.columns]

        if conf_cols:
            st.dataframe(
                latest[conf_cols].style.format({
                    c: "{:.2f}" for c in conf_cols if c != "country"
                }, na_rep="—").background_gradient(
                    subset=[c for c in conf_cols if c != "country"],
                    cmap="RdYlGn", vmin=0, vmax=1
                ),
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# AI Chatbot dialog
# ---------------------------------------------------------------------------
@st.dialog("🤖 EM Sovereign AI Assistant", width="large")
def open_chatbot():
    """Floating chat assistant powered by Claude."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown(
        '<p style="color:#8B929E;font-size:13px;margin-top:-8px;margin-bottom:16px;">'
        "Ask me anything about the model, variables, data sources, or how to interpret the dashboard."
        "</p>",
        unsafe_allow_html=True,
    )

    # Render conversation history
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask a question about the model or data…")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-user-msg">🧑 {user_input}</div>', unsafe_allow_html=True)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.error("ANTHROPIC_API_KEY environment variable is not set. Please add it to use the AI assistant.")
            return

        client = anthropic.Anthropic(api_key=api_key)

        # Build messages list (exclude system from messages list)
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]

        # Stream response
        response_placeholder = st.empty()
        full_response = ""

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=CHATBOT_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                full_response += text_chunk
                response_placeholder.markdown(
                    f'<div class="chat-assistant-msg">🤖 {full_response}▌</div>',
                    unsafe_allow_html=True,
                )

        response_placeholder.markdown(
            f'<div class="chat-assistant-msg">🤖 {full_response}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

    if st.session_state.chat_messages:
        if st.button("🗑 Clear conversation", key="chat_clear"):
            st.session_state.chat_messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="EM Sovereign Alpha",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()

    # --- Header ---
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
        <span style="font-size: 32px;">🌍</span>
        <div>
            <h1 style="margin:0; font-size: 28px; letter-spacing: -0.5px;">
                EM Sovereign Alpha
            </h1>
            <p style="margin:0; color: #8B929E; font-size: 14px;">
                Systematic Emerging Market Sovereign Debt Research Platform
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider" style="margin-top:8px;"></div>', unsafe_allow_html=True)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary",
        "🎯 Country Scores",
        "💼 Portfolio Detail",
        "🕐 Weekly History",
        "📈 Market Data & Coverage",
    ])

    with tab1:
        render_executive_summary()
    with tab2:
        render_scores()
    with tab3:
        render_portfolio_detail()
    with tab4:
        render_weekly_history()
    with tab5:
        with st.container():
            render_market_data()
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_coverage()

    # --- Floating AI chat button ---
    st.markdown('<div class="chat-fab-wrapper">', unsafe_allow_html=True)
    if st.button("🤖", key="chat_fab", help="AI Research Assistant"):
        open_chatbot()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()