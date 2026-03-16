"""
EM Sovereign Alpha — Research Dashboard
========================================
Streamlit dashboard for the EM Sovereign Allocation Research Platform.

Run from the project root:
    cd em-sovereign-alpha-main
    streamlit run dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config & styling
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")

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

# Human-readable labels for raw column / ticker names
COLUMN_LABELS: dict[str, str] = {
    "y10y":                       "10Y Yield (%)",
    "hard_spread_proxy":          "Spread vs US (%)",
    "fx_level_local_per_usd":     "FX Rate (Local/USD)",
    "fx_usd_ret":                 "Daily FX Return",
    "fx_ret_20d":                 "FX Return (20d)",
    "fx_vol_20d":                 "FX Volatility (20d)",
    "fx_drawdown_60d":            "FX Drawdown (60d)",
    "embi_spread_proxy":          "EM Spread (%)",
    "embi_spread_20d_chg":        "EM Spread Change (20d)",
    "credit_risk_proxy":          "Credit Risk Proxy",
    "credit_risk_20d_chg":        "Credit Risk Change (20d)",
    "local_ret_proxy_usd":        "Local Return (USD)",
    "local_ret_20d":              "Local Return (20d)",
    "yield_60d_chg":              "Yield Change (60d)",
    "signal_confidence":          "Signal Confidence",
    "hard_w":                     "Hard Currency Wt",
    "local_w":                    "Local Currency Wt",
    "local_share":                "Local Share",
    "active_w":                   "Active Weight",
    "bench_w":                    "Benchmark Wt",
    "w_change":                   "Weight Change",
    "duration_tilt_years":        "Duration Tilt (yrs)",
    "score":                      "Score",
    "score_adj":                  "Adj Score",
    "weight":                     "Weight",
    "yield_coverage_60d":         "Yield Coverage (60d)",
    "spread_coverage_60d":        "Spread Coverage (60d)",
    "credit_proxy_coverage_60d":  "Credit Coverage (60d)",
    "fx_coverage_60d":            "FX Coverage (60d)",
    "embi_coverage_60d":          "EM Spread Coverage (60d)",
    "country":                    "Country",
}

MACRO_LABELS: dict[str, str] = {
    "DGS10":                  "US 10Y Treasury Yield",
    "DGS2":                   "US 2Y Treasury Yield",
    "BAMLEMCBPIOAS":          "EM Credit Spread (OAS)",
    "BAMLEMHBHYCRPIOAS":      "EM High Yield Spread (OAS)",
    "BAMLH0A0HYM2":           "US High Yield Spread (OAS)",
    "em_hy_ig_spread":        "EM HY Premium over IG",
    "^VIX":                   "VIX (Volatility Index)",
    "BZ=F":                   "Brent Crude Oil",
    "CL=F":                   "WTI Crude Oil",
    "HG=F":                   "Copper Futures",
    "GC=F":                   "Gold Futures",
    "DTWEXEMEGS":             "USD Trade-Weighted Index",
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
    avg_duration = (latest_port["duration_tilt_years"] * latest_port["weight"]).sum()
    hard_total = latest_port["hard_w"].sum()
    local_total = latest_port["local_w"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Countries", str(n_countries), "Active in universe")
    with c2:
        metric_card("Top Allocation", f"{top_weight:.1%}", top_country)
    with c3:
        metric_card("Avg Portfolio Duration", f"{avg_duration:+.2f}yr",
                     "Short" if avg_duration < 0 else "Long" if avg_duration > 0 else "Neutral")
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

        st.markdown(
            f'<p style="color:{COLORS["muted"]};font-size:12px;margin-top:10px;">'
            "Trade signals reflect portfolio rebalancing actions — what needs to change from the current allocation. "
            "A country can show HOLD even with a top score if it is already at its target weight, "
            "or SELL with a positive score if it was previously overweight."
            "</p>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Top & Bottom Performers ---
    scores = load_parquet("country_scores_daily.parquet")
    if scores is not None and not scores.empty:
        latest_scores_date = scores["date"].max()
        latest_scores = scores[scores["date"] == latest_scores_date].copy()
        latest_scores["score_adj"] = latest_scores["score"] * latest_scores["signal_confidence"]
        latest_scores = (
            latest_scores
            .dropna(subset=["score_adj"])
            .sort_values("score_adj", ascending=False)
        )

        top3 = latest_scores.head(3)
        bot3 = latest_scores.tail(3).iloc[::-1]

        def _performer_narrative(row: pd.Series) -> list[str]:
            """Return up to 3 plain-English bullets explaining the score drivers."""
            bullets = []
            spread = row.get("hard_spread_proxy", None)
            sp_chg = row.get("embi_spread_20d_chg", None)
            lr = row.get("local_ret_20d", None)
            yc = row.get("yield_60d_chg", None)
            fx = row.get("fx_ret_20d", None)

            # Credit risk level
            if pd.notna(spread):
                spread_bps = spread * 100
                if spread > 4.0:
                    bullets.append(f"Sovereign spread is wide at {spread_bps:.0f}bps above US 10Y — elevated credit risk")
                elif spread < 1.5:
                    bullets.append(f"Sovereign spread is tight at {spread_bps:.0f}bps above US 10Y — low perceived risk")

            # Spread momentum (most important signal at 25%)
            if pd.notna(sp_chg):
                sp_chg_bps = sp_chg * 100
                if sp_chg < -0.20:
                    bullets.append(f"Spreads tightened {abs(sp_chg_bps):.0f}bps over 20 days — credit conditions improving")
                elif sp_chg > 0.20:
                    bullets.append(f"Spreads widened {sp_chg_bps:.0f}bps over 20 days — credit conditions deteriorating")

            # Local return (20%)
            if pd.notna(lr):
                if lr > 0.01:
                    bullets.append(f"20-day local return is positive (+{lr:.1%}) — bond price and/or currency tailwinds")
                elif lr < -0.01:
                    bullets.append(f"20-day local return is negative ({lr:.1%}) — bond or currency losses")

            # Yield change 60d (15%)
            if pd.notna(yc):
                if yc < -0.30:
                    bullets.append(f"10Y yield fell {abs(yc):.1f}pp over 60 days — duration momentum is bullish")
                elif yc > 0.30:
                    bullets.append(f"10Y yield rose {yc:.1f}pp over 60 days — duration momentum is bearish")

            # FX (15%)
            if pd.notna(fx):
                if fx > 0.01:
                    bullets.append(f"Currency gained {fx:.1%} vs USD over 20 days")
                elif fx < -0.01:
                    bullets.append(f"Currency lost {abs(fx):.1%} vs USD over 20 days")

            return bullets[:3] if bullets else ["Insufficient data to generate a narrative."]

        def _performer_card(row: pd.Series, rank: int, is_top: bool) -> str:
            score_val = row.get("score_adj", 0)
            color = COLORS["green"] if is_top else COLORS["red"]
            bullets = _performer_narrative(row)
            bullets_html = "".join(
                f'<li style="margin-bottom:4px;color:#C5CAD4;font-size:13px;">{b}</li>'
                for b in bullets
            )
            return f"""
            <div style="background:linear-gradient(135deg,#1A1D24 0%,#22262E 100%);
                        border:1px solid {color}33;border-left:3px solid {color};
                        border-radius:10px;padding:14px 16px;margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-weight:700;font-size:15px;color:{COLORS['text']};">
                        #{rank} &nbsp; {row['country']}
                    </span>
                    <span style="display:flex;gap:6px;align-items:center;">
                        <span style="color:{COLORS['muted']};font-size:11px;text-transform:uppercase;
                                     letter-spacing:0.05em;">Score</span>
                        <span style="color:{color};font-family:'JetBrains Mono',monospace;
                                     font-size:14px;font-weight:700;">{score_val:+.2f}</span>
                    </span>
                </div>
                <ul style="margin:0;padding-left:18px;">
                    {bullets_html}
                </ul>
            </div>"""

        st.subheader("Top & Bottom Performers")
        st.markdown(
            '<p class="section-subtitle">What the model is capturing — key drivers for the highest and lowest ranked countries</p>',
            unsafe_allow_html=True,
        )

        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("**🟢 Top 3 — Most Attractive**")
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                st.markdown(_performer_card(row, i, is_top=True), unsafe_allow_html=True)
        with col_bot:
            st.markdown("**🔴 Bottom 3 — Least Attractive**")
            for i, (_, row) in enumerate(bot3.iterrows(), 1):
                rank = len(latest_scores) - i + 1
                st.markdown(_performer_card(row, rank, is_top=False), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Carry & Value ---
    if scores is not None and not scores.empty:
        portfolio = load_parquet("portfolio_daily.parquet")

        cv_latest = latest_scores.copy()
        cv_avail = all(c in cv_latest.columns for c in ["real_yield", "fx_carry"])

        if cv_avail:
            # Merge in portfolio weights for bubble sizing
            if portfolio is not None and not portfolio.empty:
                latest_port_cv = portfolio[portfolio["date"] == portfolio["date"].max()][
                    ["country", "weight"]
                ]
                cv_latest = cv_latest.merge(latest_port_cv, on="country", how="left")
                cv_latest["weight"] = cv_latest["weight"].fillna(0.0)
            else:
                cv_latest["weight"] = 1.0 / len(cv_latest)

            cv_data = cv_latest.dropna(subset=["real_yield", "fx_carry"])

            st.subheader("Carry & Value")
            st.markdown(
                '<p class="section-subtitle">'
                "What are you being paid to own these countries? "
                "Alpha answers <em>what should I do</em> — carry answers <em>what am I earning</em>. "
                "Top-right + green = highest-conviction positions. Top-right + red = potential value trap."
                "</p>",
                unsafe_allow_html=True,
            )

            col_scatter, col_bar = st.columns([3, 2])

            with col_scatter:
                # Scatter: real yield (x) vs FX carry (y), sized by weight, colored by alpha score
                dot_colors = [
                    f"rgba(0,210,106,{min(0.9, 0.4 + abs(s))})" if s >= 0
                    else f"rgba(255,75,75,{min(0.9, 0.4 + abs(s))})"
                    for s in cv_data["score_adj"]
                ]
                scatter_fig = go.Figure()
                scatter_fig.add_trace(go.Scatter(
                    x=cv_data["real_yield"],
                    y=cv_data["fx_carry"],
                    mode="markers+text",
                    text=cv_data["country"],
                    textposition="top center",
                    textfont=dict(size=11, color=COLORS["text"]),
                    marker=dict(
                        size=cv_data["weight"].clip(lower=0.01) * 600,
                        color=dot_colors,
                        line=dict(width=1, color=COLORS["border"]),
                    ),
                    customdata=cv_data[["score_adj", "real_yield", "fx_carry", "weight"]].values,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Real Yield: %{x:.1f}%<br>"
                        "FX Carry: %{y:.1f}%<br>"
                        "Alpha Score: %{customdata[0]:+.2f}<br>"
                        "Portfolio Weight: %{customdata[3]:.1%}<extra></extra>"
                    ),
                ))
                # Quadrant lines at median
                med_ry = cv_data["real_yield"].median()
                med_fc = cv_data["fx_carry"].median()
                scatter_fig.add_vline(x=med_ry, line_dash="dot", line_color=COLORS["muted"], line_width=1)
                scatter_fig.add_hline(y=med_fc, line_dash="dot", line_color=COLORS["muted"], line_width=1)

                scatter_fig.update_layout(
                    title="Real Yield vs FX Carry (bubble = portfolio weight, colour = alpha score)",
                    xaxis_title="Real Yield — 10Y yield minus CPI inflation (%)",
                    yaxis_title="FX Carry — local rate minus US rate (%)",
                    height=420,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(scatter_fig, width="stretch")

            with col_bar:
                # Bar chart: real yield ranked, with US real yield baseline
                bar_data = cv_data.sort_values("real_yield", ascending=True)
                us_real_yield = cv_data["us_real_yield"].iloc[0] if "us_real_yield" in cv_data.columns else None

                bar_colors = [
                    COLORS["green"] if v > (us_real_yield or 0) else COLORS["red"]
                    for v in bar_data["real_yield"]
                ]
                bar_fig = go.Figure(go.Bar(
                    x=bar_data["real_yield"],
                    y=bar_data["country"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=bar_data["real_yield"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=11),
                ))
                if us_real_yield is not None and pd.notna(us_real_yield):
                    bar_fig.add_vline(
                        x=us_real_yield,
                        line_dash="dash",
                        line_color="#FFD700",
                        annotation_text=f"US real yield {us_real_yield:.1f}%",
                        annotation_font_color="#FFD700",
                        annotation_font_size=10,
                    )
                bar_fig.update_layout(
                    title="Real Yield vs US Baseline",
                    xaxis_title="Real Yield (%)",
                    height=420,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(bar_fig, width="stretch")

            # Carry decomposition table
            table_cols = {
                "country":          "Country",
                "y10y":             "Nominal Yield (%)",
                "cpi_yoy":          "CPI Inflation (%)",
                "real_yield":       "Real Yield (%)",
                "real_yield_rank":  "Real Yield Rank",
                "local_short_rate": "Local Rate (%)",
                "us_short_rate":    "US Rate (%)",
                "fx_carry":         "FX Carry (%)",
                "fx_carry_rank":    "FX Carry Rank",
            }
            avail_cols = [c for c in table_cols if c in cv_latest.columns]
            tbl = (
                cv_latest[avail_cols]
                .dropna(subset=["real_yield", "fx_carry"])
                .sort_values("real_yield", ascending=False)
                .rename(columns=table_cols)
            )
            fmt = {
                "Nominal Yield (%)":  "{:.2f}",
                "CPI Inflation (%)":  "{:.1f}",
                "Real Yield (%)":     "{:.2f}",
                "Real Yield Rank":    "{:.0f}",
                "Local Rate (%)":     "{:.2f}",
                "US Rate (%)":        "{:.2f}",
                "FX Carry (%)":       "{:.2f}",
                "FX Carry Rank":      "{:.0f}",
            }
            fmt = {k: v for k, v in fmt.items() if k in tbl.columns}
            st.dataframe(
                tbl.style.format(fmt, na_rep="—")
                .background_gradient(subset=["Real Yield (%)"], cmap="RdYlGn", vmin=-2, vmax=12)
                .background_gradient(subset=["FX Carry (%)"],   cmap="RdYlGn", vmin=-5, vmax=15),
                width="stretch",
                hide_index=True,
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

    fig.add_trace(go.Scatter(
        x=latest_port["country"],
        y=latest_port["bench_w"],
        mode="markers",
        name="EMBI Benchmark",
        marker=dict(symbol="diamond", size=10, color=COLORS["muted"]),
    ))

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

    # --- Score methodology explainer ---
    with st.expander("What is the Sovereign Alpha Score?", expanded=False):
        st.markdown("""
**The Sovereign Alpha Score answers: which EM country looks most attractive for sovereign debt investment right now, relative to its peers?**

It combines 5 signals into a single composite ranking. All signals are z-scored cross-sectionally — meaning each country is ranked against the other countries on the same date, not against its own history.

| Signal | Weight | What it measures | Direction |
|---|---|---|---|
| Spread vs US 10Y | 25% | How wide the country's sovereign spread is | Tighter = better |
| Spread change (20d) | 25% | Whether spreads are tightening or widening | Tightening = better |
| Local bond + FX return (20d) | 20% | Recent total return from holding the local bond | Positive = better |
| 10Y yield change (60d) | 15% | Whether yields are rising or falling | Falling = better |
| FX return vs USD (20d) | 15% | Whether the local currency is strengthening | Appreciating = better |

The raw score is then multiplied by a **signal confidence** factor (0–1) based on data availability over the last 60 days. Countries with patchy data are scaled down.

**Three themes the score captures:**
- **Credit risk** — Is the spread vs US Treasuries tight or wide, and is it improving or worsening?
- **Rate momentum** — Are local yields trending in a direction that generates bond price gains?
- **FX momentum** — Is the currency strengthening or weakening against the dollar?

**Important caveats:** The score is relative, not absolute — a +1.0 means best in the current universe, not that EM is broadly attractive. It is not a valuation model and has no macro overlay for global risk-off events.
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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

    display = latest[decomp_cols].copy().rename(columns=COLUMN_LABELS)
    st.dataframe(
        display.style.format({
            "Score": "{:+.4f}",
            "Signal Confidence": "{:.2f}",
            "Spread vs US (%)": "{:.2f}",
            "10Y Yield (%)": "{:.3f}",
            "Daily FX Return": "{:+.4f}",
        }, na_rep="—").background_gradient(
            subset=["Score"], cmap="RdYlGn", vmin=-1, vmax=1
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
    display = latest[display_cols].copy().rename(columns=COLUMN_LABELS)

    st.dataframe(
        display.style.format({
            "Score": "{:+.3f}",
            "Weight": "{:.1%}",
            "Hard Currency Wt": "{:.1%}",
            "Local Currency Wt": "{:.1%}",
            "Local Share": "{:.0%}",
            "Active Weight": "{:+.1%}",
            "Benchmark Wt": "{:.1%}",
            "Duration Tilt (yrs)": "{:+.2f}",
        }, na_rep="—").background_gradient(
            subset=["Weight"], cmap="Blues", vmin=0
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
        latest[snap_cols].rename(columns=COLUMN_LABELS).style.format({
            "10Y Yield (%)": "{:.3f}",
            "Spread vs US (%)": "{:.2f}",
            "FX Rate (Local/USD)": "{:.2f}",
            "Daily FX Return": "{:+.4f}",
            "EM Spread (%)": "{:.3f}",
            "EM Spread Change (20d)": "{:+.3f}",
            "Credit Risk Proxy": "{:.3f}",
            "FX Volatility (20d)": "{:.4f}",
            "FX Drawdown (60d)": "{:.3f}",
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
        st.subheader("Sovereign Credit Spread (Local 10Y − US 10Y)")
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
        # Map friendly label → raw column name for display
        macro_label_to_col = {MACRO_LABELS.get(c, c): c for c in available_macro}
        macro_labels_available = list(macro_label_to_col.keys())
        selected_labels = st.multiselect(
            "Select macro series",
            macro_labels_available,
            default=macro_labels_available[:3] if len(macro_labels_available) >= 3 else macro_labels_available,
            key="macro_series",
        )
        selected_macro = [macro_label_to_col[lbl] for lbl in selected_labels]

        if selected_macro:
            for series_name, series_label in zip(selected_macro, selected_labels):
                fig = px.line(
                    macro_recent, x="date", y=series_name,
                    title=series_label,
                    color_discrete_sequence=[COLORS["accent"]],
                )
                fig.update_layout(
                    yaxis_title=series_label,
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
        week_sorted[table_cols].rename(columns=COLUMN_LABELS).style.format({
            "Score": "{:+.3f}",
            "Weight": "{:.1%}",
            "Weight Change": "{:+.1%}",
            "Hard Currency Wt": "{:.1%}",
            "Local Currency Wt": "{:.1%}",
            "Local Share": "{:.0%}",
            "Duration Tilt (yrs)": "{:+.2f}",
        }, na_rep="—").background_gradient(
            subset=["Score"], cmap="RdYlGn", vmin=-1, vmax=1
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

        market_display = week_data.sort_values("score", ascending=False)[market_cols].rename(columns=COLUMN_LABELS)
        st.dataframe(
            market_display.style.format({
                "10Y Yield (%)": "{:.3f}",
                "Spread vs US (%)": "{:.2f}",
                "Daily FX Return": "{:+.4f}",
                "EM Spread (%)": "{:.3f}",
                "EM Spread Change (20d)": "{:+.3f}",
                "Credit Risk Proxy": "{:.3f}",
                "Signal Confidence": "{:.2f}",
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
            display_key_cols = [COLUMN_LABELS.get(c, c) for c in key_cols]

            # Heatmap
            fig = go.Figure(go.Heatmap(
                z=cov[key_cols].values,
                x=display_key_cols,
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
            renamed_num_cols = [COLUMN_LABELS.get(c, c) for c in conf_cols if c != "country"]
            st.dataframe(
                latest[conf_cols].rename(columns=COLUMN_LABELS).style.format({
                    c: "{:.2f}" for c in renamed_num_cols
                }, na_rep="—").background_gradient(
                    subset=renamed_num_cols,
                    cmap="RdYlGn", vmin=0, vmax=1
                ),
                width="stretch",
                hide_index=True,
            )


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



if __name__ == "__main__":
    main()