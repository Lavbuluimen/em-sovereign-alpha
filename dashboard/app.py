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

import numpy as np
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
    "embi_coverage_60d":          "EM Spread Coverage (60d)",
    "real_yield_coverage_60d":    "Real Yield Coverage (60d)",
    "fx_carry_coverage_60d":      "FX Carry Coverage (60d)",
    "country":                    "Country",
    # Phase 1–3 model outputs
    "regime":                     "Risk Regime",
    "score_raw":                  "Raw Score",
    "credit_quality_score":       "Credit Quality Score",
    "spread_value_blended_z":     "Spread Value Z (Blended)",
    "spread_mom_blend_z":         "Spread Momentum Z (Blend)",
    "spread_ts_z":                "Spread TS Z-Score",
    "spread_20d_chg_z":           "Spread Mom (20d) Z",
    "spread_60d_chg_z":           "Spread Mom (60d) Z",
    "spread_120d_chg_z":          "Spread Mom (120d) Z",
    "real_yield_z":               "Real Yield Z-Score",
    "fx_carry_z":                 "FX Carry Z-Score",
    "local_ret_60d_z":            "Local Return (60d) Z",
    "fx_ret_60d_z":               "FX Return (60d) Z",
    "commodity_tot_z":            "Commodity ToT Z-Score",
    "us_real_rate_tilt":          "US Real Rate Tilt",
    "local_ret_60d":              "Local Return (60d)",
    "fx_ret_60d":                 "FX Return (60d)",
    "tc_estimate":                "TC Estimate",
    "cost_aware_action":          "Cost-Aware Action",
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
    "VIX":                    "VIX (Spot)",
    "DXY":                    "US Dollar Index (DXY)",
    "em_oas":                 "EM OAS (BAMLEM)",
    "Brent":                  "Brent Crude (USD)",
    "us_real_yield":          "US Real Yield (%)",
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

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Countries", str(n_countries), "Active in universe")
    with c2:
        metric_card("Top Allocation", f"{top_weight:.1%}", top_country)
    with c3:
        metric_card("Avg Portfolio Duration", f"{avg_duration:+.2f}yr",
                     "Short" if avg_duration < 0 else "Long" if avg_duration > 0 else "Neutral")
    with c4:
        metric_card("Hard / Local", f"{hard_total:.0%} / {local_total:.0%}", "Currency split")
    with c5:
        if "regime" in latest_port.columns:
            regime_val = str(latest_port["regime"].iloc[0])
            regime_color = {"Green": COLORS["green"], "Amber": COLORS["amber"], "Red": COLORS["red"]}.get(regime_val, COLORS["muted"])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Regime</div>
                <div class="metric-value" style="color:{regime_color};">{regime_val}</div>
                <div class="metric-sub">VIX / DXY / EM OAS</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            metric_card("Risk Regime", "—", "VIX / DXY / EM OAS")

    # --- Risk Regime Sparklines ---
    _macro_spark = load_parquet("global_macro_daily.parquet")
    if _macro_spark is not None and not _macro_spark.empty:
        _mp_full = _macro_spark.sort_values("date").copy()
        if "DXY" in _mp_full.columns:
            _dxy_chg = _mp_full["DXY"].diff(60)
            _dxy_mu  = _dxy_chg.rolling(252, min_periods=60).mean()
            _dxy_sd  = _dxy_chg.rolling(252, min_periods=60).std().replace(0, float("nan"))
            _mp_full["_dxy_z"] = (_dxy_chg - _dxy_mu) / _dxy_sd
        _cutoff90 = latest_date - pd.Timedelta(days=90)
        _mp90 = _mp_full[_mp_full["date"] >= _cutoff90].copy()

        _spark_specs = [
            ("VIX",    "VIX",            "VIX Level",  20.0,  30.0),
            ("_dxy_z", "DXY 60d Z-Score","Z-Score",     1.0,   2.0),
            ("em_oas", "EM OAS",          "OAS (bp)",  400.0, 550.0),
        ]
        _spark_specs = [s for s in _spark_specs if s[0] in _mp90.columns and _mp90[s[0]].notna().any()]

        if _spark_specs:
            st.subheader("Risk Regime Inputs")
            _scols = st.columns(len(_spark_specs))
            for _cw, (col_nm, title, ylab, lo, hi) in zip(_scols, _spark_specs):
                with _cw:
                    _s = _mp90[["date", col_nm]].dropna()
                    if _s.empty:
                        continue
                    _cur = float(_s[col_nm].iloc[-1])
                    _rc = COLORS["green"] if _cur < lo else COLORS["amber"] if _cur < hi else COLORS["red"]
                    _ylo = min(_s[col_nm].min() * 1.05, lo * 0.5 if lo > 0 else lo * 1.5)
                    _yhi = max(_s[col_nm].max() * 1.05, hi * 1.3)
                    _fig_s = go.Figure()
                    _fig_s.add_hrect(y0=_ylo, y1=lo,  fillcolor="rgba(0,210,106,0.10)",  line_width=0)
                    _fig_s.add_hrect(y0=lo,   y1=hi,  fillcolor="rgba(255,179,71,0.10)", line_width=0)
                    _fig_s.add_hrect(y0=hi,   y1=_yhi, fillcolor="rgba(255,75,75,0.10)", line_width=0)
                    _fig_s.add_hline(y=lo, line_dash="dot", line_color=COLORS["amber"], line_width=1)
                    _fig_s.add_hline(y=hi, line_dash="dot", line_color=COLORS["red"],   line_width=1)
                    _fig_s.add_trace(go.Scatter(
                        x=_s["date"], y=_s[col_nm], mode="lines",
                        line=dict(color=_rc, width=2), showlegend=False,
                    ))
                    _fig_s.update_layout(
                        title=f"{title}  <b>{_cur:.2f}</b>",
                        height=220, yaxis_title=ylab,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Sans, sans-serif", color=COLORS["text"], size=11),
                        margin=dict(l=35, r=10, t=45, b=30),
                        xaxis=dict(gridcolor="#2D3139"), yaxis=dict(gridcolor="#2D3139"),
                    )
                    st.plotly_chart(_fig_s, width="stretch", key=f"risk_regime_{col_nm}")
            st.caption(
                "90-day trend of the three risk regime inputs. Shaded green/amber/red bands show "
                "the classification thresholds — any single reading in the amber band halves active risk; "
                "any reading in the red band cuts active risk to near-zero."
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Weekly actions summary ---
    if actions is not None and not actions.empty:
        st.subheader("Weekly Trade Signals")
        st.markdown('<p class="section-subtitle">Recommended portfolio rebalancing actions</p>',
                    unsafe_allow_html=True)

        action_col = "cost_aware_action" if "cost_aware_action" in actions.columns else "action"
        buys = actions[actions[action_col].str.contains("BUY", na=False)].sort_values("w_change", ascending=False)
        sells = actions[actions[action_col].str.contains("SELL", na=False)].sort_values("w_change", ascending=True)
        holds = actions[actions[action_col].str.contains("HOLD", na=False)]

        col_b, col_h, col_s = st.columns(3)

        with col_b:
            st.markdown(f"**🟢 BUY / ADD** ({len(buys)})")
            for _, row in buys.iterrows():
                chg = row.get("w_change", 0)
                tc = row.get("tc_estimate", None)
                tc_str = f' <span style="color:{COLORS["muted"]};font-size:11px;">TC≈{tc*10000:.0f}bp</span>' if pd.notna(tc) and tc else ""
                st.markdown(
                    f'{action_badge(row[action_col])} &nbsp; **{row["country"]}** '
                    f'<span style="color:{COLORS["green"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                    f'+{chg:.1%}</span>{tc_str}',
                    unsafe_allow_html=True
                )

        with col_h:
            st.markdown(f"**⚪ HOLD** ({len(holds)})")
            for _, row in holds.iterrows():
                st.markdown(
                    f'{action_badge(row[action_col])} &nbsp; **{row["country"]}**',
                    unsafe_allow_html=True
                )

        with col_s:
            st.markdown(f"**🔴 SELL / TRIM** ({len(sells)})")
            for _, row in sells.iterrows():
                chg = row.get("w_change", 0)
                tc = row.get("tc_estimate", None)
                tc_str = f' <span style="color:{COLORS["muted"]};font-size:11px;">TC≈{tc*10000:.0f}bp</span>' if pd.notna(tc) and tc else ""
                st.markdown(
                    f'{action_badge(row[action_col])} &nbsp; **{row["country"]}** '
                    f'<span style="color:{COLORS["red"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                    f'{chg:.1%}</span>{tc_str}',
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
            """Return up to 4 plain-English bullets explaining the score drivers."""
            bullets = []
            spread      = row.get("hard_spread_proxy", None)
            sp_mom_z    = row.get("spread_mom_blend_z", None)
            lr          = row.get("local_ret_60d", None)
            fx          = row.get("fx_ret_60d", None)
            fx_carry    = row.get("fx_carry", None)
            comm_z      = row.get("commodity_tot_z", None)
            us_tilt     = row.get("us_real_rate_tilt", None)

            # Credit risk level
            if pd.notna(spread):
                spread_bps = spread * 100
                if spread > 4.0:
                    bullets.append(f"Sovereign spread is wide at {spread_bps:.0f}bps above US 10Y — elevated credit risk")
                elif spread < 1.5:
                    bullets.append(f"Sovereign spread is tight at {spread_bps:.0f}bps above US 10Y — low perceived risk")

            # Spread momentum — blended 20/60/120d z-score (15% weight)
            if pd.notna(sp_mom_z):
                if sp_mom_z < -0.3:
                    bullets.append(f"Spread momentum is negative ({sp_mom_z:+.2f}z) — spreads tightening across 20/60/120d horizons")
                elif sp_mom_z > 0.3:
                    bullets.append(f"Spread momentum is positive ({sp_mom_z:+.2f}z) — spreads widening across horizons")

            # FX carry (15% weight)
            if pd.notna(fx_carry):
                if fx_carry > 2.0:
                    bullets.append(f"FX carry is attractive at +{fx_carry:.1f}% — local rate meaningfully above US rate")
                elif fx_carry < -1.0:
                    bullets.append(f"Negative FX carry ({fx_carry:.1f}%) — local rate below US funding cost")

            # Local return 60d (10% weight)
            if pd.notna(lr):
                if lr > 0.02:
                    bullets.append(f"60-day local return is positive (+{lr:.1%}) — bond and/or currency tailwinds")
                elif lr < -0.02:
                    bullets.append(f"60-day local return is negative ({lr:.1%}) — bond or currency headwinds")

            # FX momentum 60d (5% weight)
            if pd.notna(fx) and not bullets[-1:] or (pd.notna(fx) and len(bullets) < 3):
                if fx > 0.02:
                    bullets.append(f"Currency gained {fx:.1%} vs USD over 60 days")
                elif fx < -0.02:
                    bullets.append(f"Currency lost {abs(fx):.1%} vs USD over 60 days")

            # Commodity ToT (10% weight)
            if pd.notna(comm_z) and len(bullets) < 4:
                if comm_z > 0.4:
                    bullets.append(f"Commodity tailwind ({comm_z:+.2f}z) — country benefits from current commodity rally")
                elif comm_z < -0.4:
                    bullets.append(f"Commodity headwind ({comm_z:+.2f}z) — country hurt by commodity weakness")

            # US real rate global tilt
            if pd.notna(us_tilt) and len(bullets) < 4:
                if us_tilt < -0.05:
                    bullets.append(f"US real rate rising — global tilt of {us_tilt:+.2f} reduces all EM scores")
                elif us_tilt > 0.05:
                    bullets.append(f"US real rate falling — global tilt of {us_tilt:+.2f} supports all EM scores")

            return bullets[:4] if bullets else ["Insufficient data to generate a narrative."]

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

It combines 9 signals into a single composite ranking. Cross-sectional signals are z-scored against the 11-country universe on each date. Two global adjustments are applied after the cross-sectional ranking.

| Signal | Weight | What it measures | Direction |
|---|---|---|---|
| Spread value (blended TS + CS, quality-guarded) | 15% | Spread vs peers + own 2-year history; capped for distressed credits | Tighter = better |
| Spread momentum (20/60/120d blend) | 15% | Multi-horizon spread tightening | Tightening = better |
| FX carry | 15% | Local policy rate minus US rate | Positive carry = better |
| Real yield | 10% | 10Y yield minus CPI inflation | Higher real yield = better |
| Local return (60d) | 10% | 60-day total return from local bond in USD | Positive = better |
| FX momentum (60d) | 5% | 60-day FX return vs USD | Appreciation = better |
| Commodity terms-of-trade | 10% | Brent 60d return × country export sensitivity | Varies by country |
| US real rate global tilt | 15% | 60d change in US real rate (same for all countries) | Rising US rates = negative |
| Risk regime (VIX/DXY/EM OAS) | 5% | Active weight cap: Green 4%, Amber 2%, Red 0.5% | Green = full active risk |

The raw score is then multiplied by a **signal confidence** factor (0–1) based on data availability over the last 60 days. Countries with patchy data are scaled down.

> **Data coverage note:** FX carry and real yield are always zero for Colombia, Romania, Philippines, and Malaysia — FRED has no CPI or short-rate data for these countries. Their `signal_confidence` is capped at ≤0.70, which automatically reduces their active tilts.

**Four themes the score captures:**
- **Credit value** — Is the spread tight/wide vs peers and own history? Is the credit quality solid enough to warrant the carry?
- **Momentum** — Are spreads tightening across 20d, 60d, and 120d horizons simultaneously?
- **Carry & income** — What is the local rate premium and real yield advantage over US Treasuries?
- **Global context** — Is the macro regime supportive (VIX/DXY/OAS) and is US real rate pressure rising or falling?

**Important caveat:** The score is relative, not absolute — a +1.0 means best in the current universe, not that EM is broadly attractive. A country can score high while EM as a whole is in a Red regime (low active risk allowed).
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

    # --- Factor Attribution (Score Waterfall) ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Factor Attribution")
    _sig_specs = [
        ("spread_value_blended_z",  0.15,  "Spread Value",    True),
        ("spread_mom_blend_z",      0.15,  "Spread Momentum", False),
        ("fx_carry_z",              0.15,  "FX Carry",        True),
        ("real_yield_z",            0.10,  "Real Yield",      True),
        ("local_ret_60d_z",         0.10,  "Local Return",    True),
        ("fx_ret_60d_z",            0.05,  "FX Momentum",     True),
        ("commodity_tot_z",         0.10,  "Commodity ToT",   True),
        ("us_real_rate_tilt",       1.00,  "US Rate Tilt",    True),
    ]
    _sig_avail = [(c, w, lbl, pos) for c, w, lbl, pos in _sig_specs if c in latest.columns]
    if _sig_avail:
        _factor_df = latest[["country"]].copy()
        for col, w, lbl, pos in _sig_avail:
            _factor_df[lbl] = (latest[col].fillna(0.0) * w if pos else -latest[col].fillna(0.0) * w).values
        _attr_palette = ["#4DA3FF","#FF8C42","#00D26A","#B39DDB","#4DD0E1","#F48FB1","#AED581","#FFD54F"]
        _wfall = go.Figure()
        for (_, _, lbl, _), color in zip(_sig_avail, _attr_palette):
            _vals = _factor_df[lbl]
            _wfall.add_trace(go.Bar(
                name=lbl, x=_vals, y=_factor_df["country"], orientation="h",
                marker_color=[color if v >= 0 else "rgba(255,75,75,0.65)" for v in _vals],
                hovertemplate=f"<b>%{{y}}</b><br>{lbl}: %{{x:+.3f}}<extra></extra>",
            ))
        _wfall.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
        _wfall.update_yaxes(autorange="reversed")
        _wfall.update_layout(
            barmode="relative",
            title="Weighted Signal Contributions to Raw Score",
            xaxis_title="Weighted Contribution",
            height=max(380, len(latest) * 40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_wfall, width="stretch")
        st.caption(
            "Each segment shows one signal's weighted contribution to the country's raw score. "
            "Positive segments (right of zero) boost the score; negative segments reduce it. "
            "The total bar length approximates score_raw before signal confidence scaling."
        )

    # --- Score Conviction Map (Signal Confidence vs Score Scatter) ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Score Conviction Map")
    if "signal_confidence" in latest.columns:
        _conf_colors = [score_bar_color(s) for s in latest["score"]]
        _conf_fig = go.Figure()
        _conf_fig.add_trace(go.Scatter(
            x=latest["score"], y=latest["signal_confidence"],
            mode="markers+text", text=latest["country"],
            textposition="top center", textfont=dict(size=11, color=COLORS["text"]),
            marker=dict(size=14, color=_conf_colors, line=dict(width=1, color=COLORS["border"])),
            hovertemplate="<b>%{text}</b><br>Score: %{x:+.3f}<br>Confidence: %{y:.2f}<extra></extra>",
        ))
        _conf_fig.add_vline(x=0, line_dash="dot", line_color=COLORS["muted"], line_width=1)
        _conf_fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["amber"], line_width=1,
                             annotation_text="0.5 reliability threshold",
                             annotation_font_color=COLORS["amber"], annotation_font_size=10)
        _conf_fig.update_layout(
            title="Score vs Signal Confidence",
            xaxis_title="Composite Score", yaxis_title="Signal Confidence (0–1)",
            height=380, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_conf_fig, width="stretch")
        st.caption(
            "Top-right quadrant (high score, high confidence) = strong conviction longs. "
            "Top-left (positive score, low confidence) = data is thin — treat with caution. "
            "The amber dashed line marks 0.5; below it scores are significantly discounted by the signal confidence multiplier."
        )

    # --- Cross-Sectional Z-Score Dot Strip ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Cross-Sectional Z-Score Distribution")
    _strip_sigs = [
        ("spread_value_blended_z", "Spread Value"),
        ("spread_mom_blend_z",     "Spread Momentum"),
        ("fx_carry_z",             "FX Carry"),
        ("real_yield_z",           "Real Yield"),
        ("local_ret_60d_z",        "Local Return (60d)"),
        ("fx_ret_60d_z",           "FX Momentum (60d)"),
        ("commodity_tot_z",        "Commodity ToT"),
    ]
    _strip_avail = [(c, l) for c, l in _strip_sigs if c in latest.columns]
    if _strip_avail:
        _strip_fig = go.Figure()
        for col_nm, sig_label in _strip_avail:
            _z_vals = latest[col_nm].fillna(0.0).values
            _ctries = latest["country"].values
            _abbrevs = [c[:3] for c in _ctries]
            _dot_clrs = [COLORS["green"] if z > 0 else COLORS["red"] for z in _z_vals]
            _strip_fig.add_trace(go.Scatter(
                x=_z_vals, y=[sig_label] * len(_z_vals),
                mode="markers+text", text=_abbrevs,
                textposition="top center", textfont=dict(size=9, color=COLORS["muted"]),
                marker=dict(size=12, color=_dot_clrs, opacity=0.85,
                            line=dict(width=1, color=COLORS["border"])),
                showlegend=False, customdata=_ctries,
                hovertemplate=f"<b>%{{customdata}}</b><br>{sig_label}: %{{x:+.3f}}<extra></extra>",
            ))
        _strip_fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
        _strip_fig.update_layout(
            title="Where Each Country Sits on Each Signal Today",
            xaxis_title="Z-Score (cross-sectional, current date)",
            height=max(400, len(_strip_avail) * 72),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_strip_fig, width="stretch")
        st.caption(
            "Each dot is one country's z-score on that signal relative to all 11 countries today. "
            "Green = above the cross-sectional mean; red = below. "
            "A dot far to the right means the country is an outlier-high on that dimension versus its peers."
        )

    # --- Score decomposition table ---
    st.subheader("Score Components")
    st.markdown('<p class="section-subtitle">Signal breakdown by feature z-score</p>',
                unsafe_allow_html=True)

    decomp_cols = [c for c in [
        "country", "score", "score_raw", "signal_confidence",
        "spread_value_blended_z", "spread_mom_blend_z",
        "fx_carry_z", "real_yield_z",
        "local_ret_60d_z", "fx_ret_60d_z",
        "commodity_tot_z", "us_real_rate_tilt",
        "credit_quality_score",
        "hard_spread_proxy", "y10y",
    ] if c in latest.columns]

    display = latest[decomp_cols].copy().rename(columns=COLUMN_LABELS)
    z_cols = [COLUMN_LABELS.get(c, c) for c in [
        "spread_value_blended_z", "spread_mom_blend_z", "fx_carry_z",
        "real_yield_z", "local_ret_60d_z", "fx_ret_60d_z",
        "commodity_tot_z", "us_real_rate_tilt",
    ] if c in latest.columns]
    fmt = {
        "Score": "{:+.4f}",
        "Raw Score": "{:+.4f}",
        "Signal Confidence": "{:.2f}",
        "Credit Quality Score": "{:.2f}",
        "Spread vs US (%)": "{:.2f}",
        "10Y Yield (%)": "{:.3f}",
    }
    fmt.update({col: "{:+.3f}" for col in z_cols if col in display.columns})
    st.dataframe(
        display.style.format(fmt, na_rep="—").background_gradient(
            subset=["Score"], cmap="RdYlGn", vmin=-1, vmax=1
        ).background_gradient(
            subset=[c for c in z_cols if c in display.columns],
            cmap="RdYlGn", vmin=-1, vmax=1,
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

    # Merge fx_carry and fx_ret_20d from the country panel (not present in portfolio parquet)
    _cp = load_parquet("country_daily.parquet")
    if _cp is not None and not _cp.empty:
        _cp_latest_date = _cp["date"].max()
        _cp_snap = _cp[_cp["date"] == _cp_latest_date]
        _carry_merge_cols = [c for c in ["country", "fx_carry", "fx_ret_20d"] if c in _cp_snap.columns]
        if len(_carry_merge_cols) > 1:
            latest = latest.merge(_cp_snap[_carry_merge_cols], on="country", how="left")

    st.markdown(f'<p class="section-subtitle">Detailed allocation breakdown as of {latest_date.strftime("%B %d, %Y")}</p>',
                unsafe_allow_html=True)

    # --- Allocation detail table ---
    display_cols = ["country", "score", "regime", "weight", "hard_w", "local_w",
                    "local_share", "active_w", "bench_w", "duration_tilt_years"]
    display_cols = [c for c in display_cols if c in latest.columns]
    display = latest[display_cols].copy().rename(columns=COLUMN_LABELS)

    fmt = {
        "Score": "{:+.3f}",
        "Weight": "{:.1%}",
        "Hard Currency Wt": "{:.1%}",
        "Local Currency Wt": "{:.1%}",
        "Local Share": "{:.0%}",
        "Active Weight": "{:+.1%}",
        "Benchmark Wt": "{:.1%}",
        "Duration Tilt (yrs)": "{:+.2f}",
    }
    st.dataframe(
        display.style.format(fmt, na_rep="—").background_gradient(
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

    # --- Butterfly Chart: Portfolio vs Benchmark ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Portfolio vs Benchmark Deviation")
    if "bench_w" in latest.columns and "weight" in latest.columns:
        _but_df = latest[["country", "bench_w", "weight", "score"]].sort_values("bench_w", ascending=True)
        _but_clrs = [score_bar_color(s) for s in _but_df["score"]]
        _but_fig = go.Figure()
        _but_fig.add_trace(go.Bar(
            name="EMBI Benchmark",
            x=-_but_df["bench_w"], y=_but_df["country"], orientation="h",
            marker_color="#4DA3FF", opacity=0.80,
            text=_but_df["bench_w"].apply(lambda v: f"−{v:.1%}"),
            textposition="inside", textfont=dict(size=11, family="JetBrains Mono"),
            customdata=_but_df["bench_w"],
            hovertemplate="<b>%{y}</b><br>EMBI Benchmark: %{customdata:.1%}<extra></extra>",
        ))
        _but_fig.add_trace(go.Bar(
            name="Portfolio Weight",
            x=_but_df["weight"], y=_but_df["country"], orientation="h",
            marker_color=_but_clrs, opacity=0.85,
            text=_but_df["weight"].apply(lambda v: f"{v:.1%}"),
            textposition="inside", textfont=dict(size=11, family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Portfolio Weight: %{x:.1%}<extra></extra>",
        ))
        _but_fig.add_vline(x=0, line_color=COLORS["text"], line_width=1.5)
        _but_fig.update_layout(
            barmode="overlay",
            title="EMBI Benchmark (left, blue) vs Portfolio Weight (right, score-coloured)",
            xaxis_title="Weight", xaxis_tickformat=".1%",
            height=max(380, len(_but_df) * 42),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_but_fig, width="stretch")
        st.caption(
            "Left bars (blue) show each country's EMBI benchmark weight mirrored left from the centre axis. "
            "Right bars show the portfolio weight, coloured green/amber/red by alpha score. "
            "The gap between a country's two bars is the active tilt — wider gap means more conviction relative to the index."
        )

    # --- Local Allocation Gating Scatter ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Local Allocation Gating")
    if "local_share" in latest.columns and "fx_carry" in latest.columns:
        _loc = latest[["country", "local_share", "fx_carry"]].copy()
        _has_fxret = "fx_ret_20d" in latest.columns
        _loc["_fx_ret20"] = latest["fx_ret_20d"].fillna(0.0) if _has_fxret else 0.0
        _loc_sizes = (_loc["local_share"].fillna(0) * 800 + 8).clip(8, 60)
        _loc_clrs = [
            COLORS["green"] if (row["fx_carry"] or 0) > 0 and (row["_fx_ret20"] or 0) > 0
            else COLORS["red"]
            for _, row in _loc.iterrows()
        ]
        _loc_fig = go.Figure()
        _loc_fig.add_trace(go.Scatter(
            x=_loc["fx_carry"].fillna(0),
            y=_loc["_fx_ret20"].fillna(0),
            mode="markers+text", text=_loc["country"],
            textposition="top center", textfont=dict(size=11, color=COLORS["text"]),
            marker=dict(size=_loc_sizes, color=_loc_clrs, opacity=0.80,
                        line=dict(width=1, color=COLORS["border"])),
            customdata=_loc["local_share"].fillna(0),
            hovertemplate=(
                "<b>%{text}</b><br>FX Carry: %{x:.1f}%<br>"
                "FX Return (20d): %{y:.2%}<br>Local Share: %{customdata:.0%}<extra></extra>"
            ),
        ))
        _loc_fig.add_vline(x=0, line_dash="dot", line_color=COLORS["muted"], line_width=1,
                            annotation_text="carry threshold", annotation_font_color=COLORS["muted"],
                            annotation_font_size=10)
        _loc_fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1,
                            annotation_text="FX momentum threshold", annotation_font_color=COLORS["muted"],
                            annotation_font_size=10)
        _loc_fig.update_layout(
            title="FX Carry vs FX Return (20d)  —  bubble size = local currency allocation",
            xaxis_title="FX Carry — local rate minus US rate (%)",
            yaxis_title="FX Return (20d)", yaxis_tickformat=".1%",
            height=420, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_loc_fig, width="stretch")
        st.caption(
            "Countries in the top-right quadrant (positive carry AND positive FX momentum) are eligible "
            "for local currency allocation — the model gates local exposure on both conditions simultaneously. "
            "Bubble size reflects the resulting local share; only top-right countries should have nonzero bubbles."
        )

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

    # Action labels — threshold matches weekly_actions.py (50bp)
    threshold = 0.0050
    snap["action"] = "HOLD"
    snap.loc[snap["w_change"] >= threshold, "action"] = "BUY / ADD"
    snap.loc[snap["w_change"] <= -threshold, "action"] = "SELL / TRIM"

    # Merge score details if available
    if scores is not None and not scores.empty:
        score_cols = ["date", "country", "signal_confidence"]
        # Add any feature columns that exist
        for c in ["hard_spread_proxy", "y10y", "fx_ret_60d",
                   "embi_spread_proxy", "spread_value_blended_z",
                   "fx_carry", "commodity_tot_z",
                   "credit_quality_score"]:
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

    wh_action_col = "cost_aware_action" if "cost_aware_action" in week_data.columns else "action"
    buys = week_data[week_data[wh_action_col].str.contains("BUY", na=False)].sort_values("w_change", ascending=False)
    holds = week_data[week_data[wh_action_col].str.contains("HOLD", na=False)].sort_values("score", ascending=False)
    sells = week_data[week_data[wh_action_col].str.contains("SELL", na=False)].sort_values("w_change", ascending=True)

    col_b, col_h, col_s = st.columns(3)

    with col_b:
        st.markdown(f"**🟢 BUY / ADD** ({len(buys)})")
        if buys.empty:
            st.markdown(f'<span style="color:{COLORS["muted"]}">None</span>', unsafe_allow_html=True)
        for _, row in buys.iterrows():
            chg = row.get("w_change", 0)
            chg_str = f"+{chg:.1%}" if pd.notna(chg) else ""
            tc = row.get("tc_estimate", None)
            tc_str = f' <span style="color:{COLORS["muted"]};font-size:11px;">TC≈{tc*10000:.0f}bp</span>' if pd.notna(tc) and tc else ""
            st.markdown(
                f'{action_badge(row[wh_action_col])} &nbsp; **{row["country"]}** '
                f'<span style="color:{COLORS["green"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                f'{chg_str}</span>{tc_str}',
                unsafe_allow_html=True
            )

    with col_h:
        st.markdown(f"**⚪ HOLD** ({len(holds)})")
        if holds.empty:
            st.markdown(f'<span style="color:{COLORS["muted"]}">None</span>', unsafe_allow_html=True)
        for _, row in holds.iterrows():
            st.markdown(
                f'{action_badge(row[wh_action_col])} &nbsp; **{row["country"]}**',
                unsafe_allow_html=True
            )

    with col_s:
        st.markdown(f"**🔴 SELL / TRIM** ({len(sells)})")
        if sells.empty:
            st.markdown(f'<span style="color:{COLORS["muted"]}">None</span>', unsafe_allow_html=True)
        for _, row in sells.iterrows():
            chg = row.get("w_change", 0)
            chg_str = f"{chg:.1%}" if pd.notna(chg) else ""
            tc = row.get("tc_estimate", None)
            tc_str = f' <span style="color:{COLORS["muted"]};font-size:11px;">TC≈{tc*10000:.0f}bp</span>' if pd.notna(tc) and tc else ""
            st.markdown(
                f'{action_badge(row[wh_action_col])} &nbsp; **{row["country"]}** '
                f'<span style="color:{COLORS["red"]}; font-family: JetBrains Mono, monospace; font-size:13px;">'
                f'{chg_str}</span>{tc_str}',
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
        "country", "score", "regime", "weight", "w_change",
        "action", "cost_aware_action", "tc_estimate",
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
            "TC Estimate": "{:.4f}",
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

    # EMBI benchmark markers (replaces stale equal-weight flat line)
    if "bench_w" in week_sorted.columns:
        fig.add_trace(go.Scatter(
            x=week_sorted["country"],
            y=week_sorted["bench_w"],
            mode="markers",
            name="EMBI Benchmark",
            marker=dict(symbol="diamond", size=10, color=COLORS["muted"]),
        ))

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
            "embi_spread_proxy", "real_yield", "fx_carry",
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
            "yield_coverage_60d", "spread_coverage_60d", "embi_coverage_60d",
            "real_yield_coverage_60d",
            "fx_carry_coverage_60d",
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
def _check_password() -> bool:
    """Return True if the user entered the correct password."""
    pwd = st.text_input("Password", type="password", key="auth_password")
    if not pwd:
        return False
    try:
        correct = st.secrets["PASSWORD"]
    except (KeyError, FileNotFoundError):
        st.error("No PASSWORD set in .streamlit/secrets.toml")
        return False
    if pwd == correct:
        return True
    st.error("Incorrect password")
    return False


def main():
    st.set_page_config(
        page_title="EM Sovereign Alpha",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not _check_password():
        st.stop()

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

# ---------------------------------------------------------------------------
# Tab 6: Macro Forecasts
# ---------------------------------------------------------------------------
def render_macro_forecasts():
    from em.models.macro_forecast import VAR_LABELS

    forecast_dir = DATA_DIR / "macro_forecasts"
    if not forecast_dir.exists():
        st.info("No macro forecast data found. Run `python run/run_macro_forecast.py` first.")
        return

    # Discover available countries from saved files
    available = sorted(set(
        p.stem.replace("_forecast", "").replace("_", " ").title()
        for p in forecast_dir.glob("*_forecast.parquet")
    ))
    if not available:
        st.info("No forecast files found.")
        return

    country = st.selectbox("Country", available, index=0, key="macro_country")
    tag = country.lower().replace(" ", "_")

    forecast_path = forecast_dir / f"{tag}_forecast.parquet"
    irf_path = forecast_dir / f"{tag}_irf.parquet"
    granger_path = forecast_dir / f"{tag}_granger.parquet"
    meta_path = forecast_dir / f"{tag}_meta.json"

    if not forecast_path.exists():
        st.warning(f"No forecast data for {country}.")
        return

    fc = pd.read_parquet(forecast_path)
    fc["date"] = pd.to_datetime(fc["date"])

    # Model metadata
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("VAR lag order", str(meta.get("lag_order", "—")))
        with c2:
            metric_card("AIC", f'{meta.get("aic", 0):.1f}')
        with c3:
            metric_card("BIC", f'{meta.get("bic", 0):.1f}')

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Forecast fan chart ---
    st.markdown("### Forecast fan chart")
    var_options = fc["variable"].unique().tolist()
    display_labels = {v: VAR_LABELS.get(v, v) for v in var_options}
    selected_var = st.radio(
        "Variable", var_options,
        format_func=lambda x: display_labels[x],
        horizontal=True, key="fc_var",
    )

    fc_var = fc[fc["variable"] == selected_var].sort_values("date")
    history = fc_var[fc_var["step"] == 0]
    fwd = fc_var[fc_var["step"] > 0]

    fig_fc = go.Figure()

    # History line
    fig_fc.add_trace(go.Scatter(
        x=history["date"], y=history["mean"],
        mode="lines", name="Actual",
        line=dict(color=COLORS["text"], width=2),
    ))

    if not fwd.empty:
        # 95% band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fwd["date"], fwd["date"][::-1]]),
            y=pd.concat([fwd["upper_95"], fwd["lower_95"][::-1]]),
            fill="toself", fillcolor="rgba(77,163,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True,
        ))
        # 68% band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fwd["date"], fwd["date"][::-1]]),
            y=pd.concat([fwd["upper_68"], fwd["lower_68"][::-1]]),
            fill="toself", fillcolor="rgba(77,163,255,0.25)",
            line=dict(color="rgba(0,0,0,0)"), name="68% CI", showlegend=True,
        ))
        # Forecast mean
        fig_fc.add_trace(go.Scatter(
            x=fwd["date"], y=fwd["mean"],
            mode="lines", name="Forecast",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ))

    fig_fc.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{country} — {display_labels[selected_var]}",
        height=360, legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_fc, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Impulse Response Functions ---
    if irf_path.exists():
        st.markdown("### Impulse response functions")
        st.markdown(f'<p class="section-subtitle">Response to 1 std-dev shock over 12 months</p>',
                    unsafe_allow_html=True)

        irf = pd.read_parquet(irf_path)
        impulse_vars = irf["impulse"].unique().tolist()

        cols = st.columns(len(impulse_vars))
        for i, imp in enumerate(impulse_vars):
            with cols[i]:
                irf_sub = irf[irf["impulse"] == imp]
                fig_irf = go.Figure()
                for resp in irf_sub["response"].unique():
                    data = irf_sub[irf_sub["response"] == resp]
                    fig_irf.add_trace(go.Scatter(
                        x=data["step"], y=data["value"],
                        mode="lines", name=VAR_LABELS.get(resp, resp),
                        line=dict(width=2),
                    ))
                fig_irf.add_hline(y=0, line=dict(color=COLORS["muted"], width=0.5, dash="dot"))
                fig_irf.update_layout(
                    **PLOTLY_LAYOUT,
                    title=f"Shock: {VAR_LABELS.get(imp, imp)}",
                    height=260,
                    showlegend=(i == 0),
                    legend=dict(orientation="h", y=-0.25),
                    xaxis_title="Months",
                )
                st.plotly_chart(fig_irf, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Granger causality table ---
    if granger_path.exists():
        st.markdown("### Granger causality")
        granger = pd.read_parquet(granger_path)
        if not granger.empty:
            granger["cause_label"] = granger["cause"].map(VAR_LABELS).fillna(granger["cause"])
            granger["effect_label"] = granger["effect"].map(VAR_LABELS).fillna(granger["effect"])
            pivot = granger.pivot(index="cause_label", columns="effect_label", values="p_value")

            fig_g = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, COLORS["green"]], [0.05, COLORS["green"]],
                            [0.05, COLORS["muted"]], [1.0, COLORS["muted"]]],
                zmin=0, zmax=0.2,
                text=pivot.round(3).values,
                texttemplate="%{text}",
                textfont=dict(size=13),
                colorbar=dict(title="p-value", tickvals=[0, 0.05, 0.1, 0.2]),
            ))
            fig_g.update_layout(
                **PLOTLY_LAYOUT,
                title="Granger causality p-values (green = significant at 5%)",
                height=260,
                xaxis_title="Effect ←",
                yaxis_title="Cause →",
            )
            st.plotly_chart(fig_g, width="stretch")


# ---------------------------------------------------------------------------
# Tab 7: Data Surprises
# ---------------------------------------------------------------------------
def render_data_surprises():
    surprise_dir = DATA_DIR / "surprises"
    if not surprise_dir.exists():
        st.info("No surprise data found. Run `python run/run_surprise_analysis.py` first.")
        return

    surprise_path = surprise_dir / "surprise_panel.parquet"
    reg_path = surprise_dir / "regression_coefficients.parquet"
    heatmap_path = surprise_dir / "cpi_surprise_heatmap.parquet"

    if not surprise_path.exists():
        st.info("Surprise panel not found.")
        return

    surprise = pd.read_parquet(surprise_path)
    surprise["date"] = pd.to_datetime(surprise["date"])

    # --- Surprise heatmap ---
    st.markdown("### CPI surprise heatmap")
    st.markdown(f'<p class="section-subtitle">Red = hotter than expected, blue = cooler (last 12 months)</p>',
                unsafe_allow_html=True)

    surprise_col = st.radio("Surprise type", ["cpi_surprise", "rate_surprise"],
                            format_func=lambda x: x.replace("_", " ").title(),
                            horizontal=True, key="surprise_type")

    if heatmap_path.exists() and surprise_col == "cpi_surprise":
        heatmap = pd.read_parquet(heatmap_path)
    else:
        # Build on the fly for rate_surprise
        df = surprise.copy()
        df["month_str"] = df["month"].astype(str)
        recent = sorted(df["month_str"].unique())[-12:]
        df = df[df["month_str"].isin(recent)]
        heatmap = df.pivot_table(index="country", columns="month_str", values=surprise_col, aggfunc="last")

    if not heatmap.empty:
        max_abs = max(heatmap.abs().max().max(), 0.1)
        fig_hm = go.Figure(data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.tolist(),
            y=heatmap.index.tolist(),
            colorscale="RdBu_r",
            zmid=0, zmin=-max_abs, zmax=max_abs,
            text=heatmap.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Surprise (pp)"),
        ))
        fig_hm.update_layout(
            **PLOTLY_LAYOUT,
            height=max(300, 30 * len(heatmap)),
            xaxis_title="Month",
        )
        st.plotly_chart(fig_hm, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Surprise → FX scatter ---
    st.markdown("### Surprise → FX reaction")
    st.markdown(f'<p class="section-subtitle">Each dot is one country-month; line = OLS fit</p>',
                unsafe_allow_html=True)

    scatter_data = surprise.dropna(subset=[surprise_col, "fx_ret_5d"]).copy()
    if not scatter_data.empty:
        regime_colors = {"Green": COLORS["green"], "Amber": COLORS["amber"], "Red": COLORS["red"]}

        fig_sc = go.Figure()
        for regime, color in regime_colors.items():
            sub = scatter_data[scatter_data["regime"] == regime]
            if sub.empty:
                continue
            fig_sc.add_trace(go.Scatter(
                x=sub[surprise_col], y=sub["fx_ret_5d"],
                mode="markers", name=regime,
                marker=dict(color=color, size=6, opacity=0.6),
                text=sub["country"],
                hovertemplate="%{text}<br>Surprise: %{x:.2f}<br>FX 5d: %{y:.4f}<extra></extra>",
            ))

        # OLS fit line
        from numpy.polynomial.polynomial import polyfit
        x_all = scatter_data[surprise_col].values
        y_all = scatter_data["fx_ret_5d"].values
        mask = np.isfinite(x_all) & np.isfinite(y_all)
        if mask.sum() > 5:
            b, m = polyfit(x_all[mask], y_all[mask], 1)
            x_line = np.linspace(x_all[mask].min(), x_all[mask].max(), 50)
            fig_sc.add_trace(go.Scatter(
                x=x_line, y=b + m * x_line,
                mode="lines", name="OLS fit",
                line=dict(color=COLORS["accent"], width=2, dash="dash"),
            ))

        fig_sc.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            xaxis_title=surprise_col.replace("_", " ").title() + " (pp)",
            yaxis_title="5-day FX return",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_sc, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Regression coefficients ---
    st.markdown("### Panel regression coefficients")
    st.markdown(f'<p class="section-subtitle">FX 5d return ~ surprise + regime + interactions (HC1 robust SE, country FE)</p>',
                unsafe_allow_html=True)

    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if not reg.empty:
            reg["color"] = reg.apply(
                lambda r: COLORS["green"] if r["significant"] else COLORS["muted"], axis=1
            )
            reg["label"] = reg["variable"].str.replace("_", " ").str.title()
            reg["opacity"] = reg["significant"].map({True: 1.0, False: 0.4})

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(
                y=reg["label"], x=reg["beta"],
                orientation="h",
                marker=dict(color=reg["color"], opacity=reg["opacity"].tolist()),
                text=reg.apply(lambda r: f't={r["t_stat"]:.1f}', axis=1),
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig_reg.add_vline(x=0, line=dict(color=COLORS["muted"], width=0.5))
            fig_reg.update_layout(
                **PLOTLY_LAYOUT,
                height=260,
                xaxis_title="Coefficient (β)",
                showlegend=False,
            )
            st.plotly_chart(fig_reg, width="stretch")


# ---------------------------------------------------------------------------
# Tab 8: Research
# ---------------------------------------------------------------------------
def render_research():
    from em.models.macro_forecast import VAR_LABELS

    scored = load_parquet("country_scores_daily.parquet")
    portfolio = load_parquet("portfolio_daily.parquet")

    if scored is None:
        st.info("No scored data found. Run the pipeline first.")
        return

    latest_date = scored["date"].max()
    latest = scored[scored["date"] == latest_date].sort_values("score", ascending=False)

    country = st.selectbox("Country deep dive", latest["country"].tolist(), index=0, key="research_country")
    row = latest[latest["country"] == country].iloc[0]

    # --- Signal decomposition bar chart ---
    st.markdown("### Signal decomposition")
    st.markdown(f'<p class="section-subtitle">Contribution of each signal to {country}\'s composite score</p>',
                unsafe_allow_html=True)

    signal_map = {
        "Spread value": "spread_value_blended_z",
        "Spread momentum": "spread_mom_blend_z",
        "FX carry": "fx_carry_z",
        "Real yield": "real_yield_z",
        "Local return (60d)": "local_ret_60d_z",
        "FX momentum (60d)": "fx_ret_60d_z",
        "Commodity ToT": "commodity_tot_z",
        "US real rate tilt": "us_real_rate_tilt",
    }
    weights = {
        "Spread value": 0.15,
        "Spread momentum": -0.15,  # negative because widening = bad
        "FX carry": 0.15,
        "Real yield": 0.10,
        "Local return (60d)": 0.10,
        "FX momentum (60d)": 0.05,
        "Commodity ToT": 0.10,
        "US real rate tilt": 1.0,  # already scaled in score.py
    }

    decomp_rows = []
    for label, col in signal_map.items():
        if col in row.index:
            raw_val = float(row[col]) if pd.notna(row[col]) else 0.0
            w = weights.get(label, 1.0)
            contribution = raw_val * w if label != "US real rate tilt" else raw_val
            decomp_rows.append({"signal": label, "contribution": contribution})

    decomp = pd.DataFrame(decomp_rows).sort_values("contribution")

    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Bar(
        y=decomp["signal"],
        x=decomp["contribution"],
        orientation="h",
        marker=dict(
            color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in decomp["contribution"]],
        ),
        text=decomp["contribution"].apply(lambda v: f"{v:+.3f}"),
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_decomp.add_vline(x=0, line=dict(color=COLORS["muted"], width=0.5))
    fig_decomp.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        xaxis_title="Contribution to score",
        showlegend=False,
    )
    st.plotly_chart(fig_decomp, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Auto-generated research card ---
    st.markdown("### Country research card")

    score = float(row["score"]) if pd.notna(row.get("score")) else 0.0
    confidence = float(row["signal_confidence"]) if pd.notna(row.get("signal_confidence")) else 0.0
    credit_q = float(row["credit_quality_score"]) if pd.notna(row.get("credit_quality_score")) else 0.0

    # Portfolio weight and regime
    port_weight = "—"
    regime = "—"
    action = "—"
    if portfolio is not None:
        port_latest = portfolio[
            (portfolio["date"] == portfolio["date"].max()) & (portfolio["country"] == country)
        ]
        if not port_latest.empty:
            port_weight = f'{float(port_latest.iloc[0]["weight"]) * 100:.1f}%'
            regime = str(port_latest.iloc[0].get("regime", "—"))

    # Top contributing signals
    if decomp_rows:
        sorted_signals = sorted(decomp_rows, key=lambda x: abs(x["contribution"]), reverse=True)
        top_2 = sorted_signals[:2]
        top_drivers = ", ".join(
            f'{s["signal"]} ({s["contribution"]:+.3f})' for s in top_2
        )
    else:
        top_drivers = "—"

    # VAR forecast direction (if available)
    forecast_note = ""
    fc_dir = DATA_DIR / "macro_forecasts"
    tag = country.lower().replace(" ", "_")
    fc_path = fc_dir / f"{tag}_forecast.parquet"
    if fc_path.exists():
        fc = pd.read_parquet(fc_path)
        fc_fwd = fc[fc["step"] > 0]
        if not fc_fwd.empty:
            cpi_fc = fc_fwd[fc_fwd["variable"] == "cpi_yoy"]
            rate_fc = fc_fwd[fc_fwd["variable"] == "local_short_rate"]
            parts = []
            if not cpi_fc.empty:
                cpi_dir = "rising" if cpi_fc["mean"].iloc[-1] > cpi_fc["mean"].iloc[0] else "falling"
                parts.append(f"CPI {cpi_dir}")
            if not rate_fc.empty:
                rate_dir = "rising" if rate_fc["mean"].iloc[-1] > rate_fc["mean"].iloc[0] else "falling"
                parts.append(f"rates {rate_dir}")
            if parts:
                forecast_note = f"**VAR 6m outlook:** {', '.join(parts)}"

    # Latest surprise (if available)
    surprise_note = ""
    sp_path = DATA_DIR / "surprises" / "surprise_panel.parquet"
    if sp_path.exists():
        sp = pd.read_parquet(sp_path)
        sp_country = sp[sp["country"] == country].sort_values("date")
        if not sp_country.empty:
            last_sp = sp_country.iloc[-1]
            cpi_s = last_sp.get("cpi_surprise")
            if pd.notna(cpi_s):
                direction = "hotter" if cpi_s > 0 else "cooler"
                surprise_note = f"**Latest CPI surprise:** {cpi_s:+.2f}pp ({direction} than prior)"

    card_md = f"""
| Metric | Value |
|--------|-------|
| **Score** | {score:+.3f} |
| **Signal confidence** | {confidence:.0%} |
| **Credit quality** | {credit_q:.2f} |
| **Regime** | {regime} |
| **Portfolio weight** | {port_weight} |
| **Top drivers** | {top_drivers} |

{forecast_note}

{surprise_note}
"""
    st.markdown(card_md)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Fiscal snapshot table ---
    st.markdown("### Fiscal snapshot")
    st.markdown(f'<p class="section-subtitle">World Bank annual data (latest available, forward-filled)</p>',
                unsafe_allow_html=True)

    from em.country.universe import SOVEREIGN_RATINGS

    fiscal_cols = ["country", "debt_gdp", "fiscal_balance_gdp", "reserves_months",
                   "credit_quality_score"]
    available_cols = [c for c in fiscal_cols if c in latest.columns]

    if len(available_cols) > 1:
        fiscal_table = latest[available_cols].copy()
        fiscal_table["rating"] = fiscal_table["country"].map(SOVEREIGN_RATINGS).fillna("—")

        rename = {
            "debt_gdp": "Debt/GDP (%)",
            "fiscal_balance_gdp": "Fiscal Bal/GDP (%)",
            "reserves_months": "Reserves (months)",
            "credit_quality_score": "Quality Score",
            "rating": "Rating (numeric)",
        }
        fiscal_table = fiscal_table.rename(columns=rename)
        fiscal_table = fiscal_table.set_index("country")

        # Format
        for col in fiscal_table.columns:
            fiscal_table[col] = fiscal_table[col].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—"
            )

        st.dataframe(fiscal_table, width="stretch")

if __name__ == "__main__":
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Executive Summary",
        "🎯 Country Scores",
        "💼 Portfolio Detail",
        "🕐 Weekly History",
        "📈 Market Data & Coverage",
        "🔮 Macro Forecasts",
        "⚡ Data Surprises",
        "📝 Research",
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
    with tab6:
        render_macro_forecasts()
    with tab7:
        render_data_surprises()
    with tab8:
        render_research()