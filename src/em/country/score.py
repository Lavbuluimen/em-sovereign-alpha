from __future__ import annotations

import numpy as np
import pandas as pd

from em.country.universe import COMMODITY_SENSITIVITY, SOVEREIGN_RATINGS


# ── Cross-sectional z-score helpers ───────────────────────────────────────────

def zscore_cross_section(df: pd.DataFrame, col: str) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        mu = x.mean()
        sd = x.std()
        if sd is None or sd == 0 or np.isnan(sd):
            return x * 0.0
        return (x - mu) / sd
    return df.groupby("date")[col].transform(_z)


def winsorized_zscore_cross_section(
    df: pd.DataFrame, col: str, clip: float = 3.0
) -> pd.Series:
    """Cross-sectional z-score winsorized to [-clip, +clip] then rescaled to [-1, +1].

    Preserves signal magnitude (a 3-sigma outlier scores ±1.0; a 1-sigma outlier
    scores ±0.33), unlike pct_rank which discards magnitude information entirely.
    """
    def _wz(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        mu, sd = x.mean(), x.std()
        if sd is None or sd == 0 or np.isnan(sd):
            return x * 0.0
        return ((x - mu) / sd).clip(-clip, clip) / clip
    return df.groupby("date")[col].transform(_wz)


def _ts_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Time-series (rolling) z-score for a single date-indexed Series."""
    mu = series.rolling(window, min_periods=min_periods).mean()
    sd = series.rolling(window, min_periods=min_periods).std()
    z = (series - mu) / sd.replace(0, np.nan)
    return z.fillna(0.0).clip(-3.0, 3.0)


# ── Main scoring function ──────────────────────────────────────────────────────

def build_country_scores(
    panel: pd.DataFrame,
    macro_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute composite sovereign alpha scores from the country daily panel.

    Parameters
    ----------
    panel:
        Country panel produced by build_country_panel.py. Must contain at
        minimum: date, country, y10y, hard_spread_proxy, embi_spread_proxy,
        embi_spread_20d_chg, fx_usd_ret, local_ret_proxy_usd, us_real_yield.
        Carry/value signals (real_yield, fx_carry) are optional — countries
        missing these get neutral z-scores and reduced signal_confidence.
        Fiscal fundamentals (fiscal_balance_gdp, debt_gdp, reserves_months) are
        optional — used by the credit quality guard; default to conservative
        fallbacks if absent.
    macro_panel:
        Global macro panel produced by build_country_panel.py. Used to extract
        Brent prices (commodity ToT signal) and the US real rate time series.
        Optional — signals that depend on it fall back to 0 if not provided.

    Returns
    -------
    pd.DataFrame
        Input panel with additional scoring columns appended.
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"])

    # ── 0. Local return fallback ───────────────────────────────────────────────
    df["local_ret_fallback"] = df["local_ret_proxy_usd"].where(
        df["local_ret_proxy_usd"].notna(), df["fx_usd_ret"]
    )

    # ── 1. Multi-horizon momentum windows ─────────────────────────────────────
    # Spread momentum — 20d / 60d / 120d
    for days in (20, 60, 120):
        df[f"spread_{days}d_chg"] = df.groupby("country")["hard_spread_proxy"].transform(
            lambda x, d=days: x.diff(d)
        )

    # Local return momentum — 20d (Phase 1 carry-over) and 60d (Phase 2 primary)
    df["local_ret_20d"] = df.groupby("country")["local_ret_fallback"].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    df["local_ret_60d"] = df.groupby("country")["local_ret_fallback"].transform(
        lambda x: x.rolling(60, min_periods=30).sum()
    )

    # FX momentum — 20d (carry-over) and 60d (primary)
    df["fx_ret_20d"] = df.groupby("country")["fx_usd_ret"].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    df["fx_ret_60d"] = df.groupby("country")["fx_usd_ret"].transform(
        lambda x: x.rolling(60, min_periods=30).sum()
    )

    # Yield change — kept for backward compatibility (used by allocator duration tilt)
    df["yield_60d_chg"] = df.groupby("country")["y10y"].transform(lambda x: x.diff(60))

    # ── 2. Cross-sectional z-scores ────────────────────────────────────────────
    cs_cols = [
        "spread_20d_chg", "spread_60d_chg", "spread_120d_chg",
        "local_ret_60d", "fx_ret_60d",
        "hard_spread_proxy",
    ]
    for col in cs_cols:
        df[f"{col}_z"] = zscore_cross_section(df, col).fillna(0.0)

    # Carry/value signals — NaN for countries with incomplete data (e.g. Brazil,
    # Colombia, China pre-IFS coverage); fillna(0.0) gives neutral score on those signals.
    for col in ("real_yield", "fx_carry"):
        if col in df.columns:
            df[f"{col}_z"] = zscore_cross_section(df, col).fillna(0.0)
        else:
            df[f"{col}_z"] = 0.0

    # ── 3. Blended multi-horizon spread momentum ───────────────────────────────
    df["spread_mom_blend_z"] = (
        0.25 * df["spread_20d_chg_z"] +
        0.50 * df["spread_60d_chg_z"] +
        0.25 * df["spread_120d_chg_z"]
    )

    # ── 4. Time-series spread z-score (per-country 2-year rolling) ─────────────
    df["spread_ts_z"] = df.groupby("country")["hard_spread_proxy"].transform(
        lambda x: (
            (x - x.rolling(504, min_periods=120).mean())
            / x.rolling(504, min_periods=120).std().replace(0, np.nan)
        )
    ).fillna(0.0).clip(-3.0, 3.0)

    # Blended spread value: 50% cross-sectional (cheap vs peers) +
    #                       50% time-series (cheap vs own history)
    df["spread_value_blended_z"] = (
        0.50 * df["hard_spread_proxy_z"] +
        0.50 * df["spread_ts_z"]
    )

    # ── 5. Credit quality guard ────────────────────────────────────────────────
    # Composite credit quality score [0, 1] combining four dimensions.
    # (a) Ratings
    MAX_RATING = 20.0
    df["rating_norm"] = df["country"].map(SOVEREIGN_RATINGS).fillna(5.0) / MAX_RATING

    # (b–d) Fiscal fundamentals — use conservative defaults if columns absent
    fiscal_bal = df["fiscal_balance_gdp"] if "fiscal_balance_gdp" in df.columns else pd.Series(-3.0, index=df.index)
    debt_gdp   = df["debt_gdp"]           if "debt_gdp"           in df.columns else pd.Series(60.0, index=df.index)
    reserves   = df["reserves_months"]    if "reserves_months"    in df.columns else pd.Series(3.0,  index=df.index)

    df["fiscal_norm"]   = ((fiscal_bal.fillna(-3.0) + 10.0) / 20.0).clip(0.0, 1.0)
    df["debt_norm"]     = (1.0 - (debt_gdp.fillna(60.0) / 150.0)).clip(0.0, 1.0)
    df["reserves_norm"] = (reserves.fillna(3.0) / 12.0).clip(0.0, 1.0)

    df["credit_quality_score"] = (
        0.30 * df["rating_norm"]  +
        0.25 * df["fiscal_norm"]  +
        0.25 * df["debt_norm"]    +
        0.20 * df["reserves_norm"]
    )

    # Guard: zero out the positive spread value contribution when:
    #   - spread > 700 bp (absolute level too wide — likely distressed), OR
    #   - composite credit quality score < 0.30 (poor quality across all dimensions)
    spread_too_wide = df["hard_spread_proxy"] > 7.0
    low_quality     = df["credit_quality_score"] < 0.30
    cap_positive    = spread_too_wide | low_quality

    # Apply guard to both the cross-sectional z-score and the blended value signal
    df["hard_spread_proxy_z"] = df["hard_spread_proxy_z"].where(
        ~(cap_positive & (df["hard_spread_proxy_z"] > 0)), 0.0
    )
    df["spread_value_blended_z"] = df["spread_value_blended_z"].where(
        ~(cap_positive & (df["spread_value_blended_z"] > 0)), 0.0
    )

    # ── 6. Commodity terms-of-trade signal ────────────────────────────────────
    # Brent 60d return time-series z-score (same for all countries on a given date),
    # scaled by each country's commodity sensitivity coefficient.
    if macro_panel is not None and "Brent" in macro_panel.columns:
        mp = macro_panel.copy()
        mp["date"] = pd.to_datetime(mp["date"])
        brent = mp.set_index("date")["Brent"].sort_index()
        brent_60d_ret = brent.pct_change(60)
        brent_z_ts = _ts_zscore(brent_60d_ret, window=252, min_periods=60)
        df["brent_z"] = df["date"].map(brent_z_ts).fillna(0.0)
    else:
        df["brent_z"] = 0.0

    df["commodity_sensitivity"] = df["country"].map(COMMODITY_SENSITIVITY).fillna(0.0)
    df["commodity_tot"] = df["commodity_sensitivity"] * df["brent_z"]
    df["commodity_tot_z"] = winsorized_zscore_cross_section(df, "commodity_tot").fillna(0.0)

    # ── 7. Score raw (cross-sectional signals, 80% of final weight budget) ─────
    #
    # Final target weights (Post Phase 1–2, per evaluation Section 8):
    #   spread_value_blended_z : 15%   (incl. quality guard + TS blend)
    #   spread_mom_blend_z     : 15%   (multi-horizon 20d/60d/120d)
    #   fx_carry_z             : 15%   (0 for 4 zero-coverage countries)
    #   real_yield_z           : 10%   (0 for 4 zero-coverage countries)
    #   local_ret_60d_z        : 10%
    #   fx_ret_60d_z           :  5%
    #   commodity_tot_z        : 10%
    #   ─────────────────────────────
    #   subtotal               : 80%  (cross-sectional)
    #   us_real_rate_tilt      : 15%  (global — applied after winsorized z-score)
    #   risk_regime_penalty    :  5%  (global — applied in allocator)
    #   ─────────────────────────────
    #   total                  : 100%
    df["score_raw"] = (
        0.15 * df["spread_value_blended_z"]  +
        0.15 * (-df["spread_mom_blend_z"])   +  # widening = bad
        0.15 * df["fx_carry_z"]              +
        0.10 * df["real_yield_z"]            +
        0.10 * df["local_ret_60d_z"]         +
        0.05 * df["fx_ret_60d_z"]            +
        0.10 * df["commodity_tot_z"]
    )

    # ── 8. Signal coverage flags ───────────────────────────────────────────────
    df["has_yield_data"]      = df["y10y"].notna().astype(float)
    df["has_spread_data"]     = df["hard_spread_proxy"].notna().astype(float)
    df["has_embi_data"]       = df["embi_spread_proxy"].notna().astype(float)
    df["has_real_yield_data"] = df["real_yield"].notna().astype(float) if "real_yield" in df.columns else 0.0
    df["has_fx_carry_data"]   = df["fx_carry"].notna().astype(float)   if "fx_carry"   in df.columns else 0.0

    for flag, col in [
        ("yield_coverage_60d",      "has_yield_data"),
        ("spread_coverage_60d",     "has_spread_data"),
        ("embi_coverage_60d",       "has_embi_data"),
        ("real_yield_coverage_60d", "has_real_yield_data"),
        ("fx_carry_coverage_60d",   "has_fx_carry_data"),
    ]:
        df[flag] = df.groupby("country")[col].transform(
            lambda x: x.rolling(60, min_periods=20).mean()
        ).fillna(0.0)

    # Signal confidence weights mirror score_raw weight proportions so that
    # each missing signal degrades confidence proportionally.
    # Countries with partial data (e.g. Brazil, Colombia, China): real_yield + fx_carry = 0
    #   → signal_confidence ≤ 0.70 → score naturally shrunk → smaller active tilts.
    df["signal_confidence"] = (
        0.30 * df["yield_coverage_60d"]      +
        0.20 * df["spread_coverage_60d"]     +
        0.20 * df["embi_coverage_60d"]       +
        0.15 * df["real_yield_coverage_60d"] +
        0.15 * df["fx_carry_coverage_60d"]
    ).clip(0.0, 1.0)

    # ── 9. Winsorized z-score (replaces pct_rank — preserves signal magnitude) ─
    df["score_scaled"] = winsorized_zscore_cross_section(df, "score_raw").fillna(0.0)

    # ── 10. US real rate global tilt (15% weight, applied post-scaling) ────────
    # Rising US real rates = opportunity cost for EM risk = negative tilt for all.
    # Applied as an additive term AFTER winsorized z-score so it shifts all
    # country scores by the same amount (does not affect relative ranking, but
    # reduces absolute score magnitude when US rates are rising).
    if "us_real_yield" in df.columns:
        # One time-series value per date (identical across all countries)
        us_ry_ts = (
            df[["date", "us_real_yield"]]
            .drop_duplicates("date")
            .set_index("date")["us_real_yield"]
            .sort_index()
        )
        us_ry_60d_chg = us_ry_ts.diff(60)
        us_real_rate_z_ts = _ts_zscore(us_ry_60d_chg, window=252, min_periods=60)
        df["us_real_rate_z"]    = df["date"].map(us_real_rate_z_ts).fillna(0.0)
        df["us_real_rate_tilt"] = -0.15 * df["us_real_rate_z"]
    else:
        df["us_real_rate_z"]    = 0.0
        df["us_real_rate_tilt"] = 0.0

    # ── 11. Final score ────────────────────────────────────────────────────────
    df["score"] = (
        (df["score_scaled"] + df["us_real_rate_tilt"]) * df["signal_confidence"]
    ).clip(-1.0, 1.0)

    return df
