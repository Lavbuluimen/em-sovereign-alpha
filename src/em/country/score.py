from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_cross_section(df: pd.DataFrame, col: str) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        mu = x.mean()
        sd = x.std()
        if sd is None or sd == 0 or np.isnan(sd):
            return x * 0.0
        return (x - mu) / sd
    return df.groupby("date")[col].transform(_z)


def pct_rank_cross_section(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].transform(lambda x: x.rank(pct=True))


def build_country_scores(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"])

    df["local_ret_fallback"] = df["local_ret_proxy_usd"].where(
        df["local_ret_proxy_usd"].notna(),
        df["fx_usd_ret"],
    )

    df["local_ret_20d"] = df.groupby("country")["local_ret_fallback"].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    df["fx_ret_20d"] = df.groupby("country")["fx_usd_ret"].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    df["yield_60d_chg"] = df.groupby("country")["y10y"].transform(
        lambda x: x.diff(60)
    )
    feature_cols = [
        "local_ret_20d",
        "fx_ret_20d",
        "yield_60d_chg",
        "hard_spread_proxy",
        "embi_spread_20d_chg",
    ]

    for col in feature_cols:
        df[f"{col}_z"] = zscore_cross_section(df, col).fillna(0.0)

    # Sovereign alpha score using EMBI/yield-spread signals.
    # Weights:
    #   25% spread level    — cross-sectional cheapness of credit
    #   20% local return    — FX + duration momentum
    #   15% yield trend     — rising yields = deterioration (negative)
    #   25% spread momentum — spread widening = deterioration (negative)
    #   15% FX return       — currency momentum
    df["score_raw"] = (
        0.25 * df["hard_spread_proxy_z"] +
        0.20 * df["local_ret_20d_z"] +
        0.15 * (-df["yield_60d_chg_z"]) +
        0.25 * (-df["embi_spread_20d_chg_z"]) +
        0.15 * df["fx_ret_20d_z"]
    )

    # Availability flags
    df["has_yield_data"] = df["y10y"].notna().astype(float)
    df["has_spread_data"] = df["hard_spread_proxy"].notna().astype(float)
    df["has_embi_data"] = df["embi_spread_proxy"].notna().astype(float)

    # Rolling coverage (60-day windows)
    df["yield_coverage_60d"] = df.groupby("country")["has_yield_data"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    ).fillna(0.0)

    df["spread_coverage_60d"] = df.groupby("country")["has_spread_data"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    ).fillna(0.0)

    df["embi_coverage_60d"] = df.groupby("country")["has_embi_data"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    ).fillna(0.0)

    # Overall confidence: 40% yield, 30% spread, 30% EMBI proxy
    df["signal_confidence"] = (
        0.4 * df["yield_coverage_60d"] +
        0.3 * df["spread_coverage_60d"] +
        0.3 * df["embi_coverage_60d"]
    ).clip(0.0, 1.0)

    df["score_pct"] = pct_rank_cross_section(df, "score_raw").fillna(0.5)
    df["score_scaled"] = (2.0 * df["score_pct"] - 1.0).clip(-1.0, 1.0)

    df["score"] = df["score_scaled"] * df["signal_confidence"]

    return df