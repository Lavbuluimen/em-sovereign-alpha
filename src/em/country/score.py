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

    # Fallback return if local bond return proxy is missing
    df["local_ret_fallback"] = df["local_ret_proxy_usd"].where(
        df["local_ret_proxy_usd"].notna(),
        df["fx_usd_ret"],
    )

    # Momentum features
    df["local_ret_20d"] = df.groupby("country")["local_ret_fallback"].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    df["local_ret_60d"] = df.groupby("country")["local_ret_fallback"].transform(
        lambda x: x.rolling(60, min_periods=30).sum()
    )
    df["fx_ret_20d"] = df.groupby("country")["fx_usd_ret"].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )

    # Trend / valuation features
    df["spread_20d_chg"] = df.groupby("country")["hard_spread_proxy"].transform(
        lambda x: x.diff(20)
    )
    df["yield_60d_chg"] = df.groupby("country")["y10y"].transform(
        lambda x: x.diff(60)
    )

    feature_cols = [
        "local_ret_20d",
        "local_ret_60d",
        "fx_ret_20d",
        "spread_20d_chg",
        "yield_60d_chg",
        "hard_spread_proxy",
    ]

    for col in feature_cols:
        df[f"{col}_z"] = zscore_cross_section(df, col).fillna(0.0)

    # Raw composite score
    df["score_raw"] = (
        0.35 * df["hard_spread_proxy_z"] +
        0.35 * df["local_ret_20d_z"] +
        0.30 * (-df["yield_60d_chg_z"])
    )

    # Normalize to PM-friendly scale
    df["score_pct"] = pct_rank_cross_section(df, "score_raw").fillna(0.5)
    df["score_scaled"] = (2.0 * df["score_pct"] - 1.0).clip(-1.0, 1.0)

    # Default downstream score
    df["score"] = df["score_scaled"]

    return df