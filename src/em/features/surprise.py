"""
Macro data surprise analysis for EM sovereign countries.

Computes CPI and rate surprises (actual minus naive forecast), then tests
whether surprises predict FX and spread movements via panel regression.

All inputs come from country_daily.parquet and portfolio_daily.parquet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class SurpriseResult:
    surprise_panel: pd.DataFrame    # monthly: date, country, cpi_surprise, rate_surprise, fx_ret_5d, spread_chg_5d, regime
    regression: pd.DataFrame        # coefficients: variable, beta, t_stat, p_value, significant
    heatmap_data: pd.DataFrame      # pivot: index=country, columns=month, values=surprise


def _compute_monthly_surprises(
    country_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly CPI and rate surprises per country.

    Surprise = actual_t - actual_{t-1} (random walk naive forecast).
    Also captures the 5-day forward FX return and spread change after each
    month-end observation date.
    """
    df = country_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"])

    # Resample to monthly (last observation per country-month)
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby(["country", "month"])
        .agg({
            "date": "last",
            "cpi_yoy": "last",
            "local_short_rate": "last",
            "fx_usd_ret": lambda x: x.tail(5).sum(),          # 5-day forward FX return
            "hard_spread_proxy": "last",
        })
        .reset_index()
    )
    monthly = monthly.sort_values(["country", "month"])

    # Surprises: change from prior month
    monthly["cpi_surprise"] = monthly.groupby("country")["cpi_yoy"].diff()
    monthly["rate_surprise"] = monthly.groupby("country")["local_short_rate"].diff()

    # 5-day forward spread change (proxy for market reaction)
    monthly["spread_chg_5d"] = monthly.groupby("country")["hard_spread_proxy"].diff()

    # FX return is already a 5-day sum from the agg above
    monthly = monthly.rename(columns={"fx_usd_ret": "fx_ret_5d"})

    monthly = monthly.dropna(subset=["cpi_surprise"])
    return monthly[["date", "country", "month", "cpi_surprise", "rate_surprise",
                     "fx_ret_5d", "spread_chg_5d"]]


def _merge_regime(
    surprise_panel: pd.DataFrame,
    portfolio_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add regime column from portfolio panel (nearest date match)."""
    if portfolio_panel is None or "regime" not in portfolio_panel.columns:
        surprise_panel["regime"] = "Green"
        return surprise_panel

    port = portfolio_panel.copy()
    port["date"] = pd.to_datetime(port["date"])

    # Get one regime per date (they're identical across countries on a given date)
    regime_daily = (
        port.groupby("date")["regime"]
        .first()
        .reset_index()
    )

    # Merge on nearest date
    surprise_panel = surprise_panel.sort_values("date")
    regime_daily = regime_daily.sort_values("date")
    surprise_panel = pd.merge_asof(
        surprise_panel, regime_daily,
        on="date", direction="nearest", tolerance=pd.Timedelta("7D"),
    )
    surprise_panel["regime"] = surprise_panel["regime"].fillna("Green")
    return surprise_panel


def _run_panel_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Panel regression: fx_ret_5d ~ cpi_surprise + rate_surprise + regime dummies + interactions.

    Returns coefficient table.
    """
    reg = df.dropna(subset=["cpi_surprise", "rate_surprise", "fx_ret_5d"]).copy()
    if len(reg) < 20:
        return pd.DataFrame(columns=["variable", "beta", "t_stat", "p_value", "significant"])

    # Regime dummies (Green is base)
    reg["regime_amber"] = (reg["regime"] == "Amber").astype(float)
    reg["regime_red"] = (reg["regime"] == "Red").astype(float)

    # Interaction terms
    reg["cpi_x_amber"] = reg["cpi_surprise"] * reg["regime_amber"]
    reg["cpi_x_red"] = reg["cpi_surprise"] * reg["regime_red"]

    # Country fixed effects
    country_dummies = pd.get_dummies(reg["country"], prefix="fe", drop_first=True, dtype=float)

    X_cols = ["cpi_surprise", "rate_surprise", "regime_amber", "regime_red",
              "cpi_x_amber", "cpi_x_red"]
    X = pd.concat([reg[X_cols], country_dummies], axis=1)
    X = sm.add_constant(X)
    y = reg["fx_ret_5d"]

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")  # heteroskedasticity-robust SEs
    except Exception:
        return pd.DataFrame(columns=["variable", "beta", "t_stat", "p_value", "significant"])

    # Extract main coefficients (skip country FE and constant for display)
    display_vars = ["cpi_surprise", "rate_surprise", "regime_amber", "regime_red",
                    "cpi_x_amber", "cpi_x_red"]
    rows = []
    for var in display_vars:
        if var in model.params.index:
            rows.append({
                "variable": var,
                "beta": round(float(model.params[var]), 6),
                "t_stat": round(float(model.tvalues[var]), 2),
                "p_value": round(float(model.pvalues[var]), 4),
                "significant": model.pvalues[var] < 0.05,
            })
    return pd.DataFrame(rows)


def _build_heatmap(surprise_panel: pd.DataFrame, col: str = "cpi_surprise") -> pd.DataFrame:
    """Pivot surprise data into a heatmap: countries x months."""
    df = surprise_panel.copy()
    df["month_str"] = df["month"].astype(str)
    # Last 12 months
    recent_months = sorted(df["month_str"].unique())[-12:]
    df = df[df["month_str"].isin(recent_months)]
    pivot = df.pivot_table(index="country", columns="month_str", values=col, aggfunc="last")
    return pivot


def run_surprise_analysis(
    country_panel: pd.DataFrame,
    portfolio_panel: pd.DataFrame | None = None,
) -> SurpriseResult:
    """Run full surprise analysis.

    Parameters
    ----------
    country_panel : pd.DataFrame
        The country_daily.parquet panel.
    portfolio_panel : pd.DataFrame | None
        The portfolio_daily.parquet panel (for regime data).

    Returns
    -------
    SurpriseResult
    """
    surprise = _compute_monthly_surprises(country_panel)
    surprise = _merge_regime(surprise, portfolio_panel)
    regression = _run_panel_regression(surprise)
    heatmap = _build_heatmap(surprise, "cpi_surprise")

    return SurpriseResult(
        surprise_panel=surprise,
        regression=regression,
        heatmap_data=heatmap,
    )