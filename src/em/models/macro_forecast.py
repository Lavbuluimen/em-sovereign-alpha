"""
VAR-based macro forecasting for EM sovereign countries.

Fits a Vector Autoregression on monthly CPI, short-term interest rate, and FX
for a given country. Produces:
  - h-step-ahead forecasts with confidence bands
  - impulse response functions
  - Granger causality p-value matrix

Uses statsmodels VAR. All inputs come from the existing country_daily.parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class MacroForecastResult:
    country: str
    variables: list[str]
    forecast: pd.DataFrame          # columns: variable, step, mean, lower_68, upper_68, lower_95, upper_95
    irf: pd.DataFrame               # columns: impulse, response, step, value
    granger: pd.DataFrame           # columns: cause, effect, f_stat, p_value, significant
    lag_order: int
    aic: float
    bic: float
    stationarity: pd.DataFrame      # columns: variable, adf_stat, p_value, stationary, differenced


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resample_to_monthly(
    country_panel: pd.DataFrame,
    country: str,
    cols: list[str],
) -> pd.DataFrame:
    """Extract one country, resample to month-end, forward-fill gaps."""
    df = country_panel[country_panel["country"] == country].copy()
    df = df.set_index("date").sort_index()
    monthly = df[cols].resample("ME").last().dropna(how="all")
    return monthly.ffill().dropna()


def _check_stationarity(series: pd.Series, name: str, alpha: float = 0.05) -> dict:
    """ADF test; return dict with test result."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "variable": name,
        "adf_stat": round(float(result[0]), 4),
        "p_value": round(float(result[1]), 4),
        "stationary": result[1] < alpha,
        "differenced": False,
    }


def _ensure_stationary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Difference non-stationary columns. Return (transformed df, stationarity records)."""
    records = []
    out = df.copy()
    for col in df.columns:
        rec = _check_stationarity(df[col], col)
        if not rec["stationary"]:
            out[col] = df[col].diff()
            rec_diff = _check_stationarity(out[col].dropna(), col)
            rec_diff["differenced"] = True
            records.append(rec_diff)
        else:
            records.append(rec)
    out = out.dropna()
    return out, records


# ── Main function ────────────────────────────────────────────────────────────

VAR_COLS = ["cpi_yoy", "local_short_rate", "fx_level_local_per_usd"]
VAR_LABELS = {"cpi_yoy": "CPI YoY (%)", "local_short_rate": "Policy Rate (%)", "fx_level_local_per_usd": "FX (local/USD)"}


def run_macro_forecast(
    country_panel: pd.DataFrame,
    country: str = "Brazil",
    max_lags: int = 6,
    forecast_steps: int = 6,
    irf_steps: int = 12,
    var_cols: list[str] | None = None,
) -> MacroForecastResult:
    """Run full VAR analysis for one country.

    Parameters
    ----------
    country_panel : pd.DataFrame
        The country_daily.parquet panel.
    country : str
        Country name matching the panel.
    max_lags : int
        Maximum lag order to test; best selected by AIC.
    forecast_steps : int
        Number of months ahead to forecast.
    irf_steps : int
        Number of months for impulse response functions.
    var_cols : list[str] | None
        Columns to include in the VAR. Defaults to VAR_COLS.

    Returns
    -------
    MacroForecastResult
    """
    if var_cols is None:
        var_cols = list(VAR_COLS)

    # 1. Resample to monthly
    monthly = _resample_to_monthly(country_panel, country, var_cols)
    if len(monthly) < 36:
        raise ValueError(f"Insufficient data for {country}: {len(monthly)} months (need ≥36)")

    # 2. Stationarity check + differencing
    monthly_transformed, stationarity_records = _ensure_stationary(monthly)
    stationarity_df = pd.DataFrame(stationarity_records)

    # 3. Fit VAR, select lag order by AIC
    model = VAR(monthly_transformed)
    lag_results = model.select_order(maxlags=min(max_lags, len(monthly_transformed) // 3 - 1))
    best_lag = max(lag_results.aic, 1)  # at least 1 lag

    fitted = model.fit(best_lag)

    # 4. Forecast with confidence intervals
    fc = fitted.forecast_interval(
        monthly_transformed.values[-best_lag:],
        steps=forecast_steps,
        alpha=0.32,  # 68% CI
    )
    fc_mean, fc_lower_68, fc_upper_68 = fc

    fc_95 = fitted.forecast_interval(
        monthly_transformed.values[-best_lag:],
        steps=forecast_steps,
        alpha=0.05,  # 95% CI
    )
    _, fc_lower_95, fc_upper_95 = fc_95

    # Build forecast dataframe
    last_date = monthly_transformed.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=forecast_steps, freq="ME")

    forecast_rows = []
    for i, dt in enumerate(forecast_dates):
        for j, col in enumerate(var_cols):
            forecast_rows.append({
                "date": dt,
                "variable": col,
                "step": i + 1,
                "mean": float(fc_mean[i, j]),
                "lower_68": float(fc_lower_68[i, j]),
                "upper_68": float(fc_upper_68[i, j]),
                "lower_95": float(fc_lower_95[i, j]),
                "upper_95": float(fc_upper_95[i, j]),
            })
    forecast_df = pd.DataFrame(forecast_rows)

    # If variables were differenced, cumsum back to levels for interpretability
    for rec in stationarity_records:
        if rec["differenced"]:
            col = rec["variable"]
            last_level = float(monthly[col].iloc[-1])
            mask = forecast_df["variable"] == col
            for field in ("mean", "lower_68", "upper_68", "lower_95", "upper_95"):
                forecast_df.loc[mask, field] = last_level + forecast_df.loc[mask, field].cumsum().values

    # Append history for charting
    history_rows = []
    # Last 24 months of actual data
    history_window = monthly.iloc[-24:]
    for dt, row in history_window.iterrows():
        for col in var_cols:
            history_rows.append({
                "date": dt,
                "variable": col,
                "step": 0,
                "mean": float(row[col]),
                "lower_68": float(row[col]),
                "upper_68": float(row[col]),
                "lower_95": float(row[col]),
                "upper_95": float(row[col]),
            })
    history_df = pd.DataFrame(history_rows)
    forecast_df = pd.concat([history_df, forecast_df], ignore_index=True)

    # 5. Impulse Response Functions
    irf_result = fitted.irf(irf_steps)
    irf_rows = []
    for i, impulse_col in enumerate(var_cols):
        for j, response_col in enumerate(var_cols):
            for step in range(irf_steps + 1):
                irf_rows.append({
                    "impulse": impulse_col,
                    "response": response_col,
                    "step": step,
                    "value": float(irf_result.irfs[step, j, i]),
                })
    irf_df = pd.DataFrame(irf_rows)

    # 6. Granger causality
    granger_rows = []
    for cause in var_cols:
        for effect in var_cols:
            if cause == effect:
                continue
            try:
                test_data = monthly_transformed[[effect, cause]].dropna()
                if len(test_data) < best_lag + 10:
                    continue
                result = grangercausalitytests(test_data, maxlag=best_lag, verbose=False)
                # Extract F-test result for the best lag
                f_stat = result[best_lag][0]["ssr_ftest"][0]
                p_val = result[best_lag][0]["ssr_ftest"][1]
                granger_rows.append({
                    "cause": cause,
                    "effect": effect,
                    "f_stat": round(float(f_stat), 3),
                    "p_value": round(float(p_val), 4),
                    "significant": p_val < 0.05,
                })
            except Exception:
                granger_rows.append({
                    "cause": cause,
                    "effect": effect,
                    "f_stat": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                })
    granger_df = pd.DataFrame(granger_rows)

    return MacroForecastResult(
        country=country,
        variables=var_cols,
        forecast=forecast_df,
        irf=irf_df,
        granger=granger_df,
        lag_order=best_lag,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        stationarity=stationarity_df,
    )