"""Sovereign portfolio backtest vs EMBI GD (proxied by iShares EMB ETF).

Methodology
-----------
- Weights lagged 1 business day (signal at close t → position at t+1)
- Hard-currency return:  -6.5yr × Δy10y / 100 + y10y/252   (price + carry)
- Local-currency return: local_ret_proxy_usd + y10y/252     (price+FX + carry)
- Carry accrual: y10y / 252 per day per country (running yield).
  Countries missing y10y use ASSUMED_CARRY_PCT as fallback.
  Carry is added to both hard and local legs since EMB ETF total return
  includes coupon reinvestment and our price proxies do not.
- Portfolio return:      Σ (hard_w[t-1] × hard_ret[t] + local_w[t-1] × local_ret[t])
- Active return:         port_ret − bench_ret (EMB ETF, which is already total return)
- Burn-in: drop 2015; report from 2016-01-01
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HARD_DURATION       = 6.5   # years, EMBI GD approximate modified duration
START_DATE          = "2016-01-01"
TRADING_DAYS        = 252
ASSUMED_CARRY_PCT   = 6.0   # % p.a. fallback yield for countries missing y10y data


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------

def _hard_ret(y10y_chg: pd.Series) -> pd.Series:
    """Price return for a hard-currency EM sovereign bond (duration approx)."""
    return -HARD_DURATION * y10y_chg / 100


def build_daily_returns(
    portfolio: pd.DataFrame,
    country_panel: pd.DataFrame,
    emb_ret: pd.Series,
    start: str = START_DATE,
) -> pd.DataFrame:
    """Compute daily portfolio, benchmark, and active returns.

    Parameters
    ----------
    portfolio:      portfolio_daily.parquet  (flat, columns: date, country, hard_w, local_w, regime)
    country_panel:  country_daily.parquet    (flat, columns: date, country, y10y_chg, local_ret_proxy_usd)
    emb_ret:        pd.Series indexed by date, EMB ETF daily return
    start:          first date to include in output

    Returns
    -------
    DataFrame with columns: date, port_ret, bench_ret, active_ret
    """
    # ---- prepare country panel -----------------------------------------------
    cp = country_panel[["date", "country", "y10y", "y10y_chg", "local_ret_proxy_usd"]].copy()
    cp["date"] = pd.to_datetime(cp["date"])
    cp["hard_ret"] = _hard_ret(cp["y10y_chg"])

    # Daily carry accrual: running yield / 252. Fill missing y10y with assumption.
    cp["carry_daily"] = cp["y10y"].fillna(ASSUMED_CARRY_PCT) / 100 / TRADING_DAYS

    # ---- prepare portfolio weights -------------------------------------------
    pw = portfolio[["date", "country", "hard_w", "local_w", "regime"]].copy()
    pw["date"] = pd.to_datetime(pw["date"])

    # pivot to wide: index=date, columns=country
    hard_w  = pw.pivot(index="date", columns="country", values="hard_w").sort_index()
    local_w = pw.pivot(index="date", columns="country", values="local_w").sort_index()

    # LAG weights by 1 business day
    hard_w_lag  = hard_w.shift(1)
    local_w_lag = local_w.shift(1)

    # ---- pivot returns to wide ------------------------------------------------
    hard_r  = cp.pivot(index="date", columns="country", values="hard_ret").sort_index()
    local_r = cp.pivot(index="date", columns="country", values="local_ret_proxy_usd").sort_index()
    carry_r = cp.pivot(index="date", columns="country", values="carry_daily").sort_index()

    # align columns and dates
    countries = hard_w_lag.columns
    all_dates = hard_w_lag.index.union(hard_r.index)
    hard_w_lag  = hard_w_lag.reindex(all_dates).ffill()
    local_w_lag = local_w_lag.reindex(all_dates).ffill()
    hard_r      = hard_r.reindex(all_dates)[countries]
    local_r     = local_r.reindex(all_dates)[countries]
    carry_r     = carry_r.reindex(all_dates)[countries]

    # ---- portfolio return: Σ (w_hard * (r_hard + carry) + w_local * (r_local + carry))
    # carry_r is the same daily accrual for both hard and local legs (same issuer yield)
    total_w_lag = hard_w_lag + local_w_lag
    port_ret = (hard_w_lag * hard_r).sum(axis=1, min_count=1) + \
               (local_w_lag * local_r).sum(axis=1, min_count=1) + \
               (total_w_lag * carry_r).sum(axis=1, min_count=1)

    # ---- benchmark -----------------------------------------------------------
    bench_ret = emb_ret.reindex(all_dates)

    # ---- assemble and filter -------------------------------------------------
    df = pd.DataFrame({
        "port_ret":   port_ret,
        "bench_ret":  bench_ret,
    })
    df["active_ret"] = df["port_ret"] - df["bench_ret"]
    df.index.name = "date"
    df = df.loc[start:].dropna(subset=["port_ret", "bench_ret"])
    return df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _annualised_ret(r: pd.Series) -> float:
    return float(r.mean() * TRADING_DAYS)


def _annualised_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def _sharpe(r: pd.Series) -> float:
    vol = _annualised_vol(r)
    return _annualised_ret(r) / vol if vol > 0 else np.nan


def _max_drawdown(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def _calmar(r: pd.Series) -> float:
    mdd = _max_drawdown(r)
    return _annualised_ret(r) / abs(mdd) if mdd < 0 else np.nan


def _information_ratio(active: pd.Series) -> float:
    te = _annualised_vol(active)
    return _annualised_ret(active) / te if te > 0 else np.nan


def _monthly_win_rate(port: pd.Series, bench: pd.Series) -> float:
    pm = (1 + port).resample("ME").prod() - 1
    bm = (1 + bench).resample("ME").prod() - 1
    wins = (pm > bm).sum()
    return float(wins / len(pm)) if len(pm) > 0 else np.nan


def compute_metrics(returns: pd.DataFrame) -> dict:
    """Return dict of annualised metrics."""
    p = returns["port_ret"]
    b = returns["bench_ret"]
    a = returns["active_ret"]

    metrics: dict = {
        "ann_return_port":   round(_annualised_ret(p),  4),
        "ann_return_bench":  round(_annualised_ret(b),  4),
        "ann_return_active": round(_annualised_ret(a),  4),
        "sharpe_port":       round(_sharpe(p),           4),
        "sharpe_bench":      round(_sharpe(b),           4),
        "information_ratio": round(_information_ratio(a), 4),
        "tracking_error":    round(_annualised_vol(a),   4),
        "max_dd_port":       round(_max_drawdown(p),     4),
        "max_dd_bench":      round(_max_drawdown(b),     4),
        "calmar_port":       round(_calmar(p),            4),
        "monthly_win_rate":  round(_monthly_win_rate(p, b), 4),
        "n_days":            int(len(returns)),
        "start_date":        str(returns.index.min().date()),
        "end_date":          str(returns.index.max().date()),
    }

    # ---- year-by-year --------------------------------------------------------
    yearly: dict = {}
    for yr, grp in returns.groupby(returns.index.year):
        yearly[int(yr)] = {
            "port":   round(_annualised_ret(grp["port_ret"]),  4),
            "bench":  round(_annualised_ret(grp["bench_ret"]), 4),
            "active": round(_annualised_ret(grp["active_ret"]), 4),
            "sharpe": round(_sharpe(grp["port_ret"]),           4),
            "ir":     round(_information_ratio(grp["active_ret"]), 4),
        }
    metrics["yearly"] = yearly
    return metrics


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def country_attribution(
    portfolio: pd.DataFrame,
    country_panel: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Each country's contribution to active return over the full period.

    contribution_i = Σ_t (w_i[t-1] - bench_w_i[t-1]) × ret_i[t]
    where ret_i is the blended hard+local return.
    """
    cp = country_panel[["date", "country", "y10y", "y10y_chg", "local_ret_proxy_usd"]].copy()
    cp["date"] = pd.to_datetime(cp["date"])
    cp["hard_ret"] = _hard_ret(cp["y10y_chg"])
    cp["carry_daily"] = cp["y10y"].fillna(ASSUMED_CARRY_PCT) / 100 / TRADING_DAYS

    pw = portfolio[["date", "country", "hard_w", "local_w", "bench_w"]].copy()
    pw["date"] = pd.to_datetime(pw["date"])

    # blended total weight and return per country/day
    hw  = pw.pivot(index="date", columns="country", values="hard_w").sort_index()
    lw  = pw.pivot(index="date", columns="country", values="local_w").sort_index()
    bw  = pw.pivot(index="date", columns="country", values="bench_w").sort_index()
    hr  = cp.pivot(index="date", columns="country", values="hard_ret").sort_index()
    lr  = cp.pivot(index="date", columns="country", values="local_ret_proxy_usd").sort_index()
    cr  = cp.pivot(index="date", columns="country", values="carry_daily").sort_index()

    active_w = (hw + lw - bw).shift(1)

    all_dates = active_w.index.union(hr.index)
    active_w = active_w.reindex(all_dates).ffill()
    hr = hr.reindex(all_dates)
    lr = lr.reindex(all_dates)
    cr = cr.reindex(all_dates)
    hw_lag = hw.shift(1).reindex(all_dates).ffill()
    lw_lag = lw.shift(1).reindex(all_dates).ffill()
    total_w = hw_lag + lw_lag
    # blended country return (weighted avg of hard and local, including carry)
    safe_w = total_w.replace(0, np.nan)
    blended_ret = (hw_lag * hr + lw_lag * lr) / safe_w + cr
    blended_ret = blended_ret.fillna(0)

    # contribution to active return = active_w * blended_ret
    contrib = (active_w * blended_ret).loc[returns.index]
    summary = contrib.sum().rename("active_contribution")
    ann_factor = TRADING_DAYS / len(returns) if len(returns) > 0 else 1
    attr = pd.DataFrame({
        "active_contribution_total": summary,
        "active_contribution_ann":   (summary * ann_factor).round(4),
    })
    attr.index.name = "country"
    return attr.reset_index()


def regime_attribution(
    portfolio: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Average annualised active return when portfolio's dominant regime is Green/Amber/Red."""
    pw = portfolio[["date", "country", "regime", "hard_w", "local_w"]].copy()
    pw["date"] = pd.to_datetime(pw["date"])

    # dominant regime on each date = regime of highest-weighted country
    total_w = pw.assign(total_w=pw["hard_w"] + pw["local_w"])
    dominant = (
        total_w.sort_values("total_w", ascending=False)
               .groupby("date")
               .first()
               .reset_index()[["date", "regime"]]
    )
    dominant = dominant.set_index("date")["regime"]
    dominant.index = pd.to_datetime(dominant.index)

    ret_df = returns.copy()
    ret_df["regime"] = dominant.reindex(ret_df.index)

    rows = []
    for regime, grp in ret_df.dropna(subset=["regime"]).groupby("regime"):
        rows.append({
            "regime":           regime,
            "n_days":           len(grp),
            "ann_port_ret":     round(_annualised_ret(grp["port_ret"]),   4),
            "ann_active_ret":   round(_annualised_ret(grp["active_ret"]), 4),
            "sharpe":           round(_sharpe(grp["port_ret"]),            4),
            "ir":               round(_information_ratio(grp["active_ret"]), 4),
        })
    return pd.DataFrame(rows)
