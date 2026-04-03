from __future__ import annotations

from typing import Optional

import pandas as pd

from em.country.data import pull_yield10y_panel
from em.data.fred import fetch_fred_series, fetch_many_fred_series
from em.data.yahoo import fetch_many_yahoo_close


def build_country_daily_panel(
    fx_tickers: dict[str, str],
    yield10y_stooq: dict[str, str],
    start: str = "2015-01-01",
    local_duration_years: float = 5.0,
    yield10y_fred: Optional[dict[str, str]] = None,
    yield10y_ifs: Optional[dict[str, str]] = None,
    yield10y_oecd: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    fx = fetch_many_yahoo_close(fx_tickers, start=start)
    y10_daily = pull_yield10y_panel(
        yield10y_stooq, start=start,
        fred_fallback=yield10y_fred,
        ifs_fallback=yield10y_ifs,
        oecd_fallback=yield10y_oecd,
    )
    us10_daily = fetch_fred_series("DGS10", start=start).rename("us10y")

    if fx.empty:
        raise RuntimeError("FX panel is empty. Cannot build country panel.")

    idx = pd.date_range(start=start, end=fx.index.max(), freq="B")

    fx_a = fx.reindex(idx).ffill()
    y10_a = y10_daily.reindex(idx).ffill()
    us10_a = us10_daily.reindex(idx).ffill()

    # Yahoo FX is generally local per USD, so local currency USD return is approx negative pct change
    fx_usd_ret = -fx_a.pct_change(fill_method=None)

    y10_chg = y10_a.diff()
    us10_chg = us10_a.diff()

    hard_spread_proxy = y10_a.sub(us10_a, axis=0)
    local_ret_proxy_usd = (-local_duration_years * (y10_chg / 100.0)) + fx_usd_ret

    rows = []
    for dt in idx:
        for country in fx_tickers:
            rows.append(
                {
                    "date": dt,
                    "country": country,
                    "fx_level_local_per_usd": fx_a.at[dt, country] if country in fx_a.columns else pd.NA,
                    "fx_usd_ret": fx_usd_ret.at[dt, country] if country in fx_usd_ret.columns else pd.NA,
                    "y10y": y10_a.at[dt, country] if country in y10_a.columns else pd.NA,
                    "y10y_chg": y10_chg.at[dt, country] if country in y10_chg.columns else pd.NA,
                    "us10y": us10_a.loc[dt] if dt in us10_a.index else pd.NA,
                    "us10y_chg": us10_chg.loc[dt] if dt in us10_chg.index else pd.NA,
                    "hard_spread_proxy": hard_spread_proxy.at[dt, country]
                    if country in hard_spread_proxy.columns
                    else pd.NA,
                    "local_ret_proxy_usd": local_ret_proxy_usd.at[dt, country]
                    if country in local_ret_proxy_usd.columns
                    else pd.NA,
                }
            )

    panel = pd.DataFrame(rows).sort_values(["date", "country"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def build_global_macro_panel(
    fred_series: dict[str, str],
    yahoo_tickers: dict[str, str],
    start: str = "2015-01-01",
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    fred_df = fetch_many_fred_series(fred_series, start=start, api_key=fred_api_key)
    yahoo_df = fetch_many_yahoo_close(yahoo_tickers, start=start)

    if fred_df.empty and yahoo_df.empty:
        return pd.DataFrame()

    max_end = None
    if not fred_df.empty:
        max_end = fred_df.index.max()
    if not yahoo_df.empty:
        max_end = yahoo_df.index.max() if max_end is None else max(max_end, yahoo_df.index.max())

    idx = pd.date_range(start=start, end=max_end, freq="B")

    fred_a = fred_df.reindex(idx).ffill() if not fred_df.empty else pd.DataFrame(index=idx)
    yahoo_a = yahoo_df.reindex(idx).ffill() if not yahoo_df.empty else pd.DataFrame(index=idx)

    df = pd.concat([fred_a, yahoo_a], axis=1).sort_index()
    df.index.name = "date"
    return df.reset_index()