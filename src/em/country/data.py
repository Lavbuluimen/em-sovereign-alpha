
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yfinance as yf


STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"  # daily CSV


@dataclass
class PanelBuildConfig:
    start: str = "2015-01-01"
    local_duration_years: float = 5.0


def _download_stooq_daily(symbol: str, start: str) -> pd.Series:
    url = STOOQ_CSV.format(symbol=symbol.lower())
    df = pd.read_csv(url)
    if "Date" not in df.columns:
        raise RuntimeError(f"Unexpected Stooq format for {symbol}: {df.columns.tolist()}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if "Close" not in df.columns:
        raise RuntimeError(f"Stooq CSV missing Close for {symbol}: {df.columns.tolist()}")

    s = df["Close"].astype(float)
    s = s.loc[s.index >= pd.to_datetime(start)]
    s.name = symbol
    return s


def _download_yf_close(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for ticker={ticker}")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s.sort_index()


def pull_fx_panel(fx_tickers: Dict[str, str], start: str) -> pd.DataFrame:
    out = {}
    for country, tkr in fx_tickers.items():
        out[country] = _download_yf_close(tkr, start=start)
    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()


def pull_yield10y_panel(yield_tickers: Dict[str, str], start: str) -> pd.DataFrame:
    out = {}
    for country, sym in yield_tickers.items():
        try:
            out[country] = _download_stooq_daily(sym, start=start)
        except Exception:
            # keep column but empty (will stay NaN after reindex)
            out[country] = pd.Series(dtype=float, name=country)
    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()


def build_country_daily_panel(
    fx_tickers: Dict[str, str],
    yield10y_tickers: Dict[str, str],
    start: str = "2015-01-01",
    local_duration_years: float = 5.0,
) -> pd.DataFrame:
    """
    Daily country panel with:
      - fx_usd_ret (USD return of local currency; assumes Yahoo quotes LOCAL per USD)
      - y10y level (%)
      - y10y_chg daily change (percentage points)
      - us10y level (%), us10y_chg
      - hard_spread_proxy = y10y - us10y
      - local_ret_proxy_usd ≈ -Duration*(Δy/100) + fx_usd_ret
    """
    fx = pull_fx_panel(fx_tickers, start=start)
    y10 = pull_yield10y_panel(yield10y_tickers, start=start)
    us10 = _download_stooq_daily("10YUSY.B", start=start).rename("US")

    # Align calendars WITHOUT concatenating duplicate column names
    idx = fx.index.union(y10.index).union(us10.index).sort_values()

    fx_a = fx.reindex(idx)
    y10_a = y10.reindex(idx)
    us10_a = us10.reindex(idx)

    # FX USD returns: LOCAL per USD quote -> USD return approx = -pct_change
    fx_usd_ret = -fx_a.pct_change(fill_method=None)

    y10_chg = y10_a.diff()     # percentage points
    us10_chg = us10_a.diff()   # percentage points

    spread_proxy = y10_a.sub(us10_a, axis=0)

    local_ret_proxy_usd = (-local_duration_years * (y10_chg / 100.0)) + fx_usd_ret

    # Long panel
    records = []
    for dt in idx:
        us10y_val = us10_a.loc[dt] if dt in us10_a.index else float("nan")
        us10y_chg_val = us10_chg.loc[dt] if dt in us10_chg.index else float("nan")

        for country in fx_tickers.keys():
            records.append({
                "date": dt,
                "country": country,
                "fx_level_local_per_usd": float(fx_a.at[dt, country]) if country in fx_a.columns and pd.notna(fx_a.at[dt, country]) else float("nan"),
                "fx_usd_ret": float(fx_usd_ret.at[dt, country]) if country in fx_usd_ret.columns and pd.notna(fx_usd_ret.at[dt, country]) else float("nan"),
                "y10y": float(y10_a.at[dt, country]) if country in y10_a.columns and pd.notna(y10_a.at[dt, country]) else float("nan"),
                "y10y_chg": float(y10_chg.at[dt, country]) if country in y10_chg.columns and pd.notna(y10_chg.at[dt, country]) else float("nan"),
                "us10y": float(us10y_val) if pd.notna(us10y_val) else float("nan"),
                "us10y_chg": float(us10y_chg_val) if pd.notna(us10y_chg_val) else float("nan"),
                "hard_spread_proxy": float(spread_proxy.at[dt, country]) if country in spread_proxy.columns and pd.notna(spread_proxy.at[dt, country]) else float("nan"),
                "local_ret_proxy_usd": float(local_ret_proxy_usd.at[dt, country]) if country in local_ret_proxy_usd.columns and pd.notna(local_ret_proxy_usd.at[dt, country]) else float("nan"),
            })

    panel = pd.DataFrame.from_records(records)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "country"]).reset_index(drop=True)

    return panel
