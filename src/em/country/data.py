

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

_YIELD_CACHE_DIR = Path("data/cache/yields")


STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"  # daily CSV

_STOOQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


@dataclass
class PanelBuildConfig:
    start: str = "2015-01-01"
    local_duration_years: float = 5.0


def _download_stooq_daily(symbol: str, start: str, retries: int = 3) -> pd.Series:
    url = STOOQ_CSV.format(symbol=symbol.lower())
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)  # 2s, 4s back-off
        try:
            resp = requests.get(url, headers=_STOOQ_HEADERS, timeout=30)
            resp.raise_for_status()
            text = resp.text
            if "<html" in text[:200].lower():
                raise RuntimeError(f"Stooq returned HTML for {symbol} (rate-limited or blocked)")
            df = pd.read_csv(io.StringIO(text))
            if "Date" not in df.columns:
                raise RuntimeError(f"Unexpected Stooq format for {symbol}: {df.columns.tolist()}")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            if "Close" not in df.columns:
                raise RuntimeError(f"Stooq CSV missing Close for {symbol}: {df.columns.tolist()}")
            s = df["Close"].astype(float)
            s = s.loc[s.index >= pd.to_datetime(start)]
            # Stooq returns 0.0 for missing dates on some tickers — replace with NaN
            s = s.replace(0.0, float("nan"))
            s.name = symbol
            return s
        except Exception as exc:
            last_exc = exc
    raise last_exc


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


def _fetch_fred_yield_daily(fred_id: str, start: str) -> pd.Series:
    """Fetch a monthly FRED yield series and linearly interpolate to daily."""
    from em.data.fred import fetch_fred_series
    s = fetch_fred_series(fred_id, start=start)
    if s.empty:
        raise RuntimeError(f"FRED returned no data for {fred_id}")
    bday_idx = pd.bdate_range(start=s.index.min(), end=s.index.max())
    # Reindex to daily (creates NaN for non-month-end dates), then interpolate
    daily = s.reindex(bday_idx.union(s.index)).sort_index()
    daily = daily.interpolate(method="time").reindex(bday_idx)
    return daily


def _save_yield_cache(country: str, series: pd.Series) -> None:
    """Persist a yield series to disk so it survives network outages."""
    _YIELD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _YIELD_CACHE_DIR / f"{country.lower().replace(' ', '_')}_10y.parquet"
    series.to_frame(name="y10y").to_parquet(path)


def _load_yield_cache(country: str, start: str) -> Optional[pd.Series]:
    """Load a cached yield series from disk, filtered to start date."""
    path = _YIELD_CACHE_DIR / f"{country.lower().replace(' ', '_')}_10y.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        s = df["y10y"].copy()
        s.index = pd.to_datetime(s.index)
        s = s.loc[s.index >= pd.to_datetime(start)]
        if s.dropna().empty:
            return None
        s.name = country
        return s
    except Exception:
        return None


def pull_yield10y_panel(
    yield_tickers: Dict[str, str],
    start: str,
    fred_fallback: Optional[Dict[str, str]] = None,
    ifs_fallback: Optional[Dict[str, str]] = None,
    oecd_fallback: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Download daily 10Y yields.

    Source priority per country:
      1. Stooq (daily CSV)
      2. FRED OECD monthly series → linearly interpolated to daily
      3. IMF IFS FIGB_PA monthly → linearly interpolated to daily
      4. OECD SDMX DF_FINMARK IRLT → linearly interpolated to daily
      5. Local disk cache (data/cache/yields/) — survives network outages
    """
    # Pre-fetch IFS data in one batch request (more efficient than per-country)
    ifs_data: Dict[str, pd.Series] = {}
    if ifs_fallback:
        from em.data.ifs import fetch_ifs_yield10y
        ifs_data = fetch_ifs_yield10y(ifs_fallback, start=start)

    # Pre-fetch OECD data in one batch request
    oecd_data: Dict[str, pd.Series] = {}
    if oecd_fallback:
        from em.data.oecd import fetch_oecd_yield10y
        # Convert start date "YYYY-MM-DD" → "YYYY-MM" for OECD API
        oecd_start = start[:7] if len(start) > 7 else start
        oecd_data = fetch_oecd_yield10y(oecd_fallback, start=oecd_start)
        if oecd_data:
            print(f"  [info] OECD pre-fetch succeeded for: {list(oecd_data.keys())}")
        else:
            print("  [warn] OECD pre-fetch returned no data")

    out = {}
    for country, sym in yield_tickers.items():
        # 1. Stooq
        try:
            series = _download_stooq_daily(sym, start=start)
            if series.dropna().empty:
                raise RuntimeError(f"Stooq returned empty series for {sym}")
            out[country] = series
            _save_yield_cache(country, series)
            continue
        except Exception as stooq_exc:
            print(f"  [warn] Stooq failed for {country} ({sym}): {stooq_exc}")

        # 2. FRED OECD monthly → interpolated daily
        if fred_fallback and country in fred_fallback:
            try:
                series = _fetch_fred_yield_daily(fred_fallback[country], start=start)
                series.name = country
                out[country] = series
                _save_yield_cache(country, series)
                print(f"  [info] Using FRED fallback for {country} ({fred_fallback[country]})")
                continue
            except Exception as fred_exc:
                print(f"  [warn] FRED fallback failed for {country}: {fred_exc}")

        # 3. IMF IFS FIGB_PA monthly → interpolated daily
        if country in ifs_data:
            out[country] = ifs_data[country]
            _save_yield_cache(country, ifs_data[country])
            print(f"  [info] Using IMF IFS fallback for {country}")
            continue

        # 4. OECD SDMX DF_FINMARK IRLT → interpolated daily
        if country in oecd_data:
            out[country] = oecd_data[country]
            _save_yield_cache(country, oecd_data[country])
            print(f"  [info] Using OECD SDMX fallback for {country}")
            continue

        # 5. Local disk cache
        cached = _load_yield_cache(country, start)
        if cached is not None:
            out[country] = cached
            print(f"  [info] Using local cache for {country} (all live sources failed)")
            continue

        # All sources failed
        print(f"  [error] No yield data for {country} — will remain NaN")
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
