from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import requests
from pandas_datareader import data as pdr

# Direct CSV download — used as fallback when pandas_datareader fails.
# No API key required; FRED serves public series at this endpoint.
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_FRED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}
_TIMEOUT = 30


def _fetch_fred_direct(series_id: str, start: str) -> pd.Series:
    """Fetch a FRED series via direct CSV download (fallback)."""
    url = _FRED_CSV_URL.format(series_id=series_id)
    resp = requests.get(url, headers=_FRED_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    if "<html" in resp.text[:200].lower():
        raise RuntimeError(f"FRED returned HTML for {series_id}")
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or len(df.columns) < 2:
        raise RuntimeError(f"FRED CSV empty for {series_id}")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # FRED uses "." for missing values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).set_index("date").sort_index()
    s = df["value"]
    s = s.loc[s.index >= pd.to_datetime(start)]
    if s.empty:
        raise RuntimeError(f"FRED CSV has no data after {start} for {series_id}")
    s.name = series_id
    return s


def fetch_fred_series(
    series_id: str,
    start: str = "2015-01-01",
    api_key: Optional[str] = None,  # kept for compatibility, not used
) -> pd.Series:
    """Fetch a FRED series. Tries pandas_datareader first, then direct CSV download."""
    # Primary: pandas_datareader (FRED REST API)
    try:
        df = pdr.DataReader(series_id, "fred", start)
        if df is not None and not df.empty:
            s = df[series_id].copy()
            s.index = pd.to_datetime(s.index)
            s.name = series_id
            return s.sort_index()
    except Exception:
        pass

    # Fallback: direct CSV download
    try:
        return _fetch_fred_direct(series_id, start)
    except Exception:
        pass

    return pd.Series(dtype=float, name=series_id)


def fetch_many_fred_series(
    series_map: dict[str, str],
    start: str = "2015-01-01",
    api_key: Optional[str] = None,  # kept for compatibility, not used
) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}

    for name, series_id in series_map.items():
        try:
            out[name] = fetch_fred_series(series_id=series_id, start=start, api_key=api_key)
        except Exception:
            out[name] = pd.Series(dtype=float, name=name)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()
