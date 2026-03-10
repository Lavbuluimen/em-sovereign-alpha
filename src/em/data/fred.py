from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas_datareader import data as pdr


def fetch_fred_series(
    series_id: str,
    start: str = "2015-01-01",
    api_key: Optional[str] = None,  # kept for compatibility, not used
) -> pd.Series:
    try:
        df = pdr.DataReader(series_id, "fred", start)
    except Exception:
        return pd.Series(dtype=float, name=series_id)

    if df is None or df.empty:
        return pd.Series(dtype=float, name=series_id)

    s = df[series_id].copy()
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s.sort_index()


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