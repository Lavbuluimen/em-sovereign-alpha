from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_yahoo_close(
    ticker: str,
    start: str = "2015-01-01",
) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"No Yahoo data returned for ticker={ticker}")

    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s.sort_index()


def fetch_many_yahoo_close(
    ticker_map: dict[str, str],
    start: str = "2015-01-01",
) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}

    for name, ticker in ticker_map.items():
        try:
            out[name] = fetch_yahoo_close(ticker=ticker, start=start)
        except Exception:
            out[name] = pd.Series(dtype=float, name=name)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()