
from __future__ import annotations
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

def _yf_close(tickers: dict[str, str], start: str) -> pd.DataFrame:
    px = yf.download(list(tickers.values()), start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    inv = {v: k for k, v in tickers.items()}
    px = px.rename(columns=inv)
    px.index = pd.to_datetime(px.index)
    return px.sort_index()

def pull_market_features(start: str = "2015-01-01") -> pd.DataFrame:
    # EM proxy + commodities + UST 10y yield (FRED)
    etf_tickers = {"EMB": "EMB"}
    commodity_tickers = {
        "Brent": "BZ=F",
        "WTI": "CL=F",
        "Copper": "HG=F",
        "Gold": "GC=F",
    }
    fred_series = {"DGS10": "DGS10"}  # 10Y Treasury yield

    emb_px = _yf_close(etf_tickers, start=start)
    cmdty_px = _yf_close(commodity_tickers, start=start)

    fred_df = pdr.DataReader(list(fred_series.values()), "fred", start)
    fred_df = fred_df.rename(columns={v: k for k, v in fred_series.items()})
    fred_df.index = pd.to_datetime(fred_df.index)

    df = pd.concat([emb_px, cmdty_px, fred_df], axis=1).sort_index()

    out = pd.DataFrame(index=df.index)
    out["EMB_ret"] = df["EMB"].pct_change()
    for c in ["Brent", "WTI", "Copper", "Gold"]:
        out[f"{c}_ret"] = df[c].pct_change()
    out["DGS10_chg"] = df["DGS10"].diff()  # yield change in percentage points

    return out.dropna(how="all")
