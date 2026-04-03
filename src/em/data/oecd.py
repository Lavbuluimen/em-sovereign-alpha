"""Fetch 10Y government bond yields from the OECD SDMX API.

Dataset:  OECD.SDD.STES / DSD_STES@DF_FINMARK
Measure:  IRLT — long-term interest rate, government bond yield (% p.a., monthly)
Coverage: Brazil (BRA), Colombia (COL), China (CHN) — confirmed working.
          Turkey is not present in this dataset.

Returns monthly series, linearly interpolated to business-day daily frequency.
"""
from __future__ import annotations

import io
import warnings
from typing import Dict

import pandas as pd
import requests

_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
    "{country_str}.M.........?startPeriod={start}"
)
_HEADERS = {"Accept": "application/vnd.sdmx.data+csv;version=2"}
_TIMEOUT = 30


def fetch_oecd_yield10y(
    country_codes: Dict[str, str],
    start: str = "2015-01",
) -> Dict[str, pd.Series]:
    """Fetch monthly 10Y govt bond yields from OECD and interpolate to daily.

    Parameters
    ----------
    country_codes : dict mapping country name → OECD ISO-3 code
                    e.g. {"Brazil": "BRA", "Colombia": "COL", "China": "CHN"}
    start         : first period in YYYY-MM format

    Returns
    -------
    dict mapping country name → pd.Series of daily yields (% p.a.),
    business-day index, linearly interpolated from monthly OECD observations.
    Missing countries (no IRLT rows) are omitted from the result.
    """
    codes = list(country_codes.values())
    country_str = "+".join(codes)
    url = _URL.format(country_str=country_str, start=start)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        warnings.warn(f"OECD SDMX fetch failed: {exc}")
        return {}

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        warnings.warn(f"OECD SDMX CSV parse failed: {exc}")
        return {}

    if "MEASURE" not in df.columns or "IRLT" not in df["MEASURE"].values:
        warnings.warn("OECD SDMX: no IRLT rows in response")
        return {}

    irlt = df[df["MEASURE"] == "IRLT"][["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    irlt["TIME_PERIOD"] = pd.to_datetime(irlt["TIME_PERIOD"])
    irlt = irlt.dropna(subset=["OBS_VALUE"]).sort_values("TIME_PERIOD")

    # Reverse lookup: OECD code → country name
    code_to_name = {v: k for k, v in country_codes.items()}

    result: Dict[str, pd.Series] = {}
    for code, grp in irlt.groupby("REF_AREA"):
        name = code_to_name.get(code)
        if name is None:
            continue
        monthly = grp.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
        monthly.index = pd.DatetimeIndex(monthly.index)

        # Interpolate monthly → business-daily
        bday_idx = pd.bdate_range(start=monthly.index.min(), end=monthly.index.max())
        daily = monthly.reindex(bday_idx.union(monthly.index)).sort_index()
        daily = daily.interpolate(method="time").reindex(bday_idx)
        daily.name = name
        result[name] = daily

    return result
