from __future__ import annotations

import io
import warnings

import pandas as pd
import requests

_BIS_CBPOL_URL = (
    "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0"
    "/{key}?format=csv&startPeriod={start}"
)
_TIMEOUT = 30


def fetch_bis_policy_rates(
    countries: dict[str, str],
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Fetch BIS central bank policy rates.

    Parameters
    ----------
    countries:
        Mapping of {country_name: ISO 3166-1 alpha-2 code}.
    start:
        Start date string (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Long-format with columns [date, country, local_short_rate].
        date is a monthly Timestamp (first of month).
        Returns empty DataFrame if fetch fails.
    """
    iso2_codes = list(countries.values())
    iso2_to_name = {v.upper(): k for k, v in countries.items()}
    key = "M." + "+".join(iso2_codes)
    start_period = start[:7]  # YYYY-MM

    url = _BIS_CBPOL_URL.format(key=key, start=start_period)
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        warnings.warn(f"BIS policy rate fetch failed: {exc}", stacklevel=2)
        return pd.DataFrame(columns=["date", "country", "local_short_rate"])

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        warnings.warn(f"BIS CSV parse failed: {exc}", stacklevel=2)
        return pd.DataFrame(columns=["date", "country", "local_short_rate"])

    # Expected columns: FREQ, REF_AREA, ..., TIME_PERIOD, OBS_VALUE, ...
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        warnings.warn(
            f"BIS: unexpected CSV columns: {list(df.columns)}", stacklevel=2
        )
        return pd.DataFrame(columns=["date", "country", "local_short_rate"])

    df = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])

    df["date"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["date"])

    df["country"] = df["REF_AREA"].str.upper().map(iso2_to_name)
    df = df.dropna(subset=["country"])

    df = df.rename(columns={"OBS_VALUE": "local_short_rate"})[
        ["date", "country", "local_short_rate"]
    ]
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df
