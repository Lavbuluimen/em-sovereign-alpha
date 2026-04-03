from __future__ import annotations

import warnings

import pandas as pd
import requests

# IMF IFS SDMX JSON REST API base URL
_IFS_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"

# CPI indicator — raw index; we compute YoY % change from it
_CPI_INDICATOR = "PCPI_IX"

# Short-term rate indicators in priority order
_RATE_INDICATORS = ["FIMM_PA", "FPOLM_PA", "FIDR_PA"]

# Government bond yield (benchmark long-term rate, % p.a.) — used as 10Y proxy
_BOND_YIELD_INDICATOR = "FIGB_PA"

_TIMEOUT = 60  # seconds per request


def _fetch_ifs_series(
    countries_iso2: list[str],
    indicator: str,
    start: str = "2014-01",
) -> dict[str, pd.Series]:
    """Fetch one IFS monthly indicator for multiple countries.

    Returns a dict mapping iso2 country code → monthly pd.Series (float).
    Countries with no data are omitted from the result.
    """
    country_key = "+".join(countries_iso2)
    url = f"{_IFS_BASE}/M.{country_key}.{indicator}"
    params = {"startPeriod": start}

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        warnings.warn(f"IMF IFS fetch failed ({indicator}): {exc}", stacklevel=3)
        return {}

    try:
        dataset = data["CompactData"]["DataSet"]
    except (KeyError, TypeError):
        return {}

    if not dataset or "Series" not in dataset:
        return {}

    raw_series = dataset["Series"]
    # API returns a list for multiple countries, a dict for a single country
    if isinstance(raw_series, dict):
        raw_series = [raw_series]

    result: dict[str, pd.Series] = {}
    for series in raw_series:
        iso2 = series.get("@REF_AREA", "")
        obs = series.get("Obs", [])
        if not obs:
            continue
        if isinstance(obs, dict):
            obs = [obs]

        records = []
        for o in obs:
            period = o.get("@TIME_PERIOD")
            value = o.get("@OBS_VALUE")
            if period is None or value is None:
                continue
            try:
                records.append((pd.Period(period, freq="M"), float(value)))
            except (ValueError, TypeError):
                continue

        if not records:
            continue

        idx, vals = zip(*records)
        s = pd.Series(vals, index=pd.PeriodIndex(idx), dtype=float)
        s = s.sort_index()
        result[iso2] = s

    return result


def fetch_ifs_panel(
    countries: dict[str, str],
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Fetch monthly CPI YoY and short-term rate from IMF IFS for given countries.

    Parameters
    ----------
    countries:
        Mapping of {country_name: ISO 3166-1 alpha-2 code}.
        Example: {"Colombia": "CO", "Malaysia": "MY"}
    start:
        Start date string (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Long-format with columns [date, country, cpi_yoy, local_short_rate].
        ``date`` is a monthly Timestamp (first of month); forward-fill to daily
        is left to the caller.
        Returns empty DataFrame if all fetches fail.
    """
    iso2_list = list(countries.values())
    name_map = {v: k for k, v in countries.items()}
    start_period = start[:7]  # "YYYY-MM"

    # ── CPI index → YoY % change ─────────────────────────────────────────────
    cpi_raw = _fetch_ifs_series(iso2_list, _CPI_INDICATOR, start="2013-01")
    cpi_yoy: dict[str, pd.Series] = {}
    for iso2, s in cpi_raw.items():
        yoy = s.pct_change(periods=12) * 100
        yoy = yoy[yoy.index >= pd.Period(start_period, freq="M")]
        cpi_yoy[iso2] = yoy

    # ── Short-term rate with fallback indicators ──────────────────────────────
    rate_data: dict[str, pd.Series] = {}
    for indicator in _RATE_INDICATORS:
        missing = [c for c in iso2_list if c not in rate_data]
        if not missing:
            break
        fetched = _fetch_ifs_series(missing, indicator, start=start_period)
        for iso2, s in fetched.items():
            if iso2 not in rate_data:
                rate_data[iso2] = s

    # ── Assemble long-format panel ────────────────────────────────────────────
    all_iso2 = set(cpi_yoy) | set(rate_data)
    if not all_iso2:
        return pd.DataFrame(columns=["date", "country", "cpi_yoy", "local_short_rate"])

    rows = []
    for iso2 in all_iso2:
        country_name = name_map.get(iso2, iso2)
        cpi_s   = cpi_yoy.get(iso2, pd.Series(dtype=float))
        rate_s  = rate_data.get(iso2, pd.Series(dtype=float))
        all_periods = cpi_s.index.union(rate_s.index)

        for period in all_periods:
            rows.append({
                "date":             period.to_timestamp(),
                "country":          country_name,
                "cpi_yoy":          cpi_s.get(period, float("nan")),
                "local_short_rate": rate_s.get(period, float("nan")),
            })

    if not rows:
        return pd.DataFrame(columns=["date", "country", "cpi_yoy", "local_short_rate"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


def fetch_ifs_yield10y(
    countries: dict[str, str],
    start: str = "2015-01-01",
) -> dict[str, pd.Series]:
    """Fetch monthly government bond yield (FIGB_PA) from IMF IFS.

    FIGB_PA is the IMF IFS benchmark government bond yield (% p.a.), typically
    the 10Y rate or closest available tenor.  Used as a fallback when Stooq and
    FRED are unavailable for a country.

    Parameters
    ----------
    countries:
        Mapping of {country_name: ISO 3166-1 alpha-2 code}.
    start:
        Start date string (YYYY-MM-DD).

    Returns
    -------
    dict mapping country_name → daily pd.Series (linearly interpolated from monthly).
    Countries with no IFS data are omitted.
    """
    iso2_list = list(countries.values())
    name_map = {v: k for k, v in countries.items()}
    start_period = start[:7]

    raw = _fetch_ifs_series(iso2_list, _BOND_YIELD_INDICATOR, start=start_period)
    if not raw:
        print(f"  [warn] IMF IFS {_BOND_YIELD_INDICATOR} returned no data for: {iso2_list}")
        return {}

    result: dict[str, pd.Series] = {}
    for iso2, monthly_s in raw.items():
        country_name = name_map.get(iso2)
        if country_name is None:
            continue
        # Convert PeriodIndex → DatetimeIndex (first of month)
        ts = monthly_s.copy()
        ts.index = ts.index.to_timestamp()
        if ts.dropna().empty:
            continue
        # Linearly interpolate monthly → business-day frequency
        bday_idx = pd.bdate_range(start=ts.index.min(), end=ts.index.max())
        daily = ts.reindex(bday_idx.union(ts.index)).sort_index()
        daily = daily.interpolate(method="time").reindex(bday_idx)
        daily.name = country_name
        result[country_name] = daily

    return result
