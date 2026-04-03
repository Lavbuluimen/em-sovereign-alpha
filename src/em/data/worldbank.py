from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd


def _fetch_via_rest_api(
    indicators: dict[str, str],
    countries_iso3: dict[str, str],
    start_year: int,
) -> pd.DataFrame:
    """Fallback: call the World Bank REST API directly without wbdata.

    Uses the v2 JSON endpoint: api.worldbank.org/v2/country/{iso}/indicator/{code}
    """
    try:
        import requests
    except ImportError:
        return pd.DataFrame(columns=["date", "country"] + list(indicators.keys()))

    rows: list[dict] = []
    for col_name, wb_code in indicators.items():
        for country_name, iso3 in countries_iso3.items():
            url = (
                f"https://api.worldbank.org/v2/country/{iso3}/indicator/{wb_code}"
                f"?format=json&date={start_year}:2025&per_page=100&mrv=20"
            )
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or len(payload) < 2:
                    continue
                for item in payload[1] or []:
                    if item.get("value") is None:
                        continue
                    year = int(item["date"])
                    if year < start_year:
                        continue
                    rows.append({
                        "date": pd.Timestamp(f"{year}-01-01"),
                        "country": country_name,
                        col_name: float(item["value"]),
                    })
            except Exception as exc:
                warnings.warn(
                    f"World Bank REST API failed for {country_name}/{wb_code}: {exc}",
                    stacklevel=3,
                )
                continue

    if not rows:
        return pd.DataFrame(columns=["date", "country"] + list(indicators.keys()))

    df = pd.DataFrame(rows)
    # Aggregate in case multiple indicator columns need combining
    df = df.groupby(["date", "country"]).first().reset_index()
    return df.sort_values(["country", "date"]).reset_index(drop=True)


def fetch_worldbank_panel(
    indicators: dict[str, str],
    countries_iso3: dict[str, str],
    start_year: int = 2010,
) -> pd.DataFrame:
    """Fetch World Bank annual indicators and return a long-format daily panel.

    Parameters
    ----------
    indicators:
        Mapping of {column_name: World Bank indicator code}.
        Example: {"fiscal_balance_gdp": "GC.NLD.TOTL.GD.ZS"}
    countries_iso3:
        Mapping of {country_name: ISO 3166-1 alpha-3 code}.
        Example: {"Brazil": "BRA"}
    start_year:
        First year to include. Rows before this year are dropped.

    Returns
    -------
    pd.DataFrame
        Long-format with columns [date, country, *indicator_cols].
        ``date`` is a pandas Timestamp at annual frequency (Jan 1 of each year).
        The caller is responsible for forward-filling to business-day frequency.
        Returns an empty DataFrame (with the correct columns) if wbdata is not
        installed or all fetches fail.
    """
    wbdata_ok = False
    raw: Optional[pd.DataFrame] = None

    try:
        import wbdata  # type: ignore[import]
        wbdata_ok = True
    except ImportError:
        warnings.warn(
            "wbdata is not installed — trying direct REST API fallback. "
            "Run: pip install wbdata",
            stacklevel=2,
        )

    if wbdata_ok:
        iso3_list = list(countries_iso3.values())
        wb_codes = {wb_code: col_name for col_name, wb_code in indicators.items()}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = wbdata.get_dataframe(wb_codes, country=iso3_list)
        except Exception as exc:
            warnings.warn(
                f"wbdata fetch failed ({exc}); falling back to direct REST API.",
                stacklevel=2,
            )
            raw = None

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        warnings.warn("Trying World Bank REST API fallback...", stacklevel=2)
        return _fetch_via_rest_api(indicators, countries_iso3, start_year)

    # wbdata 1.1+ returns MultiIndex (country, date) with string year labels
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]

    # Convert string year ("2023") → Timestamp (Jan 1 of that year)
    raw["date"] = pd.to_datetime(raw["date"].astype(str) + "-01-01", errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw = raw[raw["date"].dt.year >= start_year]

    # Rename columns to our internal names
    wb_codes = {wb_code: col_name for col_name, wb_code in indicators.items()}
    raw = raw.rename(columns=wb_codes)

    col_order = ["date", "country"] + list(indicators.keys())
    raw = raw[[c for c in col_order if c in raw.columns]]

    raw = raw.sort_values(["country", "date"]).reset_index(drop=True)
    return raw
