from __future__ import annotations

import warnings

import pandas as pd


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
    try:
        import wbdata  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "wbdata is not installed — fiscal fundamentals will be unavailable. "
            "Run: pip install wbdata",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["date", "country"] + list(indicators.keys()))

    iso3_list = list(countries_iso3.values())
    wb_codes = {wb_code: col_name for col_name, wb_code in indicators.items()}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = wbdata.get_dataframe(wb_codes, country=iso3_list)
    except Exception as exc:
        warnings.warn(f"World Bank fetch failed: {exc}", stacklevel=2)
        return pd.DataFrame(columns=["date", "country"] + list(indicators.keys()))

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "country"] + list(indicators.keys()))

    # wbdata 1.1+ returns MultiIndex (country, date) with string year labels
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]

    # Convert string year ("2023") → Timestamp (Jan 1 of that year)
    raw["date"] = pd.to_datetime(raw["date"].astype(str) + "-01-01", errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw = raw[raw["date"].dt.year >= start_year]

    # Rename columns to our internal names
    raw = raw.rename(columns=wb_codes)

    col_order = ["date", "country"] + list(indicators.keys())
    raw = raw[[c for c in col_order if c in raw.columns]]

    raw = raw.sort_values(["country", "date"]).reset_index(drop=True)
    return raw
