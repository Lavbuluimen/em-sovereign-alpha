from __future__ import annotations

import datetime
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
        Example: {"fiscal_balance_gdp": "GC.BAL.CASH.GD.ZS"}
    countries_iso3:
        Mapping of {country_name: ISO 3166-1 alpha-3 code}.
        Example: {"Brazil": "BRA"}
    start_year:
        First year to request. Data is returned from Jan 1 of this year.

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

    iso3_to_name = {v: k for k, v in countries_iso3.items()}
    iso3_list = list(countries_iso3.values())

    start_dt = datetime.datetime(start_year, 1, 1)
    end_dt   = datetime.datetime(datetime.date.today().year, 12, 31)

    rows: list[dict] = []
    for col_name, wb_code in indicators.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = wbdata.get_dataframe(
                    {wb_code: col_name},
                    country=iso3_list,
                    date=(start_dt, end_dt),
                    convert_date=True,
                )
        except Exception:
            continue

        if raw is None or raw.empty:
            continue

        # wbdata returns MultiIndex (country, date) or single-level; normalise
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index()
            raw.columns = [c.lower() for c in raw.columns]
            # columns: country, date, col_name
            raw["country"] = raw["country"].map(
                lambda x: iso3_to_name.get(x, x)
            )
        else:
            raw = raw.reset_index()
            raw.columns = [c.lower() for c in raw.columns]

        rows.append(raw[["date", "country", col_name]])

    if not rows:
        return pd.DataFrame(columns=["date", "country"] + list(indicators.keys()))

    # Merge all indicator frames on [date, country]
    merged = rows[0]
    for frame in rows[1:]:
        merged = merged.merge(frame, on=["date", "country"], how="outer")

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["country", "date"]).reset_index(drop=True)
    return merged
