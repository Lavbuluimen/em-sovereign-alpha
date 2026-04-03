from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


# WEO indicator codes
WEO_DEBT_CODE = "GGXWDG_NGDP"     # General government gross debt (% GDP)
WEO_FISCAL_CODE = "GGXCNL_NGDP"   # General government net lending/borrowing (% GDP)
WEO_RESERVES_CODE = "ARA_BOP"     # Reserve assets in months of prospective imports

# WEO release to download — update each April/October when IMF publishes a new edition
_WEO_YEAR = 2024
_WEO_RELEASE = "Oct"


def fetch_imf_weo_panel(
    countries_iso3: dict[str, str],
    start_year: int = 2010,
    cache_dir: Path | None = None,
    include_reserves: bool = False,
) -> pd.DataFrame:
    """Fetch IMF WEO fiscal_balance_gdp, debt_gdp (and optionally reserves_months).

    Downloads the IMF World Economic Outlook dataset via the ``weo`` library.
    The downloaded CSV is cached in ``cache_dir`` (or the system temp dir) so
    subsequent calls skip the network request.

    Parameters
    ----------
    countries_iso3:
        Mapping of {country_name: ISO 3166-1 alpha-3 code}.
    start_year:
        First year to include. Rows before this year are dropped.
    cache_dir:
        Directory to cache the WEO CSV. Defaults to the system temp dir.
    include_reserves:
        If True, also extract ``reserves_months`` from the WEO series
        ``ARA_BOP`` (reserve assets in months of prospective imports).
        Use as a fallback when the World Bank fetch fails.

    Returns
    -------
    pd.DataFrame
        Long-format with columns [date, country, fiscal_balance_gdp, debt_gdp]
        plus ``reserves_months`` when ``include_reserves=True``.
        ``date`` is a pandas Timestamp at annual frequency (Jan 1 of each year).
        The caller is responsible for forward-filling to business-day frequency.
        Returns an empty DataFrame if the ``weo`` library is not installed or
        the download fails.
    """
    try:
        import weo  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "weo is not installed — IMF fiscal data will be unavailable. "
            "Run: pip install weo",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["date", "country", "fiscal_balance_gdp", "debt_gdp"])

    import tempfile, os

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir())
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"weo_{_WEO_YEAR}_{_WEO_RELEASE.lower()}.csv"

    # Download only if not already cached
    if not cache_file.exists():
        try:
            orig_dir = os.getcwd()
            os.chdir(cache_dir)
            downloaded_path, _ = weo.download(_WEO_YEAR, _WEO_RELEASE)
            os.rename(downloaded_path, cache_file)
            os.chdir(orig_dir)
        except Exception as exc:
            warnings.warn(f"IMF WEO download failed: {exc}", stacklevel=2)
            return pd.DataFrame(columns=["date", "country", "fiscal_balance_gdp", "debt_gdp"])

    out_cols = ["date", "country", "fiscal_balance_gdp", "debt_gdp"]
    if include_reserves:
        out_cols.append("reserves_months")

    try:
        w = weo.WEO(str(cache_file))
        debt_wide    = w.getc(WEO_DEBT_CODE)
        fiscal_wide  = w.getc(WEO_FISCAL_CODE)
        reserves_wide = w.getc(WEO_RESERVES_CODE) if include_reserves else None
    except Exception as exc:
        warnings.warn(f"IMF WEO parsing failed: {exc}", stacklevel=2)
        return pd.DataFrame(columns=out_cols)

    iso3_list = list(countries_iso3.values())
    name_map = {v: k for k, v in countries_iso3.items()}

    rows = []
    for iso3 in iso3_list:
        if iso3 not in debt_wide.columns or iso3 not in fiscal_wide.columns:
            warnings.warn(f"IMF WEO: {iso3} not found in dataset", stacklevel=2)
            continue

        debt_s   = debt_wide[iso3].dropna()
        fiscal_s = fiscal_wide[iso3].dropna()

        # PeriodIndex (annual) → Timestamp Jan 1
        years = debt_s.index.union(fiscal_s.index)
        for period in years:
            year = int(str(period))
            if year < start_year:
                continue
            row: dict = {
                "date":               pd.Timestamp(f"{year}-01-01"),
                "country":            name_map[iso3],
                "debt_gdp":           debt_wide.loc[period, iso3] if period in debt_wide.index else float("nan"),
                "fiscal_balance_gdp": fiscal_wide.loc[period, iso3] if period in fiscal_wide.index else float("nan"),
            }
            if include_reserves and reserves_wide is not None and iso3 in reserves_wide.columns:
                row["reserves_months"] = (
                    reserves_wide.loc[period, iso3] if period in reserves_wide.index else float("nan")
                )
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(rows)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df
