from __future__ import annotations

from pathlib import Path

import pandas as pd

from em.country.universe import (
    COUNTRY_ISO3,
    CPI_YOY_FRED,
    FISCAL_WB_INDICATORS,
    FX_TICKERS,
    GLOBAL_MACRO_FRED,
    GLOBAL_MACRO_YAHOO,
    IFS_COUNTRIES,
    IMF_WEO_FISCAL_COLS,
    SHORT_RATE_FRED,
    YIELD10Y_STOOQ,
)
from em.data.embi import build_embi_spread_panel
from em.data.fred import fetch_many_fred_series
from em.data.ifs import fetch_ifs_panel
from em.data.imf import fetch_imf_weo_panel
from em.data.sovereign import build_country_daily_panel, build_global_macro_panel
from em.data.worldbank import fetch_worldbank_panel


def _merge_monthly_fred(
    country_panel: pd.DataFrame,
    series_map: dict[str, str],
    col_name: str,
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Fetch a monthly FRED series per country, forward-fill to daily, merge into panel.

    series_map keys must match the 'country' values in country_panel.
    Entries with key "US" are skipped (handled separately as a scalar join).
    """
    country_keys = {k: v for k, v in series_map.items() if k != "US"}
    raw = fetch_many_fred_series(country_keys, start=start)

    if raw.empty:
        country_panel[col_name] = float("nan")
        return country_panel

    bday_idx = pd.bdate_range(
        start=country_panel["date"].min(), end=country_panel["date"].max()
    )
    daily = raw.reindex(bday_idx).ffill()

    long = (
        daily
        .reset_index()
        .rename(columns={"index": "date"})
        .melt(id_vars="date", var_name="country", value_name=col_name)
    )
    country_panel = country_panel.merge(long, on=["date", "country"], how="left")
    return country_panel


def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build country panel: FX returns, 10Y yields, hard spread proxy
    country_panel = build_country_daily_panel(
        fx_tickers=FX_TICKERS,
        yield10y_stooq=YIELD10Y_STOOQ,
        start="2015-01-01",
        local_duration_years=5.0,
    )

    # Add EMBI spread proxy columns (derived from hard_spread_proxy)
    country_panel = build_embi_spread_panel(country_panel)

    # ── Monthly carry / value signals ────────────────────────────────────────
    # CPI YoY (%) — used to compute real_yield = y10y - cpi_yoy
    country_panel = _merge_monthly_fred(
        country_panel, CPI_YOY_FRED, "cpi_yoy", start="2014-01-01"
    )

    # Local short-term interbank rate (%) — used to compute fx_carry
    country_panel = _merge_monthly_fred(
        country_panel, SHORT_RATE_FRED, "local_short_rate", start="2014-01-01"
    )

    # US short rate — same value for every country on a given date
    us_short_raw = fetch_many_fred_series({"US": SHORT_RATE_FRED["US"]}, start="2014-01-01")
    if not us_short_raw.empty:
        bday_idx = pd.bdate_range(
            start=country_panel["date"].min(), end=country_panel["date"].max()
        )
        us_short_daily = us_short_raw["US"].reindex(bday_idx).ffill()
        country_panel["us_short_rate"] = country_panel["date"].map(us_short_daily)
    else:
        country_panel["us_short_rate"] = float("nan")

    # US CPI YoY — used to compute us_real_yield for the dashboard bar chart baseline
    if "US" in CPI_YOY_FRED:
        us_cpi_raw = fetch_many_fred_series({"US": CPI_YOY_FRED["US"]}, start="2014-01-01")
        if not us_cpi_raw.empty:
            bday_idx = pd.bdate_range(
                start=country_panel["date"].min(), end=country_panel["date"].max()
            )
            us_cpi_daily = us_cpi_raw["US"].reindex(bday_idx).ffill()
            country_panel["us_cpi_yoy"] = country_panel["date"].map(us_cpi_daily)
        else:
            country_panel["us_cpi_yoy"] = float("nan")

    # ── IMF IFS: fill CPI and short rate for non-OECD countries ──────────────
    # Colombia, Malaysia, Philippines, Romania have no FRED coverage.
    # Fetch from IMF IFS SDMX API and fill NaN rows only (FRED takes precedence).
    ifs_panel = fetch_ifs_panel(IFS_COUNTRIES, start="2014-01-01")
    if not ifs_panel.empty:
        bday_idx = pd.bdate_range(
            start=country_panel["date"].min(), end=country_panel["date"].max()
        )
        for col in ("cpi_yoy", "local_short_rate"):
            for country_name, iso2 in IFS_COUNTRIES.items():
                country_ifs = ifs_panel[ifs_panel["country"] == country_name][["date", col]].copy()
                if country_ifs.empty or country_ifs[col].isna().all():
                    continue
                country_ifs = country_ifs.set_index("date")[col]
                country_ifs.index = pd.DatetimeIndex(country_ifs.index)
                daily = country_ifs.reindex(bday_idx).ffill()
                mask = (country_panel["country"] == country_name) & country_panel[col].isna()
                country_panel.loc[mask, col] = country_panel.loc[mask, "date"].map(daily)

    # Derived carry / value features
    country_panel["real_yield"] = country_panel["y10y"] - country_panel["cpi_yoy"]
    country_panel["fx_carry"] = (
        country_panel["local_short_rate"] - country_panel["us_short_rate"]
    )
    country_panel["us_real_yield"] = country_panel["us10y"] - country_panel["us_cpi_yoy"]

    # ── Fiscal fundamentals: IMF WEO (debt, fiscal balance) ──────────────────
    # Covers all 11 countries; World Bank GC.DOD.TOTL.GD.ZS misses several.
    # Annual series, forward-filled within each country group to daily.
    imf_panel = fetch_imf_weo_panel(COUNTRY_ISO3, start_year=2010)
    if not imf_panel.empty:
        country_panel = country_panel.merge(imf_panel, on=["date", "country"], how="left")
        for col in IMF_WEO_FISCAL_COLS:
            country_panel[col] = country_panel.groupby("country")[col].transform(
                lambda x: x.ffill()
            )
    else:
        for col in IMF_WEO_FISCAL_COLS:
            country_panel[col] = float("nan")

    # ── Reserves from World Bank (IMF WEO does not carry this series) ─────────
    wb_panel = fetch_worldbank_panel(FISCAL_WB_INDICATORS, COUNTRY_ISO3, start_year=2010)
    if not wb_panel.empty:
        country_panel = country_panel.merge(wb_panel, on=["date", "country"], how="left")
        for col in FISCAL_WB_INDICATORS:
            country_panel[col] = country_panel.groupby("country")[col].transform(
                lambda x: x.ffill()
            )
    else:
        for col in FISCAL_WB_INDICATORS:
            country_panel[col] = float("nan")

    # Cross-sectional ranks (1 = best) among countries with data on each date
    country_panel["real_yield_rank"] = (
        country_panel.groupby("date")["real_yield"]
        .rank(ascending=False, na_option="bottom")
    )
    country_panel["fx_carry_rank"] = (
        country_panel.groupby("date")["fx_carry"]
        .rank(ascending=False, na_option="bottom")
    )

    # Save country panel
    country_path = out_dir / "country_daily.parquet"
    country_panel.to_parquet(country_path)

    # Build global macro panel (US rates, VIX, commodities + global EM OAS)
    macro_panel = build_global_macro_panel(
        fred_series=GLOBAL_MACRO_FRED,
        yahoo_tickers=GLOBAL_MACRO_YAHOO,
        start="2015-01-01",
    )

    # Derived feature: EM HY/IG spread — measures risk appetite within EM credit
    if "em_hy_oas" in macro_panel.columns and "em_oas" in macro_panel.columns:
        macro_panel["em_hy_ig_spread"] = macro_panel["em_hy_oas"] - macro_panel["em_oas"]

    macro_path = out_dir / "global_macro_daily.parquet"
    macro_panel.to_parquet(macro_path)

    print(f"✅ Saved {country_path}")
    print(f"✅ Saved {macro_path}")
    print(country_panel.tail(10))

    print("\nCountries:", sorted(country_panel["country"].unique().tolist()))
    print("Date range:", country_panel["date"].min(), "->", country_panel["date"].max())
    print("Missing y10y count:",            int(country_panel["y10y"].isna().sum()))
    print("Missing embi_spread_proxy count:", int(country_panel["embi_spread_proxy"].isna().sum()))
    print("Missing fx count:",              int(country_panel["fx_level_local_per_usd"].isna().sum()))
    print("Missing cpi_yoy count:",         int(country_panel["cpi_yoy"].isna().sum()))
    print("Missing local_short_rate count:", int(country_panel["local_short_rate"].isna().sum()))
    print("Missing real_yield count:",         int(country_panel["real_yield"].isna().sum()))
    print("Missing fx_carry count:",           int(country_panel["fx_carry"].isna().sum()))
    print("Missing fiscal_balance_gdp count:", int(country_panel["fiscal_balance_gdp"].isna().sum()))
    print("Missing debt_gdp count:",           int(country_panel["debt_gdp"].isna().sum()))
    print("Missing reserves_months count:",    int(country_panel["reserves_months"].isna().sum()))


if __name__ == "__main__":
    main()
