from __future__ import annotations

from pathlib import Path

from em.country.universe import (
    CDS_WGB_SLUGS,
    FX_TICKERS,
    GLOBAL_MACRO_FRED,
    GLOBAL_MACRO_YAHOO,
    YIELD10Y_FRED,
)
from em.data.cds import build_cds_panel
from em.data.sovereign import build_country_daily_panel, build_global_macro_panel


def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    country_panel = build_country_daily_panel(
        fx_tickers=FX_TICKERS,
        yield10y_fred=YIELD10Y_FRED,
        start="2015-01-01",
        local_duration_years=5.0,
    )

    # Build CDS panel
    cds_df, cds_source = build_cds_panel(
        slug_map=CDS_WGB_SLUGS,
        start="2015-01-01",
        raw_dir="data/raw/cds",
        use_web_fallback=True,
    )

    # Align CDS to business-day country panel
    country_panel = country_panel.sort_values(["date", "country"]).copy()

    cds_daily = cds_df.reindex(sorted(cds_df.index.union(country_panel["date"].unique()))).sort_index().ffill()
    cds_source_daily = cds_source.reindex(cds_daily.index).ffill()

    country_panel["cds_5y"] = country_panel.apply(
        lambda r: cds_daily.at[r["date"], r["country"]]
        if r["country"] in cds_daily.columns and r["date"] in cds_daily.index
        else None,
        axis=1,
    )

    country_panel["cds_source"] = country_panel.apply(
        lambda r: cds_source_daily.at[r["date"], r["country"]]
        if r["country"] in cds_source_daily.columns and r["date"] in cds_source_daily.index
        else None,
        axis=1,
    )

    country_panel["cds_20d_chg"] = country_panel.groupby("country")["cds_5y"].transform(lambda x: x.diff(20))

    country_path = out_dir / "country_daily.parquet"
    country_panel.to_parquet(country_path)

    macro_panel = build_global_macro_panel(
        fred_series=GLOBAL_MACRO_FRED,
        yahoo_tickers=GLOBAL_MACRO_YAHOO,
        start="2015-01-01",
    )

    macro_path = out_dir / "global_macro_daily.parquet"
    macro_panel.to_parquet(macro_path)

    print(f"✅ Saved {country_path}")
    print(f"✅ Saved {macro_path}")
    print(country_panel.tail(10))

    print("\nCountries:", sorted(country_panel["country"].unique().tolist()))
    print("Date range:", country_panel["date"].min(), "->", country_panel["date"].max())
    print("Missing y10y count:", int(country_panel["y10y"].isna().sum()))
    print("Missing cds_5y count:", int(country_panel["cds_5y"].isna().sum()))
    print("Missing fx count:", int(country_panel["fx_level_local_per_usd"].isna().sum()))


if __name__ == "__main__":
    main()