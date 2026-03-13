from __future__ import annotations

from pathlib import Path

from em.country.universe import (
    FX_TICKERS,
    GLOBAL_MACRO_FRED,
    GLOBAL_MACRO_YAHOO,
    YIELD10Y_STOOQ,
)
from em.data.embi import build_embi_spread_panel
from em.data.sovereign import build_country_daily_panel, build_global_macro_panel


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
    print("Missing y10y count:", int(country_panel["y10y"].isna().sum()))
    print("Missing embi_spread_proxy count:", int(country_panel["embi_spread_proxy"].isna().sum()))
    print("Missing fx count:", int(country_panel["fx_level_local_per_usd"].isna().sum()))


if __name__ == "__main__":
    main()
