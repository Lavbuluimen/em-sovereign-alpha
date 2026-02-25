
from __future__ import annotations

from pathlib import Path
import pandas as pd

from em.country.universe import FX_TICKERS, YIELD10Y_TICKERS
from em.country.data import build_country_daily_panel

def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_country_daily_panel(
        fx_tickers=FX_TICKERS,
        yield10y_tickers=YIELD10Y_TICKERS,
        start="2015-01-01",
        local_duration_years=5.0,
    )

    path = out_dir / "country_daily.parquet"
    panel.to_parquet(path)

    print(f"✅ Saved {path}")
    print(panel.tail(10))
    print("\nCountries:", sorted(panel["country"].unique().tolist()))
    print("Date range:", panel["date"].min(), "->", panel["date"].max())
    print("Missing y10y count:", int(panel["y10y"].isna().sum()))
    print("Missing fx count:", int(panel["fx_level_local_per_usd"].isna().sum()))

if __name__ == "__main__":
    main()
