from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from em.models.macro_forecast import run_macro_forecast


# Countries with sufficient monthly CPI + rate data on FRED (OECD members).
# Colombia, Romania, Philippines, Malaysia are excluded (zero FRED coverage).
FORECAST_COUNTRIES = [
    "Brazil", "Mexico", "Chile", "South Africa",
    "Poland", "Hungary", "Indonesia",
]


def main() -> None:
    panel = pd.read_parquet("data/processed/country_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    out_dir = Path("data/processed/macro_forecasts")
    out_dir.mkdir(parents=True, exist_ok=True)

    for country in FORECAST_COUNTRIES:
        try:
            result = run_macro_forecast(panel, country=country)
        except Exception as e:
            print(f"⚠️  {country}: {e}")
            continue

        tag = country.lower().replace(" ", "_")
        result.forecast.to_parquet(out_dir / f"{tag}_forecast.parquet")
        result.irf.to_parquet(out_dir / f"{tag}_irf.parquet")
        result.granger.to_parquet(out_dir / f"{tag}_granger.parquet")
        result.stationarity.to_parquet(out_dir / f"{tag}_stationarity.parquet")

        meta = {
            "country": country,
            "lag_order": int(result.lag_order),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "variables": result.variables,
        }
        (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))

        print(f"✅ {country}: lag={result.lag_order}, AIC={result.aic:.1f}")
        print(f"   Granger significant pairs: "
              f"{result.granger[result.granger['significant']][['cause','effect']].values.tolist()}")

    print(f"\n✅ All forecasts saved to {out_dir}")


if __name__ == "__main__":
    main()