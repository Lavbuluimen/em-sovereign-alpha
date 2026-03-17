from __future__ import annotations

from pathlib import Path

import pandas as pd

from em.features.surprise import run_surprise_analysis


def main() -> None:
    panel = pd.read_parquet("data/processed/country_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    portfolio = None
    port_path = Path("data/processed/portfolio_daily.parquet")
    if port_path.exists():
        portfolio = pd.read_parquet(port_path)
        portfolio["date"] = pd.to_datetime(portfolio["date"])

    result = run_surprise_analysis(panel, portfolio_panel=portfolio)

    out_dir = Path("data/processed/surprises")
    out_dir.mkdir(parents=True, exist_ok=True)

    result.surprise_panel.to_parquet(out_dir / "surprise_panel.parquet")
    result.regression.to_parquet(out_dir / "regression_coefficients.parquet")
    result.heatmap_data.to_parquet(out_dir / "cpi_surprise_heatmap.parquet")

    print(f"✅ Saved surprise data to {out_dir}")
    print(f"\nObservations: {len(result.surprise_panel)}")
    print(f"Countries: {sorted(result.surprise_panel['country'].unique().tolist())}")
    print(f"\nRegression coefficients:")
    print(result.regression.to_string(index=False))


if __name__ == "__main__":
    main()