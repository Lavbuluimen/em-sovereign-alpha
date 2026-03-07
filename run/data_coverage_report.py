from __future__ import annotations

from pathlib import Path
import pandas as pd


def report(path: Path, group_col: str | None = None):
    if not path.exists():
        print(f"\n{path.name}: not found")
        return

    df = pd.read_parquet(path)
    print(f"\n=== {path.name} ===")

    if group_col and group_col in df.columns:
        cov = (
            df.groupby(group_col)
            .apply(lambda x: x.notna().mean())
        )
        print(cov)
    else:
        cov = pd.DataFrame({
            "column": df.columns,
            "coverage": [df[c].notna().mean() for c in df.columns],
        }).sort_values("coverage")
        print(cov)


def main():
    base = Path("data/processed")
    report(base / "country_daily.parquet", group_col="country")
    report(base / "country_scores_daily.parquet", group_col="country")
    report(base / "portfolio_daily.parquet", group_col="country")
    report(base / "mkt_features.parquet")


if __name__ == "__main__":
    main()