from __future__ import annotations

from pathlib import Path
import pandas as pd


def report_file(path: Path, group_col: str | None = None, cols: list[str] | None = None) -> None:
    if not path.exists():
        print(f"\n{path.name}: not found")
        return

    df = pd.read_parquet(path)
    print(f"\n=== {path.name} ===")

    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df = df[[group_col] + cols] if group_col else df[cols]

    if group_col and group_col in df.columns:
        value_cols = [c for c in df.columns if c != group_col]
        cov = df.groupby(group_col)[value_cols].apply(lambda x: x.notna().mean())
        print(cov)
    else:
        cov = pd.DataFrame(
            {
                "column": df.columns,
                "coverage": [df[c].notna().mean() for c in df.columns],
            }
        ).sort_values("coverage")
        print(cov)


def main() -> None:
    base = Path("data/processed")

    report_file(
        base / "country_daily.parquet",
        group_col="country",
        cols=["y10y", "y10y_source", "cds_5y", "cds_source", "fx_usd_ret", "hard_spread_proxy"],
    )

    report_file(
        base / "country_scores_daily.parquet",
        group_col="country",
        cols=["score", "signal_confidence"],
    )

    report_file(
        base / "portfolio_daily.parquet",
        group_col="country",
        cols=["weight", "hard_w", "local_w", "local_share"],
    )

    report_file(base / "global_macro_daily.parquet")


if __name__ == "__main__":
    main()