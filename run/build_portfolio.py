
from __future__ import annotations

from pathlib import Path
import pandas as pd

from em.portfolio.allocator import allocate_daily


def main() -> None:
    panel = pd.read_parquet("data/processed/country_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country", "date"])

    scored = pd.read_parquet("data/processed/country_scores_daily.parquet")
    scored["date"] = pd.to_datetime(scored["date"])

    # Build US10y 20d trend (same across countries per date)
    us = (
        panel[["date", "us10y"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    us["us10y_chg_20d"] = us["us10y"].diff(20)
    us = us.reset_index()

    # scored ALREADY contains fx_ret_20d (computed in score.py)
    scored2 = scored.merge(us[["date", "us10y_chg_20d"]], on="date", how="left")

    port = allocate_daily(scored2, max_active=0.04, cash_buffer=0.00)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "portfolio_daily.parquet"
    port.to_parquet(out_path)

    print(f"✅ Saved {out_path}")
    last_dt = port["date"].max()
    snap = port[port["date"] == last_dt].sort_values("weight", ascending=False)

    print("\nLatest snapshot:", last_dt)
    print(snap[["country","score","weight","hard_w","local_w","local_share","duration_tilt_years"]].to_string(index=False))


if __name__ == "__main__":
    main()
