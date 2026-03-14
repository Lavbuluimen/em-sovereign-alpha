
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

    port = allocate_daily(scored, max_active=0.04, cash_buffer=0.00)

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
