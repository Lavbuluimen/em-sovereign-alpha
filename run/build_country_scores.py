from __future__ import annotations

from pathlib import Path

import pandas as pd

from em.country.score import build_country_scores


def main() -> None:
    panel = pd.read_parquet("data/processed/country_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    scored = build_country_scores(panel)

    out_path = Path("data/processed") / "country_scores_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path)

    print(f"✅ Saved {out_path}")
    last_dt = scored["date"].max()
    snap = scored[scored["date"] == last_dt].sort_values("score", ascending=False)

    cols = [c for c in [
        "country", "score", "signal_confidence",
        "score_raw", "hard_spread_proxy", "y10y", "fx_usd_ret",
        "real_yield", "fx_carry",
    ] if c in snap.columns]

    print("\nLatest scores:", last_dt)
    print(snap[cols].to_string(index=False))


if __name__ == "__main__":
    main()
