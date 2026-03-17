from __future__ import annotations

from pathlib import Path

import pandas as pd

from em.country.score import build_country_scores


def main() -> None:
    panel = pd.read_parquet("data/processed/country_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    macro_panel = pd.read_parquet("data/processed/global_macro_daily.parquet")
    macro_panel["date"] = pd.to_datetime(macro_panel["date"])

    scored = build_country_scores(panel, macro_panel=macro_panel)

    out_path = Path("data/processed") / "country_scores_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path)

    print(f"✅ Saved {out_path}")
    last_dt = scored["date"].max()
    snap = scored[scored["date"] == last_dt].sort_values("score", ascending=False)

    cols = [c for c in [
        "country", "score", "signal_confidence",
        "score_raw", "us_real_rate_tilt",
        "spread_value_blended_z", "spread_mom_blend_z",
        "fx_carry_z", "real_yield_z", "commodity_tot_z",
        "credit_quality_score",
        "real_yield_coverage_60d", "fx_carry_coverage_60d",
        "hard_spread_proxy", "y10y", "fx_usd_ret",
        "real_yield", "fx_carry",
    ] if c in snap.columns]

    print("\nLatest scores:", last_dt)
    print(snap[cols].to_string(index=False))


if __name__ == "__main__":
    main()
