from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from em.portfolio.allocator import allocate_daily

# Phase 3.4: minimum calendar days before allowing an active_w sign reversal.
_MIN_HOLD_DAYS = 5


def _apply_min_holding(port: pd.DataFrame) -> pd.DataFrame:
    """Suppress active-weight sign reversals that occur within MIN_HOLD_DAYS.

    Iterates per country chronologically. When a reversal is detected within
    the minimum holding window, the new row's active_w is reverted to the
    previous sign (magnitude kept). Weights are adjusted accordingly.
    """
    port = port.sort_values(["country", "date"]).copy()

    for country, grp in port.groupby("country"):
        idxs = grp.index.tolist()
        last_flip_date: pd.Timestamp | None = None
        prev_sign: float = 0.0

        for idx in idxs:
            cur_aw   = float(port.loc[idx, "active_w"])
            cur_sign = float(np.sign(cur_aw))
            cur_date = pd.Timestamp(port.loc[idx, "date"])

            if cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign:
                # Detected a sign reversal
                if last_flip_date is not None:
                    days_since = (cur_date - last_flip_date).days
                    if days_since < _MIN_HOLD_DAYS:
                        # Revert: keep the previous direction, same magnitude
                        port.loc[idx, "active_w"] = abs(cur_aw) * prev_sign
                        # Recompute weight from bench + revised active
                        port.loc[idx, "weight"] = (
                            float(port.loc[idx, "bench_w"]) + float(port.loc[idx, "active_w"])
                        )
                        continue  # don't update flip date or prev_sign
                last_flip_date = cur_date

            if cur_sign != 0:
                prev_sign = cur_sign

    return port


def main() -> None:
    scored = pd.read_parquet("data/processed/country_scores_daily.parquet")
    scored["date"] = pd.to_datetime(scored["date"])

    macro_panel = pd.read_parquet("data/processed/global_macro_daily.parquet")
    macro_panel["date"] = pd.to_datetime(macro_panel["date"])

    port = allocate_daily(
        scored,
        macro_panel=macro_panel,
        max_active=0.04,
        target_te=0.04,
        cash_buffer=0.00,
    )

    # Phase 3.4: enforce minimum holding period on active weight reversals
    port = _apply_min_holding(port)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "portfolio_daily.parquet"
    port.to_parquet(out_path)

    print(f"✅ Saved {out_path}")
    last_dt = port["date"].max()
    snap = port[port["date"] == last_dt].sort_values("weight", ascending=False)

    print("\nLatest snapshot:", last_dt)
    cols = [c for c in [
        "country", "score", "regime", "weight", "bench_w", "active_w",
        "hard_w", "local_w", "local_share", "duration_tilt_years",
    ] if c in snap.columns]
    print(snap[cols].to_string(index=False))


if __name__ == "__main__":
    main()
