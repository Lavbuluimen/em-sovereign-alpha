from __future__ import annotations

from pathlib import Path

import pandas as pd


# Round-trip transaction cost estimates for EM sovereign bonds.
# Applied proportionally to the size of each weight change.
_TC_HARD_ONE_WAY  = 0.0015   # 15 bp one-way — hard-currency benchmark bonds
_TC_LOCAL_ONE_WAY = 0.0010   # 10 bp one-way — liquid local-currency bonds


def main() -> None:
    port = pd.read_parquet("data/processed/portfolio_daily.parquet")
    port["date"] = pd.to_datetime(port["date"])
    port = port.sort_values(["country", "date"])

    # Build weekly snapshots using last available trading day per country-week
    port["week"] = port["date"].dt.to_period("W").astype(str)

    rows = []
    for (country, week), g in port.groupby(["country", "week"]):
        g = g.sort_values("date")
        rows.append(g.iloc[-1])

    snap = pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)
    snap["w_change"] = snap.groupby("country")["weight"].diff()

    last_dt = snap["date"].max()
    cur = snap[snap["date"] == last_dt].copy()

    # Phase 1.5: threshold raised from 25 bp to 50 bp
    threshold = 0.0050  # 50 bps

    cur["action"] = "HOLD"
    cur.loc[cur["w_change"] >= threshold,  "action"] = "BUY / ADD"
    cur.loc[cur["w_change"] <= -threshold, "action"] = "SELL / TRIM"
    cur["trade_size_hint"] = cur["w_change"].fillna(0.0)

    # Phase 3.3: transaction cost model
    # Estimate one-way cost as a fraction of the total position being traded.
    # cost = |w_change| / weight × (hard_pct × TC_HARD + local_pct × TC_LOCAL)
    total_w = cur["weight"].abs().replace(0, float("nan"))
    hard_pct  = (cur["hard_w"]  / total_w).fillna(0.0).clip(0.0, 1.0)
    local_pct = (cur["local_w"] / total_w).fillna(0.0).clip(0.0, 1.0)
    blended_tc = hard_pct * _TC_HARD_ONE_WAY + local_pct * _TC_LOCAL_ONE_WAY
    cur["tc_estimate"] = (cur["w_change"].abs() * blended_tc).fillna(0.0)

    # Flag trades where the estimated cost exceeds half the weight change —
    # a rough signal that alpha may not cover execution cost.
    cur["cost_aware_action"] = cur["action"]
    marginal_threshold = cur["w_change"].abs() * 0.5
    high_cost = (cur["action"] != "HOLD") & (cur["tc_estimate"] > marginal_threshold)
    cur.loc[high_cost, "cost_aware_action"] = "HOLD (cost)"

    cur = cur.sort_values("trade_size_hint", ascending=False)

    out_path = Path("data/processed") / "weekly_actions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cur.to_parquet(out_path)

    print(f"✅ Saved {out_path}")
    print("\nWeekly Actions (as-of):", last_dt)
    cols = [
        "country",
        "action",
        "cost_aware_action",
        "weight",
        "w_change",
        "tc_estimate",
        "hard_w",
        "local_w",
        "local_share",
        "duration_tilt_years",
        "regime",
    ]
    available = [c for c in cols if c in cur.columns]
    print(cur[available].to_string(index=False))


if __name__ == "__main__":
    main()
