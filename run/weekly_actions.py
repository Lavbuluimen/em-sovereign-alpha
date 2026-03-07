from __future__ import annotations

from pathlib import Path

import pandas as pd


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

    threshold = 0.0025  # 25 bps
    cur["action"] = "HOLD"
    cur.loc[cur["w_change"] >= threshold, "action"] = "BUY / ADD"
    cur.loc[cur["w_change"] <= -threshold, "action"] = "SELL / TRIM"
    cur["trade_size_hint"] = cur["w_change"].fillna(0.0)

    cur = cur.sort_values("trade_size_hint", ascending=False)

    out_path = Path("data/processed") / "weekly_actions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cur.to_parquet(out_path)

    print(f"✅ Saved {out_path}")
    print("\nWeekly Actions (as-of):", last_dt)
    cols = [
        "country",
        "action",
        "weight",
        "w_change",
        "hard_w",
        "local_w",
        "local_share",
        "duration_tilt_years",
    ]
    print(cur[cols].to_string(index=False))


if __name__ == "__main__":
    main()