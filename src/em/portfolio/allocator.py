from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_benchmark(countries: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(countries), index=countries, dtype=float)


def active_from_scores(scores: pd.Series) -> pd.Series:
    s = scores.copy().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if np.allclose(s.values, 0.0):
        return pd.Series(0.0, index=s.index, dtype=float)

    raw = s / (np.abs(s).sum() + 1e-12)
    raw = raw - raw.mean()  # zero-sum active overlay
    return raw


def allocate_daily(
    scored: pd.DataFrame,
    max_active: float = 0.04,
    cash_buffer: float = 0.00,
) -> pd.DataFrame:
    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "country"])

    out_rows: list[dict] = []

    for dt, g in df.groupby("date"):
        countries = g["country"].tolist()
        bench = equal_weight_benchmark(countries)

        scores = pd.Series(g["score"].values, index=countries, dtype=float)
        active = active_from_scores(scores)

        max_abs = float(np.max(np.abs(active.values))) if len(active) else 0.0
        if max_abs > 0:
            active = active * (max_active / max_abs)

        active = active.clip(lower=-max_active, upper=max_active)

        # Combine with benchmark
        w = bench + active

        # Long-only floor
        w = w.clip(lower=0.0)

        # Normalize to 1 - cash_buffer
        total = float(w.sum())
        if total > 0:
            w = w * ((1.0 - cash_buffer) / total)
        else:
            w = bench * (1.0 - cash_buffer)

        # Local vs hard split heuristic
        score_rank = scores.rank(pct=True)
        local_share = pd.Series(0.35, index=countries, dtype=float)

        # Higher conviction -> allow more local
        local_share = local_share + (score_rank - 0.5) * 0.20  # +/-10%

        # Add FX momentum effect if available
        if "fx_ret_20d" in g.columns:
            fx20 = pd.Series(g["fx_ret_20d"].values, index=countries, dtype=float).fillna(0.0)
            local_share = local_share + np.sign(fx20) * 0.05

        local_share = local_share.clip(0.10, 0.60)

        local_w = w * local_share
        hard_w = w - local_w

        # Duration tilt proxy using US 10Y trend
        duration_tilt = 0.0
        if "us10y_chg_20d" in g.columns:
            vals = pd.Series(g["us10y_chg_20d"]).dropna()
            if len(vals) > 0:
                val = float(vals.iloc[0])
                duration_tilt = float(np.clip(-val * 1.0, -0.25, 0.25))

        for c in countries:
            out_rows.append(
                {
                    "date": dt,
                    "country": c,
                    "score": float(scores.loc[c]),
                    "bench_w": float(bench.loc[c]),
                    "active_w": float(active.loc[c]),
                    "weight": float(w.loc[c]),
                    "hard_w": float(hard_w.loc[c]),
                    "local_w": float(local_w.loc[c]),
                    "local_share": float(local_share.loc[c]),
                    "duration_tilt_years": duration_tilt,
                }
            )

    out = pd.DataFrame(out_rows).sort_values(["date", "country"]).reset_index(drop=True)
    return out