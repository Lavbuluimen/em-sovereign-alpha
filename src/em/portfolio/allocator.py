from __future__ import annotations

import numpy as np
import pandas as pd

from em.country.universe import EMBI_WEIGHTS, FOMC_DATES


def embi_benchmark(countries: list[str]) -> pd.Series:
    return pd.Series(
        [EMBI_WEIGHTS.get(c, 0.0) for c in countries],
        index=countries,
        dtype=float,
    )


def active_from_scores(scores: pd.Series) -> pd.Series:
    s = scores.copy().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if np.allclose(s.values, 0.0):
        return pd.Series(0.0, index=s.index, dtype=float)

    raw = s / (np.abs(s).sum() + 1e-12)
    raw = raw - raw.mean()  # zero-sum active overlay
    return raw


def _classify_regime(
    vix: float,
    dxy_z: float,
    em_oas: float,
) -> str:
    """Three-regime classifier based on VIX, DXY 60d z-score, and EM OAS.

    Returns "Green", "Amber", or "Red".
    """
    if vix > 30 or dxy_z > 2.0 or em_oas > 550:
        return "Red"
    if vix > 20 or dxy_z > 1.0 or em_oas > 400:
        return "Amber"
    return "Green"


def _regime_max_active(regime: str, base_max_active: float) -> float:
    if regime == "Red":
        return 0.005
    if regime == "Amber":
        return min(0.02, base_max_active * 0.5)
    return base_max_active


def _fomc_scale(dt: pd.Timestamp) -> float:
    """Return a max_active scaling factor based on proximity to FOMC meetings.

    Within ±3 calendar days of a scheduled FOMC date, active risk is halved to
    reduce whipsaw around scheduled volatility events.
    """
    fomc_ts = pd.to_datetime(FOMC_DATES)
    min_days = min(abs((dt - d).days) for d in fomc_ts)
    return 0.5 if min_days <= 3 else 1.0


def allocate_daily(
    scored: pd.DataFrame,
    macro_panel: pd.DataFrame | None = None,
    max_active: float = 0.04,
    target_te: float = 0.04,
    cash_buffer: float = 0.00,
) -> pd.DataFrame:
    """Allocate portfolio weights from country scores.

    Parameters
    ----------
    scored:
        Output of build_country_scores(). Must contain: date, country, score.
        Optional columns used if present: fx_ret_20d, fx_carry, yield_60d_chg.
    macro_panel:
        Global macro panel (date, VIX, DXY, em_oas, ...).
        Used for the risk regime overlay and FOMC calendar scaling.
        If None, regime defaults to "Green" (full active risk).
    max_active:
        Maximum absolute active weight vs EMBI benchmark per country (Green regime).
    target_te:
        Target ex-ante tracking error (L2-norm approximation). Active weights
        are scaled so sqrt(sum(active^2)) <= target_te before the per-country cap.
    cash_buffer:
        Fraction of portfolio held in cash (default 0).
    """
    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "country"])

    # ── Pre-compute DXY 60d z-score time series from macro panel ──────────────
    dxy_z_map: dict[pd.Timestamp, float] = {}
    vix_map:   dict[pd.Timestamp, float] = {}
    oas_map:   dict[pd.Timestamp, float] = {}

    if macro_panel is not None:
        mp = macro_panel.copy()
        mp["date"] = pd.to_datetime(mp["date"])
        mp = mp.sort_values("date").set_index("date")

        if "DXY" in mp.columns:
            dxy = mp["DXY"].dropna()
            dxy_60d_z = (
                (dxy - dxy.rolling(252, min_periods=60).mean())
                / dxy.rolling(252, min_periods=60).std()
            ).fillna(0.0)
            dxy_z_map = dxy_60d_z.to_dict()

        if "VIX" in mp.columns:
            vix_map = mp["VIX"].fillna(20.0).to_dict()

        if "em_oas" in mp.columns:
            oas_map = mp["em_oas"].fillna(300.0).to_dict()

    out_rows: list[dict] = []

    for dt, g in df.groupby("date"):
        countries = g["country"].tolist()
        bench = embi_benchmark(countries)

        # ── Regime classification ──────────────────────────────────────────────
        vix    = vix_map.get(dt, 18.0)
        dxy_z  = dxy_z_map.get(dt, 0.0)
        em_oas = oas_map.get(dt, 300.0)
        regime = _classify_regime(float(vix), float(dxy_z), float(em_oas))

        # Effective max_active: shrunk by regime + FOMC proximity
        effective_max = _regime_max_active(regime, max_active)
        effective_max *= _fomc_scale(dt)

        # ── Active overlay ─────────────────────────────────────────────────────
        scores = pd.Series(g["score"].values, index=countries, dtype=float)
        active = active_from_scores(scores)

        # TE constraint: scale so L2-norm of active <= target_te
        te_approx = float(np.sqrt((active ** 2).sum()))
        if te_approx > target_te and te_approx > 0:
            active = active * (target_te / te_approx)

        # Per-country cap at effective_max_active
        max_abs = float(np.max(np.abs(active.values))) if len(active) else 0.0
        if max_abs > 0:
            active = active * (effective_max / max_abs)
        active = active.clip(lower=-effective_max, upper=effective_max)

        # ── Combine with benchmark ─────────────────────────────────────────────
        w = bench + active

        # ── Country caps relative to EMBI benchmark ────────────────────────────
        # Hard ceiling: max(2.5× EMBI weight, 5%)
        # Hard floor  : max(0.25× EMBI weight, 0.5%)
        for c in countries:
            embi_w = float(bench.get(c, 0.0))
            w[c] = min(float(w[c]), max(embi_w * 2.5, 0.05))
            w[c] = max(float(w[c]), max(embi_w * 0.25, 0.005))

        # Normalize to 1 - cash_buffer
        total = float(w.sum())
        if total > 0:
            w = w * ((1.0 - cash_buffer) / total)
        else:
            w = bench * (1.0 - cash_buffer)

        # ── Local vs hard currency split ───────────────────────────────────────
        # Baseline = 0% local (EMBI is hard-currency; any local is an active bet).
        # Add local exposure only when BOTH fx_carry > 0 AND fx_ret_20d > 0.
        # In Amber/Red regimes force local_share = 0 (hard-currency only).
        score_rank = scores.rank(pct=True)
        local_share = pd.Series(0.0, index=countries, dtype=float)

        if regime == "Green":
            if "fx_carry" in g.columns and "fx_ret_20d" in g.columns:
                fx_carry = pd.Series(
                    g["fx_carry"].values, index=countries, dtype=float
                ).fillna(0.0)
                fx_mom = pd.Series(
                    g["fx_ret_20d"].values, index=countries, dtype=float
                ).fillna(0.0)
                both_positive = (fx_carry > 0) & (fx_mom > 0)
                local_share = (
                    both_positive.astype(float)
                    * (score_rank - 0.5).clip(0.0)
                    * 0.40
                )

        local_share = local_share.clip(0.0, 0.40)

        # ── Duration tilt ──────────────────────────────────────────────────────
        # Zero in Amber/Red (risk-off regimes: stay near benchmark duration).
        duration_tilts = pd.Series(0.0, index=countries, dtype=float)
        if regime == "Green" and "yield_60d_chg" in g.columns:
            yc = pd.Series(
                g["yield_60d_chg"].values, index=countries, dtype=float
            ).fillna(0.0)
            duration_tilts = pd.Series(
                np.clip(-yc.values * 1.0, -0.25, 0.25),
                index=countries,
                dtype=float,
            )

        local_w = w * local_share
        hard_w  = w - local_w

        for c in countries:
            out_rows.append(
                {
                    "date":               dt,
                    "country":            c,
                    "score":              float(scores.loc[c]),
                    "regime":             regime,
                    "bench_w":            float(bench.loc[c]),
                    "active_w":           float(active.loc[c]),
                    "weight":             float(w.loc[c]),
                    "hard_w":             float(hard_w.loc[c]),
                    "local_w":            float(local_w.loc[c]),
                    "local_share":        float(local_share.loc[c]),
                    "duration_tilt_years": float(duration_tilts.loc[c]),
                }
            )

    out = pd.DataFrame(out_rows).sort_values(["date", "country"]).reset_index(drop=True)
    return out
