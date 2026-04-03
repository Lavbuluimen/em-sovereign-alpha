"""Run the sovereign portfolio backtest and save results to data/backtests/.

Outputs
-------
data/backtests/sovereign_returns.parquet    daily port / bench / active returns
data/backtests/sovereign_metrics.json       full metrics dict (incl. year-by-year)
data/backtests/sovereign_country_attr.parquet
data/backtests/sovereign_regime_attr.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from em.backtest.sovereign import (
    build_daily_returns,
    compute_metrics,
    country_attribution,
    regime_attribution,
)
from em.ingestion.market import pull_market_features


OUT_DIR = Path("data/backtests")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load inputs
    # ------------------------------------------------------------------ #
    print("Loading portfolio weights …")
    portfolio = pd.read_parquet("data/processed/portfolio_daily.parquet")
    portfolio["date"] = pd.to_datetime(portfolio["date"])

    print("Loading country panel …")
    country_panel = pd.read_parquet("data/processed/country_daily.parquet")
    country_panel["date"] = pd.to_datetime(country_panel["date"])

    print("Fetching EMB ETF returns (Yahoo Finance) …")
    mkt = pull_market_features(start="2015-01-01")
    emb_ret: pd.Series = mkt["EMB_ret"].rename("bench_ret")
    emb_ret.index = pd.to_datetime(emb_ret.index)

    # ------------------------------------------------------------------ #
    # 2. Build daily returns (lag weights, apply return proxies)
    # ------------------------------------------------------------------ #
    print("Computing daily returns …")
    returns = build_daily_returns(portfolio, country_panel, emb_ret)
    returns.index = pd.to_datetime(returns.index)

    out_ret = OUT_DIR / "sovereign_returns.parquet"
    returns.reset_index().to_parquet(out_ret, index=False)
    print(f"  Saved {out_ret}  ({len(returns)} days, "
          f"{returns.index.min().date()} → {returns.index.max().date()})")

    # ------------------------------------------------------------------ #
    # 3. Metrics
    # ------------------------------------------------------------------ #
    print("Computing metrics …")
    metrics = compute_metrics(returns)

    out_metrics = OUT_DIR / "sovereign_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved {out_metrics}")

    # ------------------------------------------------------------------ #
    # 4. Attribution
    # ------------------------------------------------------------------ #
    print("Computing country attribution …")
    c_attr = country_attribution(portfolio, country_panel, returns)
    out_cattr = OUT_DIR / "sovereign_country_attr.parquet"
    c_attr.to_parquet(out_cattr, index=False)
    print(f"  Saved {out_cattr}")

    print("Computing regime attribution …")
    r_attr = regime_attribution(portfolio, returns)
    out_rattr = OUT_DIR / "sovereign_regime_attr.parquet"
    r_attr.to_parquet(out_rattr, index=False)
    print(f"  Saved {out_rattr}")

    # ------------------------------------------------------------------ #
    # 5. Summary print
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SOVEREIGN BACKTEST  —  Summary")
    print("=" * 60)
    print(f"  Period:              {metrics['start_date']}  →  {metrics['end_date']}")
    print(f"  Trading days:        {metrics['n_days']}")
    print(f"  Ann. return  Port:   {metrics['ann_return_port']:+.2%}   "
          f"Bench: {metrics['ann_return_bench']:+.2%}   "
          f"Active: {metrics['ann_return_active']:+.2%}")
    print(f"  Sharpe (port):       {metrics['sharpe_port']:.2f}")
    print(f"  Info ratio:          {metrics['information_ratio']:.2f}")
    print(f"  Tracking error:      {metrics['tracking_error']:.2%}")
    print(f"  Max DD (port):       {metrics['max_dd_port']:.2%}   "
          f"Bench: {metrics['max_dd_bench']:.2%}")
    print(f"  Calmar (port):       {metrics['calmar_port']:.2f}")
    print(f"  Monthly win rate:    {metrics['monthly_win_rate']:.1%}")

    print("\n  Year-by-year (annualised):")
    print(f"  {'Year':>4}  {'Port':>7}  {'Bench':>7}  {'Active':>7}  {'Sharpe':>6}  {'IR':>6}")
    for yr, y in sorted(metrics["yearly"].items()):
        print(f"  {yr:>4}  {y['port']:>+7.2%}  {y['bench']:>+7.2%}  "
              f"{y['active']:>+7.2%}  {y['sharpe']:>6.2f}  {y['ir']:>6.2f}")

    print("\n  Country attribution (top 5 by |active contribution|):")
    top5 = c_attr.reindex(
        c_attr["active_contribution_ann"].abs().nlargest(5).index
    )
    for _, row in top5.iterrows():
        print(f"    {row['country']:<16}  {row['active_contribution_ann']:>+.4f}")

    print("\n  Regime attribution:")
    for _, row in r_attr.sort_values("regime").iterrows():
        print(f"    {row['regime']:<8}  n={row['n_days']:>4}  "
              f"port={row['ann_port_ret']:>+.2%}  active={row['ann_active_ret']:>+.2%}  "
              f"IR={row['ir']:>6.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
