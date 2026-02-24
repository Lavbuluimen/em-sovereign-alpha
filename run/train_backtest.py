
from __future__ import annotations

from pathlib import Path
import json

from em.models.baseline import run_baseline_backtest

def main() -> None:
    res = run_baseline_backtest(
        features_path="data/processed/mkt_features.parquet",
        min_train=252,
        refit_every=5,
        alpha=10.0,
    )

    out_dir = Path("data/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "baseline_predictions.parquet"
    metrics_path = out_dir / "baseline_metrics.json"

    res.predictions.to_parquet(pred_path)
    metrics_path.write_text(json.dumps(res.metrics, indent=2))

    print("✅ Saved predictions:", pred_path)
    print("✅ Saved metrics:", metrics_path)
    print("\n--- Metrics ---")
    for k, v in res.metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
