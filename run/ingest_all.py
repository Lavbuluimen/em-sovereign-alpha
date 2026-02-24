
from __future__ import annotations
from pathlib import Path
from em.ingestion.market import pull_market_features

def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pull_market_features(start="2015-01-01")
    path = out_dir / "mkt_features.parquet"
    df.to_parquet(path)

    print(f"✅ Saved {path}")
    print(df.tail(5))

if __name__ == "__main__":
    main()
