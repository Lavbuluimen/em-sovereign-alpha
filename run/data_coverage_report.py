import pandas as pd

panel = pd.read_parquet("data/processed/country_daily.parquet")

coverage = panel.groupby("country").agg({
    "y10y": lambda x: x.notna().mean(),
    "fx_usd_ret": lambda x: x.notna().mean(),
    "hard_spread_proxy": lambda x: x.notna().mean()
})

coverage = coverage.rename(columns=lambda c: f"{c}_coverage")

print("\nData coverage by country:\n")
print(coverage.sort_values("y10y_coverage"))