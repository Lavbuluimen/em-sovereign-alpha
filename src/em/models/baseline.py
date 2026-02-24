
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["Brent_ret", "WTI_ret", "Copper_ret", "Gold_ret", "DGS10_chg"]
TARGET_COL = "EMB_ret"


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]


def load_features(path: str = "data/processed/mkt_features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_index()
    return df


def make_xy(df: pd.DataFrame, feature_cols: Iterable[str] = FEATURE_COLS) -> tuple[pd.DataFrame, pd.Series]:
    """
    Predict next-day EMB_ret using same-day features.
    y_t = EMB_ret_{t+1}
    X_t = features_t
    """
    X = df.loc[:, list(feature_cols)].copy()
    y = df[TARGET_COL].rolling(5).sum().shift(-5)  # predicts 5-day forward return
    # drop rows where any needed values are missing
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    X = data[list(feature_cols)]
    y = data["y"]
    return X, y


def walk_forward_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    min_train: int = 252,      # ~1 trading year
    refit_every: int = 5,      # refit weekly (approx)
    alpha: float = 10.0,       # ridge strength
) -> pd.Series:
    """
    Expanding-window walk-forward prediction.
    Predict y[t] using data strictly before t.
    """
    idx = X.index
    n = len(idx)
    preds = pd.Series(index=idx, dtype=float)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=0)),
    ])

    last_fit = None
    for i in range(min_train, n):
        # refit schedule
        if (last_fit is None) or ((i - last_fit) >= refit_every):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            model.fit(X_train.values, y_train.values)
            last_fit = i

        x_i = X.iloc[i:i+1].values
        preds.iloc[i] = float(model.predict(x_i)[0])

    return preds


def compute_metrics(df_pred: pd.DataFrame) -> dict[str, float]:
    """
    Metrics for:
      - forecast accuracy (hit rate)
      - strategy based on sign(pred)
    Assumes df_pred has columns: y_true, y_pred
    """
    d = df_pred.dropna().copy()
    y_true = d["y_true"].values
    y_pred = d["y_pred"].values

    # hit rate: sign match (ignore zeros)
    hit = np.sign(y_true) == np.sign(y_pred)
    hit_rate = float(np.mean(hit))

    # simple strategy: take sign(pred) exposure in EMB next day
    strat = np.sign(y_pred) * y_true
    strat_mean = float(np.mean(strat))
    strat_std = float(np.std(strat, ddof=1)) if len(strat) > 1 else np.nan

    # daily sharpe -> annualized (sqrt(252))
    sharpe = float((strat_mean / strat_std) * np.sqrt(252)) if strat_std and strat_std > 0 else np.nan

    # out-of-sample R^2 (vs zero-mean benchmark)
    # R2 = 1 - SSE/SST
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - sse / sst) if sst > 0 else np.nan

    return {
        "n_obs": float(len(d)),
        "hit_rate": hit_rate,
        "strategy_mean_daily": strat_mean,
        "strategy_vol_daily": strat_std,
        "strategy_sharpe_ann": sharpe,
        "oos_r2": r2,
    }


def run_baseline_backtest(
    features_path: str = "data/processed/mkt_features.parquet",
    min_train: int = 252,
    refit_every: int = 5,
    alpha: float = 10.0,
) -> BacktestResult:
    df = load_features(features_path)
    X, y = make_xy(df)

    y_pred = walk_forward_ridge(X, y, min_train=min_train, refit_every=refit_every, alpha=alpha)

    out = pd.DataFrame(index=X.index)
    out["y_true"] = y
    out["y_pred"] = y_pred

    metrics = compute_metrics(out)
    return BacktestResult(predictions=out, metrics=metrics)
