from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error


def compute_forward_returns(price: pd.Series, horizon: int = 5, log: bool = True) -> pd.Series:
    """Compute forward returns (shifted into the future by horizon).

    If log=True, returns log(p[t+h]/p[t]); else simple pct change.
    """
    if log:
        ret = np.log(price.shift(-horizon) / price)
    else:
        ret = price.shift(-horizon).pct_change(periods=horizon)
    ret.name = f"ret_fwd_{horizon}"
    return ret


@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    predictions: pd.DataFrame  # columns: ['y_true','y_pred'] indexed by date


def walk_forward_regression(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> BacktestResult:
    """Walk-forward Ridge regression with expanding window and OOS metrics."""
    # Align and drop NA
    df = X.join(y.to_frame(name="y"), how="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping data for features and target after dropping NA")

    Xc = df.drop(columns=["y"])  # features
    yc = df["y"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_preds = pd.Series(index=yc.index, dtype=float)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", Ridge(alpha=alpha, random_state=0)),
    ])

    for train_idx, test_idx in tscv.split(Xc):
        X_train, X_test = Xc.iloc[train_idx], Xc.iloc[test_idx]
        y_train, y_test = yc.iloc[train_idx], yc.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        oos_preds.iloc[test_idx] = preds

    valid = ~oos_preds.isna()
    y_true = yc[valid]
    y_pred = oos_preds[valid]

    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else np.nan
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean()) if len(y_true) > 0 else np.nan

    metrics = {"r2": float(r2), "mae": float(mae), "directional_accuracy": dir_acc}
    preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    return BacktestResult(metrics=metrics, predictions=preds_df)

