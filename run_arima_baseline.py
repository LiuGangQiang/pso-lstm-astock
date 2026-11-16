# -*- coding: utf-8 -*-
"""Pre-M5 Reinforcement Experiment (B1): ARIMA baseline (AIC order selection), implemented purely in NumPy.

Motivation: The statistical baseline for S1–S4 is only Persistence, and reviewers may question whether "the model is only worse than the naive baseline."
Add a standard baseline from the literature: Apply ARIMA(p,1,0) to log(close), with p ∈ {0..4} and order selected based on the AIC from the training fold,
Recursive multi-step prediction of the t+h closing price; fully consistent with the walk-forward 5-fold protocol (scaler calibration matches the baseline:
The prediction is evaluated in the min-max inverse-scaled space, directly comparable to Persistence/APSO-LSTM).

Note: The point forecast of GARCH(1,1) is the same as that of the mean model (conditional mean recursion), and there is no incremental improvement in the RMSE comparison.
Therefore, the point forecast component of the strengthened baseline is represented by ARIMA, and GARCH is only discussed in the paper.

Output: output/arima_parts/<stock>.csv (each row = stock × horizon × fold),
Subsequently, the s4_cells structure is incorporated into output/b1_cells.csv for ratio comparison."""
import argparse
import os

import numpy as np
import pandas as pd

from pso_lstm import metrics
from run_m1_baseline import POOL, fold_bounds, load_series
from select_stocks import CANDIDATES

HORIZONS = [1, 5, 20]
PARTS_DIR = os.path.join("output", "arima_parts")


def aic_ar(diff: np.ndarray, p: int) -> float:
    """AIC for a least-squares AR(p) fit on the differenced series (σ² is the residual variance; includes an intercept)."""
    n = len(diff)
    if p == 0:
        resid = diff - diff.mean()
        sigma2 = max(np.mean(resid ** 2), 1e-12)
        return n * np.log(sigma2) + 2 * 1
    X, y = [], []
    for t in range(p, n):
        X.append(np.concatenate([[1.0], diff[t - p:t]]))
        y.append(diff[t])
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = max(np.mean(resid ** 2), 1e-12)
    k = p + 2          # p AR coefficients + constant + σ²
    return len(y) * np.log(sigma2) + 2 * k


def arima_forecast(train_log: np.ndarray, horizon: int) -> np.ndarray:
    """For training-fold log prices: first-difference, select p by AIC, recursively forecast horizon steps, and return log-scale predictions."""
    diff = np.diff(train_log)
    p_best = min(range(5), key=lambda p: aic_ar(diff, p))
    n = len(diff)
    if p_best == 0:
        c = diff.mean()
        fc_diff = np.full(horizon, c)
        beta = None
    else:
        X, y = [], []
        for t in range(p_best, n):
            X.append(np.concatenate([[1.0], diff[t - p_best:t]]))
            y.append(diff[t])
        beta, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
        hist = list(diff)
        fc_diff = []
        for _ in range(horizon):
            x = np.concatenate([[1.0], hist[-p_best:]])
            d = float(x @ beta)
            fc_diff.append(d)
            hist.append(d)
        fc_diff = np.asarray(fc_diff)
    last = train_log[-1]
    return last + np.cumsum(fc_diff)


def run_stock(code: str):
    close = load_series(code)
    n = len(close)
    rows = []
    tier = {c: t for t, codes in POOL.items() for c in codes}[code]
    for fold in range(5):
        train_end, test_end = fold_bounds(n, fold)
        train_raw = close.values[:train_end]
        test_raw = close.values[train_end:test_end]
        test_len = len(test_raw)
        # Walk-forward rolling origin: Each test point t re-selects the order and performs fitting using all the data preceding it.
        # Extrapolate by another h steps. Evaluation target = close[origin + h - 1] (compared with LSTM's realized
        # H-day labels are fully aligned. V3.2 review c1 fix: The previous version was mistakenly compared with close[origin]);
        # Number of valid samples = test_len - h + 1 (samples with incomplete h-day windows at the boundaries are discarded,
        # Consistent with the label removal rules of the LSTM agreement).
        log_all = np.log(np.maximum(close.values, 1e-6))
        for horizon in HORIZONS:
            n_eval = test_len - horizon + 1
            preds = np.empty(n_eval)
            for i in range(n_eval):
                origin = train_end + i
                train_log = log_all[:origin]
                fc = arima_forecast(train_log, horizon)   # Log space prediction of length h
                preds[i] = np.exp(np.clip(fc[-1], -20, 20))
            targets = test_raw[horizon - 1:]
            row = dict(stock=code, name=CANDIDATES[code][0], tier=tier,
                       horizon=f"+{horizon}d", fold=fold, seed=0, model="ARIMA",
                       arch="aic-ar", **metrics(stocks, preds), elapsed_s=0.0)
            rows.append(row)
            print(f"  {code} +{horizon}d fold{fold} RMSE={row['RMSE']:.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=None)
    args = ap.parse_args()
    os.makedirs(PARTS_DIR, exist_ok=True)
    tiers = {c: t for t, codes in POOL.items() for c in codes}
    stocks = args.stocks.split(",") if args.stocks else list(tiers)
    total = 0
    for code in stocks:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        done_h = set()
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done_h = set(zip(prev.horizon, prev.fold))
        rows = [r for r in run_stock(code)
                if (r["horizon"], r["fold"]) not in done_h]
        if rows:
            pd.DataFrame(rows).to_csv(part, mode="a", index=False,
                                      header=not os.path.exists(part),
                                      encoding="utf-8-sig")
            total += len(rows)
        print(f"[done] {code} {CANDIDATES[code][0]}Addition{len(rows)} row ", flush=True)
    print(f"of{total} row ")


if __name__ == "__main__":
    main()
