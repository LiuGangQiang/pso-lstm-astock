# -*- coding: utf-8 -*-
"""M1 baseline rerun: 12 stocks × {Persistence, APSO-LSTM}, walk-forward 5-fold × multiple seeds.

Protocol (Proposal Document V1.1, Chapter 5, Section 5.1):
- Anchored walk-forward 5-fold: Fold k uses the first (50 + 10k)% for training, followed by 10% for testing (k = 0..4),
  The training window increases, and the test window covers 5 non-overlapping intervals in the second half of the sample;
- Multiple seeds: 5 seeds by default, report mean ± standard deviation;
- Determinism: Torch fixed single-threaded + explicit seed (the prototype found that the RMSE fluctuated by about 5% for two runs with the same seed);
- Leakage prevention: The scaler is only fitted within the training fold;
- Incremental disk writing: Completed (target × horizon × fold × seed × model) combinations are automatically skipped, allowing for batch processing/checkpoint resume.

Usage:
  python run_m1_baseline.py --stocks 600036,600519 --horizons 1,5,20 --seeds 0,1,2,3,4"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, apso_weight_search, eval_mse, make_sequences,
                      metrics, train_model)
from select_stocks import CANDIDATES, fetch_daily

# 12 stocks (tiers, codes) finalized in V1.1 — Volatility is the actual measured value from select_stocks.py
POOL = {
    'low_volatility': ["600900", "600036", "600519", "600887"],
    'medium_volatility': ["000333", "600276", "600031", "002415"],
    'high_volatility': ["688981", "300750", "300274", "300308"],
}
SEQ_LEN, HIDDEN, EPOCHS = 20, 16, 40          # Consistent with the prototype APSO-LSTM configuration
SPLIT_START = 0.5     # End of First Fold Training (Sample Ratio)
FOLD_STEP = 0.1       # 10% advance for each fold
N_FOLDS = 5
# Search budget (overridden by command line in main; M2 uses 100×50 from V1.2 onwards)
APSO_PARTICLES, APSO_ITERS = 15, 20
VAL_FITNESS = False   # V1.3: Internal validation slice for fitness-based training segments (last 20%)
FITNESS_MODE = "train"   # train / val / composite (V1.3+ experimental variant)
WEIGHT_PENALTY = 0.0     # Weight norm penalty coefficient λ (V1.4 candidate route b)
FITNESS_SUBSAMPLE = 0    # PSO Fitness Sequence Subsampling Number (0=Full; Multi-Fidelity Search Acceleration)
PARTS_DIR = os.path.join("output", "m1_parts")  # One CSV file per stock to avoid concurrent write conflicts


def load_series(code: str) -> pd.Series:
    path = fetch_daily(code)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    close = df.set_index("date")["close"].astype(float)
    return close[close > 0]


def fold_bounds(n: int, fold: int) -> tuple:
    """For fold k: train on [0, train_end) and test on [train_end, test_end)."""
    train_end = int(n * (SPLIT_START + FOLD_STEP * fold))
    test_end = int(n * (SPLIT_START + FOLD_STEP * (fold + 1)))
    return train_end, test_end


def prepare_fold(close: pd.Series, seq_len: int, horizon: int, fold: int):
    """Fold-local preprocessing: fit the scaler only on the training fold (leakage-prevention protocol from Paper 02)."""
    n = len(close)
    train_end, test_end = fold_bounds(n, fold)
    train_raw = close.values[:train_end].copy()
    test_raw = close.values[train_end:test_end].copy()
    lo, hi = train_raw.min(), train_raw.max()
    scale = lambda v: ((v - lo) / (hi - lo)).astype(np.float32)
    train_s, test_s = scale(train_raw), scale(test_raw)
    x_tr, y_tr = make_sequences(train_s, seq_len, horizon)
    x_te, y_te = make_sequences(np.concatenate([train_s[-seq_len:], test_s]),
                                seq_len, horizon)
    inv = lambda v: np.asarray(v) * (hi - lo) + lo
    return train_s, x_tr, y_tr, x_te, y_te, inv


def run_combo(close, code, tier, horizon, fold, seed):
    """Single (stock × horizon × fold × seed): Persistence + APSO-LSTM."""
    out = []
    train_s, x_tr, y_tr, x_te, y_te, inv = prepare_fold(close, SEQ_LEN, horizon, fold)
    actual = inv(np.asarray(y_te))
    base = dict(stock=code, name=CANDIDATES[code][0], tier=tier,
                horizon=f"+{horizon}d", fold=fold, seed=seed)

    # Persistence (not related to seeds)
    if seed == 0:
        pred = inv(np.asarray(x_te[:, -1, 0]))
        out.append({**base, "model": "Persistence", **metrics(actual, pred), "elapsed_s": 0.0})

    # APSO-LSTM (Paper 12 approach: PSO evolution of initial weights + Adam fine-tuning)
    t0 = time.time()
    inner_train, val = train_s, None
    if VAL_FITNESS or FITNESS_MODE != "train":
        n_val = max(1, int(len(train_s) * 0.2))
        inner_train, val = train_s[:-n_val], train_s[-n_val:]
    best_vec, _, _ = apso_weight_search(inner_train, seq_len=SEQ_LEN, hidden=HIDDEN,
                                        n_particles=APSO_PARTICLES, iters=APSO_ITERS,
                                        seed=seed, horizon=horizon, verbose=False,
                                        val_close=val, fitness_mode=FITNESS_MODE,
                                        weight_penalty=WEIGHT_PENALTY,
                                        fitness_subsample=FITNESS_SUBSAMPLE)
    m = LSTMRegressor(hidden=HIDDEN)
    m.set_vector(best_vec)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    out.append({**base, "model": "APSO-LSTM", **metrics(actual, pred),
                "elapsed_s": round(time.time() - t0, 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=None, help='Comma-separated stock codes; default is all 12')
    ap.add_argument("--horizons", default="1,3,5,10,20")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--apso-particles", type=int, default=15)
    ap.add_argument("--apso-iters", type=int, default=20)
    ap.add_argument("--tag", default="m1", help='Output partition output/<tag>_parts/')
    ap.add_argument("--val-fitness", action="store_true",
                    help='V1.3: PSO fitness uses internal validation slices from the training set (last 20%)')
    ap.add_argument("--fitness-mode", default="train", choices=["train", "val", "composite"])
    ap.add_argument("--weight-penalty", type=float, default=0.0, help='Weight-norm penalty λ')
    ap.add_argument("--fitness-subsample", type=int, default=0, help='Number of PSO fitness subsampling sequences (0=full)')
    args = ap.parse_args()

    global APSO_PARTICLES, APSO_ITERS, PARTS_DIR, VAL_FITNESS, FITNESS_MODE, WEIGHT_PENALTY, FITNESS_SUBSAMPLE
    APSO_PARTICLES, APSO_ITERS = args.apso_particles, args.apso_iters
    VAL_FITNESS = args.val_fitness
    FITNESS_MODE = args.fitness_mode
    WEIGHT_PENALTY = args.weight_penalty
    FITNESS_SUBSAMPLE = args.fitness_subsample
    PARTS_DIR = os.path.join("output", f"{args.tag}_parts")

    torch.set_num_threads(1)  # Determinism: Single-threaded + Explicit Seed
    os.makedirs("output", exist_ok=True)

    tiers = {c: t for t, codes in POOL.items() for c in codes}
    stocks = args.stocks.split(",") if args.stocks else list(tiers)
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    done = {}
    for code in stocks:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done[code] = set(zip(prev.stock, prev.horizon, prev.fold, prev.seed, prev.model))
    os.makedirs(PARTS_DIR, exist_ok=True)

    total = 0
    for code in stocks:
        close = load_series(code)
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        for horizon in horizons:
            for fold in folds:
                for seed in seeds:
                    rows = []
                    for r in run_combo(close, code, tiers[code], horizon, fold, seed):
                        key = (r["stock"], r["horizon"], r["fold"], r["seed"], r["model"])
                        if key in done.get(code, set()):
                            continue
                        rows.append(r)
                        done.setdefault(code, set()).add(key)
                        print(f"[done] {code} {CANDIDATES[code][0]} +{horizon}d fold{fold} "
                              f"seed{seed} {r['model']} RMSE={r['RMSE']:.3f} {r['elapsed_s']}s", flush=True)
                    if rows:
                        # Execute by combination: Even if the order is killed due to timeout, the completed results will not be lost.
                        pd.DataFrame(rows).to_csv(part, mode="a", index=False,
                                                  header=not os.path.exists(part),
                                                  encoding="utf-8-sig")
                        total += len(rows)
    print(f"Rows written/appended in this batch: {total}; directory: {PARTS_DIR}/ (one CSV per stock)" if total else 'No new rows in this batch (all combinations completed)')


if __name__ == "__main__":
    main()
