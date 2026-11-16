# -*- coding: utf-8 -*-
"""M4 / S3: feature-ablation experiment (Proposal V1.0, Section 5.3, H5).

Starting from the s1v14 APSO-LSTM configuration (fixed s20,h16; 100×50; λ=0.1; subsample 600;
five walk-forward folds × five seeds), compare three input-feature levels:

  price: close only; uses existing s1v14_parts results and is not rerun here
  tech: close + four technical indicators (five features):
             ret1: log return log(close_t / close_{t-1})
             ma20r: moving-average deviation close / MA20 - 1
             rsi14: 14-day relative strength index (SMA method)
             vol20: 20-day rolling return volatility
  sent: tech + three sentiment/capital proxies (eight features):
             overnight: overnight gap open / prev_close - 1
             range_: intraday range (high - low) / prev_close
             volr: volume ratio volume / MA20(volume)

Protocol:
- Every indicator uses only data available at or before t (causal; no future leakage).
- Fit each min-max scaler only on the training fold; apply those parameters to the test fold.
- Use the same target as the price baseline: close scaled with training-fold min-max parameters;
  evaluate in the same inverse-scaled space, ensuring comparable ablations (H5 uses ratios to Persistence).
- Drop leading NaN rows created by indicator construction before sequence generation; valid sample counts differ slightly by stock.
- Write incrementally to output/s3_parts/<stock>.csv and support checkpoint resume.

Usage:
  python run_s3.py --stocks 600036 --horizons 5 --folds 0 --seeds 0,1,2
  python s3_driver.py --start 0 --count 12
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, apso_weight_search, make_sequences_multi,
                      metrics, train_model)
from run_m1_baseline import POOL, fold_bounds, load_series
from select_stocks import CANDIDATES

SEQ_LEN, HIDDEN, EPOCHS = 20, 16, 40        # Fully consistent with the s1v14 baseline configuration
APSO_P, APSO_I = 100, 50                    # V1.2 Budget
WEIGHT_PENALTY, SUBSAMPLE = 0.1, 600        # V1.4/V1.5 Configuration
PARTS_DIR = os.path.join("output", "s3_parts")

TECH_COLS = ["ret1", "ma20r", "rsi14", "vol20"]
SENT_COLS = ["overnight", "range_", "volr"]
FEATURE_SETS = {"tech": TECH_COLS, "sent": TECH_COLS + SENT_COLS}


def build_features(code: str, fset: str):
    """Construct the causal feature matrix and closing-price series. Returns feat_raw [T, F+1]
    (including raw close prices) and close_raw [T]. Column 0 of feat_raw is close; scaling is fold-local.
    """
    from select_stocks import fetch_daily
    path = fetch_daily(code)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    close, open_, high, low, volume = (df["close"].astype(float), df["open"].astype(float),
                                       df["high"].astype(float), df["low"].astype(float),
                                       df["volume"].astype(float))
    prev_close = close.shift(1)
    feat = pd.DataFrame({"close": close})
    feat["ret1"] = np.log(close / prev_close)
    feat["ma20r"] = close / close.rolling(20).mean() - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    feat["rsi14"] = 100.0 - 100.0 / (1.0 + rs)
    feat.loc[loss == 0.0, "rsi14"] = 100.0
    feat["vol20"] = feat["ret1"].rolling(20).std()
    if fset == "sent":
        feat["overnight"] = open_ / prev_close - 1.0
        feat["range_"] = (high - low) / prev_close
        feat["volr"] = volume / volume.rolling(20).mean()
    feat = feat.dropna().reset_index(drop=True)
    return feat[["close"] + FEATURE_SETS[fset]].values.astype(np.float64), feat["close"].values.astype(np.float64)


def _minmax(train_seg, full):
    lo, hi = float(train_seg.min()), float(train_seg.max())
    if hi - lo < 1e-12:
        return np.zeros_like(full, dtype=np.float32), lo, hi
    return ((full - lo) / (hi - lo)).astype(np.float32), lo, hi


def prepare_fold_multi(feat_raw, close_raw, seq_len, horizon, fold):
    """Multivariate in-fold scaling: For "close", use the same min-max as the baseline; for the remaining features, each is scaled according to the min-max of the training fold."""
    n = len(close_raw)
    train_end, test_end = fold_bounds(n, fold)
    cols, lo_close, hi_close = [], None, None
    scaled = []
    for j in range(feat_raw.shape[1]):
        s, lo, hi = _minmax(feat_raw[:train_end, j], feat_raw[:, j])
        scaled.append(s)
        if j == 0:
            lo_close, hi_close = lo, hi
    feat_s = np.stack(scaled, axis=1)          # [T, F]
    close_s = feat_s[:, 0].copy()
    train_feat, test_feat = feat_s[:train_end], feat_s[train_end:test_end]
    tail_feat = np.concatenate([train_feat[-seq_len:], test_feat], axis=0)
    tail_close = np.concatenate([close_s[:train_end][-seq_len:],
                                 close_s[train_end:test_end]], axis=0)
    x_tr, y_tr = make_sequences_multi(train_feat, close_s[:train_end], seq_len, horizon)
    x_te, y_te = make_sequences_multi(tail_feat, tail_close, seq_len, horizon)
    inv = lambda v: np.asarray(v) * (hi_close - lo_close) + lo_close
    return x_tr, y_tr, x_te, y_te, inv


def run_fset(fset, feat_raw, close_raw, code, tier, horizon, fold, seed):
    t0 = time.time()
    x_tr, y_tr, x_te, y_te, inv = prepare_fold_multi(feat_raw, close_raw, SEQ_LEN, horizon, fold)
    actual = inv(y_te)
    # APSO search only accesses the training segment (the scaling parameter also only uses the training segment)
    tr_end = int(len(close_raw) * (0.5 + 0.1 * fold))
    train_feat = np.stack([_minmax(feat_raw[:tr_end, j], feat_raw[:tr_end, j])[0]
                           for j in range(feat_raw.shape[1])], axis=1)
    best_vec, _, _ = apso_weight_search(
        train_feat, seq_len=SEQ_LEN, hidden=HIDDEN, n_particles=APSO_P, iters=APSO_I,
        seed=seed, horizon=horizon, verbose=False, weight_penalty=WEIGHT_PENALTY,
        fitness_subsample=SUBSAMPLE, n_features=train_feat.shape[1])
    m = LSTMRegressor(hidden=HIDDEN, n_features=train_feat.shape[1])
    m.set_vector(best_vec)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    return dict(stock=code, name=CANDIDATES[code][0], tier=tier, horizon=f"+{horizon}d",
                fold=fold, seed=seed, model=f"APSO-LSTM-{fset}", arch=f"s{SEQ_LEN},h{HIDDEN}",
                n_feat=train_feat.shape[1],
                **metrics(actual, pred), elapsed_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", required=True)
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()

    torch.set_num_threads(1)
    tiers = {c: t for t, codes in POOL.items() for c in codes}
    os.makedirs(PARTS_DIR, exist_ok=True)

    part = os.path.join(PARTS_DIR, f"{args.stocks}.csv")
    done = set()
    if os.path.exists(part):
        prev = pd.read_csv(part, dtype={"stock": str})
        done = set(zip(prev.stock, prev.horizon, prev.fold, prev.seed, prev.model))

    data = {}
    total = 0
    for fset in ("tech", "sent"):
        feat_raw, close_raw = build_features(args.stocks, fset)
        data[fset] = (feat_raw, close_raw)
    for horizon in [int(h) for h in args.horizons.split(",")]:
        for fold in [int(f) for f in args.folds.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                for fset in ("tech", "sent"):
                    key = (args.stocks, f"+{horizon}d", fold, seed, f"APSO-LSTM-{fset}")
                    if key in done:
                        continue
                    r = run_fset(fset, *data[fset], args.stocks, tiers[args.stocks],
                                 horizon, fold, seed)
                    pd.DataFrame([r]).to_csv(part, mode="a", index=False,
                                             header=not os.path.exists(part),
                                             encoding="utf-8-sig")
                    total += 1
                    print(f"[done] {args.stocks} +{horizon}d fold{fold} seed{seed} "
                          f"{r['model']} RMSE={r['RMSE']:.2f} {r['elapsed_s']}s", flush=True)
    print(f"Write{total} row " if total else 'No new rows')


if __name__ == "__main__":
    main()
