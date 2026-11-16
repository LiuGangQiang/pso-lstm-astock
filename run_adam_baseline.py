# -*- coding: utf-8 -*-
"""M5 Pre-Reinforcement Experiment (B2): Adam-only control with the same protocol (no PSO pre-search).

Motivation: To address the question of whether "the performance of APSO-LSTM is merely a result of computational power / whether PSO pre-search contributes."
Same LSTM (s20, h16), same walk-forward 5-fold × 5 seeds, same Adam fine-tuning
(40 epochs, lr=0.005, same seed), the only difference: no APSO weight search,
The initial weights use the Glorot/orthogonal initialization from Paper 12 (init_weights_paper12).
If there is no significant difference between Adam-only and APSO-LSTM, then the incremental contribution of the PSO pre-search is zero;
This is also further evidence that "the value of PSO lies in basin selection rather than accuracy."

Output: output/adam_parts/<stock>.csv."""
import argparse
import os
import time

import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, init_weights_paper12, metrics, train_model)
from run_m1_baseline import POOL, load_series, prepare_fold
from select_stocks import CANDIDATES

SEQ_LEN, HIDDEN, EPOCHS = 20, 16, 40
PARTS_DIR = os.path.join("output", "adam_parts")


def run_one(close, code, tier, horizon, fold, seed):
    t0 = time.time()
    train_s, x_tr, y_tr, x_te, y_te, inv = prepare_fold(close, SEQ_LEN, horizon, fold)
    actual = inv(y_te)
    m = LSTMRegressor(hidden=HIDDEN)
    init_weights_paper12(m, n_features=1, seed=seed)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    return dict(stock=code, name=CANDIDATES[code][0], tier=tier, horizon=f"+{horizon}d",
                fold=fold, seed=seed, model="Adam-only", arch=f"s{SEQ_LEN},h{HIDDEN}",
                **metrics(actual, pred), elapsed_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=None)
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(PARTS_DIR, exist_ok=True)
    tiers = {c: t for t, codes in POOL.items() for c in codes}
    stocks = args.stocks.split(",") if args.stocks else list(tiers)
    total = 0
    for code in stocks:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        done = set()
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done = set(zip(prev.horizon, prev.fold, prev.seed))
        close = load_series(code)
        rows = []
        for horizon in [int(h) for h in args.horizons.split(",")]:
            for fold in [int(f) for f in args.folds.split(",")]:
                for seed in [int(s) for s in args.seeds.split(",")]:
                    if (f"+{horizon}d", fold, seed) in done:
                        continue
                    rows.append(run_one(close, code, tiers[code], horizon, fold, seed))
        if rows:
            pd.DataFrame(rows).to_csv(part, mode="a", index=False,
                                      header=not os.path.exists(part),
                                      encoding="utf-8-sig")
            total += len(rows)
        print(f"[done] {code}Addition{len(rows)} row ", flush=True)
    print(f"of{total} row ")


if __name__ == "__main__":
    main()
