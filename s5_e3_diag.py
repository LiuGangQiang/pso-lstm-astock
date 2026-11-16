# -*- coding: utf-8 -*-
"""E3 Pre-Diagnosis V3.5 Revision: Verify the reachable MSE gap (structural signal strength) for each structure of the synthetic generator.

V3.5 Revision (Review Annotation c13):
  1. Seed bug fix – torch.manual_seed is set before model construction (in the old version of LSTMRegressor,
     weight initialization occurred before seeding within train_model, and the initialization was not reproducible);
  2. Add a held-out control — evaluate the out-of-sample MSE on the last 500 points outside the training segment (using the same min-max scaling)
     out-of-sample MSE to test whether the "structural signal" leads to a generalizable gap rather than merely overfitting the training segment.

Output: output/s5_e3_diag_v35.csv"""
import os
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch

from s5_e3 import N_TRAIN, gen_series
from pso_lstm import LSTMRegressor, eval_mse, make_sequences, train_model

GRID = [(10, 8), (40, 8), (10, 32), (40, 32), (60, 32)]
SEEDS = [0, 1, 2]
OUT = os.path.join("output", "s5_e3_diag_v35.csv")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(1)
    rs = [args.r] if args.r is not None else [0.6, 0.95]
    seeds = [args.seed] if args.seed is not None else SEEDS
    rows = []
    for r in rs:
        y = gen_series(r, 0)
        tr = y[:N_TRAIN]
        ho = y[N_TRAIN:]
        lo, hi = tr.min(), tr.max()
        tr_s = ((tr - lo) / (hi - lo)).astype(np.float32)
        ho_s = ((ho - lo) / (hi - lo)).astype(np.float32)
        for seq, hid in GRID:
            for seed in seeds:
                torch.manual_seed(seed)  # Seeding must be done before model construction (c13 fix)
                m = LSTMRegressor(hidden=hid)
                train_model(m, make_sequences(tr_s, seq, 1)[0],
                            make_sequences(tr_s, seq, 1)[1],
                            epochs=30, lr=0.01, seed=seed, verbose=False)
                xtr, ytr = make_sequences(tr_s, seq, 1)
                xho, yho = make_sequences(ho_s, seq, 1)
                rows.append(dict(r=r, seq=seq, hid=hid, seed=seed,
                                 train_mse=eval_mse(m, xtr, ytr),
                                 heldout_mse=eval_mse(m, xho, yho)))
                print(f"r={r} seq={seq} h={hid} seed={seed}: "
                      f"train={rows[-1]['train_mse']:.4f} heldout={rows[-1]['heldout_mse']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print()
    print('== Summary (average of 3 seeds) ==')
    print(df.groupby(["r", "seq", "hid"])[["train_mse", "heldout_mse"]].mean().round(4))


if __name__ == "__main__":
    main()
