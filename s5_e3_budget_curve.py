# -*- coding: utf-8 -*-
"""E2 Supplement: Budget curve for fixed architecture weight search (V3.5, review annotation c11).

The protocol is consistent with the archived P3 probe parameter by parameter (synthetic r=0.95, data seed=0, seq=10, λ=0,
Subsampling 600, apso_weight_search), only the search budget is changed:
  B1 = 25 particles × 12 generations (300 evaluations)
  B2 = 50 particles × 25 generations (1250 evaluations)
  B3 = 100 particles × 50 generations (5000 evaluations; same as P3)
Each budget × {h8, h32} × 10 search seeds (0–9).

Purpose: To test whether the "low-budget bias toward low-dimensional networks" weakens as the budget increases (a directional signal of budget–dimensionality coupling,
exploratory analysis).

Usage:
  python s5_e3_budget_curve.py                  # All (resume from breakpoint)
  python s5_e3_budget_curve.py --budget 50x25 --hid 32 --seed 3
Output: output/s5_e3_parts/s5_e3_budget_curve.csv"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pso_lstm import apso_weight_search
from s5_e3 import N_TRAIN, gen_series

OUT = os.path.join("output", "s5_e3_parts", "s5_e3_budget_curve.csv")

BUDGETS = {"25x12": (25, 12), "50x25": (50, 25), "100x50": (100, 50)}
HIDS = (8, 32)
SEEDS = range(10)


def run_one(budget_name, hid, seed):
    y = gen_series(0.95, 0)
    tr = y[:N_TRAIN]
    lo, hi = tr.min(), tr.max()
    tr_s = ((tr - lo) / (hi - lo)).astype(np.float32)
    n_particles, iters = BUDGETS[budget_name]
    best, fit, hist = apso_weight_search(
        tr_s, seq_len=10, hidden=hid, n_particles=n_particles, iters=iters,
        seed=seed, horizon=1, verbose=False,
        weight_penalty=0.0, fitness_subsample=600)
    return dict(budget=budget_name, n_particles=n_particles, iters=iters,
                hidden=hid, seed=seed, gbest_fit=float(fit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=None)
    ap.add_argument("--hid", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.budget is not None and args.hid is not None and args.seed is not None:
        print(run_one(args.budget, args.hid, args.seed))
        return

    done = set()
    if os.path.exists(OUT):
        df = pd.read_csv(OUT)
        done = set(zip(df.budget, df.hidden, df.seed))
        print(f"Completed {len(done)}Set, resume from breakpoint")
    combos = [(b, h, s) for b in BUDGETS for h in HIDS for s in SEEDS
              if (b, h, s) not in done]
    print(f"Remaining {len(combos)}Group")
    rows = []
    for b, h, s in combos:
        rows.append(run_one(b, h, s))
        print(f"{b} h{h} seed{s}: fit={rows[-1]['gbest_fit']:.5f}", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
        print("saved ->", OUT)


if __name__ == "__main__":
    main()
