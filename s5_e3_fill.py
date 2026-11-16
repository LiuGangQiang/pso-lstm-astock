# -*- coding: utf-8 -*-
"""S5-E3 Supplement: Fill in the missing cells for the λ=0 control group at the low signal levels (r=0.2, 0.4),
so that E2 covers the entire 5 r × 2 λ grid (a total of 200 sets of main experiments).

The results are appended to output/s5_e3_parts/s5_e3.csv (isomorphic with the original 160 groups).

Usage:
  python s5_e3_fill.py                # Time-budget continued run (default 250 seconds)
  python s5_e3_fill.py --budget 600"""
import argparse
import os
import time

import pandas as pd

from s5_e3 import run_one

RESULT_DIR = os.path.join("output", "s5_e3_parts")
CSV_PATH = os.path.join(RESULT_DIR, "s5_e3.csv")

CELLS = [(0.2, 0.0), (0.4, 0.0)]
SEEDS = range(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=250)
    args = ap.parse_args()

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done = set(zip(df.r, df.lam, df.seed))
        print(f"Completed {len(done)}Set, resume from breakpoint")

    combos = [(r, lam, seed) for (r, lam) in CELLS for seed in SEEDS
              if (r, lam, seed) not in done]
    print(f"Remaining {len(combos)}Group")
    t_start = time.time()
    rows, n_new = [], 0
    for r, lam, seed in combos:
        try:
            row = run_one(r, lam, seed)
        except Exception as e:  # Failure of a single group does not interrupt the entire batch
            row = dict(r=r, lam=lam, seed=seed, error=str(e))
        rows.append(row)
        n_new += 1
        print(f"[{n_new}] r={r} λ={lam} seed={seed} "
              f"chosen=s{row.get('chosen_seq','?')},h{row.get('chosen_hidden','?')}", flush=True)
        if time.time() - t_start > args.budget:
            print('Time budget exhausted; saved results from this batch')
            break
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, mode="a", index=False,
                  header=not os.path.exists(CSV_PATH))
        print(f"Completed in this batch: {len(rows)}Group{CSV_PATH}")


if __name__ == "__main__":
    main()
