# -*- coding: utf-8 -*-
"""S1 Scan Driver: Slice the (stock × horizon × fold) combination list and launch multiple
run_m1_baseline subprocesses (each subprocess runs 5 seeds sequentially internally), and returns only after all have waited.

Usage: python s1_driver.py --start 0 --count 8
Completed combinations are automatically skipped by the parts CSV of run_m1_baseline (safe for checkpoint resume)."""
import argparse
import subprocess
import sys

from run_m1_baseline import POOL

STOCKS = [c for tier in ('low_volatility', 'medium_volatility', 'high_volatility') for c in POOL[tier]]
HORIZONS = [1, 3, 5, 10, 20]
FOLDS = [0, 1, 2, 3, 4]
COMBOS = [(s, h, f) for s in STOCKS for h in HORIZONS for f in FOLDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--apso-particles", type=int, default=100)
    ap.add_argument("--apso-iters", type=int, default=50)
    ap.add_argument("--weight-penalty", type=float, default=0.1)
    ap.add_argument("--fitness-subsample", type=int, default=600)
    ap.add_argument("--horizons", default="1,3,5,10,20", help='Horizon list used to enumerate combinations')
    ap.add_argument("--tag", default="s1v14")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    combos = [(s, h, f) for s in STOCKS for h in horizons for f in FOLDS]
    slice_ = combos[args.start:args.start + args.count]
    if not slice_:
        print('Combination index out of range; all combinations completed')
        return
    procs = []
    for s, h, f in slice_:
        cmd = [sys.executable, "run_m1_baseline.py",
               "--stocks", s, "--horizons", str(h), "--folds", str(f),
               "--seeds", "0,1,2,3,4",
               "--apso-particles", str(args.apso_particles),
               "--apso-iters", str(args.apso_iters),
               "--weight-penalty", str(args.weight_penalty),
               "--fitness-subsample", str(args.fitness_subsample),
               "--tag", args.tag]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    print(f"Launched {len(procs)} subprocesses; combinations {args.start}..{args.start + len(slice_) - 1}")
    for p in procs:
        p.wait()
    print('All subprocesses in this batch have finished')


if __name__ == "__main__":
    main()
