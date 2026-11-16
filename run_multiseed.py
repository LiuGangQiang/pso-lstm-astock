# -*- coding: utf-8 -*-
"""Checklist item 5: replicate E2 synthetic controls across generator seeds 1 and 2; seed 0 is the original archive.

The original E2 main runs and no_signal/mem controls all used the single series gen_series(seed=0).
This script reproduces the same combinations with generator seeds 1 and 2:
  main: r∈{0.2,0.4,0.6,0.8,0.95} × λ∈{0.0,0.1} × optseed 0..19 (200 per generator seed)
  no_signal_std: r=0, λ=0.1, 20 optseeds (matching s5_e3_controls.py)
  mem20/mem60: r=0.6, λ=0.1, mem=20/60, 20 optseeds
Total: 260 combinations per generator seed. Optimization seeds remain 0..19; only the generated series changes.
Broad-initialization and random-search controls probe the search mechanism rather than signal tracking and are outside this replication.

Usage: python run_multiseed.py [chunk_size=60] [workers=6]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from s5_e3 import RS, gen_series, N_TRAIN
from pso_lstm import unified_pso_search

RESULT_DIR = os.path.join("output", "s5_e3_parts")
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "s5_e3_multiseed.csv")

GEN_SEEDS = [1, 2]
OPT_SEEDS = range(20)


def all_combos():
    combos = []
    for gs in GEN_SEEDS:
        for r in RS:
            for lam in (0.1, 0.0):
                for s in OPT_SEEDS:
                    combos.append(("main", gs, r, lam, s, 40))
        for s in OPT_SEEDS:
            combos.append(("no_signal_std", gs, 0.0, 0.1, s, 40))
            combos.append(("mem20", gs, 0.6, 0.1, s, 20))
            combos.append(("mem60", gs, 0.6, 0.1, s, 60))
    return combos


def run_one(kind, gs, r, lam, s, mem):
    y = gen_series(r, gs, None if kind == "main" else mem)
    tr_raw = y[:N_TRAIN]
    lo, hi = tr_raw.min(), tr_raw.max()
    tr_s = ((tr_raw - lo) / (hi - lo)).astype(np.float32)
    best, fit, hist, decode = unified_pso_search(
        tr_s, n_particles=15, iters=20, seed=s, horizon=1,
        weight_penalty=lam, fitness_subsample=600)
    seq_len, hidden = decode(best)
    return dict(kind=kind, genseed=gs, r=r, lam=lam, seed=s, mem=mem,
                chosen_seq=seq_len, chosen_hidden=hidden, gbest_fit=float(fit))


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done = set(zip(df.kind, df.genseed, df.r, df.lam, df.seed, df.mem))
    rem = [c for c in all_combos() if (c[0], c[1], c[2], c[3], c[4], c[5]) not in done]
    print(f"Remaining {len(rem)}Group, this block is taken{min(chunk, len(rem))}Group{workers}Concurrency")
    block = rem[:chunk]
    if not block:
        print("all done")
        return

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, *c): c for c in block}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                c = futs[fut]
                rows.append(dict(kind=c[0], genseed=c[1], r=c[2], lam=c[3],
                                 seed=c[4], mem=c[5], error=str(e)))
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, mode="a", index=False, header=not os.path.exists(CSV_PATH))
    ok = sum(1 for r_ in rows if "error" not in r_)
    print(f"Chunk completed {ok}/{len(rows)}Set, time taken{time.time()-t0:.0f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
