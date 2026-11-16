# -*- coding: utf-8 -*-
"""S5-E3 Control Experiment: Excluding the initialization prior to test the robustness of the "change in selection distribution."

Background (V3.4 Review Annotation c4): The initial position of the structure segment for the unified-encoding search is at [-1,1],
After linear mapping by decode_struct, the initial seq is concentrated at the median of 35 (range 30–40), and hidden is only
{16,32}——coincides with the E2 final state concentration region, and the evidence for "restoring the implanted structure L=40" cannot rule out the initialization prior.

This script contains four control groups (20 seeds per group, with the protocol fully consistent with the main experiment in s5_e3.py):
  1. wide_init: The initialization of the structure segment is expanded to [-5,5] (covering all reachable structures),
     r ∈ {0.2, 0.6, 0.95} × λ = 0.1 and r = 0.6 × λ = 0;
  2. random_search: Uniform random search control with the same budget (300 fitness evaluations),
     r=0.6 λ=0.1, candidate structures are sampled from [-5,5]², and weights are truncated according to the Glorot scheme;
  3. no_signal: r=0 (pure noise), one group each for std-init and wide-init——If both groups converge to the same region, then the convergence is driven by the penalty term/noise rather than signal recovery;
  4. multi_mem: Implanted memory L∈{20,60} (r=0.6 λ=0.1, std-init)——
     Check whether the selection distribution moves with the implanted structure (distribution changes rather than a fixed anchor point).

Usage:
  python s5_e3_controls.py                          # Run the next batch (time budget, resume from breakpoint)
  python s5_e3_controls.py --control wide_init --r 0.6 --lam 0.1 --seed 0"""
import argparse
import os

import numpy as np
import pandas as pd

from pso_lstm import (LSTMRegressor, UNI_HIDDEN, UNI_SEQ_BOUNDS, _weight_dim,
                      decode_struct, eval_mse, init_weights_paper12,
                      make_sequences, unified_pso_search)
from s5_e3 import N_TRAIN, gen_series

RESULT_DIR = os.path.join("output", "s5_e3_parts")
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "s5_e3_controls.csv")

SEEDS = range(20)
N_CAND = 300  # Random Search Budget = 15 particles × 20 iterations = 300 fitness evaluations


def _scale(y: np.ndarray) -> np.ndarray:
    tr_raw = y[:N_TRAIN]
    lo, hi = tr_raw.min(), tr_raw.max()
    return ((tr_raw - lo) / (hi - lo)).astype(np.float32)


def run_pso(control, r, lam, seed, mem, struct_init):
    y = _scale(gen_series(r, seed, mem=mem))
    info = {}
    best, fit, hist, decode = unified_pso_search(
        y, n_particles=15, iters=20, seed=seed, horizon=1,
        weight_penalty=lam, fitness_subsample=600,
        struct_init=struct_init, init_info=info)
    seq_len, hidden = decode(best)
    return dict(control=control, r=r, lam=lam, seed=seed, mem=mem,
                chosen_seq=seq_len, chosen_hidden=hidden, gbest_fit=float(fit),
                init_seq_med=info.get("init_seq_med"), init_hid_med=info.get("init_hid_med"))


def run_random_search(r, lam, seed, mem=40):
    """Comparison with uniform random search under the same budget: Structure and weights are sampled independently, and the fitness protocol is replicated line by line as unified."""
    y = _scale(gen_series(r, seed, mem=mem))
    d_max = _weight_dim(int(UNI_HIDDEN.max()))
    rng = np.random.default_rng(seed + 4242)
    fx_all_cache = {}

    def fitness(vec):
        seq_len, hidden = decode_struct(vec[0], vec[1], UNI_SEQ_BOUNDS, UNI_HIDDEN)
        if seq_len not in fx_all_cache:
            x, yy = make_sequences(y, seq_len, 1)
            if 600 and len(x) > 600:
                idx = np.random.RandomState(seed).choice(len(x), 600, replace=False)
                x, yy = x[idx], yy[idx]
            fx_all_cache[seq_len] = (x, yy)
        x, yy = fx_all_cache[seq_len]
        model = LSTMRegressor(hidden=hidden)
        model.set_vector(vec[2:2 + model.get_vector().size])
        mse = eval_mse(model, x, yy)
        if lam > 0:
            mse = mse + lam * float(np.mean(vec[2:] ** 2))
        return mse

    best_fit, best_vec = np.inf, None
    for i in range(N_CAND):
        vec = np.empty(2 + d_max)
        vec[0] = rng.uniform(-5, 5)
        vec[1] = rng.uniform(-5, 5)
        probe = LSTMRegressor(hidden=int(UNI_HIDDEN.max()))
        init_weights_paper12(probe, n_features=1, seed=seed * 1000 + i)
        vec[2:] = probe.get_vector()
        f = fitness(vec)
        if f < best_fit:
            best_fit, best_vec = f, vec
    seq_len, hidden = decode_struct(best_vec[0], best_vec[1], UNI_SEQ_BOUNDS, UNI_HIDDEN)
    return dict(control="random_search", r=r, lam=lam, seed=seed, mem=mem,
                chosen_seq=seq_len, chosen_hidden=hidden, gbest_fit=float(best_fit),
                init_seq_med=np.nan, init_hid_med=np.nan)


def all_combos():
    combos = []
    # 1) Wide coverage initialization
    for r, lam in [(0.2, 0.1), (0.6, 0.1), (0.95, 0.1), (0.6, 0.0)]:
        for seed in SEEDS:
            combos.append(("wide_init", r, lam, seed, 40))
    # 2) Randomized search control
    for seed in SEEDS:
        combos.append(("random_search", 0.6, 0.1, seed, 40))
    # 3) No signal control
    for seed in SEEDS:
        combos.append(("no_signal_std", 0.0, 0.1, seed, 40))
        combos.append(("no_signal_wide", 0.0, 0.1, seed, 40))
    # 4) Multiple memory lengths
    for mem in (20, 60):
        for seed in SEEDS:
            combos.append((f"mem{mem}", 0.6, 0.1, seed, mem))
    return combos


def run_one(control, r, lam, seed, mem):
    if control == "random_search":
        return run_random_search(r, lam, seed, mem)
    struct_init = (-5.0, 5.0) if control in ("wide_init", "no_signal_wide") else (-1.0, 1.0)
    return run_pso(control, r, lam, seed, mem, struct_init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control")
    ap.add_argument("--r", type=float)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mem", type=int, default=40)
    ap.add_argument("--budget", type=float, default=260)
    args = ap.parse_args()

    if args.control is not None and args.r is not None and args.seed is not None:
        print(run_one(args.control, args.r, args.lam, args.seed, args.mem))
        return

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done = set(zip(df.control, df.r, df.lam, df.seed))
        print(f"Completed {len(done)}Set, resume from breakpoint")
    combos = [c for c in all_combos() if (c[0], c[1], c[2], c[3]) not in done]
    print(f"Remaining {len(combos)}Group")

    import time
    t_start, rows = time.time(), []
    for control, r, lam, seed, mem in combos:
        try:
            row = run_one(control, r, lam, seed, mem)
        except Exception as e:
            row = dict(control=control, r=r, lam=lam, seed=seed, mem=mem, error=str(e))
        rows.append(row)
        print(f"[{len(rows)}] {control} r={r} lam={lam} seed={seed} -> "
              f"s{row.get('chosen_seq')},h{row.get('chosen_hidden')} fit={row.get('gbest_fit', float('nan')):.4f}")
        if time.time() - t_start > args.budget:
            break
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, mode="a", index=False, header=not os.path.exists(CSV_PATH))
        print(f"Completed in this batch: {len(rows)}Group{CSV_PATH}")


if __name__ == "__main__":
    main()
