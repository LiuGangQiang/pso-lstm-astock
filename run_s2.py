# -*- coding: utf-8 -*-
"""M3 / S2: Joint Encoding Bias Diagnosis Experiment (Proposal Document V1.5, Chapter 5, Section 5.2).

Models (two, plus the existing APSO-LSTM from s1v14 as a fixed architecture control):
  1. Unified-PSO   —— Naive joint encoding (architecture segment + weight segment with the same particle), V1.4 fair configuration (λ=0.1 + sub-sampling 600)
  2. Staged-PSO    —— Staged joint encoding (outer-layer structure PSO + inner-layer small APSO), V1.4 configuration

Design (H3): h ∈ {5, 10} pure price; 12 stocks × 5 folds × 5 seeds.
After each run, the best particle is fine-tuned using Adam (40 epochs, lr=0.005) and then re-evaluated,
This is completely consistent with the APSO-LSTM protocol of s1v14.

Usage:
  python run_s2.py --stocks 600276 --horizons 10 --folds 0 --seeds 0,1,2
  python s2_driver.py --start 0 --count 8"""
import argparse
import os
import time

import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, eval_mse, make_sequences, metrics,
                      staged_pso_search, train_model, unified_pso_search)
from run_m1_baseline import POOL, fold_bounds, load_series, prepare_fold
from select_stocks import CANDIDATES

EPOCHS = 40
PARTS_DIR = os.path.join("output", "s2_parts")
# V1.4 Equitable Allocation
WEIGHT_PENALTY = 0.1
SUBSAMPLE = 600
# Budget: Unified encoding follows the prototype 15×20; phased outer layer 12×10 + inner layer 10×8
UNI_P, UNI_I = 15, 20
OUT_P, OUT_I, IN_P, IN_I = 12, 10, 10, 8


def run_model(model_name, close, code, tier, horizon, fold, seed):
    """Return a row of test-fold metrics for this model."""
    t0 = time.time()

    if model_name == "Unified-PSO":
        # The sequence length of the fitness in the structural search period is determined by the structural segments of each particle themselves (the default window of 20 is only used for
        # Take the training segment); Fine-tuning and evaluation data are reconstructed after selecting s (V3.2 review c7 fix).
        train_s, _, _, _, _, _ = prepare_fold(close, 20, horizon, fold)
        best_vec, fit, hist, decode = unified_pso_search(
            train_s, n_particles=UNI_P, iters=UNI_I, seed=seed, horizon=horizon,
            weight_penalty=WEIGHT_PENALTY, fitness_subsample=SUBSAMPLE)
        s, h = decode(best_vec)
        m = LSTMRegressor(hidden=h)
        m.set_vector(best_vec[2:2 + m.get_vector().size])
    else:  # Staged-PSO
        train_s, _, _, _, _, _ = prepare_fold(close, 20, horizon, fold)
        best_vec, fit, hist, decode = staged_pso_search(
            train_s, outer_particles=OUT_P, outer_iters=OUT_I, seed=seed, horizon=horizon,
            inner_particles=IN_P, inner_iters=IN_I, weight_penalty=WEIGHT_PENALTY,
            fitness_subsample=SUBSAMPLE, verbose=False)
        s, h = decode(best_vec)
        # Optimal inner-layer weights: Re-run the inner-layer search once to obtain the weights (deterministic with the same seed), then inject them.
        from pso_lstm import apso_weight_search
        wvec, _, _ = apso_weight_search(
            train_s, seq_len=s, hidden=h, n_particles=IN_P, iters=IN_I, seed=seed,
            horizon=horizon, verbose=False, weight_penalty=WEIGHT_PENALTY,
            fitness_subsample=SUBSAMPLE)
        m = LSTMRegressor(hidden=h)
        m.set_vector(wvec)

    # Reconstruct fine-tuning/evaluation data based on the selected sequence length s from the search (c7: legacy constant 20-day window)
    train_s, x_tr, y_tr, x_te, y_te, inv = prepare_fold(close, s, horizon, fold)
    actual = inv(y_te)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    row = dict(stock=code, name=CANDIDATES[code][0], tier=tier, horizon=f"+{horizon}d",
               fold=fold, seed=seed, model=model_name, arch=f"s{s},h{h}",
               **metrics(actual, pred), elapsed_s=round(time.time() - t0, 1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", required=True)
    ap.add_argument("--horizons", default="5,10")
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

    close = load_series(args.stocks)
    total = 0
    for horizon in [int(h) for h in args.horizons.split(",")]:
        for fold in [int(f) for f in args.folds.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                for model in ["Unified-PSO", "Staged-PSO"]:
                    key = (args.stocks, f"+{horizon}d", fold, seed, model)
                    if key in done:
                        continue
                    r = run_model(model, close, args.stocks, tiers[args.stocks],
                                  horizon, fold, seed)
                    pd.DataFrame([r]).to_csv(part, mode="a", index=False,
                                             header=not os.path.exists(part),
                                             encoding="utf-8-sig")
                    total += 1
                    print(f"[done] {args.stocks} +{horizon}d fold{fold} seed{seed} "
                          f"{model} {r['arch']} RMSE={r['RMSE']:.2f} {r['elapsed_s']}s", flush=True)
    print(f"Write{total} row " if total else 'No new rows')


if __name__ == "__main__":
    main()
