# -*- coding: utf-8 -*-
"""H2 偏置分解诊断：结构边际 vs 权重边际的适应度方差（开题文档第三章 H2）。

对 Unified-PSO 的 gbest 粒子：
  - 结构边际：固定 gbest 权重段，随机换 K 个结构 (seq_len, hidden)，算适应度 std；
  - 权重边际：固定 gbest 结构，换 K 组 Glorot/正交初始化权重，算适应度 std。
若 权重边际 std >> 结构边际 std，则 H2 成立（结构信号被权重段淹没）。

用法：python h2_diagnose.py
"""
import numpy as np
import torch

from pso_lstm import (LSTMRegressor, UNI_HIDDEN, UNI_SEQ_BOUNDS, _weight_dim,
                      decode_struct, eval_mse, init_weights_paper12,
                      make_sequences, unified_pso_search)
from run_m1_baseline import prepare_fold
from run_s2 import SUBSAMPLE, WEIGHT_PENALTY
from select_stocks import CANDIDATES

STOCKS = ["600036", "600276", "300750", "300308"]
HORIZON = 10
FOLD, SEEDS, K = 0, [0, 1, 2], 20


def uni_fitness(vec, cache, train_s, horizon, seed):
    """与 unified_pso_search 完全一致的适应度（子采样按种子固定 + 全段惩罚）。

    V3.5 修订（复核批注 c5）：结构解码对齐 unified_pso_search 的 decode_struct
    坐标映射（旧版把坐标直接当 seq_len/hidden 用，属归档 V2 及之前的旧解码语义）。
    """
    seq_len, hidden = decode_struct(vec[0], vec[1], UNI_SEQ_BOUNDS, UNI_HIDDEN)
    if seq_len not in cache:
        x, y = make_sequences(train_s, seq_len, horizon)
        if SUBSAMPLE and len(x) > SUBSAMPLE:
            idx = np.random.RandomState(seed).choice(len(x), SUBSAMPLE, replace=False)
            x, y = x[idx], y[idx]
        cache[seq_len] = (x, y)
    x, y = cache[seq_len]
    m = LSTMRegressor(hidden=hidden)
    m.set_vector(vec[2:2 + m.get_vector().size])
    mse = eval_mse(m, x, y)
    return mse + WEIGHT_PENALTY * float(np.mean(vec[2:] ** 2))


def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    stocks = [args.stock] if args.stock else STOCKS
    seeds = [args.seed] if args.seed is not None else SEEDS

    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    print(f"{'标的':<8}{'种子':<4}{'gbest结构':<10}{'结构边际std':<12}{'权重边际std':<12}{'比值(权/结)'}")
    for code in stocks:
        from run_m1_baseline import load_series
        close = load_series(code)
        train_s, *_ = prepare_fold(close, 20, HORIZON, FOLD)
        for seed in seeds:
            gbest, fit, _, decode = unified_pso_search(
                train_s, n_particles=15, iters=20, seed=seed, horizon=HORIZON,
                weight_penalty=WEIGHT_PENALTY, fitness_subsample=SUBSAMPLE)
            s, h = decode(gbest)
            cache = {}
            # 结构边际：权重段不动，随机结构
            struct_fits = []
            for _ in range(K):
                v = gbest.copy()
                # 结构坐标在 [-5,5] 均匀采样（decode_struct 的完整可达域）
                v[0] = rng.uniform(-5, 5)
                v[1] = rng.uniform(-5, 5)
                struct_fits.append(uni_fitness(v, cache, train_s, HORIZON, seed))
            # 权重边际：结构不动，随机 Glorot 权重（按 h64 生成再截取）
            weight_fits = []
            for k in range(K):
                v = gbest.copy()
                probe = LSTMRegressor(hidden=int(UNI_HIDDEN.max()))
                init_weights_paper12(probe, n_features=1, seed=seed * 1000 + 100 + k)
                v[2:] = probe.get_vector()
                weight_fits.append(uni_fitness(v, cache, train_s, HORIZON, seed))
            ss, sw = np.std(struct_fits), np.std(weight_fits)
            print(f"{CANDIDATES[code][0]:<8}{seed:<4}s{s},h{h:<6}"
                  f"{ss:<12.5f}{sw:<12.5f}{sw / (ss + 1e-12):.2f}")


if __name__ == "__main__":
    main()
