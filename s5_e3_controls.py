# -*- coding: utf-8 -*-
"""S5-E3 对照实验：排除初始化先验，检验"选择分布改变"的稳健性。

背景（V3.4 复核批注 c4）：统一编码搜索的结构段初始位置在 [-1,1]，
经 decode_struct 线性映射后初始 seq 集中在中位 35（区间 30–40）、hidden 仅
{16,32}——与 E2 终态集中区重合，"恢复植入结构 L=40"的证据无法排除初始化先验。

本脚本四组对照（每组 20 seeds，协议与 s5_e3.py 主实验完全一致）：
  1. wide_init：结构段初始化扩至 [-5,5]（覆盖全部可达结构），
     r ∈ {0.2,0.6,0.95}×λ=0.1 及 r=0.6×λ=0；
  2. random_search：同预算（300 次适应度评估）均匀随机搜索对照，
     r=0.6 λ=0.1，候选结构采样自 [-5,5]²、权重按 Glorot 方案截取；
  3. no_signal：r=0（纯噪声），std-init 与 wide-init 各一组——
     若两组都收敛到同一区域，则该收敛是罚项/噪声驱动而非信号恢复；
  4. multi_mem：植入记忆 L∈{20,60}（r=0.6 λ=0.1，std-init）——
     检验选择分布是否随植入结构移动（分布改变而非固定锚点）。

用法：
  python s5_e3_controls.py                          # 跑下一批（时间盒，断点续跑）
  python s5_e3_controls.py --control wide_init --r 0.6 --lam 0.1 --seed 0
"""
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
N_CAND = 300  # 随机搜索预算 = 15 粒子 × 20 迭代 = 300 次适应度评估


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
    """同预算均匀随机搜索对照：结构与权重独立采样，适应度协议逐行复刻 unified。"""
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
    # 1) 广覆盖初始化
    for r, lam in [(0.2, 0.1), (0.6, 0.1), (0.95, 0.1), (0.6, 0.0)]:
        for seed in SEEDS:
            combos.append(("wide_init", r, lam, seed, 40))
    # 2) 随机搜索对照
    for seed in SEEDS:
        combos.append(("random_search", 0.6, 0.1, seed, 40))
    # 3) 无信号对照
    for seed in SEEDS:
        combos.append(("no_signal_std", 0.0, 0.1, seed, 40))
        combos.append(("no_signal_wide", 0.0, 0.1, seed, 40))
    # 4) 多记忆长度
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
        print(f"已完成 {len(done)} 组，断点续跑")
    combos = [c for c in all_combos() if (c[0], c[1], c[2], c[3]) not in done]
    print(f"剩余 {len(combos)} 组")

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
        print(f"本批完成 {len(rows)} 组 -> {CSV_PATH}")


if __name__ == "__main__":
    main()
