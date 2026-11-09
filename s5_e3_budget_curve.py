# -*- coding: utf-8 -*-
"""E2 补充：固定架构权重搜索的预算曲线（V3.5，复核批注 c11）。

协议与已归档 P3 探针逐参数一致（合成 r=0.95、数据 seed=0、seq=10、λ=0、
子采样 600、apso_weight_search），仅改变搜索预算：
  B1 = 25 粒子 × 12 代（300 次评估）
  B2 = 50 粒子 × 25 代（1250 次评估）
  B3 = 100 粒子 × 50 代（5000 次评估；与 P3 相同）
每个预算 × {h8, h32} × 10 个搜索种子（0–9）。

用途：检验"低预算偏向低维网络"是否随预算增大而减弱（预算–维度耦合的方向性信号，
探索性分析）。

用法：
  python s5_e3_budget_curve.py                  # 全部（断点续跑）
  python s5_e3_budget_curve.py --budget 50x25 --hid 32 --seed 3
输出：output/s5_e3_parts/s5_e3_budget_curve.csv
"""
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
        print(f"已完成 {len(done)} 组，断点续跑")
    combos = [(b, h, s) for b in BUDGETS for h in HIDS for s in SEEDS
              if (b, h, s) not in done]
    print(f"剩余 {len(combos)} 组")
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
