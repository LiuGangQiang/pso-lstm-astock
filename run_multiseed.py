# -*- coding: utf-8 -*-
"""编辑清单第 5 项：E2 合成对照的多生成器种子复现（gen seed 1、2；seed 0 为原档）。

原始 E2 主运行与 no_signal/mem 对照全部基于 gen_series(seed=0) 的单一生成序列。
本脚本在生成器种子 1、2 上复现与原文相同口径的组合：
  main:   r∈{0.2,0.4,0.6,0.8,0.95} × λ∈{0.0,0.1} × optseed 0..19（200/种子）
  no_signal_std: r=0, λ=0.1, 20 optseeds（与 s5_e3_controls.py 口径一致）
  mem20/mem60:   r=0.6, λ=0.1, mem=20/60, 20 optseeds
合计 260 组/生成种子。优化种子保持 0..19 不变，只改变生成序列。
广覆盖初始化与随机搜索对照探测的是搜索机制本身（与信号追踪无关），不在本复现范围。

用法：python run_multiseed.py [块大小=60] [并发=6]
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
    print(f"剩余 {len(rem)} 组，本块取 {min(chunk, len(rem))} 组，{workers} 并发")
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
    print(f"本块完成 {ok}/{len(rows)} 组，用时 {time.time()-t0:.0f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
