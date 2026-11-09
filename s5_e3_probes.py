# -*- coding: utf-8 -*-
"""E3/E2 判决探针重跑（V3.2 复核 c0/c7 修复后，2026-11-09）。

三个探针（均用合成数据 r=0.95, seed 递增）：
  P1 staged：分阶段编码（外层结构 12×10 + 内层 10×8，λ=0）8 seeds
  P2 budget：统一编码 6× 预算（30×60，λ=0）6 seeds
  P3 fixed ：固定架构纯权重搜索（100×50，λ=0）h8 vs h32 5 seeds
输出：output/s5_e3_parts/s5_e3_probes.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from s5_e3 import N_TRAIN, gen_series
from pso_lstm import apso_weight_search, staged_pso_search, unified_pso_search

OUT = os.path.join("output", "s5_e3_parts", "s5_e3_probes.csv")
y = gen_series(0.95, 0)
tr = y[:N_TRAIN]
lo, hi = tr.min(), tr.max()
tr_s = ((tr - lo) / (hi - lo)).astype(np.float32)

rows = []

# P1 staged，8 seeds
for seed in range(8):
    t0 = time.time()
    best, fit, hist, decode = staged_pso_search(
        tr_s, outer_particles=12, outer_iters=10, seed=seed, horizon=1,
        inner_particles=10, inner_iters=8, weight_penalty=0.0,
        fitness_subsample=600, verbose=False)
    s, h = decode(best)
    rows.append(dict(probe="staged", seed=seed, chosen_seq=s, chosen_hidden=h,
                     gbest_fit=float(fit), elapsed_s=round(time.time() - t0, 1)))
    print("P1", seed, f"s{s},h{h}", flush=True)

# P2 budget 30x60，6 seeds
for seed in range(6):
    t0 = time.time()
    best, fit, hist, decode = unified_pso_search(
        tr_s, n_particles=30, iters=60, seed=seed, horizon=1,
        weight_penalty=0.0, fitness_subsample=600)
    s, h = decode(best)
    rows.append(dict(probe="budget30x60", seed=seed, chosen_seq=s, chosen_hidden=h,
                     gbest_fit=float(fit), elapsed_s=round(time.time() - t0, 1)))
    print("P2", seed, f"s{s},h{h}", flush=True)

# P3 fixed h8 vs h32，5 seeds
for seed in range(5):
    for hid in (8, 32):
        t0 = time.time()
        best, fit, hist = apso_weight_search(
            tr_s, seq_len=10, hidden=hid, n_particles=100, iters=50,
            seed=seed, horizon=1, verbose=False,
            weight_penalty=0.0, fitness_subsample=600)
        rows.append(dict(probe=f"fixed_h{hid}", seed=seed, chosen_seq=10,
                         chosen_hidden=hid, gbest_fit=float(fit),
                         elapsed_s=round(time.time() - t0, 1)))
        print(f"P3 seed{seed} h{hid} fit={fit:.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print("probes ->", OUT)
