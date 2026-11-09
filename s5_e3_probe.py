# -*- coding: utf-8 -*-
"""E3 判决探针：固定架构下 PSO 纯权重搜索能否兑现 h32 的容量优势。

若 PSO(100x50, lam=0) 在 h32 上达不到 Adam 显示的 0.0036 量级，
则"信号对整个 PSO 管线不可见"闭环；若能达到，失明严格发生在结构层。
"""
import sys
sys.path.insert(0, ".")
import numpy as np

from s5_e3 import gen_series, N_TRAIN
from pso_lstm import apso_weight_search

y = gen_series(0.95, 0)
tr = y[:N_TRAIN]
lo, hi = tr.min(), tr.max()
tr_s = ((tr - lo) / (hi - lo)).astype(np.float32)

for hid in (8, 32):
    best, fit, hist = apso_weight_search(tr_s, seq_len=10, hidden=hid,
        n_particles=100, iters=50, seed=0, horizon=1, verbose=False,
        weight_penalty=0.0, fitness_subsample=600)
    print(f"fixed s10 h{hid}: PSO(100x50, lam=0) gbest MSE = {fit:.4f}")
print("对照（Adam 30 epochs 诊断）: h8 ≈ 0.0176, h32 ≈ 0.0036")
