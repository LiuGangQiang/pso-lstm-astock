# -*- coding: utf-8 -*-
"""E3 Decision Probe: Under a fixed architecture, can PSO pure weight search deliver on the capacity advantages of h32?

If PSO(100x50, lam=0) does not reach the 0.0036 magnitude shown by Adam on h32,
then the closed loop "signal is not visible to the entire PSO pipeline"; if it can be achieved, the blindness occurs strictly at the structural level."""
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
print('Control (Adam diagnostic with 30 epochs): h8 ≈ 0.0176, h32 ≈ 0.0036')
