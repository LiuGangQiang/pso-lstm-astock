# -*- coding: utf-8 -*-
"""E3 前置诊断 V3.5 修订版：验证合成生成器各结构的可达 MSE 差距（结构信号强度）。

V3.5 修订（复核批注 c13）：
  1. seed bug 修复——torch.manual_seed 在模型构造之前设置（旧版 LSTMRegressor
     的权重初始化发生在 train_model 内部 seeding 之前，初始化不可复现）；
  2. 增加 held-out 对照——在训练段之外的最后 500 点（同一 min-max 缩放）上评估
     样本外 MSE，检验"结构信号"是否带来可泛化的差距而非仅训练段过拟合。

输出：output/s5_e3_diag_v35.csv
"""
import os
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch

from s5_e3 import N_TRAIN, gen_series
from pso_lstm import LSTMRegressor, eval_mse, make_sequences, train_model

GRID = [(10, 8), (40, 8), (10, 32), (40, 32), (60, 32)]
SEEDS = [0, 1, 2]
OUT = os.path.join("output", "s5_e3_diag_v35.csv")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(1)
    rs = [args.r] if args.r is not None else [0.6, 0.95]
    seeds = [args.seed] if args.seed is not None else SEEDS
    rows = []
    for r in rs:
        y = gen_series(r, 0)
        tr = y[:N_TRAIN]
        ho = y[N_TRAIN:]
        lo, hi = tr.min(), tr.max()
        tr_s = ((tr - lo) / (hi - lo)).astype(np.float32)
        ho_s = ((ho - lo) / (hi - lo)).astype(np.float32)
        for seq, hid in GRID:
            for seed in seeds:
                torch.manual_seed(seed)  # 必须在模型构造前 seeding（c13 修复）
                m = LSTMRegressor(hidden=hid)
                train_model(m, make_sequences(tr_s, seq, 1)[0],
                            make_sequences(tr_s, seq, 1)[1],
                            epochs=30, lr=0.01, seed=seed, verbose=False)
                xtr, ytr = make_sequences(tr_s, seq, 1)
                xho, yho = make_sequences(ho_s, seq, 1)
                rows.append(dict(r=r, seq=seq, hid=hid, seed=seed,
                                 train_mse=eval_mse(m, xtr, ytr),
                                 heldout_mse=eval_mse(m, xho, yho)))
                print(f"r={r} seq={seq} h={hid} seed={seed}: "
                      f"train={rows[-1]['train_mse']:.4f} heldout={rows[-1]['heldout_mse']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print()
    print("== 汇总（3 seeds 均值）==")
    print(df.groupby(["r", "seq", "hid"])[["train_mse", "heldout_mse"]].mean().round(4))


if __name__ == "__main__":
    main()
