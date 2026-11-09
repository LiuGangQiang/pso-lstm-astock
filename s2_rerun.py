# -*- coding: utf-8 -*-
"""S2 重跑驱动（V3.2 复核 c0/c7 修复后，2026-11-09）。

最小单元 = (stock, horizon, fold, seed)，各进程独立写
output/s2_parts_run/<stock>_h<h>_f<f>_s<s>.csv，避免并发写冲突；
全部完成后由 merge_s2_run.py 合并为规范 parts。

用法：python s2_rerun.py --stock 600276 --horizon 5 --fold 0 --seed 0
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, ".")
torch.set_num_threads(1)  # 多进程并行，单线程避免 thrash

from run_m1_baseline import POOL, load_series
from run_s2 import run_model

OUT_DIR = os.path.join("output", "s2_parts_run")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--horizon", required=True, type=int)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    tiers = {c: t for t, codes in POOL.items() for c in codes}
    tag = f"{args.stock}_h{args.horizon}_f{args.fold}_s{args.seed}"
    part = os.path.join(OUT_DIR, f"{tag}.csv")
    if os.path.exists(part):
        print(tag, "skip")
        return

    close = load_series(args.stock)
    rows = []
    for model in ("Unified-PSO", "Staged-PSO"):
        t0 = time.time()
        r = run_model(model, close, args.stock, tiers[args.stock],
                      args.horizon, args.fold, args.seed)
        r["wall_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(tag, model, r["arch"], f"RMSE={r['RMSE']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(part, index=False, encoding="utf-8-sig")
    print(tag, "OK", flush=True)


if __name__ == "__main__":
    main()
