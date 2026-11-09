# -*- coding: utf-8 -*-
"""S5-E1 补充：招商银行 600036 的 400 日短窗探针（补齐"四面板"设计）。

设计对齐恒瑞短窗探针（s5_e1.py --win 400）：λ ∈ {0, 0.05, 0.1, 0.5} × 20 种子。
结果追加到 output/s5_e1_parts/s5_e1_win400.csv（stock 列区分标的）。

用法：
  python s5_e1_cmb400.py                # 时间盒续跑（默认 260 秒）
  python s5_e1_cmb400.py --budget 600   # 指定时间预算
"""
import argparse
import os
import time

import pandas as pd

from s5_e1 import run_one

RESULT_DIR = os.path.join("output", "s5_e1_parts")
CSV_PATH = os.path.join(RESULT_DIR, "s5_e1_win400.csv")

GRID = [0.0, 0.05, 0.1, 0.5]
SEEDS = range(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=260)
    args = ap.parse_args()

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, dtype={"stock": str})
        done = set(zip(df.stock, df.lam, df.seed))
        print(f"已完成 {len(done)} 组，断点续跑")

    combos = [(lam, seed) for lam in GRID for seed in SEEDS
              if ("600036", lam, seed) not in done]
    print(f"剩余 {len(combos)} 组")
    t_start = time.time()
    rows, n_new = [], 0
    for lam, seed in combos:
        try:
            row = run_one("600036", lam, seed, win=400)
        except Exception as e:  # 单组失败不中断整批
            row = dict(stock="600036", horizon=5, lam=lam, seed=seed, win=400,
                       error=str(e))
        rows.append(row)
        n_new += 1
        print(f"[{n_new}] 600036 λ={lam} seed={seed} "
              f"share={row.get('penalty_share', float('nan')):.3f} "
              f"norm={row.get('gbest_norm', float('nan')):.2f} "
              f"ratio={row.get('ratio', float('nan')):.3f}", flush=True)
        if time.time() - t_start > args.budget:
            print("时间盒用尽，保存本批结果")
            break
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, mode="a", index=False,
                  header=not os.path.exists(CSV_PATH))
        print(f"本批完成 {len(rows)} 组 -> {CSV_PATH}")


if __name__ == "__main__":
    main()
