# -*- coding: utf-8 -*-
"""S5-E3 补充：补齐 λ=0 对照组在低信号档 (r=0.2, 0.4) 的缺失单元格，
使 E2 覆盖完整的 5 r × 2 λ 网格（共 200 组主实验）。

结果追加到 output/s5_e3_parts/s5_e3.csv（与原有 160 组同构）。

用法：
  python s5_e3_fill.py                # 时间盒续跑（默认 250 秒）
  python s5_e3_fill.py --budget 600
"""
import argparse
import os
import time

import pandas as pd

from s5_e3 import run_one

RESULT_DIR = os.path.join("output", "s5_e3_parts")
CSV_PATH = os.path.join(RESULT_DIR, "s5_e3.csv")

CELLS = [(0.2, 0.0), (0.4, 0.0)]
SEEDS = range(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=250)
    args = ap.parse_args()

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done = set(zip(df.r, df.lam, df.seed))
        print(f"已完成 {len(done)} 组，断点续跑")

    combos = [(r, lam, seed) for (r, lam) in CELLS for seed in SEEDS
              if (r, lam, seed) not in done]
    print(f"剩余 {len(combos)} 组")
    t_start = time.time()
    rows, n_new = [], 0
    for r, lam, seed in combos:
        try:
            row = run_one(r, lam, seed)
        except Exception as e:  # 单组失败不中断整批
            row = dict(r=r, lam=lam, seed=seed, error=str(e))
        rows.append(row)
        n_new += 1
        print(f"[{n_new}] r={r} λ={lam} seed={seed} "
              f"chosen=s{row.get('chosen_seq','?')},h{row.get('chosen_hidden','?')}", flush=True)
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
