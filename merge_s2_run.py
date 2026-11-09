# -*- coding: utf-8 -*-
"""合并 S2 重跑的 600 个单元 csv（output/s2_parts_run/）为规范输出。

产出：
  output/s2_results.csv        —— 全量 1200 行（600 单元 × 2 模型）
  output/s2_parts/<stock>.csv  —— 每股一个 part（与 run_s2.py 旧格式一致）

2026-11-09，V3.2 复核 c0/c7 修复后的重跑结果。
"""
import glob
import os

import pandas as pd

OUT_DIR = os.path.join("output", "s2_parts_run")
PARTS_DIR = os.path.join("output", "s2_parts")
os.makedirs(PARTS_DIR, exist_ok=True)

frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.csv")))]
df = pd.concat(frames, ignore_index=True)
assert len(df) == 1200, f"期望 1200 行，实际 {len(df)}"
assert df.groupby(["stock", "horizon", "fold", "seed"]).size().eq(2).all(), \
    "每个单元应有 Unified/Staged 两行"

df = df.sort_values(["stock", "horizon", "fold", "seed", "model"])
df.to_csv("output/s2_results.csv", index=False)

for stock, g in df.groupby("stock"):
    g.to_csv(os.path.join(PARTS_DIR, f"{stock}.csv"), index=False)

print(f"merged rows={len(df)}, stocks={df['stock'].nunique()}, "
      f"parts written to {PARTS_DIR}")
