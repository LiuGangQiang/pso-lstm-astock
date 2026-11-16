# -*- coding: utf-8 -*-
"""Merge the 600 unit CSVs (output/s2_parts_run/) from the S2 rerun into a standardized output.

Output:
  output/s2_results.csv        —— Full 1200 rows (600 units × 2 models)
  output/s2_parts/<stock>.csv  —— One part per unit (consistent with the old format of run_s2.py)

2026-11-09, V3.2 Review of the rerun results after the c0/c7 fix."""
import glob
import os

import pandas as pd

OUT_DIR = os.path.join("output", "s2_parts_run")
PARTS_DIR = os.path.join("output", "s2_parts")
os.makedirs(PARTS_DIR, exist_ok=True)

frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.csv")))]
df = pd.concat(frames, ignore_index=True)
assert len(df) == 1200, f"Expected 1200 rows; actual: {len(df)}"
assert df.groupby(["stock", "horizon", "fold", "seed"]).size().eq(2).all(), \
    'Each cell should contain two rows: Unified and Staged'

df = df.sort_values(["stock", "horizon", "fold", "seed", "model"])
df.to_csv("output/s2_results.csv", index=False)

for stock, g in df.groupby("stock"):
    g.to_csv(os.path.join(PARTS_DIR, f"{stock}.csv"), index=False)

print(f"merged rows={len(df)}, stocks={df['stock'].nunique()}, "
      f"parts written to {PARTS_DIR}")
