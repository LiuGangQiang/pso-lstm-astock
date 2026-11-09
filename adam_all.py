# -*- coding: utf-8 -*-
"""并行跑全部标的的 Adam-only 对照（run_adam_baseline 的批量入口）。"""
import os
import subprocess
import sys

from run_m1_baseline import POOL

stocks = [c for t in ("低波动", "中波动", "高波动") for c in POOL[t]]
todo = [s for s in stocks if not os.path.exists(os.path.join("output", "adam_parts", f"{s}.csv"))]
print("待跑:", todo, flush=True)

# 6 路并行子进程（无 lambda，直接 Popen）
procs = [subprocess.Popen([sys.executable, "run_adam_baseline.py", "--stocks", s])
         for s in todo[:6]]
for s in todo[6:]:
    while True:
        for i, p in enumerate(procs):
            if p.poll() is not None:
                procs[i] = subprocess.Popen([sys.executable, "run_adam_baseline.py", "--stocks", s])
                break
        else:
            import time
            time.sleep(2)
            continue
        break
for p in procs:
    p.wait()
print("全部完成", flush=True)
