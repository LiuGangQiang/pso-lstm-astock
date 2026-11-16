# -*- coding: utf-8 -*-
"""Run Adam-only controls for all stocks in parallel (batch entry point for run_adam_baseline)."""
import os
import subprocess
import sys

from run_m1_baseline import POOL

stocks = [c for t in ('low_volatility', 'medium_volatility', 'high_volatility') for c in POOL[t]]
todo = [s for s in stocks if not os.path.exists(os.path.join("output", "adam_parts", f"{s}.csv"))]
print('Pending:', todo, flush=True)

# 6-way parallel subprocesses (no lambda, direct Popen)
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
print('All Complete', flush=True)
