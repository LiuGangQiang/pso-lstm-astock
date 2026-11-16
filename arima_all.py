# -*- coding: utf-8 -*-
"""Run ARIMA baselines for all stocks in parallel (batch entry point for run_arima_baseline)."""
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from run_arima_baseline import PARTS_DIR, run_stock
from run_m1_baseline import POOL

stocks = [c for t in ('low_volatility', 'medium_volatility', 'high_volatility') for c in POOL[t]]
todo = [s for s in stocks
        if not (os.path.exists(os.path.join(PARTS_DIR, f"{s}.csv"))
                and len(pd.read_csv(os.path.join(PARTS_DIR, f"{s}.csv"))) >= 15)]
print('Pending:', todo, flush=True)

with ProcessPoolExecutor(max_workers=6) as ex:
    for code, rows in zip(todo, ex.map(run_stock, todo)):
        pd.DataFrame(rows).to_csv(os.path.join(PARTS_DIR, f"{code}.csv"),
                                  index=False, encoding="utf-8-sig")
        print(code, "done", len(rows), flush=True)
print('All Complete', flush=True)
