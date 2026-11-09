# -*- coding: utf-8 -*-
"""预算曲线并行驱动（分块 + 容错，Windows spawn 友好）。"""
import os

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from s5_e3_budget_curve import BUDGETS, HIDS, OUT, SEEDS, run_one


def main():
    done = set()
    if os.path.exists(OUT):
        df = pd.read_csv(OUT)
        done = set(zip(df.budget, df.hidden, df.seed))
    combos = [(b, h, s) for b in BUDGETS for h in HIDS for s in SEEDS
              if (b, h, s) not in done]
    print(f"剩余 {len(combos)} 组")
    block, rows = combos[:20], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, b, h, s): (b, h, s) for b, h, s in block}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"FAIL {c}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
    print(f"本块完成 {len(rows)}/{len(block)} -> {OUT}")


if __name__ == "__main__":
    main()
