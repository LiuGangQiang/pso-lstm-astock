# -*- coding: utf-8 -*-
"""对照实验并行驱动：ProcessPoolExecutor + 断点续跑，每次调用处理一块。

用法：python run_controls_pool.py [块大小=40] [并发=8]
"""
import os
import sys
import time

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from s5_e3_controls import CSV_PATH, all_combos, run_one


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done = set(zip(df.control, df.r, df.lam, df.seed))
    rem = [c for c in all_combos() if (c[0], c[1], c[2], c[3]) not in done]
    print(f"剩余 {len(rem)} 组，本块取 {min(chunk, len(rem))} 组，{workers} 并发")
    block = rem[:chunk]

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, c[0], c[1], c[2], c[3], c[4]): c for c in block}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append(dict(control=c[0], r=c[1], lam=c[2], seed=c[3], mem=c[4],
                                 error=str(e)))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, mode="a", index=False, header=not os.path.exists(CSV_PATH))
    ok = sum(1 for r in rows if "error" not in r)
    print(f"本块完成 {ok}/{len(rows)} 组，用时 {time.time()-t0:.0f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
