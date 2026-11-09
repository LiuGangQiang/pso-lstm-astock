# -*- coding: utf-8 -*-
"""S3 驱动：并行拉起 run_s3 子进程。用法：python s3_driver.py --start 0 --count 8"""
import argparse
import subprocess
import sys

from run_m1_baseline import POOL

STOCKS = [c for tier in ("低波动", "中波动", "高波动") for c in POOL[tier]]
HORIZONS = [1, 5, 20]
FOLDS = [0, 1, 2, 3, 4]
COMBOS = [(s, h, f) for s in STOCKS for h in HORIZONS for f in FOLDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=8)
    args = ap.parse_args()

    slice_ = COMBOS[args.start:args.start + args.count]
    if not slice_:
        print("组合索引超出范围")
        return
    procs = []
    for s, h, f in slice_:
        cmd = [sys.executable, "run_s3.py", "--stocks", s,
               "--horizons", str(h), "--folds", str(f), "--seeds", "0,1,2,3,4"]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    print(f"并行启动 {len(procs)} 个子进程，组合 {args.start}..{args.start + len(slice_) - 1}")


if __name__ == "__main__":
    main()
