# -*- coding: utf-8 -*-
"""S1 扫描驱动：把 (标的 × horizon × 折) 组合列表切片，并行拉起多个
run_m1_baseline 子进程（每个子进程内部串行 5 种子），全部 wait 后才返回。

用法：python s1_driver.py --start 0 --count 8
已完成的组合由 run_m1_baseline 的 parts CSV 自动跳过（断点续跑安全）。
"""
import argparse
import subprocess
import sys

from run_m1_baseline import POOL

STOCKS = [c for tier in ("低波动", "中波动", "高波动") for c in POOL[tier]]
HORIZONS = [1, 3, 5, 10, 20]
FOLDS = [0, 1, 2, 3, 4]
COMBOS = [(s, h, f) for s in STOCKS for h in HORIZONS for f in FOLDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--apso-particles", type=int, default=100)
    ap.add_argument("--apso-iters", type=int, default=50)
    ap.add_argument("--weight-penalty", type=float, default=0.1)
    ap.add_argument("--fitness-subsample", type=int, default=600)
    ap.add_argument("--horizons", default="1,3,5,10,20", help="组合枚举用的 horizon 列表")
    ap.add_argument("--tag", default="s1v14")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    combos = [(s, h, f) for s in STOCKS for h in horizons for f in FOLDS]
    slice_ = combos[args.start:args.start + args.count]
    if not slice_:
        print("组合索引超出范围，全部完成")
        return
    procs = []
    for s, h, f in slice_:
        cmd = [sys.executable, "run_m1_baseline.py",
               "--stocks", s, "--horizons", str(h), "--folds", str(f),
               "--seeds", "0,1,2,3,4",
               "--apso-particles", str(args.apso_particles),
               "--apso-iters", str(args.apso_iters),
               "--weight-penalty", str(args.weight_penalty),
               "--fitness-subsample", str(args.fitness_subsample),
               "--tag", args.tag]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    print(f"并行启动 {len(procs)} 个子进程，组合 {args.start}..{args.start + len(slice_) - 1}")
    for p in procs:
        p.wait()
    print("本批全部子进程结束")


if __name__ == "__main__":
    main()
