# -*- coding: utf-8 -*-
"""S3 前台主控：进程池并行跑全部组合，增量落盘、可反复中断续跑。

每次 Bash 调用在前台运行本脚本（≤280s），超时被杀也不丢结果；
再次调用自动跳过已完成组合。

用法：python s3_master.py --workers 14 --budget 260
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from run_m1_baseline import POOL
from run_s3 import (PARTS_DIR, build_features, run_fset)
from select_stocks import CANDIDATES

STOCKS = [c for tier in ("低波动", "中波动", "高波动") for c in POOL[tier]]
HORIZONS = [1, 5, 20]
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [0, 1, 2, 3, 4]
FSETS = ("tech", "sent")

_CACHE = {}


def _data(code, fset):
    if (code, fset) not in _CACHE:
        _CACHE[(code, fset)] = build_features(code, fset)
    return _CACHE[(code, fset)]


def _job(arg):
    code, fset, horizon, fold, seed = arg
    import traceback
    try:
        import torch
        torch.set_num_threads(1)
        feat_raw, close_raw = _data(code, fset)
        tiers = {c: t for t, codes in POOL.items() for c in codes}
        r = run_fset(fset, feat_raw, close_raw, code, tiers[code], horizon, fold, seed)
        # 每 job 独立临时文件，避免多进程并发追加同一 CSV
        tmp = os.path.join(PARTS_DIR, f"tmp_{code}_{fset}_h{horizon}_f{fold}_s{seed}.csv")
        pd.DataFrame([r]).to_csv(tmp, index=False, encoding="utf-8-sig")
        return f"[done] {code} +{horizon}d fold{fold} seed{seed} {r['model']} RMSE={r['RMSE']:.2f} {r['elapsed_s']}s"
    except Exception:
        return f"[ERROR] {code} {fset} h{horizon} f{fold} s{seed}\n{traceback.format_exc()}"


def merge_tmps():
    """把 tmp_*.csv 折叠进 <stock>.csv（去重），返回折叠行数。"""
    tmps = glob.glob(os.path.join(PARTS_DIR, "tmp_*.csv"))
    if not tmps:
        return 0
    add = pd.concat([pd.read_csv(p, dtype={"stock": str}) for p in tmps], ignore_index=True)
    n = 0
    for code, grp in add.groupby("stock"):
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            grp = pd.concat([prev, grp], ignore_index=True)
        grp = grp.drop_duplicates(subset=["stock", "horizon", "fold", "seed", "model"],
                                  keep="last")
        grp.to_csv(part, index=False, encoding="utf-8-sig")
        n += len(grp)
    for p in tmps:
        os.remove(p)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--budget", type=int, default=260, help="本轮最长秒数")
    args = ap.parse_args()

    os.makedirs(PARTS_DIR, exist_ok=True)
    merged = merge_tmps()

    done = set()
    for code in STOCKS:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done |= set(zip(prev.stock, prev.horizon, prev.fold, prev.seed, prev.model))

    jobs = []
    for code in STOCKS:
        for horizon in HORIZONS:
            for fold in FOLDS:
                for seed in SEEDS:
                    for fset in FSETS:
                        if (code, f"+{horizon}d", fold, seed, f"APSO-LSTM-{fset}") not in done:
                            jobs.append((code, fset, horizon, fold, seed))
    print(f"待跑 {len(jobs)} 组，workers={args.workers}", flush=True)
    if not jobs:
        print("全部完成")
        return

    t0 = time.time()
    n = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(_job, jobs, chunksize=1):
            n += 1
            if msg.startswith("[ERROR]"):
                failed += 1
                print(msg, flush=True)
            elif n % 25 == 0 or time.time() - t0 > args.budget:
                print(msg, flush=True)
            if time.time() - t0 > args.budget:
                print(f"[budget] 本轮 {n} 组（失败 {failed}），剩余 {len(jobs) - n} 组下次续跑", flush=True)
                ex.shutdown(wait=False, cancel_futures=True)
                return
    print(f"本轮完成 {n} 组（失败 {failed}）" + ("，全部结束" if n == len(jobs) and not failed else ""), flush=True)


if __name__ == "__main__":
    main()
