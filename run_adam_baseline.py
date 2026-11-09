# -*- coding: utf-8 -*-
"""M5 前补强实验（B2）：等协议 Adam-only 对照（无 PSO 预搜索）。

动机：回应"APSO-LSTM 的表现只是算力堆出来的 / PSO 预搜索是否有贡献"。
同一 LSTM（s20,h16）、同一 walk-forward 5 折 × 5 种子、同一 Adam 精调
（40 epochs, lr=0.005, seed 一致），唯一区别：不做 APSO 权重搜索，
初始权重用论文 12 的 Glorot/正交初始化（init_weights_paper12）。
若 Adam-only 与 APSO-LSTM 无显著差异，则 PSO 预搜索的增量为零；
这也是"PSO 价值在 basin selection 而非精度"的又一佐证。

输出：output/adam_parts/<stock>.csv。
"""
import argparse
import os
import time

import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, init_weights_paper12, metrics, train_model)
from run_m1_baseline import POOL, load_series, prepare_fold
from select_stocks import CANDIDATES

SEQ_LEN, HIDDEN, EPOCHS = 20, 16, 40
PARTS_DIR = os.path.join("output", "adam_parts")


def run_one(close, code, tier, horizon, fold, seed):
    t0 = time.time()
    train_s, x_tr, y_tr, x_te, y_te, inv = prepare_fold(close, SEQ_LEN, horizon, fold)
    actual = inv(y_te)
    m = LSTMRegressor(hidden=HIDDEN)
    init_weights_paper12(m, n_features=1, seed=seed)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    return dict(stock=code, name=CANDIDATES[code][0], tier=tier, horizon=f"+{horizon}d",
                fold=fold, seed=seed, model="Adam-only", arch=f"s{SEQ_LEN},h{HIDDEN}",
                **metrics(actual, pred), elapsed_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=None)
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(PARTS_DIR, exist_ok=True)
    tiers = {c: t for t, codes in POOL.items() for c in codes}
    stocks = args.stocks.split(",") if args.stocks else list(tiers)
    total = 0
    for code in stocks:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        done = set()
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done = set(zip(prev.horizon, prev.fold, prev.seed))
        close = load_series(code)
        rows = []
        for horizon in [int(h) for h in args.horizons.split(",")]:
            for fold in [int(f) for f in args.folds.split(",")]:
                for seed in [int(s) for s in args.seeds.split(",")]:
                    if (f"+{horizon}d", fold, seed) in done:
                        continue
                    rows.append(run_one(close, code, tiers[code], horizon, fold, seed))
        if rows:
            pd.DataFrame(rows).to_csv(part, mode="a", index=False,
                                      header=not os.path.exists(part),
                                      encoding="utf-8-sig")
            total += len(rows)
        print(f"[done] {code} 新增 {len(rows)} 行", flush=True)
    print(f"共 {total} 行")


if __name__ == "__main__":
    main()
