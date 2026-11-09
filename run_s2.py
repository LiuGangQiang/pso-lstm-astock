# -*- coding: utf-8 -*-
"""M3 / S2：联合编码偏置诊断实验（开题文档 V1.5 第五章 5.2）。

模型（两个，外加 s1v14 已有 APSO-LSTM 作固定架构对照）：
  1. Unified-PSO   —— 朴素联合编码（架构段+权重段同粒子），V1.4 公平配置（λ=0.1 + 子采样600）
  2. Staged-PSO    —— 分阶段联合编码（外层结构 PSO + 内层小型 APSO），V1.4 配置

设计（H3）：h ∈ {5, 10} 纯价格；12 标的 × 5 折 × 5 种子。
每个 run 结束后用最佳粒子经 Adam 精调（40 epochs, lr=0.005）再评估，
与 s1v14 的 APSO-LSTM 协议完全一致。

用法：
  python run_s2.py --stocks 600276 --horizons 10 --folds 0 --seeds 0,1,2
  python s2_driver.py --start 0 --count 8
"""
import argparse
import os
import time

import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, eval_mse, make_sequences, metrics,
                      staged_pso_search, train_model, unified_pso_search)
from run_m1_baseline import POOL, fold_bounds, load_series, prepare_fold
from select_stocks import CANDIDATES

EPOCHS = 40
PARTS_DIR = os.path.join("output", "s2_parts")
# V1.4 公平配置
WEIGHT_PENALTY = 0.1
SUBSAMPLE = 600
# 预算：统一编码沿用原型 15×20；分阶段外层 12×10 + 内层 10×8
UNI_P, UNI_I = 15, 20
OUT_P, OUT_I, IN_P, IN_I = 12, 10, 10, 8


def run_model(model_name, close, code, tier, horizon, fold, seed):
    """返回该模型在测试折上的指标行。"""
    t0 = time.time()

    if model_name == "Unified-PSO":
        # 结构搜索期 fitness 的序列长度由各粒子自己的结构段决定（默认窗口 20 仅用于
        # 取训练段）；精调与评估数据在选定 s 后重建（V3.2 复核 c7 修复）。
        train_s, _, _, _, _, _ = prepare_fold(close, 20, horizon, fold)
        best_vec, fit, hist, decode = unified_pso_search(
            train_s, n_particles=UNI_P, iters=UNI_I, seed=seed, horizon=horizon,
            weight_penalty=WEIGHT_PENALTY, fitness_subsample=SUBSAMPLE)
        s, h = decode(best_vec)
        m = LSTMRegressor(hidden=h)
        m.set_vector(best_vec[2:2 + m.get_vector().size])
    else:  # Staged-PSO
        train_s, _, _, _, _, _ = prepare_fold(close, 20, horizon, fold)
        best_vec, fit, hist, decode = staged_pso_search(
            train_s, outer_particles=OUT_P, outer_iters=OUT_I, seed=seed, horizon=horizon,
            inner_particles=IN_P, inner_iters=IN_I, weight_penalty=WEIGHT_PENALTY,
            fitness_subsample=SUBSAMPLE, verbose=False)
        s, h = decode(best_vec)
        # 内层最优权重：重跑一次内层搜索拿到权重（确定性同种子），再注入
        from pso_lstm import apso_weight_search
        wvec, _, _ = apso_weight_search(
            train_s, seq_len=s, hidden=h, n_particles=IN_P, iters=IN_I, seed=seed,
            horizon=horizon, verbose=False, weight_penalty=WEIGHT_PENALTY,
            fitness_subsample=SUBSAMPLE)
        m = LSTMRegressor(hidden=h)
        m.set_vector(wvec)

    # 按搜索选中的序列长度 s 重建精调/评估数据（c7：旧版恒用 20 日窗口）
    train_s, x_tr, y_tr, x_te, y_te, inv = prepare_fold(close, s, horizon, fold)
    actual = inv(y_te)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    row = dict(stock=code, name=CANDIDATES[code][0], tier=tier, horizon=f"+{horizon}d",
               fold=fold, seed=seed, model=model_name, arch=f"s{s},h{h}",
               **metrics(actual, pred), elapsed_s=round(time.time() - t0, 1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", required=True)
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()

    torch.set_num_threads(1)
    tiers = {c: t for t, codes in POOL.items() for c in codes}
    os.makedirs(PARTS_DIR, exist_ok=True)

    part = os.path.join(PARTS_DIR, f"{args.stocks}.csv")
    done = set()
    if os.path.exists(part):
        prev = pd.read_csv(part, dtype={"stock": str})
        done = set(zip(prev.stock, prev.horizon, prev.fold, prev.seed, prev.model))

    close = load_series(args.stocks)
    total = 0
    for horizon in [int(h) for h in args.horizons.split(",")]:
        for fold in [int(f) for f in args.folds.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                for model in ["Unified-PSO", "Staged-PSO"]:
                    key = (args.stocks, f"+{horizon}d", fold, seed, model)
                    if key in done:
                        continue
                    r = run_model(model, close, args.stocks, tiers[args.stocks],
                                  horizon, fold, seed)
                    pd.DataFrame([r]).to_csv(part, mode="a", index=False,
                                             header=not os.path.exists(part),
                                             encoding="utf-8-sig")
                    total += 1
                    print(f"[done] {args.stocks} +{horizon}d fold{fold} seed{seed} "
                          f"{model} {r['arch']} RMSE={r['RMSE']:.2f} {r['elapsed_s']}s", flush=True)
    print(f"写入 {total} 行" if total else "无新增")


if __name__ == "__main__":
    main()
