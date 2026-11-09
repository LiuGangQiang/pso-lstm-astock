# -*- coding: utf-8 -*-
"""M5 前补强实验（B1）：ARIMA 基线（AIC 选阶），纯 numpy 实现。

动机：S1–S4 的统计基线只有 Persistence，审稿人可能质疑"模型只比朴素基线差"。
补一个文献标准基线：对 log(close) 做 ARIMA(p,1,0)，p ∈ {0..4} 按训练折 AIC 选阶，
递归多步预测 t+h 收盘价；与 walk-forward 5 折协议完全一致（缩放器口径同基线：
预测在 min-max 反缩放空间评估，与 Persistence / APSO-LSTM 直接可比）。

注：GARCH(1,1) 的点预测与均值模型相同（条件均值递推），对 RMSE 对比无增量，
故基准补强的点预测部分以 ARIMA 为代表，GARCH 仅在论文中作讨论。

输出：output/arima_parts/<stock>.csv（每行 = stock × horizon × fold），
随后由 s4_cells 结构并入 output/b1_cells.csv 供比值对比。
"""
import argparse
import os

import numpy as np
import pandas as pd

from pso_lstm import metrics
from run_m1_baseline import POOL, fold_bounds, load_series
from select_stocks import CANDIDATES

HORIZONS = [1, 5, 20]
PARTS_DIR = os.path.join("output", "arima_parts")


def aic_ar(diff: np.ndarray, p: int) -> float:
    """差分序列上 AR(p) 的最小二乘拟合 AIC（σ² 用残差方差，含常数项）。"""
    n = len(diff)
    if p == 0:
        resid = diff - diff.mean()
        sigma2 = max(np.mean(resid ** 2), 1e-12)
        return n * np.log(sigma2) + 2 * 1
    X, y = [], []
    for t in range(p, n):
        X.append(np.concatenate([[1.0], diff[t - p:t]]))
        y.append(diff[t])
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = max(np.mean(resid ** 2), 1e-12)
    k = p + 2          # p 个 AR 系数 + 常数 + σ²
    return len(y) * np.log(sigma2) + 2 * k


def arima_forecast(train_log: np.ndarray, horizon: int) -> np.ndarray:
    """训练折 log 价格：一阶差分 → AIC 选 p → 递归预测 horizon 步，返回 log 空间预测。"""
    diff = np.diff(train_log)
    p_best = min(range(5), key=lambda p: aic_ar(diff, p))
    n = len(diff)
    if p_best == 0:
        c = diff.mean()
        fc_diff = np.full(horizon, c)
        beta = None
    else:
        X, y = [], []
        for t in range(p_best, n):
            X.append(np.concatenate([[1.0], diff[t - p_best:t]]))
            y.append(diff[t])
        beta, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
        hist = list(diff)
        fc_diff = []
        for _ in range(horizon):
            x = np.concatenate([[1.0], hist[-p_best:]])
            d = float(x @ beta)
            fc_diff.append(d)
            hist.append(d)
        fc_diff = np.asarray(fc_diff)
    last = train_log[-1]
    return last + np.cumsum(fc_diff)


def run_stock(code: str):
    close = load_series(code)
    n = len(close)
    rows = []
    tier = {c: t for t, codes in POOL.items() for c in codes}[code]
    for fold in range(5):
        train_end, test_end = fold_bounds(n, fold)
        train_raw = close.values[:train_end]
        test_raw = close.values[train_end:test_end]
        test_len = len(test_raw)
        # walk-forward 滚动原点：每个测试点 t 用其之前的全部数据重新选阶+拟合，
        # 再 h 步外推。评估目标 = close[origin + h - 1]（与 LSTM 的 realized
        # h-day 标签完全对齐，V3.2 复核 c1 修复：旧版误与 close[origin] 比较）；
        # 有效样本数 = test_len - h + 1（边界处 h 日窗口不完整的样本弃用，
        # 与 LSTM 协议的标签清除规则一致）。
        log_all = np.log(np.maximum(close.values, 1e-6))
        for horizon in HORIZONS:
            n_eval = test_len - horizon + 1
            preds = np.empty(n_eval)
            for i in range(n_eval):
                origin = train_end + i
                train_log = log_all[:origin]
                fc = arima_forecast(train_log, horizon)   # 长度为 h 的 log 空间预测
                preds[i] = np.exp(np.clip(fc[-1], -20, 20))
            targets = test_raw[horizon - 1:]
            row = dict(stock=code, name=CANDIDATES[code][0], tier=tier,
                       horizon=f"+{horizon}d", fold=fold, seed=0, model="ARIMA",
                       arch="aic-ar", **metrics(targets, preds), elapsed_s=0.0)
            rows.append(row)
            print(f"  {code} +{horizon}d fold{fold} RMSE={row['RMSE']:.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=None)
    args = ap.parse_args()
    os.makedirs(PARTS_DIR, exist_ok=True)
    tiers = {c: t for t, codes in POOL.items() for c in codes}
    stocks = args.stocks.split(",") if args.stocks else list(tiers)
    total = 0
    for code in stocks:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        done_h = set()
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done_h = set(zip(prev.horizon, prev.fold))
        rows = [r for r in run_stock(code)
                if (r["horizon"], r["fold"]) not in done_h]
        if rows:
            pd.DataFrame(rows).to_csv(part, mode="a", index=False,
                                      header=not os.path.exists(part),
                                      encoding="utf-8-sig")
            total += len(rows)
        print(f"[done] {code} {CANDIDATES[code][0]} 新增 {len(rows)} 行", flush=True)
    print(f"共 {total} 行")


if __name__ == "__main__":
    main()
