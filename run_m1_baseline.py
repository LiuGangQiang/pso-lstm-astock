# -*- coding: utf-8 -*-
"""M1 基线复跑：12 只标的 × {Persistence, APSO-LSTM}，walk-forward 5 折 × 多种子。

协议（开题文档 V1.1 第五章 5.1）：
- 锚定式 walk-forward 5 折：第 k 折用前 (50 + 10k)% 训练、随后 10% 测试（k = 0..4），
  训练窗口递增，测试窗口覆盖样本后半段 5 段互不重叠区间；
- 多种子：默认 5 个种子，报告均值 ± 标准差；
- 确定性：torch 固定单线程 + 显式种子（原型发现同种子两次运行 RMSE 波动约 5%）；
- 防泄漏：缩放器只在训练折内拟合；
- 增量落盘：已完成的（标的 × horizon × fold × seed × model）组合自动跳过，可分批/断点续跑。

用法：
  python run_m1_baseline.py --stocks 600036,600519 --horizons 1,5,20 --seeds 0,1,2,3,4
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from pso_lstm import (LSTMRegressor, apso_weight_search, eval_mse, make_sequences,
                      metrics, train_model)
from select_stocks import CANDIDATES, fetch_daily

# V1.1 定案的 12 只标的（层级, 代码）——波动率为 select_stocks.py 实测值
POOL = {
    "低波动": ["600900", "600036", "600519", "600887"],
    "中波动": ["000333", "600276", "600031", "002415"],
    "高波动": ["688981", "300750", "300274", "300308"],
}
SEQ_LEN, HIDDEN, EPOCHS = 20, 16, 40          # 与原型 APSO-LSTM 配置一致
SPLIT_START = 0.5     # 第一折训练终点（样本比例）
FOLD_STEP = 0.1       # 每折推进 10%
N_FOLDS = 5
# 搜索预算（main 中由命令行覆盖；V1.2 起 M2 用 100×50）
APSO_PARTICLES, APSO_ITERS = 15, 20
VAL_FITNESS = False   # V1.3：适应度用训练段内部验证切片（最后 20%）
FITNESS_MODE = "train"   # train / val / composite（V1.3+ 实验变体）
WEIGHT_PENALTY = 0.0     # 权重范数惩罚系数 λ（V1.4 候选路线 b）
FITNESS_SUBSAMPLE = 0    # PSO 适应度序列子采样数（0=全量；多保真度搜索加速）
PARTS_DIR = os.path.join("output", "m1_parts")  # 每标的一份 CSV，避免并行写冲突


def load_series(code: str) -> pd.Series:
    path = fetch_daily(code)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    close = df.set_index("date")["close"].astype(float)
    return close[close > 0]


def fold_bounds(n: int, fold: int) -> tuple:
    """第 fold 折：[0, train_end) 训练，[train_end, test_end) 测试。"""
    train_end = int(n * (SPLIT_START + FOLD_STEP * fold))
    test_end = int(n * (SPLIT_START + FOLD_STEP * (fold + 1)))
    return train_end, test_end


def prepare_fold(close: pd.Series, seq_len: int, horizon: int, fold: int):
    """折内 80/20 思路：缩放器只用训练折拟合（论文 02 防泄漏协议）。"""
    n = len(close)
    train_end, test_end = fold_bounds(n, fold)
    train_raw = close.values[:train_end].copy()
    test_raw = close.values[train_end:test_end].copy()
    lo, hi = train_raw.min(), train_raw.max()
    scale = lambda v: ((v - lo) / (hi - lo)).astype(np.float32)
    train_s, test_s = scale(train_raw), scale(test_raw)
    x_tr, y_tr = make_sequences(train_s, seq_len, horizon)
    x_te, y_te = make_sequences(np.concatenate([train_s[-seq_len:], test_s]),
                                seq_len, horizon)
    inv = lambda v: np.asarray(v) * (hi - lo) + lo
    return train_s, x_tr, y_tr, x_te, y_te, inv


def run_combo(close, code, tier, horizon, fold, seed):
    """单个（标的 × horizon × 折 × 种子）：Persistence + APSO-LSTM。"""
    out = []
    train_s, x_tr, y_tr, x_te, y_te, inv = prepare_fold(close, SEQ_LEN, horizon, fold)
    actual = inv(np.asarray(y_te))
    base = dict(stock=code, name=CANDIDATES[code][0], tier=tier,
                horizon=f"+{horizon}d", fold=fold, seed=seed)

    # Persistence（与种子无关）
    if seed == 0:
        pred = inv(np.asarray(x_te[:, -1, 0]))
        out.append({**base, "model": "Persistence", **metrics(actual, pred), "elapsed_s": 0.0})

    # APSO-LSTM（论文 12 路线：PSO 演化初始权重 + Adam 精调）
    t0 = time.time()
    inner_train, val = train_s, None
    if VAL_FITNESS or FITNESS_MODE != "train":
        n_val = max(1, int(len(train_s) * 0.2))
        inner_train, val = train_s[:-n_val], train_s[-n_val:]
    best_vec, _, _ = apso_weight_search(inner_train, seq_len=SEQ_LEN, hidden=HIDDEN,
                                        n_particles=APSO_PARTICLES, iters=APSO_ITERS,
                                        seed=seed, horizon=horizon, verbose=False,
                                        val_close=val, fitness_mode=FITNESS_MODE,
                                        weight_penalty=WEIGHT_PENALTY,
                                        fitness_subsample=FITNESS_SUBSAMPLE)
    m = LSTMRegressor(hidden=HIDDEN)
    m.set_vector(best_vec)
    train_model(m, x_tr, y_tr, epochs=EPOCHS, lr=0.005, seed=seed)
    with torch.no_grad():
        pred = inv(m(torch.from_numpy(x_te)).numpy())
    out.append({**base, "model": "APSO-LSTM", **metrics(actual, pred),
                "elapsed_s": round(time.time() - t0, 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=None, help="逗号分隔代码，默认全部 12 只")
    ap.add_argument("--horizons", default="1,3,5,10,20")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--apso-particles", type=int, default=15)
    ap.add_argument("--apso-iters", type=int, default=20)
    ap.add_argument("--tag", default="m1", help="输出分区 output/<tag>_parts/")
    ap.add_argument("--val-fitness", action="store_true",
                    help="V1.3：PSO 适应度用训练段内部验证切片（最后 20%）")
    ap.add_argument("--fitness-mode", default="train", choices=["train", "val", "composite"])
    ap.add_argument("--weight-penalty", type=float, default=0.0, help="权重范数惩罚 λ")
    ap.add_argument("--fitness-subsample", type=int, default=0, help="PSO 适应度子采样序列数（0=全量）")
    args = ap.parse_args()

    global APSO_PARTICLES, APSO_ITERS, PARTS_DIR, VAL_FITNESS, FITNESS_MODE, WEIGHT_PENALTY, FITNESS_SUBSAMPLE
    APSO_PARTICLES, APSO_ITERS = args.apso_particles, args.apso_iters
    VAL_FITNESS = args.val_fitness
    FITNESS_MODE = args.fitness_mode
    WEIGHT_PENALTY = args.weight_penalty
    FITNESS_SUBSAMPLE = args.fitness_subsample
    PARTS_DIR = os.path.join("output", f"{args.tag}_parts")

    torch.set_num_threads(1)  # 确定性：单线程 + 显式种子
    os.makedirs("output", exist_ok=True)

    tiers = {c: t for t, codes in POOL.items() for c in codes}
    stocks = args.stocks.split(",") if args.stocks else list(tiers)
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    done = {}
    for code in stocks:
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        if os.path.exists(part):
            prev = pd.read_csv(part, dtype={"stock": str})
            done[code] = set(zip(prev.stock, prev.horizon, prev.fold, prev.seed, prev.model))
    os.makedirs(PARTS_DIR, exist_ok=True)

    total = 0
    for code in stocks:
        close = load_series(code)
        part = os.path.join(PARTS_DIR, f"{code}.csv")
        for horizon in horizons:
            for fold in folds:
                for seed in seeds:
                    rows = []
                    for r in run_combo(close, code, tiers[code], horizon, fold, seed):
                        key = (r["stock"], r["horizon"], r["fold"], r["seed"], r["model"])
                        if key in done.get(code, set()):
                            continue
                        rows.append(r)
                        done.setdefault(code, set()).add(key)
                        print(f"[done] {code} {CANDIDATES[code][0]} +{horizon}d fold{fold} "
                              f"seed{seed} {r['model']} RMSE={r['RMSE']:.3f} {r['elapsed_s']}s", flush=True)
                    if rows:
                        # 逐组合落盘：超时被杀也不丢已完成的结果
                        pd.DataFrame(rows).to_csv(part, mode="a", index=False,
                                                  header=not os.path.exists(part),
                                                  encoding="utf-8-sig")
                        total += len(rows)
    print(f"\n本批写入/追加 {total} 行到 {PARTS_DIR}/（每只标的独立文件）" if total else "本批无新增（全部已完成）")


if __name__ == "__main__":
    main()
