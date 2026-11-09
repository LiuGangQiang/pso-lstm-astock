# -*- coding: utf-8 -*-
"""S5-E1：λ 相变实验（机制理论化命题 1/2 的实证先行验证）。

设计：固定恒瑞医药 600276（主，最接近交叉点的标的）/ 招商银行 600036（稳健性），
h=+5d，生产配置（hidden=16, seq_len=20, 100 粒子 × 50 代, 适应度子采样 600），
扫描 λ ∈ {0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5} × 独立种子。

每个 (λ, seed) 运行记录：
  - 终态群体两两距离中位数（swarm_med_pairdist）——盆地收缩的直接度量
  - gbest 适应度分解（mse_part / penalty_part / penalty_share）——惩罚主导程度
  - gbest 权重范数（gbest_norm）
  - PSO+Adam 全流程后测试集 RMSE 及对 Persistence 比值（ratio）——跨种子一致性的精度口径
增量写 CSV，可断点续跑。

用法：
  python s5_e1.py                 # 跑下一批（时间盒内尽量多跑）
  python s5_e1.py --stock 600276 --lam 0.1 --seed 0   # 单次测试
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from data_fetch import fetch_daily
from pso_lstm import (LSTMRegressor, PSO, eval_mse, init_weights_paper12,
                      make_sequences, train_model)

RESULT_DIR = os.path.join("output", "s5_e1_parts")
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "s5_e1.csv")

TEST_RATIO = 0.2
HIDDEN, SEQ_LEN = 16, 20
N_PARTICLES, ITERS = 100, 50
SUBSAMPLE = 600
ADAM_EPOCHS, ADAM_LR = 40, 0.005

GRID_MAIN = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
GRID_ROBUST = [0.0, 0.05, 0.1, 0.5]
SEEDS_MAIN = range(30)
SEEDS_ROBUST = range(20)
HORIZON = 5


def all_combos():
    combos = [("600276", lam, seed) for lam in GRID_MAIN for seed in SEEDS_MAIN]
    combos += [("600036", lam, seed) for lam in GRID_ROBUST for seed in SEEDS_ROBUST]
    return combos


def load_close(code: str) -> np.ndarray:
    df = pd.read_csv(fetch_daily(code), parse_dates=["date"]).sort_values("date")
    close = df["close"].astype(float)
    close = close[close > 0]
    return close.values.astype(np.float64)


def med_pairdist(pos: np.ndarray) -> float:
    """群体位置两两欧氏距离中位数（抽样 500 对以防 O(n²·d) 过大）。"""
    n = pos.shape[0]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 500), rng.integers(0, n, 500)
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    d = d[i != j] if (i != j).any() else d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


def run_one(code: str, lam: float, seed: int, win: int = 0) -> dict:
    close = load_close(code)
    n = len(close)
    split = int(n * (1 - TEST_RATIO))
    tr_raw, te_raw = close[:split], close[split:]
    if win:  # 短窗机制探针：只用训练段前 win 天拟合缩放器，测试接其后 100 天
        tr_raw, te_raw = tr_raw[:win], tr_raw[win:win + 100]
    lo, hi = tr_raw.min(), tr_raw.max()
    scale = lambda v: ((v - lo) / (hi - lo)).astype(np.float32)
    tr_s, te_s = scale(tr_raw), scale(te_raw)

    x_tr, y_tr = make_sequences(tr_s, SEQ_LEN, HORIZON)
    x_te, y_te = make_sequences(np.concatenate([tr_s[-SEQ_LEN:], te_s]), SEQ_LEN, HORIZON)

    base = LSTMRegressor(hidden=HIDDEN)
    init_weights_paper12(base, n_features=1, seed=seed)
    dim = base.get_vector().shape[0]
    pso = PSO(dim=dim, n_particles=N_PARTICLES, iters=ITERS,
              adaptive=True, c1=1.5, c2=1.5, seed=seed, verbose=False)

    if SUBSAMPLE and len(x_tr) > SUBSAMPLE:
        idx = np.random.RandomState(seed).choice(len(x_tr), SUBSAMPLE, replace=False)
        fx, fy = x_tr[idx], y_tr[idx]
    else:
        fx, fy = x_tr, y_tr

    def fitness(vec):
        model = LSTMRegressor(hidden=HIDDEN)
        model.set_vector(vec)
        mse = eval_mse(model, fx, fy)
        pen = lam * float(np.mean(vec ** 2))
        return mse + pen

    t0 = time.time()
    best, fit = pso.run(fitness)
    pso_time = time.time() - t0

    # gbest 适应度分解（用全量训练序列评估 mse 项，与搜索期子采样口径区分）
    gbest_model = LSTMRegressor(hidden=HIDDEN)
    gbest_model.set_vector(best)
    mse_part = eval_mse(gbest_model, x_tr, y_tr)
    pen_part = lam * float(np.mean(best ** 2))

    # Adam 精调（生产配置）
    torch.manual_seed(seed)
    t0 = time.time()
    train_model(gbest_model, x_tr, y_tr, epochs=ADAM_EPOCHS, lr=ADAM_LR, seed=seed)
    adam_time = time.time() - t0
    with torch.no_grad():
        pred_s = gbest_model(torch.from_numpy(x_te)).numpy().reshape(-1)
    test_rmse = float(np.sqrt(np.mean(((pred_s - y_te) * (hi - lo)) ** 2)))
    persist_pred = np.asarray(x_te[:, -1, 0])
    persist_rmse = float(np.sqrt(np.mean(((persist_pred - y_te) * (hi - lo)) ** 2)))

    return dict(
        stock=code, horizon=HORIZON, lam=lam, seed=seed, win=win,
        swarm_med_pairdist=med_pairdist(pso.pos),
        gbest_norm=float(np.linalg.norm(best)),
        gbest_mse_part=mse_part, gbest_pen_part=pen_part,
        penalty_share=pen_part / (mse_part + pen_part + 1e-12),
        final_gbest_fit=float(fit),
        pso_time_s=round(pso_time, 1), adam_time_s=round(adam_time, 1),
        test_rmse=test_rmse, persist_rmse=persist_rmse,
        ratio=test_rmse / persist_rmse,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--budget", type=float, default=260, help="本批时间预算（秒）")
    ap.add_argument("--win", type=int, default=0, help="短窗天数（0=全长训练段）")
    args = ap.parse_args()

    if args.stock and args.lam is not None and args.seed is not None:
        row = run_one(args.stock, args.lam, args.seed, win=args.win)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        return

    csv_path = CSV_PATH if not args.win else CSV_PATH.replace(".csv", f"_win{args.win}.csv")
    done = set()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"stock": str})
        done = set(zip(df.stock, df.lam, df.seed))
        print(f"已完成 {len(done)} 组，断点续跑")

    if args.win:
        combos = [("600276", lam, seed) for lam in (0.0, 0.1) for seed in range(30)]
    else:
        combos = all_combos()
    combos = [c for c in combos if c not in done]
    print(f"剩余 {len(combos)} 组")
    t_start = time.time()
    rows, n_new = [], 0
    for code, lam, seed in combos:
        try:
            row = run_one(code, lam, seed, win=args.win)
        except Exception as e:  # 单组失败不中断整批
            row = dict(stock=code, horizon=HORIZON, lam=lam, seed=seed, error=str(e))
        rows.append(row)
        n_new += 1
        print(f"[{n_new}] {code} λ={lam} seed={seed} "
              f"pairdist={row.get('swarm_med_pairdist', float('nan')):.2f} "
              f"share={row.get('penalty_share', float('nan')):.3f} "
              f"ratio={row.get('ratio', float('nan')):.3f}")
        if time.time() - t_start > args.budget:
            break
    if rows:
        df = pd.DataFrame(rows)
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", index=False, header=header)
        print(f"本批完成 {len(rows)} 组 -> {csv_path}")


if __name__ == "__main__":
    main()
