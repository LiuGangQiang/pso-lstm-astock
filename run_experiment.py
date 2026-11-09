# -*- coding: utf-8 -*-
"""端到端实验：4 个模型 × 3 只 A 股的对比。

模型：
  1. LSTM      —— 固定架构基线（hidden=32, seq_len=20, Adam）
  2. GS-LSTM   —— 网格搜索超参数（论文 02 基线的精简版）
  3. PSO-LSTM  —— PSO 搜索 (seq_len, hidden, epochs)（论文 02 路线 A）
  4. APSO-LSTM —— 自适应 PSO 演化初始权重 + Adam 精调（论文 12 路线 B）

实验协议沿用论文 02 的防泄漏做法：先 80/20 分割，再用训练集拟合 MinMax 缩放器；
评价在原始价格尺度上进行。指标为论文 12 五指标 + 论文 02 的 NRMSE 与 %Similar。
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_fetch import STOCKS, fetch_daily
from pso_lstm import (LSTMRegressor, apso_weight_search, decode_hyper,
                      eval_mse, make_sequences, metrics, pso_hyper_search,
                      train_model, unified_pso_search)

RESULT_DIR = "output"
os.makedirs(RESULT_DIR, exist_ok=True)

TEST_RATIO = 0.2
HIDDEN_BASE = 32
SEQ_BASE = 20
EPOCHS_BASE = 40
HORIZONS = [1, 3, 5]  # 预测期：t+1 / t+3 / t+5 日
GRID = [(s, h) for s in (10, 20, 40) for h in (16, 32, 64)]  # GS-LSTM 网格（9 组合）


def load_series(code: str) -> pd.Series:
    path = fetch_daily(code)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    close = df.set_index("date")["close"].astype(float)
    close = close[(close > 0)]  # 剔除脏数据
    return close


def prepare(close: pd.Series, seq_len: int, horizon: int = 1):
    """先分割、后缩放（论文 02，防数据泄漏）。"""
    n = len(close)
    split = int(n * (1 - TEST_RATIO))
    train_raw, test_raw = close.values[:split], close.values[split:]
    lo, hi = train_raw.min(), train_raw.max()
    scale = lambda v: ((v - lo) / (hi - lo)).astype(np.float32)
    train_s, test_s = scale(train_raw), scale(test_raw)
    x_tr, y_tr = make_sequences(train_s, seq_len, horizon)
    x_te, y_te = make_sequences(np.concatenate([train_s[-seq_len:], test_s]),
                                seq_len, horizon)
    inv = lambda v: np.asarray(v) * (hi - lo) + lo
    return (x_tr, y_tr, x_te, y_te, inv)


def run_stock(code: str, name: str, horizon: int = 1) -> pd.DataFrame:
    print(f"\n{'=' * 60}\n### {code} {name}  horizon=+{horizon}d\n{'=' * 60}")
    close = load_series(code)
    print(f"数据区间 {close.index[0].date()} ~ {close.index[-1].date()}，共 {len(close)} 个交易日")

    # 波动率标注：年化日收益波动
    vol = close.pct_change().std() * np.sqrt(244) * 100
    print(f"年化波动率 ≈ {vol:.1f}%")

    rows, curves = [], {}

    def score(model, label, elapsed):
        """在原始价格尺度上计算指标并记录预测曲线。"""
        with torch.no_grad():
            pred_s = model(torch.from_numpy(x_te)).numpy()
        actual = inv(np.asarray(y_te))
        pred = inv(pred_s)
        rows.append((label, *metrics(actual, pred).values(), f"{elapsed:.0f}s"))
        curves[label] = pred

    # ---------- 0. Persistence 朴素基线（最后观测值 = 未来 horizon 日预测） ----------
    x_tr, y_tr, x_te, y_te, inv = prepare(close, SEQ_BASE, horizon)
    pred = inv(np.asarray(x_te[:, -1, 0]))
    rows.append(("Persistence", *metrics(inv(np.asarray(y_te)), pred).values(), "0s"))
    curves["Persistence"] = pred

    # ---------- 1. 标准 LSTM 基线 ----------
    t0 = time.time()
    x_tr, y_tr, x_te, y_te, inv = prepare(close, SEQ_BASE, horizon)
    model = LSTMRegressor(hidden=HIDDEN_BASE)
    train_model(model, x_tr, y_tr, epochs=EPOCHS_BASE, seed=0)
    score(model, "LSTM(基线)", time.time() - t0)

    # ---------- 2. GS-LSTM 网格搜索 ----------
    t0 = time.time()
    best_gs, best_mse = None, np.inf
    for s, h in GRID:
        x_tr, y_tr, _, _, _ = prepare(close, s, horizon)
        m = LSTMRegressor(hidden=h)
        mse = train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE // 2, seed=0)
        if mse < best_mse:
            best_gs, best_mse = (s, h), mse
    s, h = best_gs
    x_tr, y_tr, x_te, y_te, inv = prepare(close, s, horizon)
    m = LSTMRegressor(hidden=h)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, seed=0)
    score(m, f"GS-LSTM(s{s},h{h})", time.time() - t0)

    # ---------- 3. PSO-LSTM：搜索超参数/序列长度（论文 02 路线 A） ----------
    t0 = time.time()
    split = int(len(close) * (1 - TEST_RATIO))
    tr = close.values[:split]
    train_s = ((tr - tr.min()) / np.ptp(tr)).astype(np.float32)
    (s, h, e), fit, hist_a = pso_hyper_search(train_s, seed=0, horizon=horizon)
    x_tr, y_tr, x_te, y_te, inv = prepare(close, s, horizon)
    m = LSTMRegressor(hidden=h)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, seed=0)  # 与基线同等训练预算，公平比较
    score(m, f"PSO-LSTM(s{s},h{h})", time.time() - t0)

    # ---------- 4. APSO-LSTM：演化初始权重 + Adam 精调（论文 12 路线 B） ----------
    t0 = time.time()
    x_tr, y_tr, x_te, y_te, inv = prepare(close, SEQ_BASE, horizon)
    best_vec, fit, hist_b = apso_weight_search(train_s,
                                               seq_len=SEQ_BASE, hidden=16, seed=0, horizon=horizon)
    m = LSTMRegressor(hidden=16)
    m.set_vector(best_vec)
    pre_mse = eval_mse(m, x_tr, y_tr)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, lr=0.005, seed=0)
    print(f"  PSO 阶段训练MSE={pre_mse:.6f} -> Adam 精调后 {eval_mse(m, x_tr, y_tr):.6f}")
    score(m, "APSO-LSTM(论文12)", time.time() - t0)

    # ---------- 5. Unified-PSO-LSTM：架构段+权重段统一编码（本研究创新点） ----------
    t0 = time.time()
    best_vec, fit, hist_c, decode = unified_pso_search(train_s, seed=0, horizon=horizon)
    s, h = decode(best_vec)
    x_tr, y_tr, x_te, y_te, inv = prepare(close, s, horizon)
    m = LSTMRegressor(hidden=h)
    m.set_vector(best_vec[2:2 + m.get_vector().size])
    pre_mse = eval_mse(m, x_tr, y_tr)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, lr=0.005, seed=0)
    print(f"  Unified 选中 s{s},h{h}；PSO 阶段训练MSE={pre_mse:.6f} -> Adam 精调后 {eval_mse(m, x_tr, y_tr):.6f}")
    score(m, f"Unified-PSO(s{s},h{h})", time.time() - t0)

    # ---------- 结果表与图 ----------
    cols = ["model", "MSE", "RMSE", "MAPE%", "MAAPE%", "SMAPE%", "TheilU", "NRMSE", "%Similar", "耗时"]
    df = pd.DataFrame(rows, columns=cols)
    print("\n" + df.to_string(index=False))

    actual = inv(np.asarray(y_te))
    dates = close.index[-len(actual):]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, "k-", lw=1.8, label="Actual")
    colors = ["#90A4AE", "#5C8DBC", "#E8A33D", "#C0504D"]
    for (label, pred), c in zip(curves.items(), colors):
        ax.plot(dates, pred, lw=1.1, color=c, label=label)
    ax.set_title(f"{code} Test Set Prediction (+{horizon}-day ahead)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f"{code}_h{horizon}_prediction.png"), dpi=120)
    plt.close(fig)
    return df


if __name__ == "__main__":
    all_rows = []
    for code, name in STOCKS.items():
        for horizon in HORIZONS:
            df = run_stock(code, name, horizon)
            df.insert(0, "horizon", f"+{horizon}d")
            df.insert(0, "stock", f"{code} {name}")
            all_rows.append(df)
    result = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(RESULT_DIR, "metrics_summary.csv")
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n汇总结果已保存：{out}")
