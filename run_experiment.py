# -*- coding: utf-8 -*-
"""End-to-end comparison of four models on three A-share stocks.

Models:
  1. LSTM: fixed-architecture baseline (hidden=32, seq_len=20, Adam)
  2. GS-LSTM: grid-searched hyperparameters (compact baseline from Paper 02)
  3. PSO-LSTM: PSO search over (seq_len, hidden, epochs) (Paper 02, Route A)
  4. APSO-LSTM: adaptive PSO evolution of initial weights + Adam fine-tuning (Paper 12, Route B)

The protocol follows Paper 02 to prevent leakage: split 80/20 first, then fit MinMax scaling on the training set.
Evaluation uses the original price scale. Metrics comprise five from Paper 12 plus NRMSE and %Similar from Paper 02.
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
HORIZONS = [1, 3, 5]  # Forecast period: t+1 / t+3 / t+5 days
GRID = [(s, h) for s in (10, 20, 40) for h in (16, 32, 64)]  # GS-LSTM Grid (9 Combinations)


def load_series(code: str) -> pd.Series:
    path = fetch_daily(code)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    close = df.set_index("date")["close"].astype(float)
    close = close[(close > 0)]  # Remove Dirty Data
    return close


def prepare(close: pd.Series, seq_len: int, horizon: int = 1):
    """Partition first, then scale (Paper 02, Preventing Data Leakage)."""
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
    print(f"Date range: {close.index[0].date()} ~ {close.index[-1].date()}, totaling {len(close)}Trading days")

    # Volatility legend: Annualized daily return volatility
    vol = close.pct_change().std() * np.sqrt(244) * 100
    print(f"Annualized volatility ≈ {vol:.1f}%")

    rows, curves = [], {}

    def score(model, label, elapsed):
        """Calculate indicators on the original price scale and record the forecast curve."""
        with torch.no_grad():
            pred_s = model(torch.from_numpy(x_te)).numpy()
        actual = inv(np.asarray(y_te))
        pred = inv(pred_s)
        rows.append((label, *metrics(actual, pred).values(), f"{elapsed:.0f}s"))
        curves[label] = pred

    # ---------- 0. Persistence Simple Baseline (Last Observed Value = Future Horizon Day Forecast) ----------
    x_tr, y_tr, x_te, y_te, inv = prepare(close, SEQ_BASE, horizon)
    pred = inv(np.asarray(x_te[:, -1, 0]))
    rows.append(("Persistence", *metrics(inv(np.asarray(y_te)), pred).values(), "0s"))
    curves["Persistence"] = pred

    # ---------- 1. Standard LSTM Baseline ----------
    t0 = time.time()
    x_tr, y_tr, x_te, y_te, inv = prepare(close, SEQ_BASE, horizon)
    model = LSTMRegressor(hidden=HIDDEN_BASE)
    train_model(model, x_tr, y_tr, epochs=EPOCHS_BASE, seed=0)
    score(model, 'LSTM (Baseline)', time.time() - t0)

    # ---------- 2. GS-LSTM Grid Search ----------
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

    # ---------- 3. PSO-LSTM: Search for Hyperparameters/Sequence Length (Paper 02, Route A) ----------
    t0 = time.time()
    split = int(len(close) * (1 - TEST_RATIO))
    tr = close.values[:split]
    train_s = ((tr - tr.min()) / np.ptp(tr)).astype(np.float32)
    (s, h, e), fit, hist_a = pso_hyper_search(train_s, seed=0, horizon=horizon)
    x_tr, y_tr, x_te, y_te, inv = prepare(close, s, horizon)
    m = LSTMRegressor(hidden=h)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, seed=0)  # Same training budget as baseline for a fair comparison
    score(m, f"PSO-LSTM(s{s},h{h})", time.time() - t0)

    # ---------- 4. APSO-LSTM: Evolutionary Initial Weights + Adam Fine-Tuning (Paper 12, Route B) ----------
    t0 = time.time()
    x_tr, y_tr, x_te, y_te, inv = prepare(close, SEQ_BASE, horizon)
    best_vec, fit, hist_b = apso_weight_search(train_s,
                                               seq_len=SEQ_BASE, hidden=16, seed=0, horizon=horizon)
    m = LSTMRegressor(hidden=16)
    m.set_vector(best_vec)
    pre_mse = eval_mse(m, x_tr, y_tr)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, lr=0.005, seed=0)
    print(f"PSO Phase Training MSE={pre_mse:.6f}-> After Adam's fine-tuning{eval_mse(m, x_tr, y_tr):.6f}")
    score(m, 'APSO-LSTM (Paper 12)', time.time() - t0)

    # ---------- 5. Unified-PSO-LSTM: Unified Encoding of Architecture Segment + Weight Segment (Innovation of This Study) ----------
    t0 = time.time()
    best_vec, fit, hist_c, decode = unified_pso_search(train_s, seed=0, horizon=horizon)
    s, h = decode(best_vec)
    x_tr, y_tr, x_te, y_te, inv = prepare(close, s, horizon)
    m = LSTMRegressor(hidden=h)
    m.set_vector(best_vec[2:2 + m.get_vector().size])
    pre_mse = eval_mse(m, x_tr, y_tr)
    train_model(m, x_tr, y_tr, epochs=EPOCHS_BASE, lr=0.005, seed=0)
    print(f"Unified Selected s{s},h{h}; PSO-stage training MSE={pre_mse:.6f}-> After Adam's fine-tuning{eval_mse(m, x_tr, y_tr):.6f}")
    score(m, f"Unified-PSO(s{s},h{h})", time.time() - t0)

    # ---------- Results Table and Figures ----------
    cols = ["model", "MSE", "RMSE", "MAPE%", "MAAPE%", "SMAPE%", "TheilU", "NRMSE", "%Similar", 'elapsed']
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
    print(f"Aggregated results have been saved:{out}")
