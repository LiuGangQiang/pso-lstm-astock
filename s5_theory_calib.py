# -*- coding: utf-8 -*-
"""命题 1 数值标定：LSTM 预测映射梯度的可计算证书与 λ* 预测。

在恒瑞 600276 全长训练段、h=+5d、hidden=16（D=1233）上：
  1. Glorot 初始化点与 λ=0.01 收敛 gbest 两点处，用 autograd 计算
     - B = max|f_θ(x_i) - y_i|（子采样 600 口径，与 E1 搜索一致）
     - L_f = mean_i ||∇_θ f_θ(x_i)||（每样本预测对参数的梯度范数均值）
     - ||∇MSE|| 精确值
     - ρ = ||θ||
  2. 充分条件证书 λ*_cert = D·B·L_f/ρ（惩罚梯度 ≥ MSE 梯度充分条件）
     精确平衡点 λ_bal = D·||∇MSE||/(2ρ)
  3. 与 E1 实测相变（λ ≤ 0.01 惩罚份额跳至 77%）对比。
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch

from data_fetch import fetch_daily
from pso_lstm import (LSTMRegressor, apso_weight_search, init_weights_paper12,
                      make_sequences)

SEQ_LEN, HIDDEN, HORIZON = 20, 16, 5
SUBSAMPLE = 600

df = pd.read_csv(fetch_daily("600276"), parse_dates=["date"]).sort_values("date")
close = df["close"].astype(float)
close = close[close > 0].values.astype(np.float64)
n = len(close)
split = int(n * 0.8)
tr_raw = close[:split]
lo, hi = tr_raw.min(), tr_raw.max()
tr_s = ((tr_raw - lo) / (hi - lo)).astype(np.float32)

x_tr, y_tr = make_sequences(tr_s, SEQ_LEN, HORIZON)
idx = np.random.RandomState(0).choice(len(x_tr), SUBSAMPLE, replace=False)
fx, fy = x_tr[idx], y_tr[idx]
fx_t = torch.from_numpy(fx)
fy_t = torch.from_numpy(fy)
D = LSTMRegressor(hidden=HIDDEN).get_vector().size
print(f"D = {D}, 子样本 = {len(fx)}")


def certificate(model, label):
    model.zero_grad()
    # 每样本梯度范数（用 loop，避免 jacobian 全量物化；抽 100 样本估 L_f）
    sub = 100
    grad_norms = []
    for i in range(sub):
        model.zero_grad()
        fi = model(fx_t[i:i + 1])
        gi = torch.autograd.grad(fi, list(model.parameters()),
                                 retain_graph=False, allow_unused=True)
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in gi if g is not None))
        grad_norms.append(gn)
    L_f = float(np.mean(grad_norms))
    L_f_max = float(np.max(grad_norms))

    model.zero_grad()
    pred = model(fx_t)
    mse = torch.mean((pred - fy_t) ** 2)
    gm = torch.autograd.grad(mse, list(model.parameters()), allow_unused=True)
    grad_mse = np.sqrt(sum(float((g ** 2).sum()) for g in gm if g is not None))
    with torch.no_grad():
        res = (pred - fy_t).abs().max().item()
    rho = float(np.sqrt(sum(float((p ** 2).sum()) for p in model.parameters())))

    lam_cert = D * res * L_f / rho
    lam_bal = D * grad_mse / (2 * rho)
    share_bal = (lam_bal * rho ** 2 / D) / (float(mse) + lam_bal * rho ** 2 / D)
    print(f"{label}: B={res:.4f} L_f(mean)={L_f:.2f} L_f(max)={L_f_max:.2f} "
          f"||gradMSE||={grad_mse:.3f} rho={rho:.2f} mse={float(mse):.5f}")
    print(f"   λ*_cert={lam_cert:.5f}  λ_bal={lam_bal:.5f}  "
          f"平衡点惩罚份额={share_bal:.3f}")


# 1. Glorot 初始化点
m0 = LSTMRegressor(hidden=HIDDEN)
init_weights_paper12(m0, n_features=1, seed=0)
certificate(m0, "Glorot 初始化点")

# 2. λ=0.01 收敛 gbest（与 E1 同配置）
best, fit, hist = apso_weight_search(tr_s, seq_len=SEQ_LEN, hidden=HIDDEN,
    n_particles=100, iters=50, seed=0, horizon=HORIZON, verbose=False,
    weight_penalty=0.01, fitness_subsample=600)
m1 = LSTMRegressor(hidden=HIDDEN)
m1.set_vector(best)
certificate(m1, "λ=0.01 gbest")
