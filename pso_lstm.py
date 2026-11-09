# -*- coding: utf-8 -*-
"""PSO × LSTM 核心库。

融合论文 02 与论文 12 两条技术路线：
- 路线 A（论文 02）：PSO 搜索 LSTM 超参数/架构/序列长度；
- 路线 B（论文 12）：自适应 PSO 演化 LSTM + 全连接层的初始权重，再由 Adam 精调。

自适应惯性权重（论文 12 式 12）：
    x_t^i = xmax - (xmax - xmin) * (V_gbest^t / V_pbest_i^t)
即按全局最优粒子与个人最优粒子的速度之比，为每个粒子动态分配惯性权重。
"""
import math

import numpy as np
import torch
import torch.nn as nn

# ---------------- LSTM 网络 ----------------

class LSTMRegressor(nn.Module):
    """单隐藏层 LSTM + 全连接输出层，对应论文 12 的 LSTM+FCL 结构。"""

    def __init__(self, n_features: int = 1, hidden: int = 32, n_layers: int = 1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

    # ---- 粒子（权重向量）<-> 网络参数 的双向映射（论文 12 的编码方案） ----
    def get_vector(self) -> np.ndarray:
        parts = []
        for p in self.lstm.parameters():
            parts.append(p.detach().numpy().ravel())
        for p in self.head.parameters():
            parts.append(p.detach().numpy().ravel())
        return np.concatenate(parts)

    def set_vector(self, vec: np.ndarray):
        vec = vec.astype(np.float32)
        idx = 0
        with torch.no_grad():
            for p in self.lstm.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(vec[idx:idx + n].reshape(p.shape)))
                idx += n
            for p in self.head.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(vec[idx:idx + n].reshape(p.shape)))
                idx += n


def init_weights_paper12(model: LSTMRegressor, n_features: int, seed: int = 0):
    """论文 12 的初始化：输入权重 Glorot，循环权重正交，偏置置零。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.lstm._all_weights:
            for name in layer:
                p = getattr(model.lstm, name)
                if name.startswith("weight_ih"):
                    bound = math.sqrt(6.0 / (n_features + 4 * model.hidden))
                    p.uniform_(-bound, bound, generator=g)
                elif name.startswith("weight_hh"):
                    hh = torch.empty(model.hidden, model.hidden)
                    nn.init.orthogonal_(hh, generator=g)
                    p.copy_(torch.cat([hh for _ in range(4)], dim=0))
                else:  # biases
                    p.zero_()
        for p in model.head.parameters():
            p.zero_() if p.dim() == 1 else p.uniform_(-0.05, 0.05, generator=g)


# ---------------- 数据序列构造 ----------------

def make_sequences(series: np.ndarray, seq_len: int, horizon: int = 1):
    """滑动窗口序列化。返回 x [N, seq_len, 1], y [N]。"""
    x, y = [], []
    for i in range(seq_len, len(series) - horizon + 1):
        x.append(series[i - seq_len:i])
        y.append(series[i + horizon - 1])
    return np.asarray(x)[..., None].astype(np.float32), np.asarray(y).astype(np.float32)


def make_sequences_multi(feat: np.ndarray, target: np.ndarray, seq_len: int,
                         horizon: int = 1):
    """多变量滑动窗口序列化（S3 特征消融）。
    feat [T, F] 多特征输入（已按训练折缩放），target [T] 缩放后的 close。
    返回 x [N, seq_len, F], y [N]（target 在 t+horizon-1 处的值）。
    """
    x, y = [], []
    for i in range(seq_len, len(feat) - horizon + 1):
        x.append(feat[i - seq_len:i])
        y.append(target[i + horizon - 1])
    return np.asarray(x).astype(np.float32), np.asarray(y).astype(np.float32)


def train_model(model: LSTMRegressor, x, y, epochs: int, lr: float = 0.01,
                seed: int = 0, verbose: bool = False) -> float:
    """Adam 训练，返回训练集 MSE。对应论文 12 的第二阶段（局部搜索）。"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep + 1}/{epochs}  train MSE={loss.item():.6f}")
    return loss.item()


def eval_mse(model: LSTMRegressor, x, y) -> float:
    with torch.no_grad():
        pred = model(torch.from_numpy(x))
    return float(torch.mean((pred - torch.from_numpy(y)) ** 2).item())


# ---------------- PSO 基类 ----------------

class PSO:
    """标准 PSO + 论文 12 的自适应惯性权重选项。"""

    def __init__(self, dim, n_particles=20, iters=30, w_max=0.9, w_min=0.4,
                 c1=1.5, c2=1.5, adaptive=False, seed=0, verbose=True):
        self.dim, self.iters = dim, iters
        self.n = n_particles
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.adaptive = adaptive
        self.verbose = verbose
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.pos = rng.uniform(-1, 1, (n_particles, dim))
        self.vel = np.zeros((n_particles, dim))
        self.pbest = self.pos.copy()
        self.pbest_fit = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_fit = np.inf
        self.history = []

    def _inertia(self) -> np.ndarray:
        if not self.adaptive:
            return np.full(self.n, self.w_max)
        # 论文 12 式(12)：按 |V_gbest| / |V_pbest_i| 比值自适应（逐粒子）
        v_g = np.linalg.norm(self.vel[self.pbest_fit.argmin()]) if self.gbest is not None else 1.0
        ratio = v_g / (np.linalg.norm(self.vel, axis=1) + 1e-12)
        ratio = np.clip(ratio, 0.0, 1.0)
        return self.w_max - (self.w_max - self.w_min) * ratio

    def run(self, fitness_fn):
        for t in range(self.iters):
            w = self._inertia()
            r1 = self.rng.random((self.n, self.dim))
            r2 = self.rng.random((self.n, self.dim))
            self.vel = (w[:, None] * self.vel
                        + self.c1 * r1 * (self.pbest - self.pos)
                        + self.c2 * r2 * (self.gbest[None, :] - self.pos
                                          if self.gbest is not None else 0))
            self.pos = np.clip(self.pos + self.vel, -5, 5)
            for i in range(self.n):
                f = fitness_fn(self.pos[i])
                if f < self.pbest_fit[i]:
                    self.pbest_fit[i] = f
                    self.pbest[i] = self.pos[i]
                    if f < self.gbest_fit:
                        self.gbest_fit = f
                        self.gbest = self.pos[i].copy()
            self.history.append(self.gbest_fit)
            if self.verbose:
                print(f"  PSO iter {t + 1}/{self.iters}  gbest MSE={self.gbest_fit:.6f}")
        return self.gbest, self.gbest_fit


# ---------------- 路线 A：PSO 调超参数/序列长度（论文 02 的精简版） ----------------

# 搜索维度：[seq_len, hidden, epochs]
BOUNDS = np.array([[10, 60], [8, 64], [10, 60]], dtype=float)  # (low, high) per dim
ALLOWED_HIDDEN = np.array([8, 16, 32, 64])


def decode_struct(v_seq: float, v_hid: float, seq_bounds, hidden_list):
    """结构坐标归一化映射（V3.2 复核 c0 修复）。

    PSO 位置被钳位在 [-5, 5]，结构坐标必须先归一化再解码，否则整个可达
    区间只能解码出 seq_bounds 下界与 hidden_list 最小值（归档 V2 及之前
    的 S2/E2 因此恒为 s10,h8，属实现 artifact）。
      seq:  [-5,5] 线性映射到 [seq_lo, seq_hi]（整数格点）
      hid:  [-5,5] 均分为 len(hidden_list) 段，取段内索引
    """
    lo, hi = seq_bounds
    seq_len = int(np.clip(lo + round((float(v_seq) + 5.0) / 10.0 * (hi - lo)),
                          lo, hi))
    k = int(np.clip(np.floor((float(v_hid) + 5.0) / 10.0 * len(hidden_list)),
                    0, len(hidden_list) - 1))
    return seq_len, int(hidden_list[k])


def decode_hyper(vec) -> tuple:
    seq_len, hidden = decode_struct(vec[0], vec[1], tuple(BOUNDS[0]), ALLOWED_HIDDEN)
    epochs = int(np.clip(round(vec[2]), *BOUNDS[2]))
    return seq_len, hidden, epochs


def pso_hyper_search(train_close, n_particles=12, iters=15, seed=0, horizon=1) -> tuple:
    """每个粒子编码 (seq_len, hidden, epochs)，适应度 = 该配置 LSTM 的训练 MSE。"""
    pso = PSO(dim=3, n_particles=n_particles, iters=iters, adaptive=False, seed=seed)

    def fitness(vec):
        seq_len, hidden, epochs = decode_hyper(vec)
        x, y = make_sequences(train_close, seq_len, horizon)
        model = LSTMRegressor(hidden=hidden)
        mse = train_model(model, x, y, epochs=max(5, epochs // 4), seed=seed)  # 搜索期用较少 epoch 加速
        return mse

    best, fit = pso.run(fitness)
    return decode_hyper(best), fit, pso.history


# ---------------- 路线 B：自适应 PSO 演化初始权重（论文 12） ----------------

def apso_weight_search(train_close, seq_len=20, hidden=16, n_particles=15,
                       iters=20, seed=0, horizon=1, verbose=True,
                       val_close=None, fitness_mode="train",
                       weight_penalty=0.0, fitness_subsample=0,
                       n_features=1) -> tuple:
    """粒子 = LSTM+FCL 的全部权重偏置；自适应惯性权重；适应度 = 前向 MSE。

    val_close（V1.3）：若给定，适应度在训练段内部的时序验证切片上前向计算，
    而非全训练段——防止大预算把训练 MSE 压进测试期爆炸的极端权重区（适应度过拟合）。
    fitness_mode："train"（默认）/ "val"（需 val_close）/ "composite"（0.5 训练 + 0.5 验证）。
    weight_penalty：权重范数惩罚系数 λ，适应度 += λ · mean(vec²)，封死极端权重区。
    fitness_subsample：PSO 阶段适应度的序列子采样数（0 = 全量；>0 时按种子固定抽取，
    属多保真度评估的搜索阶段加速，不改变最终 Adam 精调数据）。
    n_features：输入特征数（S3 特征消融；train_close 为多变量时传入 feat [T, F]）。
    """
    if n_features > 1:
        x, y = make_sequences_multi(train_close, train_close[:, 0], seq_len, horizon)
    else:
        x, y = make_sequences(train_close, seq_len, horizon)
    use_val = val_close is not None and len(val_close) > seq_len + horizon
    if use_val:
        fx, fy = make_sequences(val_close, seq_len, horizon)
    else:
        fx, fy = x, y
    if fitness_subsample and len(fx) > fitness_subsample:
        idx = np.random.RandomState(seed).choice(len(fx), fitness_subsample, replace=False)
        fx, fy = fx[idx], fy[idx]
    base = LSTMRegressor(hidden=hidden, n_features=n_features)
    init_weights_paper12(base, n_features=n_features, seed=seed)
    dim = base.get_vector().shape[0]
    pso = PSO(dim=dim, n_particles=n_particles, iters=iters,
              adaptive=True, c1=1.5, c2=1.5, seed=seed, verbose=verbose)

    def fitness(vec):
        model = LSTMRegressor(hidden=hidden, n_features=n_features)
        model.set_vector(vec)
        if fitness_mode == "val" and use_val:
            mse = eval_mse(model, fx, fy)
        elif fitness_mode == "composite" and use_val:
            mse = 0.5 * eval_mse(model, x, y) + 0.5 * eval_mse(model, fx, fy)
        else:
            mse = eval_mse(model, fx, fy)
        if weight_penalty > 0:
            mse = mse + weight_penalty * float(np.mean(vec ** 2))
        return mse

    best, fit = pso.run(fitness)
    return best, fit, pso.history


# ---------------- 路线 C：统一编码（架构段 + 权重段同粒子，本研究的创新点） ----------------

UNI_HIDDEN = np.array([8, 16, 32, 64])
UNI_SEQ_BOUNDS = (10, 60)


def _weight_dim(hidden: int, n_features: int = 1) -> int:
    """LSTM+FCL 的权重向量维度：4H(N+H) + 8H（两个偏置）+ H + 1。"""
    return 4 * hidden * (n_features + hidden) + 8 * hidden + hidden + 1


def unified_pso_search(train_close, n_particles=15, iters=20, seed=0, horizon=1,
                       weight_penalty=0.0, fitness_subsample=0,
                       struct_init=(-1.0, 1.0), init_info=None):
    """粒子 = [seq_len, hidden, w_1 ... w_Dmax]。

    架构段决定当前解的网络结构与数据序列长度；权重段按最大架构（hidden=64）填充，
    评估时只取前 D(hidden) 维注入网络（变长掩码编码）。
    适应度 = 该（架构, 权重）组合的前向训练 MSE。
    weight_penalty / fitness_subsample 与 apso_weight_search 同义（V1.4 公平配置）。
    struct_init：结构段初始采样区间（默认 (-1,1)，与全部历史归档一致；
      对照实验传 (-5,5) 即广覆盖初始化，用于排除初始化先验，见 s5_e3_controls.py）。
    init_info：传入 dict 时记录初始群结构解码中位数（排除初始化先验的对照证据）。
    """
    d_max = _weight_dim(int(UNI_HIDDEN.max()))
    dim = 2 + d_max
    pso = PSO(dim=dim, n_particles=n_particles, iters=iters, adaptive=True, seed=seed)

    # 结构段重初始化（仅在对照实验显式传入非默认区间时生效；
    # 默认路径逐位保持与历史归档一致）。用独立于 PSO 内部 rng 的流，
    # 避免扰动默认路径的随机序列。
    if tuple(struct_init) != (-1.0, 1.0):
        lo_s, hi_s = struct_init
        s_rng = np.random.default_rng(seed + 777)
        pso.pos[:, 0] = s_rng.uniform(lo_s, hi_s, n_particles)
        pso.pos[:, 1] = s_rng.uniform(lo_s, hi_s, n_particles)
    if init_info is not None:
        dec = [decode_struct(v[0], v[1], UNI_SEQ_BOUNDS, UNI_HIDDEN) for v in pso.pos]
        init_info["init_seq_med"] = float(np.median([d[0] for d in dec]))
        init_info["init_hid_med"] = float(np.median([d[1] for d in dec]))

    # 权重段不用均匀随机初始化，而是按论文 12 的 Glorot/正交方案初始化每个粒子
    # （以最大架构 h64 生成，评估时按实际架构截取前 D(hidden) 维），
    # 避免大网络因随机起点太差而在联合搜索中被系统性淘汰。
    for i in range(n_particles):
        probe = LSTMRegressor(hidden=int(UNI_HIDDEN.max()))
        init_weights_paper12(probe, n_features=1, seed=seed * 1000 + i)
        pso.pos[i, 2:] = probe.get_vector()
        pso.pbest[i] = pso.pos[i].copy()

    def decode(vec):
        seq_len, hidden = decode_struct(vec[0], vec[1], UNI_SEQ_BOUNDS, UNI_HIDDEN)
        return seq_len, hidden

    fx_all_cache = {}

    def fitness(vec):
        seq_len, hidden = decode(vec)
        if seq_len not in fx_all_cache:
            x, y = make_sequences(train_close, seq_len, horizon)
            if fitness_subsample and len(x) > fitness_subsample:
                idx = np.random.RandomState(seed).choice(len(x), fitness_subsample, replace=False)
                x, y = x[idx], y[idx]
            fx_all_cache[seq_len] = (x, y)
        x, y = fx_all_cache[seq_len]
        model = LSTMRegressor(hidden=hidden)
        model.set_vector(vec[2:2 + model.get_vector().size])
        mse = eval_mse(model, x, y)
        if weight_penalty > 0:
            mse = mse + weight_penalty * float(np.mean(vec[2:] ** 2))
        return mse

    best, fit = pso.run(fitness)
    return best, fit, pso.history, decode


# ---------------- 路线 C2：分阶段联合编码（V1.2 起 M3 的创新点实现） ----------------

STAGE_HIDDEN = np.array([8, 16, 32, 64])
STAGE_SEQ_BOUNDS = (10, 60)


def staged_pso_search(train_close, outer_particles=12, outer_iters=10, seed=0, horizon=1,
                      inner_particles=10, inner_iters=8, weight_penalty=0.1,
                      fitness_subsample=600, verbose=True):
    """分阶段联合编码：外层 PSO 只搜离散结构 (seq_len, hidden)，
    内层以小型 APSO（少量粒子×代数 + 惩罚 + 子采样）为该结构优化权重，
    外层适应度 = 内层最优权重在全训练段上的前向 MSE（含惩罚项）。

    相比朴素统一编码（17k 维权重段与 2 维结构段同粒子），搜索空间从
    D_max+2 维降到 2 维，结构维度信号不再被权重噪声淹没（对应 RQ2/H3）。
    """
    outer = PSO(dim=2, n_particles=outer_particles, iters=outer_iters,
                adaptive=False, seed=seed, verbose=verbose)

    def decode(vec):
        seq_len, hidden = decode_struct(vec[0], vec[1], STAGE_SEQ_BOUNDS, STAGE_HIDDEN)
        return seq_len, hidden

    def fitness(vec):
        seq_len, hidden = decode(vec)
        best_vec, _, _ = apso_weight_search(
            train_close, seq_len=seq_len, hidden=hidden,
            n_particles=inner_particles, iters=inner_iters, seed=seed,
            horizon=horizon, verbose=False, weight_penalty=weight_penalty,
            fitness_subsample=fitness_subsample)
        model = LSTMRegressor(hidden=hidden)
        model.set_vector(best_vec)
        x, y = make_sequences(train_close, seq_len, horizon)
        mse = eval_mse(model, x, y)
        if weight_penalty > 0:
            mse = mse + weight_penalty * float(np.mean(best_vec ** 2))
        return mse

    best, fit = outer.run(fitness)
    return best, fit, outer.history, decode


# ---------------- 评价指标（论文 12 五指标 + 论文 02 两项） ----------------

def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    n = len(a)
    err = a - p
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    mape = float(np.mean(np.abs(err / a)) * 100)
    maape = float(np.mean(np.arctan(np.abs(err / a))) * 100)
    smape = float(np.mean(2 * np.abs(err) / (np.abs(a) + np.abs(p))) * 100)
    denom = math.sqrt(np.mean(a ** 2)) + math.sqrt(np.mean(p ** 2))
    theil_u = float(rmse / denom) if denom > 0 else float("nan")
    nrmse = rmse / (a.max() - a.min())
    pct_similar = float(np.mean(np.minimum(a, p) / np.maximum(a, p)) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAPE%": mape, "MAAPE%": maape,
            "SMAPE%": smape, "TheilU": theil_u, "NRMSE": nrmse, "%Similar": pct_similar}
