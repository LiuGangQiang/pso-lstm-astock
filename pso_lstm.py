# -*- coding: utf-8 -*-
"""PSO × LSTM core library.

Combining the two technical approaches from Paper 02 and Paper 12:
- Route A (Paper 02): PSO searches for LSTM hyperparameters/architecture/sequence length;
- Route B (Paper 12): Adaptive PSO evolves the initial weights of the LSTM + fully connected layer, which are then fine-tuned by Adam.

Adaptive inertia weight (Paper 12, Equation 12):
    x_t^i = xmax - (xmax - xmin) * (V_gbest^t / V_pbest_i^t)
That is, the inertia weight is dynamically assigned to each particle based on the ratio of the velocity of the global best particle to that of the personal best particle."""
import math

import numpy as np
import torch
import torch.nn as nn

# ---------------- LSTM Network ----------------

class LSTMRegressor(nn.Module):
    """Single hidden layer LSTM + fully connected output layer, corresponding to the LSTM + fully connected layer structure in Paper 12."""

    def __init__(self, n_features: int = 1, hidden: int = 32, n_layers: int = 1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

    # ---- Bidirectional mapping between particles (weight vectors) <-> network parameters (encoding scheme from Paper 12) ----
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
    """Initialization from Paper 12: Glorot input weights, orthogonal recurrent weights, and zero biases."""
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


# ---------------- Data Sequence Construction ----------------

def make_sequences(series: np.ndarray, seq_len: int, horizon: int = 1):
    """Construct sliding-window sequences. Returns x [N, seq_len, 1] and y [N]."""
    x, y = [], []
    for i in range(seq_len, len(series) - horizon + 1):
        x.append(series[i - seq_len:i])
        y.append(series[i + horizon - 1])
    return np.asarray(x)[..., None].astype(np.float32), np.asarray(y).astype(np.float32)


def make_sequences_multi(feat: np.ndarray, target: np.ndarray, seq_len: int,
                         horizon: int = 1):
    """Multivariate sliding-window sequence construction (S3 feature ablation).
    feat [T, F]: multivariate inputs scaled within each training fold; target [T]: scaled close prices.
    Returns x [N, seq_len, F] and y [N], where the target is at t+horizon-1.
    """
    x, y = [], []
    for i in range(seq_len, len(feat) - horizon + 1):
        x.append(feat[i - seq_len:i])
        y.append(target[i + horizon - 1])
    return np.asarray(x).astype(np.float32), np.asarray(y).astype(np.float32)


def train_model(model: LSTMRegressor, x, y, epochs: int, lr: float = 0.01,
                seed: int = 0, verbose: bool = False) -> float:
    """Adam training, returns the training set MSE. Corresponds to the second stage (local search) of Paper 12."""
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


# ---------------- PSO Base Class ----------------

class PSO:
    """Standard PSO with the optional adaptive inertia weight from Paper 12."""

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
        # Paper 12 Equation (12): Adaptive according to the ratio |V_gbest| / |V_pbest_i| (particle-by-particle)
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


# ---------------- Route A: PSO Hyperparameter/Sequence Length Tuning (Condensed Version of Paper 02) ----------------

# Search Dimensions: [seq_len, hidden, epochs]
BOUNDS = np.array([[10, 60], [8, 64], [10, 60]], dtype=float)  # (low, high) per dim
ALLOWED_HIDDEN = np.array([8, 16, 32, 64])


def decode_struct(v_seq: float, v_hid: float, seq_bounds, hidden_list):
    """Normalized mapping of structural coordinates (V3.2 review fix c0).

    PSO positions are clipped to [-5, 5]. Structural coordinates must be normalized before decoding; otherwise,
    the entire reachable interval decodes only to the lower seq_bounds and smallest hidden_list value, as in the
    superseded S2/E2 archives (V2 and earlier always produced s10,h8 because of this implementation artifact).
      seq: linearly map [-5,5] to integer grid points in [seq_lo, seq_hi]
      hid: divide [-5,5] into len(hidden_list) equal intervals and select the corresponding index
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
    """Each particle encodes (seq_len, hidden, epochs); fitness is the training MSE of the corresponding LSTM."""
    pso = PSO(dim=3, n_particles=n_particles, iters=iters, adaptive=False, seed=seed)

    def fitness(vec):
        seq_len, hidden, epochs = decode_hyper(vec)
        x, y = make_sequences(train_close, seq_len, horizon)
        model = LSTMRegressor(hidden=hidden)
        mse = train_model(model, x, y, epochs=max(5, epochs // 4), seed=seed)  # Accelerate the search period with fewer epochs
        return mse

    best, fit = pso.run(fitness)
    return decode_hyper(best), fit, pso.history


# ---------------- Route B: Adaptive PSO Evolution Initial Weights (Paper 12) ----------------

def apso_weight_search(train_close, seq_len=20, hidden=16, n_particles=15,
                       iters=20, seed=0, horizon=1, verbose=True,
                       val_close=None, fitness_mode="train",
                       weight_penalty=0.0, fitness_subsample=0,
                       n_features=1) -> tuple:
    """A particle contains all LSTM+FCL weights and biases; inertia is adaptive; fitness is forward-pass MSE.

    val_close (V1.3): if supplied, evaluate fitness on a chronological validation slice within the training segment
    rather than the full training segment to discourage extreme weights that overfit the fitness objective.
    fitness_mode: 'train' (default), 'val' (requires val_close), or 'composite' (0.5 training + 0.5 validation).
    weight_penalty: coefficient λ for the weight-norm penalty; fitness += λ·mean(vec²) to penalize extreme weights.
    fitness_subsample: number of sequences sampled for PSO fitness (0 = all; >0 uses a seed-fixed sample),
    providing multi-fidelity search acceleration without changing the final Adam fine-tuning data.
    n_features: number of input features (for S3 ablation, pass feat [T, F] as train_close).
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


# ---------------- Route C: Unified Encoding (Architecture Segment + Weight Segment as Particles, the Innovative Aspect of This Study) ----------------

UNI_HIDDEN = np.array([8, 16, 32, 64])
UNI_SEQ_BOUNDS = (10, 60)


def _weight_dim(hidden: int, n_features: int = 1) -> int:
    """Weight vector dimensions for LSTM+FCL: 4H(N+H) + 8H (two biases) + H + 1."""
    return 4 * hidden * (n_features + hidden) + 8 * hidden + hidden + 1


def unified_pso_search(train_close, n_particles=15, iters=20, seed=0, horizon=1,
                       weight_penalty=0.0, fitness_subsample=0,
                       struct_init=(-1.0, 1.0), init_info=None):
    """A particle is [seq_len, hidden, w_1 ... w_Dmax].

    The architecture segment determines network structure and input sequence length. The weight segment is sized for
    the largest architecture (hidden=64); only the first D(hidden) dimensions are injected during evaluation
    (variable-length mask encoding). Fitness is forward training MSE for the (architecture, weights) combination.
    weight_penalty and fitness_subsample have the same meanings as in apso_weight_search (V1.4 fair configuration).
    struct_init: initial sampling interval for structural coordinates (default (-1,1), matching historical archives;
      controls use (-5,5) for broad initialization to rule out initialization priors; see s5_e3_controls.py).
    init_info: when a dict is supplied, record median decoded structures in the initial swarm.
    """
    d_max = _weight_dim(int(UNI_HIDDEN.max()))
    dim = 2 + d_max
    pso = PSO(dim=dim, n_particles=n_particles, iters=iters, adaptive=True, seed=seed)

    # Reinitialization of structure segments (only effective when non-default intervals are explicitly passed in the control experiment;
    # The default path remains bit-for-bit consistent with the historical archive). Using a stream independent of the RNG within the PSO,
    # Avoid perturbing the random sequence of the default path.
    if tuple(struct_init) != (-1.0, 1.0):
        lo_s, hi_s = struct_init
        s_rng = np.random.default_rng(seed + 777)
        pso.pos[:, 0] = s_rng.uniform(lo_s, hi_s, n_particles)
        pso.pos[:, 1] = s_rng.uniform(lo_s, hi_s, n_particles)
    if init_info is not None:
        dec = [decode_struct(v[0], v[1], UNI_SEQ_BOUNDS, UNI_HIDDEN) for v in pso.pos]
        init_info["init_seq_med"] = float(np.median([d[0] for d in dec]))
        init_info["init_hid_med"] = float(np.median([d[1] for d in dec]))

    # Instead of initializing the weight segments uniformly at random, each particle is initialized according to the Glorot/orthogonal scheme in Paper 12.
    # (Generated with the maximum architecture h64, and during evaluation, the first D(hidden) dimensions are taken according to the actual architecture),
    # Avoids large networks from being systematically eliminated in joint searches due to poor random starting points.
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


# ---------------- Route C2: Phased Joint Encoding (Implementation of M3 Innovations from V1.2 onwards) ----------------

STAGE_HIDDEN = np.array([8, 16, 32, 64])
STAGE_SEQ_BOUNDS = (10, 60)


def staged_pso_search(train_close, outer_particles=12, outer_iters=10, seed=0, horizon=1,
                      inner_particles=10, inner_iters=8, weight_penalty=0.1,
                      fitness_subsample=600, verbose=True):
    """Staged joint encoding: The outer-layer PSO only searches for discrete structures (seq_len, hidden),
    The inner layer uses a small APSO (small number of particles × generations + penalty + subsampling) to optimize the weights for this structure.
    Outer-layer fitness = the forward MSE of the inner-layer optimal weights over the entire training segment (including the penalty term).

    Compared to the naive unified encoding (17k-dimensional weight segment and 2-dimensional structure segment with the same particles), the search space is reduced from
    D_max+2 to 2 dimensions, and the structural dimensional signal is no longer overwhelmed by weight noise (corresponding to RQ2/H3)."""
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


# ---------------- Evaluation Metrics (Paper 12: Five Metrics + Paper 02: Two Metrics) ----------------

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
