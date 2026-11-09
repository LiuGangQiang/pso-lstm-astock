# -*- coding: utf-8 -*-
"""S5-E3：SNR 塌缩实验（"信号死亡"机制的判决性验证）。

设计：合成数据植入已知结构信号——真实记忆长度 L=40（衰减核）+ 平滑非线性，
信号强度旋钮 r = 结构化成分方差占比 ∈ {0.2,0.4,0.6,0.8,0.95}。
用生产统一编码搜索（15×20，λ=0.1，子采样 600）在合成序列上选 (seq_len, hidden)。

理论预测（信号死亡机制）：
  - λ=0.1 时，低 r 下结构选择 100% 退化为最小架构 (s10, h8)；
  - 随 r 上升，出现信号强度阈值 r*，超过后选择开始向大架构迁移；
  - λ=0 对照组应在更低的 r 就跟随信号——两组阈值的错位即 SNR 塌缩的直接度量。

用法：
  python s5_e3.py                     # 跑下一批（时间盒）
  python s5_e3.py --r 0.6 --lam 0.1 --seed 0   # 单次测试
"""
import argparse
import os

import numpy as np
import pandas as pd

from pso_lstm import unified_pso_search

RESULT_DIR = os.path.join("output", "s5_e3_parts")
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "s5_e3.csv")

RS = [0.2, 0.4, 0.6, 0.8, 0.95]
SEEDS = range(20)
N_TRAIN = 2000
TRUE_MEM = 40


def gen_series(r: float, seed: int, mem: int = None) -> np.ndarray:
    """植入结构信号的合成序列：L=40 衰减核记忆 + 平滑非线性 + 高斯噪声。

    z: AR(1) 潜过程（φ=0.98）；s_t = Σ w_k z_{t-k}（w 为归一化指数衰减核）；
    y = √r·g(s) + √(1-r)·η，g(u)=u+0.4u²-0.2u³（平滑非线性使 hidden 容量有区分度）。
    mem：植入记忆长度 L（默认 TRUE_MEM=40；对照实验传 20/60 改变植入结构）。
    """
    L = TRUE_MEM if mem is None else int(mem)
    rng = np.random.default_rng(10_000 + seed)
    n = N_TRAIN + 500
    z = np.zeros(n)
    e = rng.normal(0, 1, n)
    for t in range(1, n):
        z[t] = 0.98 * z[t - 1] + np.sqrt(1 - 0.98 ** 2) * e[t]
    k = np.arange(1, L + 1)
    w = np.exp(-(k - 1) / 15.0)
    w /= w.sum()
    s = np.zeros(n)
    for kk, wk in zip(k, w):
        s[kk:] += wk * z[:n - kk]
    s /= s.std() + 1e-12
    g = s + 0.4 * s ** 2 - 0.2 * s ** 3
    g = (g - g.mean()) / (g.std() + 1e-12)
    eta = rng.normal(0, 1, n)
    y = np.sqrt(r) * g + np.sqrt(1 - r) * eta
    return y


def run_one(r: float, lam: float, seed: int) -> dict:
    y = gen_series(r, seed)
    tr_raw = y[:N_TRAIN]
    lo, hi = tr_raw.min(), tr_raw.max()
    tr_s = ((tr_raw - lo) / (hi - lo)).astype(np.float32)

    best, fit, hist, decode = unified_pso_search(
        tr_s, n_particles=15, iters=20, seed=seed, horizon=1,
        weight_penalty=lam, fitness_subsample=600)
    seq_len, hidden = decode(best)
    return dict(r=r, lam=lam, seed=seed, chosen_seq=seq_len, chosen_hidden=hidden,
                gbest_fit=float(fit))


def all_combos():
    combos = [(r, 0.1, seed) for r in RS for seed in SEEDS]
    combos += [(r, 0.0, seed) for r in (0.6, 0.8, 0.95) for seed in SEEDS]  # λ=0 对照
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--budget", type=float, default=260)
    args = ap.parse_args()

    if args.r is not None and args.lam is not None and args.seed is not None:
        print(run_one(args.r, args.lam, args.seed))
        return

    done = set()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        done = set(zip(df.r, df.lam, df.seed))
        print(f"已完成 {len(done)} 组，断点续跑")
    combos = [c for c in all_combos() if c not in done]
    print(f"剩余 {len(combos)} 组")

    import time
    t_start, rows = time.time(), []
    for r, lam, seed in combos:
        try:
            row = run_one(r, lam, seed)
        except Exception as e:
            row = dict(r=r, lam=lam, seed=seed, error=str(e))
        rows.append(row)
        print(f"[{len(rows)}] r={r} lam={lam} seed={seed} -> "
              f"s{row.get('chosen_seq')},h{row.get('chosen_hidden')} fit={row.get('gbest_fit', float('nan')):.4f}")
        if time.time() - t_start > args.budget:
            break
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, mode="a", index=False, header=not os.path.exists(CSV_PATH))
        print(f"本批完成 {len(rows)} 组 -> {CSV_PATH}")


if __name__ == "__main__":
    main()
