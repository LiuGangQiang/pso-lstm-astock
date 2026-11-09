# -*- coding: utf-8 -*-
"""E1 离散度比较的区组置换检验（编辑清单第 6 项 / 复核 c10）。

设计：同一 seed 在 λ=0 与最大测试 λ 之间共享初始化与 600 子采样流，
样本为配对而非独立。对四个面板（恒瑞/招行 × 全样本/400 天窗口）以
seed 为区组，在区组内交换 λ 标签，重算与正文一致的 Levene 统计量
（均值的绝对偏差，双组），得到校准置换 p 值。

零假设：区组内两 λ 标签可交换（强于"仅离散度相等"的保守零假设）。
统计量与正文 nominal Levene 一致，便于直接替换口径。
B=200,000 次 Monte Carlo 置换，rng seed 0。
"""
import numpy as np
import pandas as pd

B = 200_000
RNG = np.random.default_rng(0)

full = pd.read_csv('output/s5_e1_parts/s5_e1.csv')
win = pd.read_csv('output/s5_e1_parts/s5_e1_win400.csv')

HR, CMB = 600276, 600036


def levene_W(x, y):
    """双组 Brown–Forsythe 统计量（deviations from group medians），
    与 scipy levene(center='median') 一致（正文 nominal p 值口径）。"""
    zx = np.abs(x - np.median(x))
    zy = np.abs(y - np.median(y))
    n1, n2 = len(zx), len(zy)
    zbar1, zbar2 = zx.mean(), zy.mean()
    zbar = (zx.sum() + zy.sum()) / (n1 + n2)
    ssb = n1 * (zbar1 - zbar) ** 2 + n2 * (zbar2 - zbar) ** 2
    ssw = ((zx - zbar1) ** 2).sum() + ((zy - zbar2) ** 2).sum()
    dfb, dfw = 1, n1 + n2 - 2
    return (ssb / dfb) / (ssw / dfw)


def block_permutation(a, b, B=B, rng=RNG):
    """区组内置换：每个 seed 对以 1/2 概率交换标签，重算 Levene W。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    W_obs = levene_W(a, b)
    n = len(a)
    cnt = 0
    for _ in range(B):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        if levene_W(pa, pb) >= W_obs:
            cnt += 1
    p = (cnt + 1) / (B + 1)
    return W_obs, p


def panel(df, stock, lams, label, win_size=None):
    d = df[df.stock == stock]
    if win_size is not None:
        d = d[d.win == win_size]
    d0 = d[d.lam == lams[0]].set_index('seed')['ratio']
    d1 = d[d.lam == lams[1]].set_index('seed')['ratio']
    common = d0.index.intersection(d1.index)
    a = d0.loc[common].values
    b = d1.loc[common].values
    W_obs, p_perm = block_permutation(a, b)
    # nominal（把配对样本当独立组）p 值，供对照
    from scipy import stats
    W_nom, p_nom = stats.levene(a, b, center='median')
    sd0, sd1 = a.std(ddof=1), b.std(ddof=1)
    mean_diff = b.mean() - a.mean()
    print(f'{label}: n={len(common)} seeds, λ {lams[0]} vs {lams[1]}')
    print(f'  SD {sd0:.4f} -> {sd1:.4f} (Δmean {mean_diff:+.4f})')
    print(f'  nominal Levene W={W_nom:.4f} p={p_nom:.4f} | '
          f'block-permutation p={p_perm:.4f} (B={B})')
    return dict(panel=label, n=len(common), lam_low=lams[0], lam_high=lams[1],
                sd_low=sd0, sd_high=sd1, mean_diff=mean_diff,
                W_nominal=W_nom, p_nominal=p_nom, p_blockperm=p_perm, B=B)


rows = [
    panel(full, HR, (0.0, 0.5), 'Hengrui full-length (λ=0 vs 0.5)'),
    panel(win, HR, (0.0, 0.1), 'Hengrui 400-day (λ=0 vs 0.1)', win_size=400),
    panel(full, CMB, (0.0, 0.5), 'CMB full-length (λ=0 vs 0.5)'),
    panel(win, CMB, (0.0, 0.5), 'CMB 400-day (λ=0 vs 0.5)', win_size=400),
]

out = pd.DataFrame(rows)
out.to_csv('output/s5_e1_blockperm.csv', index=False)
print()
print(out[['panel', 'p_nominal', 'p_blockperm']].to_string(index=False))
print('saved -> output/s5_e1_blockperm.csv')
