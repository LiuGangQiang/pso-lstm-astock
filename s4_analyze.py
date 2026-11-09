# -*- coding: utf-8 -*-
"""M4 / S4：波动率分组统计检验（开题文档 V1.1 第五章 5.4，H4）。

问题：APSO-LSTM 相对 Persistence 的优势（比值 RMSE_APSO / RMSE_Persist）
是否随波动率层级（低/中/高）系统变化？（H4：层级间存在显著差异；
开题预判"高波动层级 learning 模型相对占优"——S1 中 +1d 弱成立、+20d 三层重合。）

数据：
- price 三级特征共用 s1v14 的 APSO-LSTM 与 Persistence 结果
  （output/s1v14_parts/*.csv，含 h{1,3,5,10,20,40}，取 {1,5,20} 与 S3 对齐）；
- tech/sent 的 Persistence 与 price 相同（Persistence 只用收盘价），
  故分特征的三层比较直接用 s3_parts + s1v14 Persistence。

统计设计：
- 单元 = （标的, horizon, 特征集）；每个单元先对 5 折 × 5 种子共 25 次运行
  的"同折同种子"比值取均值（比值层面聚合，保持配对）；
- 层级间比较：Kruskal-Wallis H 检验（3 层级 × 每层级 4 标的，非正态小样本）；
  事后两两 Mann-Whitney U（Bonferroni 校正 ×3）；
- 层级内"是否优于 Persistence"：对 25 个配对比值做单样本 Wilcoxon（中位 vs 1）；
- 效应量：层级中位比值、比值 < 1 的比例；
- A50 子样本稳健性：12 只中属于中证 A50 的标的（名单见 A50 常量，
  以 2024 年 12 月调样后名单为准）单独重跑上述检验。

输出：output/s4_group_test.csv（检验结果汇总表）。
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

TIERS = ["低波动", "中波动", "高波动"]
HORIZONS = ["+1d", "+5d", "+20d"]
FEATURES = ["price", "tech", "sent"]

# 12 只标的中属于中证 A50 的（2024-12 调样后名单；非 A50 的不列出）
A50 = {"600900", "600036", "600519", "600887", "000333", "600276",
       "688981", "300750", "300308"}
# 注：三一重工(600031)、海康威视(002415)、阳光电源(300274)不在该期名单。


def load_price_rows():
    """s1v14 的 APSO-LSTM 与 Persistence（h{1,5,20}）。"""
    frames = []
    for p in glob.glob(os.path.join("output", "s1v14_parts", "*.csv")):
        df = pd.read_csv(p, dtype={"stock": str})
        frames.append(df[df.horizon.isin(HORIZONS)])
    return pd.concat(frames, ignore_index=True)


def load_feature_rows():
    """s3 的 APSO-LSTM-tech / -sent。"""
    frames = []
    for p in glob.glob(os.path.join("output", "s3_parts", "*.csv")):
        frames.append(pd.read_csv(p, dtype={"stock": str}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_ratios(apso: pd.DataFrame, persist: pd.DataFrame, feature: str):
    """按 (stock,horizon,fold,seed) 配对求 APSO/Persistence 的 RMSE 比值，
    再对每个 (stock,horizon) 的 25 个比值取均值，返回长表。

    Persistence 与种子无关（run_m1_baseline 只在 seed 0 记录一行），
    先按 apso 中出现的种子广播展开，保证 5 折 × 5 种子全部配对。"""
    seeds = sorted(apso.seed.unique())
    persist = persist[persist.model == "Persistence"]
    frames = []
    for s in seeds:
        p = persist.copy()
        p["seed"] = s
        frames.append(p)
    persist = pd.concat(frames, ignore_index=True)
    m = apso.merge(persist, on=["stock", "horizon", "fold", "seed"],
                   suffixes=("_a", "_p"))
    m = m[m.model_a.str.startswith("APSO-LSTM")]
    m["ratio"] = m.RMSE_a / m.RMSE_p
    g = m.groupby(["stock", "name_a", "tier_a", "horizon"]).ratio.mean().reset_index()
    g = g.rename(columns={"name_a": "name", "tier_a": "tier"})
    g["feature"] = feature
    return g


def kruskal_and_posthoc(groups: dict):
    """groups: {tier: [ratios]} → (KW H, p)，及两两 MWU（Bonferroni）。"""
    samples = [np.asarray(groups[t]) for t in TIERS if len(groups.get(t, []))]
    if len(samples) < 2:
        return None, []
    H, p_kw = stats.kruskal(*samples)
    post = []
    for i in range(len(TIERS)):
        for j in range(i + 1, len(TIERS)):
            a, b = np.asarray(groups.get(TIERS[i], [])), np.asarray(groups.get(TIERS[j], []))
            if len(a) and len(b):
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                post.append((f"{TIERS[i]} vs {TIERS[j]}", u, min(p * 3, 1.0)))
    return (H, p_kw), post


def wilcoxon_vs_1(ratios):
    r = np.asarray(ratios)
    r = r[r != 1]
    if len(r) < 6:
        return None, None
    try:
        w, p = stats.wilcoxon(r - 1)
    except ValueError:
        return None, None
    return w, p


def run_tests(cell: pd.DataFrame, label: str, rows: list):
    """cell: 某 (feature, horizon, 子样本) 的 (stock, tier, ratio) 表。"""
    groups = {t: cell[cell.tier == t].ratio.values for t in TIERS}
    (H, p_kw), post = kruskal_and_posthoc(groups)
    med = {t: (float(np.median(groups[t])) if len(groups[t]) else np.nan) for t in TIERS}
    rows.append(dict(analysis=label, test="Kruskal-Wallis(层级)", H=round(H, 4) if H else None,
                     p=p_kw, 低波动中位=med["低波动"], 中波动中位=med["中波动"],
                     高波动中位=med["高波动"], 详情="; ".join(
                         f"{t}: n={len(groups[t])}, 中位={med[t]:.3f}" if len(groups[t]) else f"{t}: 空"
                         for t in TIERS)))
    for t in TIERS:
        w, p = wilcoxon_vs_1(groups[t])
        rows.append(dict(analysis=label, test=f"Wilcoxon(中位=1, {t})", W=w, p=p,
                         低波动中位=med["低波动"], 中波动中位=med["中波动"], 高波动中位=med["高波动"]))
    for name, u, p in post:
        rows.append(dict(analysis=label, test=f"MWU Bonferroni({name})", U=u, p=p,
                         低波动中位=med["低波动"], 中波动中位=med["中波动"], 高波动中位=med["高波动"]))


def main():
    price = load_price_rows()
    feat = load_feature_rows()
    persist = price[price.model == "Persistence"]

    cells = []
    cells.append(paired_ratios(price[price.model == "APSO-LSTM"], persist, "price"))
    for fset in ("tech", "sent"):
        if len(feat):
            cells.append(paired_ratios(feat[feat.model == f"APSO-LSTM-{fset}"], persist, fset))
    allcells = pd.concat(cells, ignore_index=True)
    allcells.to_csv(os.path.join("output", "s4_cells.csv"), index=False,
                    encoding="utf-8-sig")

    rows = []
    for feature in FEATURES:
        for h in HORIZONS:
            cell = allcells[(allcells.feature == feature) & (allcells.horizon == h)]
            if not len(cell):
                continue
            run_tests(cell, f"{feature} {h} 全样本", rows)
            a50cell = cell[cell.stock.isin(A50)]
            if len(a50cell) >= 6:
                run_tests(a50cell, f"{feature} {h} A50子样本", rows)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join("output", "s4_group_test.csv"), index=False,
               encoding="utf-8-sig")
    print(f"单元数 {len(allcells)}，检验行 {len(out)}")
    print(out[out.test.str.startswith("Kruskal")].to_string(index=False))


if __name__ == "__main__":
    main()
