# -*- coding: utf-8 -*-
"""B1 推断口径精确重算（V3.5，复核批注 c1/c2/c15）。

推断统计量 = 每股 5 折比值的中位数（4.2 统一口径）；每股均值仅作 Table 3 描述。
输出 output/b1_stock_stats_v35.csv（wide 表：feature × stat × horizon）。
"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

pp_src = pd.concat([pd.read_csv(f, dtype={"stock": str})
                    for f in glob.glob("output/s1v14_parts/*.csv")], ignore_index=True)
pp = pp_src[pp_src.model == "Persistence"][["stock", "horizon", "fold", "RMSE"]].rename(
    columns={"RMSE": "pers"})

H = ["+1d", "+5d", "+20d"]


def stock_stats(df, prefix):
    if prefix == "APSO-LSTM$":  # exact match（避免 startswith 误配 tech/sent）
        d = df[df.model == "APSO-LSTM"]
    else:
        d = df[df.model.str.startswith(prefix)]
    m = d.merge(pp, on=["stock", "horizon", "fold"], how="inner")
    assert len(m) == len(d), f"{prefix}: persistence 配对缺失"
    m["ratio"] = m.RMSE / m.pers
    g = m.groupby(["stock", "horizon"]).ratio.agg(["median", "mean"]).reset_index()
    return g.pivot_table(index="stock", columns="horizon", values="median")[H], \
        g.pivot_table(index="stock", columns="horizon", values="mean")[H]


ar = pd.concat([pd.read_csv(f, dtype={"stock": str})
                for f in glob.glob("output/arima_parts/*.csv")], ignore_index=True)
s1 = pp_src[pp_src.horizon.isin(H)]
s3 = pd.concat([pd.read_csv(f, dtype={"stock": str})
                for f in glob.glob("output/s3_parts/*.csv")], ignore_index=True)
ad = pd.concat([pd.read_csv(f, dtype={"stock": str})
                for f in glob.glob("output/adam_parts/*.csv")], ignore_index=True)

med, mean = {}, {}
for feat, df_, prefix in [("arima", ar, "ARIMA"), ("price", s1, "APSO-LSTM$"),
                          ("tech", s3, "APSO-LSTM-tech"), ("sent", s3, "APSO-LSTM-sent")]:
    med[feat], mean[feat] = stock_stats(df_, prefix)

# Adam-only：per-stock 中位数 over 25 fold×seed
adm = ad.merge(pp, on=["stock", "horizon", "fold"], how="inner")
assert len(adm) == len(ad)
adm["ratio"] = adm.RMSE / adm.pers
adam_med = adm.groupby(["stock", "horizon"]).ratio.median().unstack()[H]

print("== ARIMA vs Persistence（每股中位数 对 1，双侧精确 Wilcoxon）==")
for h in H:
    d = med["arima"][h] - 1.0
    p = stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue
    print(f"  {h}: p={p:.6f}  >1: {(med['arima'][h] > 1).sum()}/12  "
          f"stockmedian={med['arima'][h].median():.4f} stockmean={med['arima'][h].mean():.4f}")

print()
print("== ARIMA vs APSO-LSTM（配对，每股中位数）==")
for h in H:
    d = med["arima"][h] - med["price"][h]
    p = stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue
    rel = med["arima"][h] / med["price"][h]
    print(f"  {h}: ARIMA 更好 {(d < 0).sum()}/12  p={p:.6f}  "
          f"相对比值 median={rel.median():.3f} mean={rel.mean():.3f}")

print()
print("== Adam-only vs APSO-LSTM（配对，每股中位数）==")
for h in H:
    d = adam_med[h] - med["price"][h]
    p = stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue
    rel = adam_med[h] / med["price"][h]
    print(f"  {h}: Adam 更好 {(d < 0).sum()}/12  p={p:.6f}  "
          f"相对比值 median={rel.median():.3f} mean={rel.mean():.3f}")

print()
print("== 描述口径（Table 3 用，每股均值→跨股中位数）==")
for feat in ["arima", "price", "tech", "sent"]:
    print(f"  {feat}: " + " / ".join(f"{mean[feat][h].median():.4f}" for h in H))

rows = []
for feat in med:
    for stat, tbl in (("median", med), ("mean", mean)):
        t = tbl[feat].copy()
        t.insert(0, "stat", stat)
        t.insert(0, "feature", feat)
        rows.append(t.reset_index())
pd.concat(rows).to_csv("output/b1_stock_stats_v35.csv", index=False, encoding="utf-8-sig")
adam_med.reset_index().melt(id_vars="stock", var_name="horizon", value_name="median") \
    .assign(feature="adam-only", stat="median") \
    .to_csv("output/b1_adam_median_v35.csv", index=False, encoding="utf-8-sig")
print("saved output/b1_stock_stats_v35.csv, output/b1_adam_median_v35.csv")
