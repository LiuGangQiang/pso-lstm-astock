# -*- coding: utf-8 -*-
"""重建 output/b1_cells.csv（V3.2 复核 c1 修复后）。

每行 = (stock, horizon, feature) 的 per-stock 比值（对 5 折取均值）：
  ratio(fold) = RMSE(feature, fold) / RMSE(Persistence, fold)
Persistence 取运行级记录值（与 LSTM 共用测试窗）。feature ∈ {arima, price,
tech, sent}；arima 为对齐版重跑（arima_parts/，2026-11-09）。
"""
import glob
import json

import pandas as pd

# ---- persistence 表（运行级记录值）----
s1 = pd.concat([pd.read_csv(f, dtype={"stock": str})
                for f in glob.glob("output/s1v14_parts/*.csv")], ignore_index=True)
pp = (s1[s1["model"] == "Persistence"][["stock", "horizon", "fold", "RMSE"]]
      .rename(columns={"RMSE": "pers"}))


def per_stock_ratios(df, model_filter, feature):
    d = df[df["model"].str.startswith(model_filter)]
    m = d.merge(pp, on=["stock", "horizon", "fold"], how="inner")
    assert len(m) == len(d), "persistence 配对缺失"
    m["ratio"] = m["RMSE"] / m["pers"]
    g = m.groupby(["stock", "name", "tier", "horizon"])["ratio"].mean().reset_index()
    g["feature"] = feature
    return g


frames = []
ar = pd.concat([pd.read_csv(f, dtype={"stock": str})
                for f in glob.glob("output/arima_parts/*.csv")], ignore_index=True)
frames.append(per_stock_ratios(ar, "ARIMA", "arima"))
frames.append(per_stock_ratios(s1[s1["horizon"].isin(["+1d", "+5d", "+20d"])],
                               "APSO-LSTM$", "price"))
s3 = pd.concat([pd.read_csv(f, dtype={"stock": str})
                for f in glob.glob("output/s3_parts/*.csv")], ignore_index=True)
frames.append(per_stock_ratios(s3, "APSO-LSTM-tech", "tech"))
frames.append(per_stock_ratios(s3, "APSO-LSTM-sent", "sent"))

out = pd.concat(frames, ignore_index=True)
out.to_csv("output/b1_cells.csv", index=False, encoding="utf-8-sig")

# ---- 对齐版 ARIMA 新比值概览 ----
piv = out[out.feature == "arima"].pivot_table(
    index="stock", columns="horizon", values="ratio")[["+1d", "+5d", "+20d"]]
summary = {"arima_stockmean": {h: round(float(piv[h].mean()), 4) for h in piv.columns},
           "arima_stockmedian": {h: round(float(piv[h].median()), 4) for h in piv.columns}}
print(json.dumps(summary, indent=1))
print(piv.round(3).to_string())
