# -*- coding: utf-8 -*-
"""从东方财富接口获取 A 股日线行情（无需 token），缓存为 CSV。

数据说明：
- klt=101 日线；fqt=2 后复权（消除分红送股造成的价格断点，建模口径与论文 02 使用的
  Yahoo Adj.Close 一致；如需预测真实交易价可改 fqt=0 不复权）。
- 字段：date, open, close, high, low, volume, amount
"""
import json
import os
import time
import urllib.request

BASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# 沪深 300 成分股中按波动率特征选取：低波动（白酒/银行）+ 高波动（创业板）
STOCKS = {
    "600519": "贵州茅台",   # 低波动，消费
    "600036": "招商银行",   # 低波动，银行
    "300750": "宁德时代",   # 高波动，创业板（涨跌幅 ±20%）
}


def market_prefix(code: str) -> str:
    """沪市(6xxxxx) secid 前缀 1.，深市(0/3xxxxx)前缀 0.。"""
    return "1" if code.startswith("6") else "0"


def fetch_daily(code: str, beg: str = "20150101", end: str = "20251231",
                cache_dir: str = "data") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{code}.csv")
    if os.path.exists(out):
        return out

    url = (f"{BASE}?secid={market_prefix(code)}.{code}"
           f"&fields1=f1,f2,f3,f4,f5,f6"
           f"&fields2=f51,f52,f53,f54,f55,f56,f57"
           f"&klt=101&fqt=2&beg={beg}&end={end}&lmt=100000")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read().decode("utf-8"))
    klines = payload["data"]["klines"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("date,open,close,high,low,volume,amount\n")
        for line in klines:
            # date,open,close,high,low,volume,amount
            f.write(line + "\n")
    print(f"[fetch] {code} {STOCKS.get(code, '')}: {len(klines)} 条日线 -> {out}")
    time.sleep(1)  # 温和限速
    return out


if __name__ == "__main__":
    for c in STOCKS:
        fetch_daily(c)
