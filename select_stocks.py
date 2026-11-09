# -*- coding: utf-8 -*-
"""M1 标的池筛选：拉取沪深 300 候选股日线，计算年化波动率，供分层选样。

候选池选取原则：均为沪深 300 成分股（各行业代表性大盘蓝筹），
覆盖低波动（金融/公用/消费）到高波动（创业板/科技）的完整谱系。
"""
import json
import os
import time
import urllib.request

BASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pool")

# 行业分散的候选池（代码: (名称, 行业)）
CANDIDATES = {
    # 金融（预期低波）
    "600036": ("招商银行", "银行"), "601398": ("工商银行", "银行"),
    "601288": ("农业银行", "银行"), "000001": ("平安银行", "银行"),
    "601318": ("中国平安", "保险"), "600030": ("中信证券", "券商"),
    # 消费（低-中波）
    "600519": ("贵州茅台", "白酒"), "000858": ("五粮液", "白酒"),
    "600887": ("伊利股份", "食品饮料"), "000333": ("美的集团", "家电"),
    "600690": ("海尔智家", "家电"), "000651": ("格力电器", "家电"),
    # 公用/能源（预期低波）
    "600900": ("长江电力", "公用事业"), "601088": ("中国神华", "煤炭"),
    "601857": ("中国石油", "石油石化"), "600028": ("中国石化", "石油石化"),
    # 医药（中波）
    "600276": ("恒瑞医药", "医药"), "300760": ("迈瑞医疗", "医疗器械"),
    "603259": ("药明康德", "医药外包"),
    # 制造（中波）
    "600031": ("三一重工", "工程机械"), "000425": ("徐工机械", "工程机械"),
    # 电子/科技（中-高波）
    "002415": ("海康威视", "电子安防"), "000063": ("中兴通讯", "通信设备"),
    "603501": ("韦尔股份", "半导体"), "688981": ("中芯国际", "半导体"),
    "300308": ("中际旭创", "光模块"),
    # 新能源（预期高波）
    "300750": ("宁德时代", "动力电池"), "002594": ("比亚迪", "新能源汽车"),
    "601012": ("隆基绿能", "光伏"), "300274": ("阳光电源", "光伏逆变"),
}


def market_prefix(code: str) -> str:
    return "1" if code.startswith("6") else "0"


def fetch_daily(code: str, beg: str = "20150101", end: str = "20251231") -> str:
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{code}.csv")
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
            f.write(line + "\n")
    print(f"[fetch] {code} {CANDIDATES[code][0]}: {len(klines)} 条")
    time.sleep(0.5)
    return out


def annualized_vol(code: str) -> tuple[float, int]:
    import math
    path = fetch_daily(code)
    closes, prev = [], None
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            c = float(parts[2])
            closes.append(c)
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    n = len(rets)
    if n < 200:
        return float("nan"), n
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252) * 100, n


if __name__ == "__main__":
    rows = []
    for code, (name, industry) in CANDIDATES.items():
        try:
            vol, n = annualized_vol(code)
            rows.append((vol, code, name, industry, n))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {code} {name}: {e}")
    rows.sort()
    print("\n代码      名称        行业          年化波动%(2015-2025)  交易日")
    for vol, code, name, industry, n in rows:
        print(f"{code}  {name:<6}  {industry:<8}  {vol:8.1f}        {n}")
