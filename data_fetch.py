# -*- coding: utf-8 -*-
"""Obtain A-share daily quotes from the East Money interface (no token required) and cache them as a CSV file.

Data description:
- klt=101 daily line; fqt=2 post-adjustment (eliminates price discontinuities caused by dividends and bonus shares; the modeling approach is consistent with the Yahoo Adj.Close used in Paper 02;
  Yahoo Adj.Close; if you need to predict the actual trading price, you can change fqt=0 to no adjustment).
- Fields: date, open, close, high, low, volume, amount"""
import json
import os
import time
import urllib.request

BASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# Selection based on volatility characteristics among CSI 300 constituents: Low volatility (liquor/banking) + High volatility (ChiNext)
STOCKS = {
    "600519": 'Kweichow Moutai',   # Low volatility, consumption
    "600036": 'China Merchants Bank',   # Low volatility, banks
    "300750": 'CATL',   # High volatility, ChiNext (price change ±20%)
}


def market_prefix(code: str) -> str:
    """Use secid prefix 1 for Shanghai (6xxxxx) and 0 for Shenzhen (0/3xxxxx)."""
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
    print(f"[fetch] {code} {STOCKS.get(code, '')}: {len(klines)}daily bars ->{out}")
    time.sleep(1)  # Moderate speed limit
    return out


if __name__ == "__main__":
    for c in STOCKS:
        fetch_daily(c)
