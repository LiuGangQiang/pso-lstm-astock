# -*- coding: utf-8 -*-
"""M1 Target Pool Screening: Retrieve the daily bars of CSI 300 candidate stocks, calculate the annualized volatility, and use it for stratified sampling.

Selection criteria for the candidate pool: All are CSI 300 constituent stocks (representative large-cap blue-chip stocks from various industries),
covering the full spectrum from low volatility (financials/utilities/consumer) to high volatility (ChiNext/technology)."""
import json
import os
import time
import urllib.request

BASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pool")

# Candidate Pool with Industry Dispersion (Code: (Name, Industry))
CANDIDATES = {
    # Financials (expected low volatility)
    "600036": ('China Merchants Bank', 'Banking'), "601398": ('ICBC', 'Banking'),
    "601288": ('Agricultural Bank of China', 'Banking'), "000001": ('Ping An Bank', 'Banking'),
    "601318": ('Ping An Insurance', 'Insurance'), "600030": ('CITIC Securities', 'Securities'),
    # Consumer staples (low-to-medium volatility)
    "600519": ('Kweichow Moutai', 'Baijiu'), "000858": ('Wuliangye', 'Baijiu'),
    "600887": ('Yili Group', 'Food and Beverage'), "000333": ('Midea Group', 'Home Appliances'),
    "600690": ('Haier Smart Home', 'Home Appliances'), "000651": ('Gree Electric', 'Home Appliances'),
    # Utilities/Energy (Expected Low Volatility)
    "600900": ('China Yangtze Power', 'Utilities'), "601088": ('China Shenhua Energy', 'Coal'),
    "601857": ('PetroChina', 'Petroleum and Petrochemicals'), "600028": ('Sinopec', 'Petroleum and Petrochemicals'),
    # Pharmaceuticals (medium volatility)
    "600276": ('Hengrui Pharmaceuticals', 'Pharmaceuticals'), "300760": ('Mindray Medical', 'Medical Devices'),
    "603259": ('WuXi AppTec', 'Pharmaceutical Outsourcing'),
    # Manufacturing (medium volatility)
    "600031": ('SANY Heavy Industry', 'Construction Machinery'), "000425": ('XCMG', 'Construction Machinery'),
    # Electronics/Technology (Medium-High Volatility)
    "002415": ('Hikvision', 'Electronic Security'), "000063": ('ZTE', 'Telecommunications Equipment'),
    "603501": ('Will Semiconductor', 'Semiconductors'), "688981": ('SMIC', 'Semiconductors'),
    "300308": ('Zhongji Innolight', 'Optical Modules'),
    # New Energy (Expected High Volatility)
    "300750": ('CATL', 'Power Batteries'), "002594": ('BYD', 'New Energy Vehicles'),
    "601012": ('LONGi Green Energy', 'Photovoltaics'), "300274": ('Sungrow Power Supply', 'Photovoltaic Inverters'),
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
    print(f"[fetch] {code} {CANDIDATES[code][0]}: {len(klines)}rows")
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
    print('Code      Name        Sector          Annualized Volatility % (2015–2025)  Trading Days')
    for vol, code, name, industry, n in rows:
        print(f"{code}  {name:<6}  {industry:<8}  {vol:8.1f}        {n}")
