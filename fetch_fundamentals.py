"""Quarterly fundamentals for every Thai stock (Finnomena public API).

Saves data/fundamentals.csv with one row per stock per quarter:
symbol, fiscal, quarter, close (price at quarter end), pe, pbv, roe,
bvps, de (debt/equity), dy (dividend yield %).

    python fetch_fundamentals.py
"""
import json
import time
import urllib.request

import pandas as pd

import config as C

BASE = "https://www.finnomena.com/market-info/api/public"
FIELDS = {"close": "close", "price_earning_ratio": "pe", "price_book_value": "pbv",
          "roe": "roe", "book_value_per_share": "bvps", "debt_to_equity": "de",
          "dividend_yield": "dy", "net_profit": "net_profit", "mkt_cap": "mkt_cap"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            err = e
            time.sleep(3 * (attempt + 1))
    raise err


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    C.DATA.mkdir(exist_ok=True)
    lst = get(f"{BASE}/stock/list?exchange=TH")["data"]
    print(f"stocks in list: {len(lst)}")
    rows, fails = [], 0
    for n, it in enumerate(lst):
        try:
            data = get(f"{BASE}/stock/summary/{it['security_id']}")["data"]
        except Exception as e:
            fails += 1
            continue
        quarters = data.get("quarter", data) if isinstance(data, dict) else data
        for q in quarters or []:
            row = dict(symbol=it["name"], fiscal=q.get("fiscal"), quarter=q.get("quarter"))
            for k, v in FIELDS.items():
                row[v] = num(q.get(k))
            rows.append(row)
        if n % 100 == 0:
            print(f"  {n}/{len(lst)}  rows {len(rows)}")
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df.to_csv(C.DATA / "fundamentals.csv", index=False)
    q9 = df[df.quarter == 9] if "quarter" in df else df
    print(f"saved {len(df)} rows, {df.symbol.nunique()} stocks, failures {fails}")
    print("fiscal years:", df.fiscal.min(), "..", df.fiscal.max())
    print("quarter values:", sorted(df.quarter.dropna().unique().tolist()))
    print("rows per fiscal year:\n", df.groupby("fiscal").size().to_string())


if __name__ == "__main__":
    main()
