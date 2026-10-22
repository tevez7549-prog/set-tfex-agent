"""Download daily price history from Yahoo Finance into data/*.csv.

Run anywhere with internet (GitHub Actions now, the VPS later):
    python fetch_data.py
"""
import sys
import time

import pandas as pd
import yfinance as yf

import config as C


def download(symbol, adjust):
    for attempt in range(3):
        try:
            df = yf.download(symbol, start=C.DATA_START, auto_adjust=adjust,
                             progress=False, threads=False)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"
                return df
        except Exception as e:  # network hiccup
            print(f"  {symbol}: {e}")
        time.sleep(2 * (attempt + 1))
    return None


def first_working(candidates, name):
    for sym in candidates:
        df = download(sym, adjust=False)
        if df is not None and len(df) > 500:
            print(f"{name}: using {sym} ({len(df)} rows, from {df.index[0].date()})")
            return df
    print(f"{name}: none of {candidates} worked")
    return None


def main():
    C.DATA.mkdir(exist_ok=True)
    (C.DATA / "stocks").mkdir(exist_ok=True)
    report = []

    for cands, name in [(C.SET50_CANDIDATES, "SET50"), (C.SET_INDEX_CANDIDATES, "SET")]:
        df = first_working(cands, name)
        if df is not None:
            df.to_csv(C.DATA / f"{name}.csv")
        report.append((name, None if df is None else len(df)))

    gold = download(C.GOLD_USD, adjust=False)
    fx = download(C.USDTHB, adjust=False)
    if gold is not None and fx is not None:
        gold.to_csv(C.DATA / "GOLD_USD.csv")
        fx.to_csv(C.DATA / "USDTHB.csv")
        # Gold in baht: USD gold price x baht per USD (same-day closes)
        fxc = fx["Close"].reindex(gold.index).ffill()
        g = gold[["Open", "High", "Low", "Close"]].mul(fxc, axis=0)
        g["Volume"] = gold["Volume"]
        g = g.dropna()
        g.to_csv(C.DATA / "GOLD_THB.csv")
        report.append(("GOLD_THB", len(g)))
    else:
        report.append(("GOLD_THB", None))

    ok = 0
    for s in C.STOCKS:
        df = download(f"{s}.BK", adjust=True)  # adjusted for dividends/splits
        if df is not None and len(df) > 250:
            df.to_csv(C.DATA / "stocks" / f"{s}.csv")
            ok += 1
        else:
            print(f"  stock {s}: no data")
    report.append(("stocks", ok))

    print("\nSummary:")
    for name, n in report:
        print(f"  {name:10s} {n}")
    if any(n is None for _, n in report[:3]):
        sys.exit(1)


if __name__ == "__main__":
    main()
