"""Download prices for (almost) every SET stock, for a point-in-time universe.

Why: round 1 used only today's big stocks, which flatters momentum
(survivorship bias). Here we take every stock listed on SET today, plus big
names that were delisted or merged away (when Yahoo still has them), and let
backtest_pit.py pick the 100 most-traded stocks AT EACH month-end.

Writes data/panel_close.parquet, data/panel_value.parquet, data/universe_report.txt
    python fetch_universe.py
"""
import json
import time
import urllib.request

import pandas as pd
import yfinance as yf

import config as C

LIST_URL = ("https://raw.githubusercontent.com/lumduan/thai-securities-data/"
            "main/thai_securities_market_set_en.json")

# Large stocks that left SET (delisted, merged, renamed) since 2008.
# Yahoo often drops these; whatever it still has is included.
DELISTED = """
DTAC INTUCH TMB GLOW BIGC MAKRO HEMRAJ TICON PTTAR SCIB ATC TUF STARK EARTH
BEC THAI TPIPP ESSO PS SAMART TTA LHBANK BAY CIMBT SCCC SPALI TCAP SIRI
BECL CK STEC ITD JAS TRUE GULF AAV NOK RS PLANB IFEC GL MACO WORK SUPER
DEMCO GUNKUL BLAND U EA BEAUTY JMART JMT KTC SINGER TKN TPIPL TASCO
""".split()


def get_list():
    with urllib.request.urlopen(LIST_URL, timeout=60) as r:
        items = json.load(r)
    return sorted({it["symbol"] for it in items if it.get("market") == "SET"})


def batch_download(symbols, size=40):
    closes, values, raws, divs, highs, lows = {}, {}, {}, {}, {}, {}
    for i in range(0, len(symbols), size):
        chunk = [f"{s}.BK" for s in symbols[i:i + size]]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=C.DATA_START, auto_adjust=False,
                                 actions=True, group_by="ticker", progress=False,
                                 threads=True)
                break
            except Exception as e:
                print("  retry", i, e)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for t in chunk:
            if t not in df.columns.get_level_values(0):
                continue
            d = df[t].dropna(subset=["Close"])
            if len(d) < 250:
                continue
            s = t[:-3]
            closes[s] = d["Adj Close"]          # total-return price
            values[s] = d["Close"] * d["Volume"]  # baht traded
            raws[s] = d["Close"]
            highs[s] = d["High"]
            lows[s] = d["Low"]
            if "Dividends" in d:
                divs[s] = d["Dividends"]
        print(f"  {min(i + size, len(symbols))}/{len(symbols)} done, kept {len(closes)}")
        time.sleep(1)
    return (pd.DataFrame(closes), pd.DataFrame(values), pd.DataFrame(raws),
            pd.DataFrame(divs), pd.DataFrame(highs), pd.DataFrame(lows))


def main():
    C.DATA.mkdir(exist_ok=True)
    listed = get_list()
    extras = [s for s in DELISTED if s not in listed]
    print(f"listed on SET today: {len(listed)}, extra delisted names to try: {len(extras)}")
    px, val, raw, div, high, low = batch_download(listed + extras)
    def naive(df):
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    px = naive(px).sort_index()
    for name, df in [("close", px), ("value", val), ("rawclose", raw), ("div", div),
                     ("high", high), ("low", low)]:
        naive(df).reindex(px.index).to_parquet(C.DATA / f"panel_{name}.parquet")
    found_extra = [s for s in extras if s in px.columns]
    report = [f"listed today: {len(listed)}",
              f"with price history: {px.shape[1]}",
              f"delisted names found on Yahoo: {len(found_extra)} of {len(extras)}: {' '.join(found_extra)}",
              f"range: {px.index[0].date()} .. {px.index[-1].date()}"]
    for y in (2008, 2012, 2016, 2020, 2024):
        n = px.loc[str(y)].notna().any().sum()
        report.append(f"stocks with data in {y}: {n}")
    text = "\n".join(report)
    print(text)
    (C.DATA / "universe_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
