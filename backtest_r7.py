"""Round 7: one trend-following portfolio across several TFEX product groups.

Idea: trend following works best across many uncorrelated markets. Proxies
(free daily data from Yahoo) for products that TFEX lists today:
  S50      SET50 Index Futures      <- TDEX.BK (SET50 ETF)
  GOLD     Gold Online Futures      <- GC=F   (USD gold)
  SILVER   Silver Online Futures    <- SI=F
  USDTHB   USD/THB Futures          <- THB=X
  EURUSD   EUR/USD Futures          <- EURUSD=X
  USDJPY   USD/JPY Futures          <- JPY=X
Rubber (JRF, RSS3) has no free history and is left out. TFEX no longer lists oil.

Each market: long/short signal decided at the close, held from the next day.
Size by volatility: weight = (10% / sqrt(6)) / 60-day annualised vol, capped
at 3x per market and 6x total notional (futures margin makes this possible).
Costs per side: S50 0.04%, gold 0.04%, silver 0.06%, FX 0.03%; rollover
0.05% per quarter. FX carry (interest-rate difference) is NOT modelled.

Signals fixed BEFORE running (3 portfolio variants):
  don55_20   long on 55-day high, exit 20-day low; short mirror
  don100_50  same with 100 / 50
  ema50_200  long when EMA50 > EMA200, short when below
Pass rule (fixed before running): portfolio Sharpe >= 0.4 in SEARCH and
VALIDATE, CAGR >= 3%/yr in VALIDATE, and beats >= 90% of runs where each
market's position series is shifted in time by a random offset (same
exposure, timing destroyed). Holdout opened once, only for passers.

    python backtest_r7.py && python holdout.py _r7
"""
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as B
import config as C

RNG = np.random.default_rng(53)
MARKETS = {"S50": "TDEX.BK", "GOLD": "GC=F", "SILVER": "SI=F",
           "USDTHB": "THB=X", "EURUSD": "EURUSD=X", "USDJPY": "JPY=X"}
COSTS = {"S50": 0.0004, "GOLD": 0.0004, "SILVER": 0.0006,
         "USDTHB": 0.0003, "EURUSD": 0.0003, "USDJPY": 0.0003}
ROLL_DAY = 0.0005 / 63
TGT = 0.10 / np.sqrt(len(MARKETS))
CAP_M, CAP_TOT = 3.0, 6.0
VARIANTS = ["don55_20", "don100_50", "ema50_200"]


def fetch():
    out = {}
    for name, sym in MARKETS.items():
        f = C.DATA / f"multi_{name}.csv"
        df = None
        for a in range(3):
            try:
                df = yf.download(sym, start="2003-01-01", auto_adjust=False,
                                 progress=False, threads=False)
                if df is not None and len(df):
                    break
            except Exception as e:
                print(" retry", sym, e)
            time.sleep(3)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s[s > 0]
        s.to_frame("Close").to_csv(f)
        out[name] = s
        print(f"{name:7s} {sym:9s} {len(s)} rows from {s.index[0].date()}")
    return pd.DataFrame(out).sort_index()


def positions(p, kind):
    """+1 / 0 / -1 per day, decided at that day's close."""
    c = p.values
    if kind.startswith("don"):
        n, m = map(int, kind[3:].split("_"))
        hiN = p.rolling(n).max().shift(1).values
        loN = p.rolling(n).min().shift(1).values
        hiM = p.rolling(m).max().shift(1).values
        loM = p.rolling(m).min().shift(1).values
        pos = np.zeros(len(c))
        s = 0
        for i in range(len(c)):
            if np.isnan(c[i]):
                pos[i] = s
                continue
            if s == 1 and c[i] < loM[i]:
                s = 0
            elif s == -1 and c[i] > hiM[i]:
                s = 0
            if s == 0:
                if c[i] > hiN[i]:
                    s = 1
                elif c[i] < loN[i]:
                    s = -1
            pos[i] = s
        return pd.Series(pos, p.index)
    e50 = p.ewm(span=50, adjust=False, min_periods=50).mean()
    e200 = p.ewm(span=200, adjust=False, min_periods=200).mean()
    return pd.Series(np.where(e200.isna(), 0, np.sign(e50 - e200)), p.index)


def run(px, pos):
    """pos: DataFrame of +1/0/-1. Returns daily portfolio returns and per-market returns."""
    R = px.pct_change(fill_method=None)
    vol = R.rolling(60, min_periods=40).std() * np.sqrt(252)
    w = (pos * (TGT / vol)).clip(-CAP_M, CAP_M)
    gross = w.abs().sum(axis=1)
    scale = (CAP_TOT / gross).clip(upper=1.0).fillna(1.0)
    w = w.mul(scale, axis=0).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    cost = pd.DataFrame({m: (w[m] - held[m]).abs() * COSTS[m] for m in px.columns})
    per = held * R.fillna(0.0) - cost - held.abs() * ROLL_DAY
    return per.sum(axis=1), per


def stats(d):
    s = B.curve_stats(d)
    return s


def main():
    C.RESULTS.mkdir(exist_ok=True)
    C.DATA.mkdir(exist_ok=True)
    px = fetch()
    px = px[px.index >= "2004-01-01"]
    px = px.ffill(limit=3)          # different market holidays
    periods = dict(B.PERIODS)
    allp = list(periods.items()) + [("holdout", (C.HOLDOUT_START, "2100"))]
    rows, hold, passed, mrows, yrs = [], [], [], [], []
    print("\ncorrelation of daily returns:\n", px.pct_change(fill_method=None).corr().round(2).to_string())
    for v in VARIANTS:
        pos = pd.DataFrame({m: positions(px[m], v) for m in px.columns})
        d, per = run(px, pos)
        res = {}
        for p, (s, e) in allp:
            st = stats(d[s:e])
            row = dict(market="TFEX_multi", variant=v, period=p, **st)
            (hold if p == "holdout" else rows).append(row)
            res[p] = row
        for p, (s, e) in periods.items():           # per-market diagnostics only
            for m in px.columns:
                ms = stats(per[m][s:e])
                mrows.append(dict(variant=v, period=p, market=m, cagr=ms["cagr"], sharpe=ms["sharpe"]))
        ok = (all(res[p]["sharpe"] >= 0.4 for p in periods)
              and res["validate"]["cagr"] >= 0.03)
        sims = {p: [] for p in periods}
        for _ in range(200):
            sh = pd.DataFrame({m: np.roll(pos[m].values, RNG.integers(250, len(pos) - 250))
                               for m in pos.columns}, index=pos.index)
            dr, _ = run(px, sh)
            for p, (s, e) in periods.items():
                sims[p].append(stats(dr[s:e])["sharpe"])
        for p in periods:
            res[p]["rand_sharpe_median"] = float(np.nanmedian(sims[p]))
            res[p]["p_value"] = float(np.mean(np.array(sims[p]) >= res[p]["sharpe"]))
        ok = ok and all(res[p]["p_value"] < C.MAX_P for p in periods)
        if ok:
            passed.append(dict(market="TFEX_multi", variant=v))
        print(f"\n{v:10s} {'PASS' if ok else 'fail'}  " + "  ".join(
            f"{p}: CAGR {res[p]['cagr']:+.3f} DD {res[p]['maxdd']:.2f} Sharpe {res[p]['sharpe']:.2f} p {res[p]['p_value']:.2f}"
            for p in periods))
        dd_ = d[:C.VALIDATE[1]]
        for y, g in dd_.groupby(dd_.index.year):
            yrs.append(dict(variant=v, year=y, ret=float((1 + g).prod() - 1)))
    pd.DataFrame(rows).to_csv(C.RESULTS / "grid_r7.csv", index=False, float_format="%.4f")
    pd.DataFrame(hold).to_csv(C.RESULTS / "_sealed_holdout_r7.csv", index=False, float_format="%.4f")
    (C.RESULTS / "passed_r7.json").write_text(json.dumps(passed, indent=1))
    mk = pd.DataFrame(mrows)
    mk.to_csv(C.RESULTS / "markets_r7.csv", index=False, float_format="%.4f")
    print("\nper-market Sharpe (diagnostic, not used to pick):\n",
          mk.pivot_table(index=["variant", "market"], columns="period", values="sharpe").round(2).to_string())
    yr = pd.DataFrame(yrs).pivot(index="year", columns="variant", values="ret")
    yr.to_csv(C.RESULTS / "years_r7.csv", float_format="%.4f")
    print("\nyearly returns:\n", yr.round(3).to_string())
    print(f"\nPassed {len(passed)} of {len(VARIANTS)} variants")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
