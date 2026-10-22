"""Round 3: dividend, low-volatility and value strategies on Thai stocks.

Universe (point-in-time, same as round 2): at each month-end, the 100 SET
stocks with the highest 60-day average baht traded and >= 1 year of history.
Hold the top N by score, equal weight, rebalance monthly (signal at the
month-end close, trade at the next day's close), 0.25% cost per side.

Grid fixed BEFORE running (5 scores x N in {10, 20} x regime on/off = 20):
  DY    trailing 12-month dividends / price (highest first; >25% treated as bad data)
  LV    lowest 1-year daily volatility
  DYLV  rank(DY) + rank(LV)
  VAL   rank(E/P) + rank(B/P), profitable stocks only     (fundamentals)
  QV    rank(E/P) + rank(B/P) + rank(ROE), ROE > 0         (fundamentals)
Fundamentals are used only 90 days after the quarter ends (report lag).

Pass rule (fixed before running):
  beats the equal-weight top-100 benchmark in CAGR and Sharpe in SEARCH and
  VALIDATE, beats >= 90% of random picks from the same candidates, and makes
  >= 3%/year in VALIDATE. Holdout opened once, only for passers.

    python backtest_r3.py && python holdout.py _r3
"""
import numpy as np
import pandas as pd

import backtest as B
import config as C

RNG = np.random.default_rng(11)
TOP_K = 100
MIN_VALIDATE_CAGR = 0.03
LAG_DAYS = 90


def rank(x):
    """Rank in [0,1], higher = better, NaN stays NaN."""
    s = pd.Series(x)
    return s.rank(pct=True).values


def load_panels():
    px = pd.read_parquet(C.DATA / "panel_close.parquet").sort_index()
    val = pd.read_parquet(C.DATA / "panel_value.parquet").reindex(px.index)
    raw = pd.read_parquet(C.DATA / "panel_rawclose.parquet").reindex(px.index)
    div = pd.read_parquet(C.DATA / "panel_div.parquet").reindex(px.index).reindex(columns=px.columns)
    return px, val, raw, div.fillna(0.0)


def fundamental_panels(px, raw):
    """E/P and B/P at today's price, ROE; only data already published."""
    f = pd.read_csv(C.DATA / "fundamentals.csv")
    f = f[f.quarter.between(1, 4)].dropna(subset=["fiscal"])
    f = f[f.symbol.isin(px.columns)]
    qend = pd.to_datetime(dict(year=f.fiscal.astype(int), month=(f.quarter * 3).astype(int),
                               day=1)) + pd.offsets.MonthEnd(0)
    f = f.assign(avail=qend + pd.Timedelta(days=LAG_DAYS), qend=qend)
    ep = pd.DataFrame(np.nan, px.index, px.columns)
    bp, roe = ep.copy(), ep.copy()
    for sym, g in f.groupby("symbol"):
        g = g.sort_values("avail")
        r = raw[sym]
        # Yahoo price at quarter end (split-consistent with today's price)
        pq = r.reindex(g.qend, method="ffill").values
        e = np.where(g.pe > 0, pq / g.pe, np.nan)          # EPS (ttm), same units as Yahoo
        b = np.where(g.pbv > 0, pq / g.pbv, np.nan)        # book value per share
        s = pd.DataFrame({"e": e, "b": b, "roe": g.roe.values}, index=g.avail.values)
        s = s[~s.index.duplicated(keep="last")].reindex(px.index, method="ffill")
        ep[sym] = s["e"] / r
        bp[sym] = s["b"] / r
        roe[sym] = s["roe"]
    return ep, bp, roe


def build_scores(px, raw, div, fund):
    R = px.pct_change(fill_method=None)
    dy = div.rolling(252, min_periods=200).sum() / raw
    dy = dy.where((dy > 0) & (dy < 0.25))
    lv = -R.rolling(252, min_periods=200).std()
    S = {"DY": lambda i, c: dy.values[i, c],
         "LV": lambda i, c: lv.values[i, c],
         "DYLV": lambda i, c: rank(dy.values[i, c]) + rank(lv.values[i, c])}
    if fund is not None:
        ep, bp, roe = fund
        epv, bpv, roev = ep.values, bp.values, roe.values

        def val(i, c):
            e, b = epv[i, c], bpv[i, c]
            ok = (e > 0) & (b > 0)
            return np.where(ok, rank(np.where(ok, e, np.nan)) + rank(np.where(ok, b, np.nan)), np.nan)

        def qv(i, c):
            e, b, q = epv[i, c], bpv[i, c], roev[i, c]
            ok = (e > 0) & (b > 0) & (q > 0)
            m = lambda x: rank(np.where(ok, x, np.nan))
            return np.where(ok, m(e) + m(b) + m(q), np.nan)
        S["VAL"], S["QV"] = val, qv
    return S


def run(px, val, setidx, score_fn, N, regime, pick="top"):
    P = px.values
    R = px.pct_change(fill_method=None).fillna(0.0).values
    AV = np.nan_to_num(val.rolling(60, min_periods=40).mean().values, nan=0.0)
    present = px.notna().values
    run_len = np.zeros(P.shape, int)
    for i in range(len(P)):
        run_len[i] = np.where(present[i], (run_len[i - 1] + 1) if i else 1, 0)
    ma200 = setidx["Close"].rolling(200).mean().reindex(px.index).ffill().values
    setc = setidx["Close"].reindex(px.index).ffill().values
    per = px.index.to_period("M")
    is_me = np.r_[per[1:] != per[:-1], True]
    k = P.shape[1]
    w, pending = np.zeros(k), None
    daily = np.zeros(len(P))
    first_valid = None
    for i in range(1, len(P)):
        r = float(w @ R[i])
        if 1 + r > 0:
            w = w * (1 + R[i]) / (1 + r)
        if pending is not None:
            r -= np.abs(pending - w).sum() * C.COST_STOCK
            w, pending = pending, None
        daily[i] = r
        if not is_me[i] or i < 260:
            continue
        cand = np.flatnonzero(run_len[i] > 252)
        cand = cand[np.argsort(-AV[i, cand])[:TOP_K]]
        new = np.zeros(k)
        if score_fn is None:                      # benchmark: hold all 100
            new[cand] = 1.0 / len(cand)
        else:
            sc = score_fn(i, cand)
            good = ~np.isnan(sc)
            if good.sum() >= 30:
                first_valid = first_valid or px.index[i]
                if not (regime and not setc[i] > ma200[i]):
                    c2, s2 = cand[good], sc[good]
                    names = (RNG.choice(c2, N, replace=False) if pick == "random"
                             else c2[np.argsort(-s2)[:N]])
                    new[names] = 1.0 / N
        pending = new
    return pd.Series(daily, px.index), first_valid


def main():
    C.RESULTS.mkdir(exist_ok=True)
    px, val, raw, div = load_panels()
    setidx = B.load("SET")
    fund = None
    if (C.DATA / "fundamentals.csv").exists():
        try:
            fund = fundamental_panels(px, raw)
        except Exception as e:
            print("fundamentals unusable:", e)
    S = build_scores(px, raw, div, fund)
    bench, _ = run(px, val, setidx, None, 0, False)
    periods = dict(B.PERIODS)
    rows, hold, passed = [], [], []
    for name, fn in S.items():
        for N in (10, 20):
            for regime in (False, True):
                v = f"{name}_top{N}_{'regime' if regime else 'always'}"
                d, fv = run(px, val, setidx, fn, N, regime)
                if fv is None:
                    print(v, "no valid data"); continue
                res = {}
                for p, (s, e) in list(periods.items()) + [("holdout", (C.HOLDOUT_START, "2100"))]:
                    s2 = max(pd.Timestamp(s), fv)          # start when the data exists
                    st, bs = B.curve_stats(d[s2:e]), B.curve_stats(bench[s2:e])
                    row = dict(variant=v, period=p, start=s2.date(), **st,
                               bench_cagr=bs["cagr"], bench_sharpe=bs["sharpe"])
                    (hold if p == "holdout" else rows).append(row)
                    res[p] = row
                ok = all(res[p]["cagr"] > res[p]["bench_cagr"] and res[p]["sharpe"] > res[p]["bench_sharpe"]
                         for p in periods) and res["validate"]["cagr"] >= MIN_VALIDATE_CAGR
                if ok:
                    sims = {p: [] for p in periods}
                    for _ in range(100):
                        dr, _ = run(px, val, setidx, fn, N, regime, pick="random")
                        for p, (s, e) in periods.items():
                            sims[p].append(B.curve_stats(dr[max(pd.Timestamp(s), fv):e])["cagr"])
                    for p in periods:
                        res[p]["rand_cagr_median"] = float(np.nanmedian(sims[p]))
                        res[p]["p_value"] = float(np.mean(np.array(sims[p]) >= res[p]["cagr"]))
                    ok = all(res[p]["p_value"] < C.MAX_P for p in periods)
                if ok:
                    passed.append(dict(market="SET_r3", variant=v))
                print(v, "PASS" if ok else "fail",
                      " ".join(f"{p}:{res[p]['cagr']:.3f}" for p in periods))
    grid = pd.DataFrame(rows).assign(market="SET_r3")
    grid.to_csv(C.RESULTS / "grid_r3.csv", index=False, float_format="%.4f")
    pd.DataFrame(hold).assign(market="SET_r3").to_csv(
        C.RESULTS / "_sealed_holdout_r3.csv", index=False, float_format="%.4f")
    import json
    (C.RESULTS / "passed_r3.json").write_text(json.dumps(passed, indent=1))
    print(f"\nPassed {len(passed)} of {grid.variant.nunique()} variants")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
