"""Round 5: EMA crossovers, RSI and Stochastic entries on SET50-type stocks.

Same universe, slots, timing, costs and pass rule as round 4 (backtest_r4.py),
so results compare directly with Donchian.

Indicators (on dividend/split-adjusted prices; Stochastic uses adjusted H/L):
  EMA 20/50/100/200, RSI(14), Stochastic slow %K(14,3) / %D(3)

Signals (fixed BEFORE running; 6 signals x filter {all, div} = 12 variants):
  ema20_50    in: EMA20 crosses above EMA50            out: EMA20 < EMA50
  ema50_200   in: EMA50 crosses above EMA200 (golden)  out: EMA50 < EMA200
  ema_stack   in: EMA20>50>100>200 becomes true         out: EMA20 < EMA50
  ema20_50_rsi in: EMA20 crosses above EMA50, close > EMA200, RSI 50-70
                                                       out: EMA20 < EMA50
  rsi_pull    in: uptrend (close>EMA200, EMA50>EMA200), RSI crosses up 40
              out: RSI > 70, close < EMA200, or 20 days
  sto_pull    in: uptrend, %K crosses above %D from below 20
              out: %K > 80, close < EMA200, or 20 days

    python backtest_r5.py && python holdout.py _r5
"""
import json

import numpy as np
import pandas as pd

import backtest as B
import backtest_r4 as R4
import config as C

RNG = np.random.default_rng(31)


def indicators(px):
    raw = pd.read_parquet(C.DATA / "panel_rawclose.parquet").reindex(px.index)[px.columns]
    adj = px / raw
    hi = pd.read_parquet(C.DATA / "panel_high.parquet").reindex(px.index)[px.columns] * adj
    lo = pd.read_parquet(C.DATA / "panel_low.parquet").reindex(px.index)[px.columns] * adj
    e = {n: px.ewm(span=n, adjust=False, min_periods=n).mean() for n in (20, 50, 100, 200)}
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hh, ll = hi.rolling(14).max(), lo.rolling(14).min()
    k = (100 * (px - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
    dd = k.rolling(3).mean()
    return e, rsi, k, dd


def signals(px):
    e, rsi, k, d = indicators(px)

    def xup(a, b):
        return (a > b) & (a.shift(1) <= b.shift(1))
    up = (px > e[200]) & (e[50] > e[200])
    stack = (e[20] > e[50]) & (e[50] > e[100]) & (e[100] > e[200])
    S = {
        "ema20_50": (xup(e[20], e[50]), e[20] < e[50], None),
        "ema50_200": (xup(e[50], e[200]), e[50] < e[200], None),
        "ema_stack": (stack & ~stack.shift(1, fill_value=False), e[20] < e[50], None),
        "ema20_50_rsi": (xup(e[20], e[50]) & (px > e[200]) & (rsi >= 50) & (rsi <= 70),
                         e[20] < e[50], None),
        "rsi_pull": (up & xup(rsi, pd.DataFrame(40.0, px.index, px.columns)),
                     (rsi > 70) | (px < e[200]), 20),
        "sto_pull": (up & xup(k, d) & (k.shift(1) < 20), (k > 80) | (px < e[200]), 20),
    }
    return {n: (a.fillna(False).astype(bool).values, b.fillna(False).astype(bool).values, h)
            for n, (a, b, h) in S.items()}


def simulate(D, ent, ext, maxhold, filt, rand_p=None):
    P, R, AV = D["P"], D["R"], D["AV"]
    T, k = P.shape
    held = np.zeros(k, bool)
    age = np.zeros(k, int)
    w = np.zeros(k)
    entry_px = np.full(k, np.nan)
    daily = np.zeros(T)
    trades, nsig = [], 0
    to_buy, to_sell = np.array([], int), np.array([], int)
    for i in range(1, T):
        r = float(w @ R[i])
        if 1 + r > 0:
            w = w * (1 + R[i]) / (1 + r)
        cost = 0.0
        for j in to_sell:
            cost += w[j] * C.COST_STOCK
            trades.append(P[i, j] / entry_px[j] - 1 - 2 * C.COST_STOCK)
            w[j], held[j], entry_px[j], age[j] = 0.0, False, np.nan, 0
        for j in to_buy:
            w[j], held[j], entry_px[j], age[j] = 1.0 / R4.SLOTS, True, P[i, j], 0
            cost += C.COST_STOCK / R4.SLOTS
        daily[i] = r - cost
        age[held] += 1
        out = held & (ext[i] | np.isnan(P[i]))
        if maxhold:
            out |= held & (age >= maxhold)
        to_sell = np.flatnonzero(out)
        free = R4.SLOTS - (held.sum() - len(to_sell))
        elig = D["univ"][i] & ~held & ~np.isnan(P[i])
        if filt == "div":
            elig &= D["divok"][i]
        sig = elig & (ent[i] if rand_p is None else (RNG.random(k) < rand_p))
        cand = np.flatnonzero(sig)
        nsig += len(cand)
        to_buy = cand[np.argsort(-AV[i, cand])[:free]] if free > 0 and len(cand) else np.array([], int)
    return pd.Series(daily, D["idx"]), np.array(trades), nsig


def main():
    C.RESULTS.mkdir(exist_ok=True)
    px, val, div = R4.load()
    D = R4.prepare(px, val, div, B.load("SET"))
    S = signals(px)
    bm = R4.bench(D)
    periods = dict(B.PERIODS)
    allp = list(periods.items()) + [("holdout", (C.HOLDOUT_START, "2100"))]
    rows, hold, passed = [], [], []
    for name, (ent, ext, mh) in S.items():
        for f in ("all", "div"):
            v = f"{name}_{f}"
            d, tr, nsig = simulate(D, ent, ext, mh, f)
            res = {}
            for p, (s, e) in allp:
                st, bs = B.curve_stats(d[s:e]), B.curve_stats(bm[s:e])
                row = dict(market="SET_r5", variant=v, period=p, **st,
                           bench_cagr=bs["cagr"], bench_sharpe=bs["sharpe"])
                (hold if p == "holdout" else rows).append(row)
                res[p] = row
            ok = (all(res[p]["cagr"] > res[p]["bench_cagr"] and res[p]["sharpe"] > res[p]["bench_sharpe"]
                      for p in periods) and res["validate"]["cagr"] >= R4.MIN_VALIDATE_CAGR)
            if ok:
                p_rate = nsig / max(1, int(D["univ"].sum()))
                sims = {p: [] for p in periods}
                for _ in range(100):
                    dr, _, _ = simulate(D, ent, ext, mh, f, rand_p=p_rate)
                    for p, (s, e) in periods.items():
                        sims[p].append(B.curve_stats(dr[s:e])["cagr"])
                for p in periods:
                    res[p]["rand_cagr_median"] = float(np.nanmedian(sims[p]))
                    res[p]["p_value"] = float(np.mean(np.array(sims[p]) >= res[p]["cagr"]))
                ok = all(res[p]["p_value"] < C.MAX_P for p in periods)
            if ok:
                passed.append(dict(market="SET_r5", variant=v))
            pf = B.pf(tr) if len(tr) else np.nan
            for p in periods:
                res[p].update(trades=len(tr), win=float(np.mean(tr > 0)) if len(tr) else np.nan, pf=pf)
            print(f"{v:22s} {'PASS' if ok else 'fail'}  trades {len(tr):4d}  win {np.mean(tr > 0):.2f}  PF {pf:.2f}  "
                  + "  ".join(f"{p}: {res[p]['cagr']:+.3f} dd {res[p]['maxdd']:.2f}" for p in periods))
    pd.DataFrame(rows).to_csv(C.RESULTS / "grid_r5.csv", index=False, float_format="%.4f")
    pd.DataFrame(hold).to_csv(C.RESULTS / "_sealed_holdout_r5.csv", index=False, float_format="%.4f")
    (C.RESULTS / "passed_r5.json").write_text(json.dumps(passed, indent=1))
    b = {p: B.curve_stats(bm[s:e]) for p, (s, e) in periods.items()}
    print("\nbenchmark:", "  ".join(f"{p}: {b[p]['cagr']:+.3f}" for p in periods))
    print(f"Passed {len(passed)} of {len(S) * 2} variants")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
