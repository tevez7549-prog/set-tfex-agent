"""Round 4: Donchian breakout on individual SET50-type stocks.

Universe (point-in-time SET50 proxy): at each month-end, the 50 SET stocks with
the highest 60-day average baht traded and >= 1 year of history. Real SET50
membership history is not freely available; this mimics it without using
today's list.
Fundamental filter (proxy, because quarterly statements are blocked from
GitHub): the stock paid a dividend in BOTH of the last two 12-month windows,
i.e. a company that has been profitable enough to pay out for 2 years.

Rules (daily closes only; the panel has no daily highs/lows):
  entry  close > highest close of the previous N days, stock in universe
  exit   close < lowest close of the previous M days
  slots  at most K positions, each 1/K of equity; when more signals than
         free slots, take the most-traded stocks first
  regime (optional) new entries only when the SET proxy is above its 200-day MA
  timing signal at today's close, trade at the next day's close
  cost   0.25% per side

Grid fixed BEFORE running (12 variants):
  (N, M) in {(20, 10), (55, 20), (100, 50)} x filter {all, dividend} x regime {off, on}, K = 10
Pass rule (same as round 3): beat the equal-weight top-50 benchmark in CAGR
and Sharpe in SEARCH and VALIDATE, make >= 3%/year in VALIDATE, and beat
>= 90% of random-entry runs (same exits, same entry rate).

    python backtest_r4.py && python holdout.py _r4
"""
import json

import numpy as np
import pandas as pd

import backtest as B
import config as C

RNG = np.random.default_rng(23)
TOP_K = 50
SLOTS = 10
MIN_VALIDATE_CAGR = 0.03
GRID = [(n, m, f, r) for n, m in [(20, 10), (55, 20), (100, 50)]
        for f in ("all", "div") for r in (False, True)]


def load():
    px = pd.read_parquet(C.DATA / "panel_close.parquet").sort_index()
    val = pd.read_parquet(C.DATA / "panel_value.parquet").reindex(px.index)
    div = (pd.read_parquet(C.DATA / "panel_div.parquet")
           .reindex(px.index).reindex(columns=px.columns).fillna(0.0))
    return px, val, div


def prepare(px, val, div, setidx):
    P = px.values
    present = px.notna().values
    run_len = np.zeros(P.shape, int)
    for i in range(len(P)):
        run_len[i] = np.where(present[i], (run_len[i - 1] + 1) if i else 1, 0)
    AV = np.nan_to_num(val.rolling(60, min_periods=40).mean().values, nan=0.0)

    # point-in-time universe, refreshed at each month-end, used the next month
    per = px.index.to_period("M")
    is_me = np.r_[per[1:] != per[:-1], True]
    univ = np.zeros(P.shape, bool)
    cur = np.zeros(P.shape[1], bool)
    for i in range(len(P)):
        univ[i] = cur
        if is_me[i]:
            cand = np.flatnonzero(run_len[i] > 252)
            cur = np.zeros(P.shape[1], bool)
            cur[cand[np.argsort(-AV[i, cand])[:TOP_K]]] = True

    d1 = div.rolling(252, min_periods=1).sum()
    d2 = d1.shift(252)
    divok = ((d1 > 0) & (d2 > 0)).values

    ma200 = setidx["Close"].rolling(200).mean().reindex(px.index).ffill().values
    setc = setidx["Close"].reindex(px.index).ffill().values
    bull = setc > ma200
    hi = {n: px.rolling(n, min_periods=n).max().shift(1).values for n in (20, 55, 100)}
    lo = {m: px.rolling(m, min_periods=m).min().shift(1).values for m in (10, 20, 50)}
    return dict(P=P, R=px.pct_change(fill_method=None).fillna(0.0).values, AV=AV,
                univ=univ, divok=divok, bull=bull, hi=hi, lo=lo, idx=px.index)


def simulate(D, n, m, filt, regime, rand_p=None):
    P, R, AV = D["P"], D["R"], D["AV"]
    hi, lo = D["hi"][n], D["lo"][m]
    T, k = P.shape
    held = np.zeros(k, bool)
    w = np.zeros(k)
    entry_px = np.full(k, np.nan)
    daily = np.zeros(T)
    trades, signals = [], 0
    to_buy, to_sell = np.array([], int), np.array([], int)
    for i in range(1, T):
        r = float(w @ R[i])
        if 1 + r > 0:
            w = w * (1 + R[i]) / (1 + r)
        # execute yesterday's decisions at today's close
        cost = 0.0
        for j in to_sell:
            cost += w[j] * C.COST_STOCK
            trades.append(P[i, j] / entry_px[j] - 1 - 2 * C.COST_STOCK)
            w[j], held[j], entry_px[j] = 0.0, False, np.nan
        for j in to_buy:
            w[j], held[j], entry_px[j] = 1.0 / SLOTS, True, P[i, j]
            cost += C.COST_STOCK / SLOTS
        daily[i] = r - cost
        # decide at today's close
        exit_sig = held & ((P[i] < lo[i]) | np.isnan(P[i]))
        to_sell = np.flatnonzero(exit_sig)
        free = SLOTS - (held.sum() - len(to_sell))
        elig = D["univ"][i] & ~held & ~np.isnan(P[i])
        if filt == "div":
            elig &= D["divok"][i]
        if regime and not D["bull"][i]:
            elig[:] = False
        if rand_p is None:
            sig = elig & (P[i] > hi[i])
        else:
            sig = elig & (RNG.random(k) < rand_p)
        cand = np.flatnonzero(sig)
        signals += len(cand)
        if free > 0 and len(cand):
            to_buy = cand[np.argsort(-AV[i, cand])[:free]]
        else:
            to_buy = np.array([], int)
    return pd.Series(daily, D["idx"]), np.array(trades), signals


def bench(D):
    """Equal weight on the whole top-50 universe, rebalanced monthly."""
    R, univ = D["R"], D["univ"]
    w = np.zeros(R.shape[1])
    daily = np.zeros(len(R))
    for i in range(1, len(R)):
        r = float(w @ R[i])
        if 1 + r > 0:
            w = w * (1 + R[i]) / (1 + r)
        if (univ[i] != univ[i - 1]).any() or (i == 1):
            new = univ[i] / max(univ[i].sum(), 1)
            r -= np.abs(new - w).sum() * C.COST_STOCK
            w = new.astype(float)
        daily[i] = r
    return pd.Series(daily, D["idx"])


def main():
    C.RESULTS.mkdir(exist_ok=True)
    px, val, div = load()
    D = prepare(px, val, div, B.load("SET"))
    bm = bench(D)
    periods = dict(B.PERIODS)
    allp = list(periods.items()) + [("holdout", (C.HOLDOUT_START, "2100"))]
    rows, hold, passed = [], [], []
    for n, m, f, reg in GRID:
        v = f"don_{n}_{m}_{f}_{'regime' if reg else 'always'}"
        d, tr, nsig = simulate(D, n, m, f, reg)
        res = {}
        for p, (s, e) in allp:
            st, bs = B.curve_stats(d[s:e]), B.curve_stats(bm[s:e])
            row = dict(market="SET_r4", variant=v, period=p, **st,
                       bench_cagr=bs["cagr"], bench_sharpe=bs["sharpe"],
                       exposure=float((d[s:e] != 0).mean()))
            (hold if p == "holdout" else rows).append(row)
            res[p] = row
        ok = (all(res[p]["cagr"] > res[p]["bench_cagr"] and res[p]["sharpe"] > res[p]["bench_sharpe"]
                  for p in periods) and res["validate"]["cagr"] >= MIN_VALIDATE_CAGR)
        if ok:
            # random entries at the same rate, same exits and slots
            elig_days = max(1, int(D["univ"].sum()))
            p_rate = nsig / elig_days
            sims = {p: [] for p in periods}
            for _ in range(100):
                dr, _, _ = simulate(D, n, m, f, reg, rand_p=p_rate)
                for p, (s, e) in periods.items():
                    sims[p].append(B.curve_stats(dr[s:e])["cagr"])
            for p in periods:
                res[p]["rand_cagr_median"] = float(np.nanmedian(sims[p]))
                res[p]["p_value"] = float(np.mean(np.array(sims[p]) >= res[p]["cagr"]))
            ok = all(res[p]["p_value"] < C.MAX_P for p in periods)
        if ok:
            passed.append(dict(market="SET_r4", variant=v))
        pf = B.pf(tr) if len(tr) else np.nan
        print(f"{v:32s} {'PASS' if ok else 'fail'}  trades {len(tr):4d}  win {np.mean(tr > 0):.2f}  PF {pf:.2f}  "
              + "  ".join(f"{p}: {res[p]['cagr']:+.3f} dd {res[p]['maxdd']:.2f}" for p in periods))
        for p in periods:
            res[p].update(trades_all=len(tr), win_all=float(np.mean(tr > 0)) if len(tr) else np.nan, pf_all=pf)
    pd.DataFrame(rows).to_csv(C.RESULTS / "grid_r4.csv", index=False, float_format="%.4f")
    pd.DataFrame(hold).to_csv(C.RESULTS / "_sealed_holdout_r4.csv", index=False, float_format="%.4f")
    (C.RESULTS / "passed_r4.json").write_text(json.dumps(passed, indent=1))
    b = {p: B.curve_stats(bm[s:e]) for p, (s, e) in periods.items()}
    print("\nbenchmark:", "  ".join(f"{p}: {b[p]['cagr']:+.3f}" for p in periods))
    print(f"Passed {len(passed)} of {len(GRID)} variants")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
