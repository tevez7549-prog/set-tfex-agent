"""Round 6: Donchian Long/Short on SET50-type stocks, as a proxy for TFEX
single stock futures (which can be shorted).

Same universe/slots/timing as round 4 (top-50 by traded value each month,
10 slots of 1/10 equity, signal at close, trade next close).

Futures modelling (conservative):
  long  P&L = total-return price change - financing (RF per year)
  short P&L = -(total-return price change) + RF     (fair-value futures)
  margin cash earns nothing; cost 0.15% per side (commission + wider spread);
  rollover 0.15% per quarter held, charged daily.

Grid fixed BEFORE running (6 variants):
  (N, M) in {(55, 20), (100, 50)} x mode:
    L     long only (futures)
    LS    long on N-day high break, short on N-day low break
    LSR   LS, but longs only when SET > MA200 and shorts only when SET < MA200
Exit long: close < lowest close of M days; exit short: close > highest close of M days.
Pass rule as round 4 (beat equal-weight top-50 in CAGR + Sharpe both periods,
>= 3%/yr in VALIDATE, beat >= 90% of random entries with random direction).

    python backtest_r6.py && python holdout.py _r6
"""
import json

import numpy as np
import pandas as pd

import backtest as B
import backtest_r4 as R4
import config as C

RNG = np.random.default_rng(41)
RF = 0.015
COST = 0.0015
ROLL_DAY = 0.0015 / 63
GRID = [(n, m, mode) for n, m in [(55, 20), (100, 50)] for mode in ("L", "LS", "LSR")]


def simulate(D, n, m, mode, rand_p=None):
    P, R, AV = D["P"], D["R"], D["AV"]
    hiN = D["hiN"][n]; loN = D["loN"][n]
    hiM = D["hiM"][m]; loM = D["loM"][m]
    T, k = P.shape
    side = np.zeros(k, int)          # +1 long, -1 short
    notion = np.zeros(k)             # currency notional per position
    entry = np.full(k, np.nan)
    E = 1.0
    daily = np.zeros(T)
    trades, nsig = [], 0
    to_open, to_close = [], []
    bull = D["bull"]
    for i in range(1, T):
        E0 = E
        Ri = np.nan_to_num(R[i])
        pnl = float((side * notion * Ri).sum())
        carry = float((notion[side == 1]).sum()) * RF / 245 - float((notion[side == -1]).sum()) * RF / 245
        roll = float(notion[side != 0].sum()) * ROLL_DAY
        notion = notion * (1 + Ri)
        E += pnl - carry - roll
        cost = 0.0
        for j in to_close:
            cost += notion[j] * COST
            trades.append(side[j] * (P[i, j] / entry[j] - 1) - 2 * COST)
            side[j], notion[j], entry[j] = 0, 0.0, np.nan
        for j, s in to_open:
            side[j], notion[j], entry[j] = s, E / R4.SLOTS, P[i, j]
            cost += notion[j] * COST
        E -= cost
        daily[i] = E / E0 - 1 if E0 > 0 else 0.0
        if E <= 0:
            break
        # decisions at today's close
        c = P[i]
        out_l = (side == 1) & ((c < loM[i]) | np.isnan(c))
        out_s = (side == -1) & ((c > hiM[i]) | np.isnan(c))
        to_close = list(np.flatnonzero(out_l | out_s))
        free = R4.SLOTS - (int((side != 0).sum()) - len(to_close))
        elig = D["univ"][i] & (side == 0) & ~np.isnan(c)
        allow_l = mode == "L" or mode == "LS" or (mode == "LSR" and bull[i])
        allow_s = mode == "LS" or (mode == "LSR" and not bull[i])
        if rand_p is None:
            sl = elig & (c > hiN[i]) if allow_l else np.zeros(k, bool)
            ss = elig & (c < loN[i]) if allow_s else np.zeros(k, bool)
        else:
            pick = elig & (RNG.random(k) < rand_p)
            coin = RNG.random(k) < 0.5
            sl = pick & coin if allow_l else np.zeros(k, bool)
            ss = pick & ~coin if allow_s else np.zeros(k, bool)
            if allow_l and not allow_s:
                sl = pick
            if allow_s and not allow_l:
                ss = pick
        cand = [(j, 1) for j in np.flatnonzero(sl)] + [(j, -1) for j in np.flatnonzero(ss)]
        nsig += len(cand)
        if free > 0 and cand:
            cand.sort(key=lambda x: -AV[i, x[0]])
            to_open = cand[:free]
        else:
            to_open = []
    return pd.Series(daily, D["idx"]), np.array(trades), nsig


def main():
    C.RESULTS.mkdir(exist_ok=True)
    px, val, div = R4.load()
    D = R4.prepare(px, val, div, B.load("SET"))
    D["hiN"] = {n: px.rolling(n, min_periods=n).max().shift(1).values for n in (55, 100)}
    D["loN"] = {n: px.rolling(n, min_periods=n).min().shift(1).values for n in (55, 100)}
    D["hiM"] = {m: px.rolling(m, min_periods=m).max().shift(1).values for m in (20, 50)}
    D["loM"] = {m: px.rolling(m, min_periods=m).min().shift(1).values for m in (20, 50)}
    bm = R4.bench(D)
    periods = dict(B.PERIODS)
    allp = list(periods.items()) + [("holdout", (C.HOLDOUT_START, "2100"))]
    rows, hold, passed, yrs = [], [], [], []
    for n, m, mode in GRID:
        v = f"don_{n}_{m}_{mode}"
        d, tr, nsig = simulate(D, n, m, mode)
        res = {}
        for p, (s, e) in allp:
            st, bs = B.curve_stats(d[s:e]), B.curve_stats(bm[s:e])
            row = dict(market="SET_r6", variant=v, period=p, **st,
                       bench_cagr=bs["cagr"], bench_sharpe=bs["sharpe"])
            (hold if p == "holdout" else rows).append(row)
            res[p] = row
        ok = (all(res[p]["cagr"] > res[p]["bench_cagr"] and res[p]["sharpe"] > res[p]["bench_sharpe"]
                  for p in periods) and res["validate"]["cagr"] >= R4.MIN_VALIDATE_CAGR)
        if ok:
            p_rate = nsig / max(1, int(D["univ"].sum()))
            sims = {p: [] for p in periods}
            for _ in range(100):
                dr, _, _ = simulate(D, n, m, mode, rand_p=p_rate)
                for p, (s, e) in periods.items():
                    sims[p].append(B.curve_stats(dr[s:e])["cagr"])
            for p in periods:
                res[p]["rand_cagr_median"] = float(np.nanmedian(sims[p]))
                res[p]["p_value"] = float(np.mean(np.array(sims[p]) >= res[p]["cagr"]))
            ok = all(res[p]["p_value"] < C.MAX_P for p in periods)
        if ok:
            passed.append(dict(market="SET_r6", variant=v))
        pf = B.pf(tr) if len(tr) else np.nan
        for p in periods:
            res[p].update(trades=len(tr), win=float(np.mean(tr > 0)) if len(tr) else np.nan, pf=pf)
        print(f"{v:18s} {'PASS' if ok else 'fail'}  trades {len(tr):4d}  win {np.mean(tr > 0):.2f}  PF {pf:.2f}  "
              + "  ".join(f"{p}: {res[p]['cagr']:+.3f} dd {res[p]['maxdd']:.2f}" for p in periods))
        dd_ = d[:C.VALIDATE[1]]
        for y, g in dd_.groupby(dd_.index.year):
            yrs.append(dict(variant=v, year=y, ret=float((1 + g).prod() - 1)))
    for y, g in bm[:C.VALIDATE[1]].groupby(bm[:C.VALIDATE[1]].index.year):
        yrs.append(dict(variant="benchmark_50", year=y, ret=float((1 + g).prod() - 1)))
    pd.DataFrame(rows).to_csv(C.RESULTS / "grid_r6.csv", index=False, float_format="%.4f")
    pd.DataFrame(hold).to_csv(C.RESULTS / "_sealed_holdout_r6.csv", index=False, float_format="%.4f")
    (C.RESULTS / "passed_r6.json").write_text(json.dumps(passed, indent=1))
    yr = pd.DataFrame(yrs).pivot(index="year", columns="variant", values="ret")
    yr.to_csv(C.RESULTS / "years_r6.csv", float_format="%.4f")
    print("\nyearly returns:\n", yr.loc[2008:].round(3).to_string())
    print(f"\nPassed {len(passed)} of {len(GRID)} variants")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
