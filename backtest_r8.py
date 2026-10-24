"""Round 8: the owner's own chart method, turned into rules.

Universe / slots / costs / timing / pass rule: same as round 4 (top-50 by
traded value each month, dividend paid in both of the last 2 years, 10 slots,
0.25% per side, signal at close and trade at the next close).

Per stock, recomputed every 5 trading days with data up to that day only:
  Trend     log-price regression from the lowest close of the last 5 years
            (the start of the trend) to today; needs >= 120 days, slope > 0,
            and close >= lower band (fit - 2 sigma) x 0.97.
  Key lines levels in the last 250 days touched by >= 3 confirmed swing
            points (highs AND lows) within +-1.5% (the "blue line"), up to 5.
  Fibonacci 250-day swing low -> swing high (low first): 0, .236, .382, .5, .618.
Entries:
  breakout  close above a line for C days (C = 1 or 2), within 10% of it
  bounce    low touched a support (line, Fibonacci or lower band) within
            1.5%, closes never more than 3% below it, today closes above it,
            and price was >= 2% above it 10 days ago (came down to it)
Hold / exit (owner: "if it breaks up and doesn't lose this line, follow it"):
  stop = base line x 0.97; after closing above the next line for C days the
  base moves up to that line; exit on close < base x 0.97 or < lower band x 0.97.

Grid fixed BEFORE running (6): entry {breakout, bounce, both} x C {1, 2}.

    python backtest_r8.py && python holdout.py _r8
"""
import json

import numpy as np
import pandas as pd

import backtest as B
import backtest_r4 as R4
import config as C

RNG = np.random.default_rng(61)
STEP, K_PIV, LB, REG_LB, TOL, TOUCH = 5, 5, 250, 1250, 0.03, 0.015
GRID = [(e, c) for e in ("breakout", "bounce", "both") for c in (1, 2)]
NLV = 10


def stock_levels(c, h, l):
    T = len(c)
    lv = np.full((T, NLV), np.nan)
    lower = np.full(T, np.nan)
    slope = np.full(T, np.nan)
    ok = np.flatnonzero(~np.isnan(c))
    if len(ok) < LB + 20:
        return lv, lower, slope
    s0 = ok[0]
    c = pd.Series(c).ffill().values
    h = pd.Series(h).ffill().fillna(pd.Series(c)).values
    l = pd.Series(l).ffill().fillna(pd.Series(c)).values
    hw = pd.Series(h).rolling(2 * K_PIV + 1, center=True).max().values
    lw = pd.Series(l).rolling(2 * K_PIV + 1, center=True).min().values
    ph = np.flatnonzero((h == hw) & (np.arange(T) >= s0))
    pl = np.flatnonzero((l == lw) & (np.arange(T) >= s0))
    logc = np.log(c)
    for t in range(s0 + LB, T, STEP):
        rows = []
        # key lines (touched as both resistance and support)
        sh = ph[(ph >= t - LB) & (ph + K_PIV <= t)]
        sl = pl[(pl >= t - LB) & (pl + K_PIV <= t)]
        prices = np.r_[h[sh], l[sl]]
        typ = np.r_[np.ones(len(sh)), np.zeros(len(sl))]
        if len(prices) >= 3:
            near = np.abs(prices[:, None] / prices[None, :] - 1) <= TOUCH
            cnt = near.sum(1)
            both = (near * typ[None, :]).sum(1)
            good = (cnt >= 3) & (both >= 1) & (both <= cnt - 1)
            order = np.argsort(-cnt)
            taken = []
            for i in order:
                if not good[i]:
                    continue
                lvl = float(np.median(prices[near[i]]))
                if all(abs(lvl / x - 1) > 2 * TOUCH for x in taken):
                    taken.append(lvl)
                if len(taken) == 5:
                    break
            rows += taken
        # fibonacci of the 250-day swing (low first)
        wh, wl = h[t - LB + 1:t + 1], l[t - LB + 1:t + 1]
        ih, il = int(np.argmax(wh)), int(np.argmin(wl))
        if il < ih:
            hh, ll = wh[ih], wl[il]
            rows += [hh - r * (hh - ll) for r in (0.0, 0.236, 0.382, 0.5, 0.618)]
        rr = np.full(NLV, np.nan)
        rr[:min(len(rows), NLV)] = rows[:NLV]
        end = min(T, t + STEP)
        lv[t:end] = rr
        # regression from the start of the trend
        a = max(s0, t - REG_LB)
        anc = a + int(np.argmin(logc[a:t + 1]))
        if t - anc >= 120:
            y = logc[anc:t + 1]
            x = np.arange(len(y), dtype=float)
            b, a0 = np.polyfit(x, y, 1)
            sd = float(np.std(y - (a0 + b * x)))
            xx = np.arange(t - anc, t - anc + (end - t), dtype=float)
            lower[t:end] = np.exp(a0 + b * xx - 2 * sd)
            slope[t:end] = b
    return lv, lower, slope


def signals(c, l, lv, lower, cdays):
    """Per stock: breakout / bounce entry flags and their base line, shape (T,)."""
    T = len(c)
    brk_b = np.full(T, np.nan)
    bnc_b = np.full(T, np.nan)

    def sh(a, k):
        return np.r_[np.full(k, np.nan), a[:-k]] if k else a
    cl = c[:, None]
    # breakout: line known at t-cdays, crossed then held cdays closes
    L0 = np.vstack([np.full((cdays, NLV), np.nan), lv[:-cdays]])
    prev = sh(c, cdays)[:, None]
    held = np.ones((T, NLV), bool)
    for k in range(cdays):
        held &= sh(c, k)[:, None] > L0
    cond = (prev <= L0) & held & (cl <= L0 * 1.10)
    brk = cond.any(1)
    if brk.any():
        brk_b[brk] = np.nanmax(np.where(cond, L0, np.nan)[brk], 1)
    # bounce: supports = lines + lower band
    S = np.hstack([lv, lower[:, None]])
    lmin = pd.Series(l).rolling(cdays).min().values[:, None]
    cmin = pd.Series(c).rolling(cdays).min().values[:, None]
    c10 = sh(c, 10)[:, None]
    cond2 = (lmin <= S * (1 + TOUCH)) & (cmin >= S * (1 - TOL)) & (cl >= S) & (c10 >= S * 1.02)
    bnc = cond2.any(1)
    if bnc.any():
        bnc_b[bnc] = np.nanmax(np.where(cond2, S, np.nan)[bnc], 1)
    return brk, brk_b, bnc, bnc_b


def simulate(D, X, entry, cdays, rand_p=None):
    P, R, AV, lv, lower = D["P"], D["R"], D["AV"], X["lv"], X["lower"]
    T, k = P.shape
    ent, base0 = X[(entry, cdays)]
    held = np.zeros(k, bool)
    base = np.full(k, np.nan)
    w = np.zeros(k)
    entry_px = np.full(k, np.nan)
    daily = np.zeros(T)
    trades, nsig = [], 0
    to_buy, to_sell, buy_base = np.array([], int), np.array([], int), {}
    for i in range(1, T):
        r = float(w @ R[i])
        if 1 + r > 0:
            w = w * (1 + R[i]) / (1 + r)
        cost = 0.0
        for j in to_sell:
            cost += w[j] * C.COST_STOCK
            trades.append(P[i, j] / entry_px[j] - 1 - 2 * C.COST_STOCK)
            w[j], held[j], entry_px[j], base[j] = 0.0, False, np.nan, np.nan
        for j in to_buy:
            w[j], held[j], entry_px[j], base[j] = 1.0 / R4.SLOTS, True, P[i, j], buy_base[j]
            cost += C.COST_STOCK / R4.SLOTS
        daily[i] = r - cost
        # manage open positions at today's close
        out = []
        for j in np.flatnonzero(held):
            c = P[i, j]
            if np.isnan(c):
                out.append(j); continue
            closes = P[max(0, i - cdays + 1):i + 1, j]
            row = lv[max(0, i - cdays + 1), j]
            up = row[(row > base[j]) & (row < closes.min())] if np.any(~np.isnan(row)) else []
            if len(up):
                base[j] = float(np.max(up))
            if c < base[j] * (1 - TOL) or (not np.isnan(lower[i, j]) and c < lower[i, j] * (1 - TOL)):
                out.append(j)
        to_sell = np.array(out, int)
        free = R4.SLOTS - (held.sum() - len(to_sell))
        elig = D["univ"][i] & D["divok"][i] & ~held & X["trend"][i] & ~np.isnan(P[i])
        if rand_p is None:
            sig = elig & ent[i]
            bb = base0[i]
        else:
            sig = elig & (RNG.random(k) < rand_p)
            row = lv[i]
            below = np.where(row < P[i][:, None], row, np.nan)
            bb = np.nanmax(np.c_[below, np.where(lower[i] < P[i], lower[i], np.nan)], 1)
            bb = np.where(np.isnan(bb), P[i] * 0.93, bb)
        cand = np.flatnonzero(sig)
        nsig += len(cand)
        if free > 0 and len(cand):
            to_buy = cand[np.argsort(-AV[i, cand])[:free]]
            buy_base = {j: float(bb[j]) for j in to_buy}
        else:
            to_buy, buy_base = np.array([], int), {}
    return pd.Series(daily, D["idx"]), np.array(trades), nsig


def build(px, D):
    raw = pd.read_parquet(C.DATA / "panel_rawclose.parquet").reindex(px.index)[px.columns]
    adj = (px / raw).values
    hi = pd.read_parquet(C.DATA / "panel_high.parquet").reindex(px.index)[px.columns].values * adj
    lo = pd.read_parquet(C.DATA / "panel_low.parquet").reindex(px.index)[px.columns].values * adj
    P = px.values
    T, N = P.shape
    used = np.flatnonzero(D["univ"].any(0))
    lv = np.full((T, N, NLV), np.nan)
    lower = np.full((T, N), np.nan)
    slope = np.full((T, N), np.nan)
    for j in used:
        lv[:, j], lower[:, j], slope[:, j] = stock_levels(P[:, j], hi[:, j], lo[:, j])
    trend = (slope > 0) & (P >= lower * (1 - TOL))
    X = {"lv": lv, "lower": lower, "trend": np.nan_to_num(trend, nan=0).astype(bool)}
    for cd in (1, 2):
        E = {e: (np.zeros((T, N), bool), np.full((T, N), np.nan)) for e in ("breakout", "bounce", "both")}
        for j in used:
            brk, bb, bnc, nb = signals(P[:, j], lo[:, j], lv[:, j], lower[:, j], cd)
            E["breakout"][0][:, j], E["breakout"][1][:, j] = brk, bb
            E["bounce"][0][:, j], E["bounce"][1][:, j] = bnc, nb
            E["both"][0][:, j] = brk | bnc
            E["both"][1][:, j] = np.where(brk, bb, nb)
        for e in E:
            X[(e, cd)] = E[e]
    print(f"levels built for {len(used)} stocks")
    return X


def main():
    C.RESULTS.mkdir(exist_ok=True)
    px, val, div = R4.load()
    D = R4.prepare(px, val, div, B.load("SET"))
    X = build(px, D)
    bm = R4.bench(D)
    periods = dict(B.PERIODS)
    allp = list(periods.items()) + [("holdout", (C.HOLDOUT_START, "2100"))]
    rows, hold, passed, yrs = [], [], [], []
    for e, cd in GRID:
        v = f"{e}_c{cd}"
        d, tr, nsig = simulate(D, X, e, cd)
        res = {}
        for p, (s, en) in allp:
            st, bs = B.curve_stats(d[s:en]), B.curve_stats(bm[s:en])
            row = dict(market="SET_r8", variant=v, period=p, **st,
                       bench_cagr=bs["cagr"], bench_sharpe=bs["sharpe"])
            (hold if p == "holdout" else rows).append(row)
            res[p] = row
        ok = (all(res[p]["cagr"] > res[p]["bench_cagr"] and res[p]["sharpe"] > res[p]["bench_sharpe"]
                  for p in periods) and res["validate"]["cagr"] >= R4.MIN_VALIDATE_CAGR)
        if ok:
            elig = max(1, int((D["univ"] & D["divok"] & X["trend"]).sum()))
            sims = {p: [] for p in periods}
            for _ in range(100):
                dr, _, _ = simulate(D, X, e, cd, rand_p=nsig / elig)
                for p, (s, en) in periods.items():
                    sims[p].append(B.curve_stats(dr[s:en])["cagr"])
            for p in periods:
                res[p]["rand_cagr_median"] = float(np.nanmedian(sims[p]))
                res[p]["p_value"] = float(np.mean(np.array(sims[p]) >= res[p]["cagr"]))
            ok = all(res[p]["p_value"] < C.MAX_P for p in periods)
        if ok:
            passed.append(dict(market="SET_r8", variant=v))
        pf = B.pf(tr) if len(tr) else np.nan
        hold_days = np.nan
        for p in periods:
            res[p].update(trades=len(tr), win=float(np.mean(tr > 0)) if len(tr) else np.nan, pf=pf,
                          avg_win=float(tr[tr > 0].mean()) if (tr > 0).any() else np.nan,
                          avg_loss=float(tr[tr <= 0].mean()) if (tr <= 0).any() else np.nan)
        print(f"{v:12s} {'PASS' if ok else 'fail'}  trades {len(tr):4d}  win {np.mean(tr > 0):.2f}  PF {pf:.2f}  "
              f"avgW {res['search']['avg_win']:+.3f} avgL {res['search']['avg_loss']:+.3f}  "
              + "  ".join(f"{p}: {res[p]['cagr']:+.3f} dd {res[p]['maxdd']:.2f}" for p in periods))
        dd_ = d[:C.VALIDATE[1]]
        for y, g in dd_.groupby(dd_.index.year):
            yrs.append(dict(variant=v, year=y, ret=float((1 + g).prod() - 1)))
    for y, g in bm[:C.VALIDATE[1]].groupby(bm[:C.VALIDATE[1]].index.year):
        yrs.append(dict(variant="benchmark_50", year=y, ret=float((1 + g).prod() - 1)))
    pd.DataFrame(rows).to_csv(C.RESULTS / "grid_r8.csv", index=False, float_format="%.4f")
    pd.DataFrame(hold).to_csv(C.RESULTS / "_sealed_holdout_r8.csv", index=False, float_format="%.4f")
    (C.RESULTS / "passed_r8.json").write_text(json.dumps(passed, indent=1))
    yr = pd.DataFrame(yrs).pivot(index="year", columns="variant", values="ret")
    yr.to_csv(C.RESULTS / "years_r8.csv", float_format="%.4f")
    print("\nyearly returns:\n", yr.loc[2008:].round(3).to_string())
    b = {p: B.curve_stats(bm[s:e]) for p, (s, e) in periods.items()}
    print("\nbenchmark:", "  ".join(f"{p}: {b[p]['cagr']:+.3f}" for p in periods))
    print(f"Passed {len(passed)} of {len(GRID)} variants")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
