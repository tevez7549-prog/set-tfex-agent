"""Backtest 3 ideas for the SET + TFEX agent.

A. S50 futures   (proxy: SET50 index, daily)        -> trend / mean-reversion grid
B. Gold futures  (proxy: COMEX gold x USDTHB, daily) -> same grid
C. Thai stocks   (~SET100, adjusted closes)         -> monthly momentum grid

Rules for honesty:
- Signals at the daily close, trades at the NEXT day's open (A/B) or close (C).
- The full grid is fixed here, before any run, and every variant is reported.
- Holdout (last ~12 months) is NOT printed here. holdout.py opens it once,
  only for variants that pass SEARCH and VALIDATE.
- Every passing variant is compared with random entries.

    python backtest.py        (reads data/*.csv, writes results/*)
"""
import json

import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(7)
PERIODS = {"search": C.SEARCH, "validate": C.VALIDATE}


# ----------------------------------------------------------------- helpers
def load(name):
    df = pd.read_csv(C.DATA / f"{name}.csv", index_col="Date", parse_dates=True)
    df = df[~df.index.duplicated()].sort_index()
    # Some index feeds have broken opens: fall back to previous close
    bad = (df["Open"] <= 0) | df["Open"].isna()
    df.loc[bad, "Open"] = df["Close"].shift(1)[bad]
    return df.dropna(subset=["Open", "Close"])


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def pf(rets):
    rets = np.asarray(rets)
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return np.inf if l == 0 and g > 0 else (g / l if l > 0 else np.nan)


def curve_stats(daily):
    daily = daily.dropna()
    if len(daily) < 20:
        return dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan)
    eq = (1 + daily).cumprod()
    years = len(daily) / 245
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    sd = daily.std()
    sharpe = daily.mean() / sd * np.sqrt(245) if sd > 0 else np.nan
    return dict(cagr=cagr, maxdd=dd, sharpe=sharpe)


# ----------------------------------------------------- A/B: signal makers
def sig_donchian(df, n, m, short):
    hh = df["High"].rolling(n).max().shift(1).values
    ll = df["Low"].rolling(n).min().shift(1).values
    xl = df["Low"].rolling(m).min().shift(1).values
    xh = df["High"].rolling(m).max().shift(1).values
    c = df["Close"].values
    pos = np.zeros(len(df))
    p = 0
    for i in range(len(df)):
        if p == 1 and c[i] < xl[i]:
            p = 0
        elif p == -1 and c[i] > xh[i]:
            p = 0
        if p == 0:
            if c[i] > hh[i]:
                p = 1
            elif short and c[i] < ll[i]:
                p = -1
        pos[i] = p
    return pd.Series(pos, df.index)


def sig_ma(df, f, s, short):
    mf = df["Close"].rolling(f).mean()
    ms = df["Close"].rolling(s).mean()
    pos = np.where(mf > ms, 1, -1 if short else 0)
    pos[ms.isna().values] = 0
    return pd.Series(pos, df.index, dtype=float)


def sig_rsi2(df, short):
    cs = df["Close"]
    r = rsi(cs, 2).values
    ma200 = cs.rolling(200).mean().values
    ma5 = cs.rolling(5).mean().values
    c = cs.values
    pos = np.zeros(len(df))
    p = 0
    for i in range(len(df)):
        if p == 1 and c[i] > ma5[i]:
            p = 0
        elif p == -1 and c[i] < ma5[i]:
            p = 0
        if p == 0 and not np.isnan(ma200[i]):
            if r[i] < 10 and c[i] > ma200[i]:
                p = 1
            elif short and r[i] > 90 and c[i] < ma200[i]:
                p = -1
        pos[i] = p
    return pd.Series(pos, df.index)


GRID_AB = []
for n, m in [(20, 10), (20, 20), (55, 20), (55, 10)]:
    for short in (False, True):
        GRID_AB.append((f"donchian_{n}_{m}_{'LS' if short else 'L'}",
                        lambda df, n=n, m=m, s=short: sig_donchian(df, n, m, s)))
for f, s in [(20, 100), (50, 200)]:
    for short in (False, True):
        GRID_AB.append((f"ma_{f}_{s}_{'LS' if short else 'L'}",
                        lambda df, f=f, s_=s, sh=short: sig_ma(df, f, s_, sh)))
for short in (False, True):
    GRID_AB.append((f"rsi2_{'LS' if short else 'L'}",
                    lambda df, sh=short: sig_rsi2(df, sh)))


# ------------------------------------------------- A/B: execution engine
def run_positions(df, target, cost):
    """target[t] decided at close t, executed at open t+1."""
    o, c = df["Open"].values, df["Close"].values
    tgt = target.values
    daily = np.zeros(len(df))
    trades = []
    pos, entry_px, entry_i = 0, None, None
    for i in range(1, len(df)):
        want = tgt[i - 1]
        r = 0.0
        if want != pos:
            # close old position at today's open
            if pos != 0:
                r += pos * (o[i] / c[i - 1] - 1) - cost
                trades.append(dict(entry=df.index[entry_i], exit=df.index[i], dir=pos,
                                   ret=pos * (o[i] / entry_px - 1) - 2 * cost,
                                   bars=i - entry_i))
            if want != 0:
                r += want * (c[i] / o[i] - 1) - cost
                entry_px, entry_i = o[i], i
            pos = want
        elif pos != 0:
            r = pos * (c[i] / c[i - 1] - 1)
        daily[i] = r
    return pd.Series(daily, df.index), pd.DataFrame(trades)


def random_pf(df, trades, cost, start, end, sims=1000):
    """PF of random entries with the same count, directions and hold lengths."""
    idx = df.index
    lo, hi = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end))
    o = df["Open"].values
    bars = trades["bars"].values
    dirs = trades["dir"].values
    out = []
    for _ in range(sims):
        rets = []
        for b, d in zip(bars, dirs):
            if hi - lo - b - 1 <= 1:
                continue
            e = RNG.integers(lo + 1, hi - b)
            rets.append(d * (o[e + b] / o[e] - 1) - 2 * cost)
        out.append(pf(rets))
    return np.array(out)


def test_ab(name, df, cost):
    rows, hold = [], []
    for vname, fn in GRID_AB:
        daily, trades = run_positions(df, fn(df), cost)
        for per, (s, e) in list(PERIODS.items()) + [("holdout", (C.HOLDOUT_START, "2100-01-01"))]:
            t = trades[(trades["entry"] >= s) & (trades["entry"] <= e)] if len(trades) else trades
            st = curve_stats(daily[s:e])
            row = dict(market=name, variant=vname, period=per, trades=len(t),
                       pf=pf(t["ret"]) if len(t) else np.nan,
                       win=(t["ret"] > 0).mean() if len(t) else np.nan,
                       avg_ret=t["ret"].mean() if len(t) else np.nan,
                       exposure=(daily[s:e] != 0).mean(), **st)
            (hold if per == "holdout" else rows).append(row)
        # random baseline for each period (filled for all, used for passers)
        for per, (s, e) in PERIODS.items():
            t = trades[(trades["entry"] >= s) & (trades["entry"] <= e)] if len(trades) else trades
            if len(t) >= 5:
                rnd = random_pf(df, t, cost, s, e)
                real = pf(t["ret"])
                for r in rows:
                    if r["variant"] == vname and r["period"] == per:
                        r["rand_pf_median"] = float(np.nanmedian(rnd))
                        r["p_value"] = float(np.mean(rnd >= real))
    return pd.DataFrame(rows), pd.DataFrame(hold)


# ------------------------------------------------------ C: momentum stocks
def load_stocks():
    closes, vals = {}, {}
    for f in sorted((C.DATA / "stocks").glob("*.csv")):
        d = pd.read_csv(f, index_col="Date", parse_dates=True)
        d = d[~d.index.duplicated()].sort_index()
        closes[f.stem] = d["Close"]
        vals[f.stem] = d["Close"] * d["Volume"]
    return pd.DataFrame(closes), pd.DataFrame(vals)


GRID_C = [(L, N, regime) for L in (126, 252) for N in (5, 10) for regime in (False, True)]

# Universe rule for momentum. Round 1: fixed list + 20M baht/day floor.
# Point-in-time mode (backtest_pit.py): top_k most-traded stocks at each month-end.
MOM_UNIV = dict(top_k=None, min_val=20e6, market="SET_stocks")


def run_momentum(px, val, setidx, L, N, regime, cost, pick=None):
    P = px.values
    R = px.pct_change(fill_method=None).fillna(0.0).values
    AV = val.rolling(60, min_periods=40).mean().values
    present = px.notna().values
    # days of continuous history ending at each day
    run = np.zeros_like(P, dtype=int)
    for i in range(len(P)):
        run[i] = np.where(present[i], (run[i - 1] + 1) if i else 1, 0)
    ma200 = setidx["Close"].rolling(200).mean().reindex(px.index).ffill().values
    setc = setidx["Close"].reindex(px.index).ffill().values
    per = px.index.to_period("M")
    is_me = np.r_[per[1:] != per[:-1], True]

    k = P.shape[1]
    w = np.zeros(k)
    daily = np.zeros(len(P))
    pending = None
    for i in range(1, len(P)):
        r = float(w @ R[i])
        if 1 + r > 0:
            w = w * (1 + R[i]) / (1 + r)
        if pending is not None:
            r -= np.abs(pending - w).sum() * cost
            w = pending
            pending = None
        daily[i] = r
        if is_me[i] and i >= L + 21:
            score = P[i - 21] / P[i - L] - 1
            av = np.nan_to_num(AV[i], nan=0.0)
            ok = (run[i] > L) & (av >= MOM_UNIV["min_val"]) & ~np.isnan(score)
            if MOM_UNIV["top_k"]:
                cand = np.flatnonzero(ok)
                keep = cand[np.argsort(-av[cand])[:MOM_UNIV["top_k"]]]
                ok = np.zeros(k, bool)
                ok[keep] = True
            new = np.zeros(k)
            if regime and not (setc[i] > ma200[i]):
                pass  # cash
            elif ok.sum() >= max(N, 1):
                cand = np.flatnonzero(ok)
                if pick == "random":
                    names = RNG.choice(cand, N, replace=False)
                elif pick == "all":
                    names = cand
                else:
                    names = cand[np.argsort(-score[cand])[:N]]
                new[names] = 1.0 / len(names)
            pending = new
    return pd.Series(daily, px.index)


def test_c(px, val, setidx):
    rows, hold = [], []
    bench = run_momentum(px, val, setidx, 252, 0, False, C.COST_STOCK, pick="all")
    for L, N, regime in GRID_C:
        vname = f"mom_L{L}_top{N}_{'regime' if regime else 'always'}"
        daily = run_momentum(px, val, setidx, L, N, regime, C.COST_STOCK)
        for per, (s, e) in list(PERIODS.items()) + [("holdout", (C.HOLDOUT_START, "2100-01-01"))]:
            st, bs = curve_stats(daily[s:e]), curve_stats(bench[s:e])
            row = dict(market=MOM_UNIV["market"], variant=vname, period=per, **st,
                       bench_cagr=bs["cagr"], bench_sharpe=bs["sharpe"],
                       exposure=(daily[s:e] != 0).mean())
            (hold if per == "holdout" else rows).append(row)
    return pd.DataFrame(rows), pd.DataFrame(hold), bench


def random_momentum(px, val, setidx, L, N, regime, sims=100):
    out = {p: [] for p in PERIODS}
    for _ in range(sims):
        d = run_momentum(px, val, setidx, L, N, regime, C.COST_STOCK, pick="random")
        for p, (s, e) in PERIODS.items():
            out[p].append(curve_stats(d[s:e])["cagr"])
    return {p: np.array(v) for p, v in out.items()}


# --------------------------------------------------------------- main
def passes_ab(g):
    s = g[g.period == "search"].iloc[0]
    v = g[g.period == "validate"].iloc[0]
    # must also beat random entries (same trades/holds) in both periods
    return (s.trades >= C.MIN_TRADES and v.trades >= C.MIN_TRADES
            and s.pf > C.MIN_PF and v.pf > C.MIN_PF
            and s.get("p_value", 1) < C.MAX_P and v.get("p_value", 1) < C.MAX_P)


def passes_c(g):
    ok = True
    for p in PERIODS:
        r = g[g.period == p].iloc[0]
        ok &= (r.cagr > r.bench_cagr) and (r.sharpe > r.bench_sharpe)
    return bool(ok)


def beats_random_c(g):
    return all(g[g.period == p]["p_value"].iloc[0] < C.MAX_P for p in PERIODS)


def main():
    C.RESULTS.mkdir(exist_ok=True)
    grids, holds, passed = [], [], []

    for name, file, cost in [("S50", "SET50", C.COST_S50), ("GOLD_THB", "GOLD_THB", C.COST_GOLD)]:
        if not (C.DATA / f"{file}.csv").exists():
            print(f"{name}: no data file, skipped")
            continue
        df = load(file)
        g, h = test_ab(name, df, cost)
        grids.append(g)
        holds.append(h)
        for v, gg in g.groupby("variant"):
            if passes_ab(gg):
                passed.append(dict(market=name, variant=v))
        print(f"{name}: {len(GRID_AB)} variants, data {df.index[0].date()}..{df.index[-1].date()}")

    px, val = load_stocks()
    if px.shape[1] < 20 or not (C.DATA / "SET.csv").exists():
        print("SET stocks: not enough data, skipped")
        px = None
    if px is not None:
        setidx = load("SET")
        run_c(px, val, setidx, grids, holds, passed)
    finish(grids, holds, passed)


def run_c(px, val, setidx, grids, holds, passed):
    g, h, bench = test_c(px, val, setidx)
    for v, gg in g.groupby("variant"):
        if passes_c(gg):
            L, N, reg = int(v.split("_")[1][1:]), int(v.split("_")[2][3:]), v.endswith("regime")
            rnd = random_momentum(px, val, setidx, L, N, reg)
            for p in PERIODS:
                m = (g.variant == v) & (g.period == p)
                real = g.loc[m, "cagr"].iloc[0]
                g.loc[m, "rand_cagr_median"] = float(np.nanmedian(rnd[p]))
                g.loc[m, "p_value"] = float(np.mean(rnd[p] >= real))
            if beats_random_c(g[g.variant == v]):
                passed.append(dict(market=MOM_UNIV["market"], variant=v))
    grids.append(g)
    holds.append(h)
    print(f"SET stocks: {px.shape[1]} names, {len(GRID_C)} variants")


def finish(grids, holds, passed, tag=""):
    grid = pd.concat(grids, ignore_index=True)
    grid.to_csv(C.RESULTS / f"grid{tag}.csv", index=False, float_format="%.4f")
    # holdout is saved but sealed: holdout.py reads it only for passers
    pd.concat(holds, ignore_index=True).to_csv(C.RESULTS / f"_sealed_holdout{tag}.csv",
                                               index=False, float_format="%.4f")
    (C.RESULTS / f"passed{tag}.json").write_text(json.dumps(passed, indent=1))
    n_total = len(grid.groupby(["market", "variant"]))
    print(f"\nPassed {len(passed)} of {n_total} variants:")
    for p in passed:
        print(" ", p)


if __name__ == "__main__":
    main()
