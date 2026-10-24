"""Show what the round-8 rules see on one stock: lines on given dates and every trade."""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")
import backtest_r8 as R8

sym = sys.argv[1] if len(sys.argv) > 1 else "TTB"
df = yf.download(f"{sym}.BK", start="2005-01-01", auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index).tz_localize(None)
c, h, l = df.Close.values, df.High.values, df.Low.values
lv, lower, slope = R8.stock_levels(c, h, l)
for d in ["2024-10-01", "2025-06-02", "2025-11-03", "2026-01-15", "2026-02-20"]:
    i = df.index.searchsorted(pd.Timestamp(d))
    keys = [x for x in lv[i] if not np.isnan(x)]
    print(f"{df.index[i].date()} close {c[i]:.2f} | lines {[round(x,2) for x in keys]} | lower band {lower[i]:.2f} | slope/yr {slope[i]*245:+.3f}")
for e, cd in [("breakout", 2), ("bounce", 1), ("both", 2)]:
    brk, bb, bnc, nb = R8.signals(c, l, lv, lower, cd)
    ent = {"breakout": brk, "bounce": bnc, "both": brk | bnc}[e]
    base0 = {"breakout": bb, "bounce": nb, "both": np.where(brk, bb, nb)}[e]
    trend = (slope > 0) & (c >= lower * 0.97)
    pos, out = None, []
    for i in range(1, len(c) - 1):
        if pos is None:
            if ent[i] and trend[i]:
                pos = dict(i=i + 1, px=c[i + 1], base=base0[i], first=base0[i])
            continue
        closes = c[max(0, i - cd + 1):i + 1]
        row = lv[max(0, i - cd + 1)]
        up = row[(row > pos["base"]) & (row < closes.min())]
        if len(up):
            pos["base"] = float(np.max(up))
        if c[i] < pos["base"] * 0.97 or (not np.isnan(lower[i]) and c[i] < lower[i] * 0.97):
            out.append((df.index[pos["i"]].date(), pos["px"], pos["first"], df.index[i + 1].date(), c[i + 1], pos["base"]))
            pos = None
    print(f"\n== {sym} {e} c{cd}: {len(out)} trades, last 12 ==")
    for t in out[-12:]:
        print(f"buy {t[0]} @{t[1]:.2f} base {t[2]:.2f} -> sell {t[3]} @{t[4]:.2f} (final base {t[5]:.2f})  {t[4]/t[1]-1:+.1%}")
    if pos:
        print(f"open: buy {df.index[pos['i']].date()} @{pos['px']:.2f} base now {pos['base']:.2f}, last {c[-1]:.2f}")
    r = np.array([t[4] / t[1] - 1 for t in out])
    if len(r):
        print(f"win {np.mean(r>0):.0%}  avg {r.mean():+.2%}  avg hold days {np.mean([(pd.Timestamp(t[3])-pd.Timestamp(t[0])).days for t in out]):.0f}")
