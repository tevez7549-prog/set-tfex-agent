"""Technical scan over all liquid SET stocks (no dividend filter).

Signals per stock (raw prices, same as TradingView):
  owner method  : break of resistance / bounce off support (regression, key lines, Fibo)
  EMA stack     : close > EMA20 > EMA50 > EMA100 > EMA200
  golden cross  : EMA50 crossed above EMA200 in the last 5 days
  52w high      : close at a new 52-week closing high
  volume surge  : today's volume >= 2x the 20-day average
  MACD cross    : MACD crossed above its signal in the last 3 days
  RSI oversold  : RSI14 < 35 while close is above EMA200 (pullback in an uptrend)
Output -> site/data/tech.json
"""
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import scan as S

SIG = {
    "breakout": "เบรกแนวต้าน",
    "bounce": "ชนแนวรับแล้วยืน",
    "ema_stack": "EMA เรียงตัวขาขึ้น",
    "golden": "Golden cross 50/200",
    "high52": "ทำ high ใหม่ 52 สัปดาห์",
    "volume": "Volume พุ่ง ≥2 เท่า",
    "macd": "MACD ตัดขึ้น",
    "rsi_dip": "RSI ต่ำในขาขึ้น",
    "near_sup": "ใกล้แนวรับ",
    "near_res": "ใกล้แนวต้าน",
}
BULL = ["breakout", "bounce", "ema_stack", "golden", "high52", "volume", "macd", "rsi_dip"]


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def r(x, nd=3):
    try:
        x = float(x)
        return round(x, nd) if np.isfinite(x) else None
    except Exception:
        return None


def one(sym, d, name, value):
    c, v = d["Close"], d["Volume"]
    a = S.analyse(d)
    e = {n: c.ewm(span=n, adjust=False).mean() for n in (20, 50, 100, 200)}
    px = float(c.iloc[-1])
    sig = []
    if a["tag"] == "BREAKOUT":
        sig.append("breakout")
    if a["tag"] == "BOUNCE":
        sig.append("bounce")
    if a["tag"] == "NEAR_SUP":
        sig.append("near_sup")
    if a["tag"] == "NEAR_RES":
        sig.append("near_res")
    if px > e[20].iloc[-1] > e[50].iloc[-1] > e[100].iloc[-1] > e[200].iloc[-1]:
        sig.append("ema_stack")
    diff = e[50] - e[200]
    if (diff.iloc[-1] > 0) and (diff.iloc[-6:-1] <= 0).any():
        sig.append("golden")
    if len(c) > 250 and px >= c.iloc[-250:].max():
        sig.append("high52")
    vr = float(v.iloc[-1] / v.iloc[-21:-1].mean()) if v.iloc[-21:-1].mean() > 0 else None
    if vr and vr >= 2 and c.iloc[-1] > c.iloc[-2]:
        sig.append("volume")
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ms = macd - macd.ewm(span=9, adjust=False).mean()
    if ms.iloc[-1] > 0 and (ms.iloc[-4:-1] <= 0).any():
        sig.append("macd")
    rs = rsi(c)
    if rs.iloc[-1] < 35 and px > e[200].iloc[-1]:
        sig.append("rsi_dip")
    score = sum(1 for s in sig if s in BULL) + (1 if a["trend_ok"] else 0)
    return dict(
        sym=sym, name=name, close=r(px), value_mb=r(value / 1e6, 1),
        chg1d=r(px / c.iloc[-2] - 1, 4), chg5d=r(px / c.iloc[-6] - 1, 4), chg1m=r(px / c.iloc[-22] - 1, 4),
        rsi=r(rs.iloc[-1], 1), vol_ratio=r(vr, 2), ema200_gap=r(px / e[200].iloc[-1] - 1, 4),
        trend_ok=a["trend_ok"], signals=sig, score=score, detail=a["signals"],
        support=None if not a["support"] else [r(a["support"][0]), a["support"][1]],
        resistance=None if not a["resistance"] else [r(a["resistance"][0]), a["resistance"][1]],
        stop=r(a["stop"]), rr=r(a["rr"], 2),
        ema={k: r(e[k].iloc[-1]) for k in e},
        hist=[r(x) for x in c.iloc[-130:].values], hist_start=str(c.index[-130].date()),
    )


def run(data, meta, out_dir, min_value_mb=10):
    rows = []
    for s, d in data.items():
        val = float((d["Close"] * d["Volume"]).tail(60).mean())
        if val < min_value_mb * 1e6 or len(d) < 260:
            continue
        try:
            rows.append(one(s, d, (meta.get(s) or {}).get("name") or s, val))
        except Exception as e:
            print("tech skip", s, e)
    rows.sort(key=lambda x: (-x["score"], -(x["value_mb"] or 0)))
    out = dict(generated=datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="minutes"),
               price_date=max(str(d.index[-1].date()) for d in data.values()),
               min_value_mb=min_value_mb, labels=SIG, bull=BULL, stocks=rows)
    (out_dir / "tech.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"tech: {len(rows)} stocks, with breakout/bounce {sum(1 for x in rows if set(x['signals']) & {'breakout', 'bounce'})}")
    return out
