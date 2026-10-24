"""Daily scan: owner's chart method on SET50-type dividend stocks.

Uses RAW (unadjusted) daily prices so levels match TradingView.
For each stock: regression channel from the start of the trend (lowest close
of the last 5 years), key lines (>= 3 swing touches, highs and lows, +-1.5%,
last 250 days), Fibonacci of the 250-day swing, then a status:
  BREAKOUT   closed above a line in the last 2 days
  BOUNCE     touched a support in the last 3 days, never closed > 3% below, closes above
  NEAR_SUP   within 3% above a support (watch)
  NEAR_RES   within 3% below a resistance (watch for breakout)
Plus basic fundamentals from Yahoo (may be missing).

    python scan.py   ->  results/scan_latest.md / .csv
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

import config as C

LIST_URL = ("https://raw.githubusercontent.com/lumduan/thai-securities-data/"
            "main/thai_securities_market_set_en.json")
TOUCH, TOL, LB, REG_LB, K = 0.015, 0.03, 250, 1250, 5


def download_all(symbols, start):
    data = {}
    for i in range(0, len(symbols), 40):
        chunk = [f"{s}.BK" for s in symbols[i:i + 40]]
        for a in range(3):
            try:
                df = yf.download(chunk, start=start, auto_adjust=False, actions=True,
                                 group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:
                print("retry", e); time.sleep(5)
        else:
            continue
        for t in chunk:
            if t in df.columns.get_level_values(0):
                d = df[t].dropna(subset=["Close"])
                if len(d) > 300:
                    d.index = pd.to_datetime(d.index).tz_localize(None)
                    data[t[:-3]] = d
        time.sleep(1)
    return data


def analyse(d):
    c, h, l = d["Close"].values, d["High"].values, d["Low"].values
    T = len(c)
    t = T - 1
    hw = pd.Series(h).rolling(2 * K + 1, center=True).max().values
    lw = pd.Series(l).rolling(2 * K + 1, center=True).min().values
    ph = np.flatnonzero(h == hw); pl = np.flatnonzero(l == lw)
    sh = ph[(ph >= t - LB) & (ph + K <= t)]; sl = pl[(pl >= t - LB) & (pl + K <= t)]
    prices = np.r_[h[sh], l[sl]]; typ = np.r_[np.ones(len(sh)), np.zeros(len(sl))]
    lines = []
    if len(prices) >= 3:
        near = np.abs(prices[:, None] / prices[None, :] - 1) <= TOUCH
        cnt = near.sum(1); both = (near * typ[None, :]).sum(1)
        for i in np.argsort(-cnt):
            if cnt[i] >= 3 and 1 <= both[i] <= cnt[i] - 1:
                lvl = float(np.median(prices[near[i]]))
                if all(abs(lvl / x[0] - 1) > 2 * TOUCH for x in lines):
                    lines.append((lvl, int(cnt[i])))
            if len(lines) == 5:
                break
    wh, wl = h[t - LB + 1:], l[t - LB + 1:]
    ih, il = int(np.argmax(wh)), int(np.argmin(wl))
    fib = {}
    if il < ih:
        hh, ll = wh[ih], wl[il]
        fib = {f"fib{r}": hh - r * (hh - ll) for r in (0.0, 0.236, 0.382, 0.5, 0.618)}
    logc = np.log(c)
    a = max(0, t - REG_LB)
    anc = a + int(np.argmin(logc[a:]))
    reg = None
    if t - anc >= 120:
        y = logc[anc:]; x = np.arange(len(y), dtype=float)
        b, a0 = np.polyfit(x, y, 1); sd = float(np.std(y - (a0 + b * x)))
        mid = a0 + b * x[-1]
        reg = dict(start=d.index[anc].date(), slope_yr=float(np.exp(b * 245) - 1),
                   lower=float(np.exp(mid - 2 * sd)), mid=float(np.exp(mid)), upper=float(np.exp(mid + 2 * sd)))
    levels = [(v, f"เส้นสำคัญ ({n} ครั้ง)") for v, n in lines]
    levels += [(v, k.replace("fib", "Fibo ")) for k, v in fib.items()]
    if reg:
        levels += [(reg["lower"], "ขอบล่าง Regression"), (reg["upper"], "ขอบบน Regression")]
    px = c[t]
    sup = sorted([x for x in levels if x[0] <= px * 1.0], key=lambda x: -x[0])
    res = sorted([x for x in levels if x[0] > px], key=lambda x: x[0])
    status, base = [], None
    for v, name in levels:                      # breakout in last 2 days
        if any(c[t - k - 1] <= v < c[t - k] for k in (0, 1)) and c[t] > v and "ขอบบน" not in name:
            status.append(f"เบรก {name} {v:.2f}"); base = max(base or 0, v)
    for v, name in sup:                          # bounce within 3 days
        if l[t - 2:t + 1].min() <= v * (1 + TOUCH) and c[t - 2:t + 1].min() >= v * (1 - TOL) \
                and c[t] >= v and c[t - 10] >= v * 1.02:
            status.append(f"ชนแนวรับ {name} {v:.2f}"); base = max(base or 0, v)
    tag = "BREAKOUT" if any(s.startswith("เบรก") for s in status) else \
          "BOUNCE" if status else None
    if not tag and sup and px <= sup[0][0] * (1 + TOL):
        tag = "NEAR_SUP"
    if not tag and res and px >= res[0][0] * (1 - TOL):
        tag = "NEAR_RES"
    trend_ok = bool(reg and reg["slope_yr"] > 0 and px >= reg["lower"] * (1 - TOL))
    stop_base = base or (sup[0][0] if sup else None)
    stop = stop_base * (1 - TOL) if stop_base else None
    target = res[0] if res else None
    rr = (target[0] - px) / (px - stop) if (target and stop and px > stop) else None
    return dict(close=px, date=d.index[t].date(), tag=tag, signals="; ".join(status),
                trend_ok=trend_ok, reg=reg, support=sup[0] if sup else None,
                resistance=target, stop=stop, rr=rr,
                lines=[round(v, 2) for v, _ in lines],
                fib={k: round(v, 2) for k, v in fib.items()})


def fundamentals(sym):
    try:
        info = yf.Ticker(f"{sym}.BK").info
        return dict(pe=info.get("trailingPE"), pbv=info.get("priceToBook"),
                    roe=info.get("returnOnEquity"), name=info.get("shortName"),
                    sector=info.get("sector"))
    except Exception:
        return {}


def main():
    C.RESULTS.mkdir(exist_ok=True)
    with urllib.request.urlopen(LIST_URL, timeout=60) as r:
        syms = sorted({x["symbol"] for x in json.load(r) if x.get("market") == "SET"})
    start = (pd.Timestamp.today() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    data = download_all(syms, start)
    print("downloaded", len(data))
    rows = []
    for s, d in data.items():
        tv = (d["Close"] * d["Volume"]).tail(60).mean()
        dv = d["Dividends"] if "Dividends" in d else pd.Series(0.0, d.index)
        last = d.index[-1]
        d1 = dv[dv.index > last - pd.DateOffset(years=1)].sum()
        d2 = dv[(dv.index <= last - pd.DateOffset(years=1)) & (dv.index > last - pd.DateOffset(years=2))].sum()
        rows.append(dict(sym=s, tv=tv, dy=d1 / d["Close"].iloc[-1], div_ok=bool(d1 > 0 and d2 > 0)))
    u = pd.DataFrame(rows).sort_values("tv", ascending=False)
    top = u.head(50)
    pick = top[top.div_ok]
    print(f"top50 by value, dividend 2y ok: {len(pick)}")
    mk = yf.download("TDEX.BK", start=start, progress=False, auto_adjust=False)
    if isinstance(mk.columns, pd.MultiIndex):
        mk.columns = mk.columns.get_level_values(0)
    ma200 = mk["Close"].rolling(200).mean().iloc[-1]
    market = f"TDEX (SET50) ปิด {mk['Close'].iloc[-1]:.2f} vs MA200 {ma200:.2f} → {'เหนือ' if mk['Close'].iloc[-1] > ma200 else 'ใต้'} MA200"
    out = []
    for _, r in pick.iterrows():
        try:
            a = analyse(data[r.sym])
        except Exception as e:
            print("skip", r.sym, e); continue
        a.update(sym=r.sym, dy=r.dy, tv_mb=r.tv / 1e6)
        if a["tag"]:
            a.update(fundamentals(r.sym))
        out.append(a)
    df = pd.DataFrame(out)
    order = {"BREAKOUT": 0, "BOUNCE": 1, "NEAR_SUP": 2, "NEAR_RES": 3}
    df["o"] = df.tag.map(order).fillna(9)
    df = df.sort_values(["o", "trend_ok", "rr"], ascending=[True, False, False])
    df.drop(columns="o").to_csv(C.RESULTS / "scan_latest.csv", index=False)
    lines = [f"# สแกน {df.date.max()}", "", market, "",
             f"หุ้นกลุ่ม SET50 (มูลค่าซื้อขายสูงสุด 50 ตัว) ที่จ่ายปันผลต่อเนื่อง 2 ปี: {len(pick)} ตัว", ""]
    for _, a in df.iterrows():
        reg = a.reg or {}
        sup = a.support; res = a.resistance
        lines.append(f"## {a.sym} [{a.tag or '-'}] ปิด {a.close:.2f}  trend {'OK' if a.trend_ok else 'NO'}")
        lines.append(f"- สัญญาณ: {a.signals or '-'}")
        if reg:
            lines.append(f"- Regression จาก {reg['start']}: ขึ้น {reg['slope_yr']:+.1%}/ปี, ล่าง {reg['lower']:.2f} กลาง {reg['mid']:.2f} บน {reg['upper']:.2f}")
        lines.append(f"- เส้นสำคัญ: {a.lines}  Fibo: {a.fib}")
        lines.append(f"- แนวรับใกล้สุด: {f'{sup[0]:.2f} ({sup[1]})' if sup else '-'} | แนวต้านถัดไป: {f'{res[0]:.2f} ({res[1]})' if res else '-'}")
        lines.append(f"- Stop: {a.stop:.2f}" if a.stop else "- Stop: -")
        lines.append(f"- R:R {a.rr:.1f}" if a.rr else "- R:R -")
        lines.append(f"- ปันผล 12 ด. {a.dy:.1%}, P/E {a.get('pe')}, P/BV {a.get('pbv')}, ROE {a.get('roe')}, มูลค่าซื้อขาย {a.tv_mb:.0f} ลบ./วัน")
        lines.append("")
    (C.RESULTS / "scan_latest.md").write_text("\n".join(lines))
    print("\n".join(lines[:6]))
    print(df[["sym", "tag", "close", "trend_ok", "signals", "stop", "rr", "dy"]].to_string(index=False))


if __name__ == "__main__":
    main()
