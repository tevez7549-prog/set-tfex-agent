"""Stock scanner with a small agent pipeline (runs daily on GitHub Actions).

  1. Screener     : SET stocks, top by traded value, paid dividends 2 years in a row,
                    chart method (regression / key lines / Fibo), EMA20/50/100/200,
                    normal dividend yield (special dividends removed)
  2. Financials   : income statement, balance sheet, cash flow from Yahoo ->
                    profit trend, debt trend, payout, dividend cover -> flags
  3. News         : Claude + web search (needs ANTHROPIC_API_KEY secret);
                    without a key falls back to Yahoo headlines only
  4. Presenter    : combines all checks into a verdict and a short view

Output -> site/data/latest.json (+ history/<date>.json, dates.json)

    python agent/pipeline.py                 # full run
    EXTRA=AOT,CPALL python agent/pipeline.py # also scan these symbols
"""
import json
import os
import re
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scan as S  # noqa: E402

OUT = ROOT / "site" / "data"
CFG = json.loads((ROOT / "agent" / "settings.json").read_text())
BKK = timezone(timedelta(hours=7))


def f(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        return round(float(x), nd)
    except Exception:
        return None


# ------------------------------------------------------------------ 1. screener
def dividend_view(d, px, eps=None):
    """Trailing-12m dividend vs the usual yearly amount -> normal yield + special flag.

    A jump vs the last 3 years alone is not enough (banks raised payouts a lot),
    so with EPS known a dividend counts as special only if it also exceeds earnings.
    Normal amount is then capped at max(usual, 70% of EPS)."""
    dv = d["Dividends"] if "Dividends" in d else pd.Series(0.0, d.index)
    dv = dv[dv > 0]
    last = d.index[-1]
    t12 = float(dv[dv.index > last - pd.DateOffset(years=1)].sum())
    prev = [float(dv[(dv.index <= last - pd.DateOffset(years=k)) &
                     (dv.index > last - pd.DateOffset(years=k + 1))].sum()) for k in (1, 2, 3)]
    prev_pos = [p for p in prev if p > 0]
    usual = float(np.median(prev_pos)) if prev_pos else t12
    jump = bool(prev_pos and t12 > 1.4 * usual)
    if eps and eps > 0:
        special = jump and t12 > eps
        normal = min(t12, max(usual, 0.7 * eps)) if special else t12
    else:
        special = jump
        normal = min(t12, usual) if special else t12
    years = {}
    for y, v in dv.groupby(dv.index.year).sum().items():
        if y >= last.year - 6:
            years[str(y)] = f(v, 3)
    return dict(dps_12m=f(t12, 3), eps=f(eps, 3), payout_12m=f(t12 / eps) if eps and eps > 0 else None, dps_usual=f(usual, 3), dy_12m=f(t12 / px),
                dy_normal=f(normal / px), special=special, paid_2y=bool(t12 > 0 and prev[0] > 0),
                by_year=years, last=[(str(i.date()), f(v, 3)) for i, v in dv.tail(4).items()])


def screener():
    with urllib.request.urlopen(S.LIST_URL, timeout=60) as r:
        meta = {x["symbol"]: x for x in json.load(r) if x.get("market") == "SET"}
    extra = [s.strip().upper() for s in (os.getenv("EXTRA", "") + "," + ",".join(CFG["watchlist"])).split(",") if s.strip()]
    syms = sorted(set(meta) | set(extra))
    start = (pd.Timestamp.today() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    data = S.download_all(syms, start)
    print("downloaded", len(data))
    rows = []
    for s, d in data.items():
        rows.append((s, float((d["Close"] * d["Volume"]).tail(60).mean())))
    tv = pd.Series(dict(rows)).sort_values(ascending=False)
    universe = list(tv.head(CFG["universe_top_value"]).index)
    universe += [s for s in extra if s in data and s not in universe]
    out = []
    for s in universe:
        d = data[s]
        try:
            a = S.analyse(d)
        except Exception as e:
            print("skip", s, e); continue
        px = a["close"]
        dvv = dividend_view(d, px)
        if dvv["special"] or (dvv["dy_12m"] or 0) >= CFG["min_div_yield"] - 0.01:
            dvv = dividend_view(d, px, trailing_eps(s))
        c = d["Close"]
        chg = {k: f(px / c.iloc[-n] - 1) if len(c) > n else None for k, n in (("1m", 21), ("3m", 63), ("1y", 245))}
        hist = [[str(i.date()), f(v, 3)] for i, v in c.tail(260).items()]
        reg = a["reg"]
        rec = dict(
            sym=s, name=meta.get(s, {}).get("name") or s, date=str(a["date"]), close=f(px, 3),
            value_mb=f(tv[s] / 1e6, 1), chg=chg, tag=a["tag"], signals=a["signals"],
            ema={k: f(a[f"ema{k}"], 3) for k in (20, 50, 100, 200)}, ema_ok=a["ema_ok"],
            trend_ok=a["trend_ok"],
            reg=None if not reg else dict(start=str(reg["start"]), slope_yr=f(reg["slope_yr"]),
                                          lower=f(reg["lower"], 3), mid=f(reg["mid"], 3), upper=f(reg["upper"], 3)),
            support=None if not a["support"] else [f(a["support"][0], 3), a["support"][1]],
            resistance=None if not a["resistance"] else [f(a["resistance"][0], 3), a["resistance"][1]],
            stop=f(a["stop"], 3), rr=f(a["rr"], 2), lines=a["lines"], fib=a["fib"],
            div=dvv, watch=s in extra, hist=hist)
        rec["pass_screen"], rec["screen_notes"] = screen_rules(rec)
        out.append(rec)
    return out, market_regime(start)


def trailing_eps(sym):
    try:
        return yf.Ticker(f"{sym}.BK").info.get("trailingEps")
    except Exception:
        return None


def screen_rules(r):
    notes, ok = [], True
    if not r["div"]["paid_2y"]:
        ok = False; notes.append("ไม่ได้จ่ายปันผลต่อเนื่อง 2 ปี")
    if r["div"]["dy_normal"] is None or r["div"]["dy_normal"] < CFG["min_div_yield"]:
        ok = False; notes.append(f"ปันผลปกติ {fmt_pct(r['div']['dy_normal'])} ต่ำกว่า {CFG['min_div_yield']:.0%}")
    if CFG["require_ema_stack"] and not r["ema_ok"]:
        ok = False; notes.append("ราคายังไม่อยู่เหนือ EMA 20/50/100/200 ครบ")
    if r["div"]["special"]:
        notes.append(f"ปันผล 12 เดือน {fmt_pct(r['div']['dy_12m'])} มีปันผลพิเศษ ใช้ปกติ {fmt_pct(r['div']['dy_normal'])}")
    if ok:
        notes.append("ผ่านเกณฑ์ปันผล + EMA")
    return ok, notes


def market_regime(start):
    try:
        mk = yf.download("TDEX.BK", start=start, progress=False, auto_adjust=False)
        if isinstance(mk.columns, pd.MultiIndex):
            mk.columns = mk.columns.get_level_values(0)
        c = mk["Close"].dropna()
        ma = c.rolling(200).mean().iloc[-1]
        return dict(name="TDEX (SET50)", close=f(c.iloc[-1], 3), ma200=f(ma, 3), above=bool(c.iloc[-1] > ma),
                    chg_1m=f(c.iloc[-1] / c.iloc[-21] - 1))
    except Exception as e:
        return dict(error=str(e))


def fmt_pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


# ------------------------------------------------------------------ 2. financials
def series(df, names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return None


def financials(r):
    t = yf.Ticker(f"{r['sym']}.BK")
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    sector = info.get("sector") or ""
    bank = "Financial" in sector
    res = dict(sector=sector, industry=info.get("industry"), name_en=info.get("longName") or info.get("shortName"),
               pe=f(info.get("trailingPE"), 2), fpe=f(info.get("forwardPE"), 2), pbv=f(info.get("priceToBook"), 2),
               roe=f(info.get("returnOnEquity")), margin=f(info.get("profitMargins")),
               payout=f(info.get("payoutRatio")), mcap_b=f((info.get("marketCap") or 0) / 1e9, 1),
               summary=(info.get("longBusinessSummary") or "")[:600])
    flags = []
    try:
        inc, bs, cf, q = t.income_stmt, t.balance_sheet, t.cashflow, t.quarterly_income_stmt
    except Exception as e:
        res["flags"] = [["warn", f"ดึงงบไม่ได้: {e}"]]
        return res
    ni = series(inc, ["Net Income Common Stockholders", "Net Income"])
    rev = series(inc, ["Total Revenue", "Operating Revenue"])
    qni = series(q, ["Net Income Common Stockholders", "Net Income"])
    debt = series(bs, ["Total Debt"])
    eq = series(bs, ["Stockholders Equity", "Common Stock Equity"])
    fcf = series(cf, ["Free Cash Flow"])
    paid = series(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"])
    yearly = {}
    for name, s in (("revenue", rev), ("net_income", ni), ("debt", debt), ("equity", eq), ("fcf", fcf), ("div_paid", paid)):
        if s is not None:
            for k, v in s.tail(4).items():
                yearly.setdefault(str(k.year), {})[name] = f(v / 1e6, 1)
    res["yearly_mb"] = dict(sorted(yearly.items()))
    if qni is not None:
        res["quarters_mb"] = {str(k.date()): f(v / 1e6, 1) for k, v in qni.tail(5).items()}

    def growth(s):
        return None if s is None or len(s) < 2 or s.iloc[-2] <= 0 else s.iloc[-1] / s.iloc[-2] - 1

    g = growth(ni)
    res["ni_growth"] = f(g)
    if g is not None:
        flags.append(["good" if g > 0.05 else "bad" if g < -0.15 else "warn", f"กำไรสุทธิปีล่าสุด {g:+.0%} จากปีก่อน"])
    if qni is not None and len(qni) >= 5 and qni.iloc[-5] > 0:
        qy = qni.iloc[-1] / qni.iloc[-5] - 1
        res["q_yoy"] = f(qy)
        flags.append(["good" if qy > 0.05 else "bad" if qy < -0.2 else "warn",
                      f"กำไรไตรมาสล่าสุด ({qni.index[-1].date()}) {qy:+.0%} YoY"])
    if ni is not None and len(ni) >= 3 and (ni.tail(3) <= 0).any():
        flags.append(["bad", "มีปีที่ขาดทุนใน 3 ปีหลัง"])
    if not bank and debt is not None and eq is not None and len(debt) >= 2:
        de = debt.iloc[-1] / eq.iloc[-1] if eq.iloc[-1] > 0 else None
        de0 = debt.iloc[-2] / eq.iloc[-2] if eq.iloc[-2] > 0 else None
        res["de"] = f(de, 2)
        if de is not None:
            lvl = "bad" if de > 2 else "warn" if de > 1 or (de0 and de > de0 * 1.25) else "good"
            flags.append([lvl, f"หนี้สินมีดอกเบี้ย/ทุน {de:.2f} เท่า" + (f" (ปีก่อน {de0:.2f})" if de0 else "")])
    if not bank and fcf is not None and paid is not None:
        last = fcf.index[-1]
        if last in paid.index and paid[last] != 0:
            cover = fcf[last] / abs(paid[last])
            res["fcf_cover"] = f(cover, 2)
            flags.append(["good" if cover >= 1.2 else "warn" if cover >= 0.8 else "bad",
                          f"กระแสเงินสดอิสระ ครอบคลุมปันผล {cover:.1f} เท่า"])
    if res["payout"] is not None:
        p = res["payout"]
        flags.append(["bad" if p > 1 else "warn" if p > 0.8 else "good", f"สัดส่วนจ่ายปันผล {p:.0%} ของกำไร"])
    if res["roe"] is not None:
        flags.append(["good" if res["roe"] >= 0.1 else "warn", f"ROE {res['roe']:.0%}"])
    if r["div"]["special"]:
        flags.append(["warn", "ปันผล 12 เดือนรวมปันผลพิเศษ ปันผลปกติต่ำกว่าที่เห็น"])
    res["flags"] = flags
    return res


# ------------------------------------------------------------------ 3. news
NEWS_PROMPT = """คุณเป็นผู้ช่วยนักลงทุนหุ้นปันผลไทย ค้นข่าวของหุ้น {sym} ({name}) ตลาดหลักทรัพย์แห่งประเทศไทย
วันนี้ {today} ให้ค้นข่าว 30 วันล่าสุดทั้งภาษาไทยและอังกฤษ (ผลประกอบการ, ปันผล/XD, ราคาเป้าหมายโบรก, ดีล, คดี, การเมือง/เศรษฐกิจที่กระทบ)

ข้อมูลที่ระบบคำนวณแล้ว (ใช้ประกอบมุมมอง ห้ามแต่งตัวเลขใหม่):
{context}

ตอบเป็น JSON อย่างเดียว ไม่มีข้อความอื่น รูปแบบ:
{{"news":[{{"date":"YYYY-MM-DD","title":"หัวข้อสั้น ภาษาไทย","impact":"positive|neutral|negative","url":"..."}}],
"targets":[{{"broker":"...","target":ตัวเลข,"date":"YYYY-MM-DD"}}],
"events":["เหตุการณ์ที่จะมาถึง เช่น วันประกาศงบ วัน XD"],
"risks":["ความเสี่ยงสำคัญ สั้นๆ"],
"sentiment":"positive|neutral|negative",
"verdict":"น่าสนใจ|รอจังหวะ|ระวัง",
"view":"มุมมอง 2-3 ประโยค ภาษาไทย รวมพื้นฐาน กราฟ ข่าว"}}
ข่าวไม่เกิน 6 ข่าว ใส่เฉพาะที่มีแหล่งจริง ถ้าไม่พบให้เป็น list ว่าง"""


def claude_news(r, fin):
    import anthropic
    client = anthropic.Anthropic()
    ctx = dict(close=r["close"], date=r["date"], tag=r["tag"], signals=r["signals"],
               support=r["support"], resistance=r["resistance"], stop=r["stop"],
               ema_ok=r["ema_ok"], div_normal=r["div"]["dy_normal"], div_12m=r["div"]["dy_12m"],
               special_dividend=r["div"]["special"], pe=fin.get("pe"), pbv=fin.get("pbv"), roe=fin.get("roe"),
               flags=[x[1] for x in fin.get("flags", [])])
    msgs = [{"role": "user", "content": NEWS_PROMPT.format(
        sym=r["sym"], name=r["name"], today=datetime.now(BKK).date(), context=json.dumps(ctx, ensure_ascii=False))}]
    sources, text = [], ""
    for _ in range(3):
        resp = client.messages.create(
            model=CFG["claude_model"], max_tokens=2500, messages=msgs,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": CFG["news_searches"]}])
        for b in resp.content:
            if b.type == "text":
                text += b.text
            elif b.type == "web_search_tool_result" and isinstance(b.content, list):
                for it in b.content:
                    u = getattr(it, "url", None)
                    if u:
                        sources.append(dict(title=getattr(it, "title", u), url=u, age=getattr(it, "page_age", None)))
        if resp.stop_reason != "pause_turn":
            break
        msgs = msgs + [{"role": "assistant", "content": resp.content}]
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group(0)) if m else {"view": text[:500]}
    seen, uniq = set(), []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"]); uniq.append(s)
    out["sources"] = uniq[:12]
    out["engine"] = "claude"
    return out


def yahoo_news(sym):
    items = []
    try:
        for n in (yf.Ticker(f"{sym}.BK").news or [])[:6]:
            c = n.get("content", n)
            url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
            items.append(dict(date=(c.get("pubDate") or "")[:10], title=c.get("title"), impact="neutral", url=url))
    except Exception:
        pass
    return dict(news=items, targets=[], events=[], risks=[], sentiment=None, verdict=None, view=None,
                sources=[], engine="yahoo")


def news_agent(r, fin, cache):
    key = r["sym"]
    old = cache.get(key)
    if old and old.get("engine") == "claude" and old.get("fetched", "") >= str(datetime.now(BKK).date() - timedelta(days=CFG["news_cache_days"] - 1)):
        old["cached"] = True
        return old
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            n = claude_news(r, fin)
        except Exception as e:
            print("claude news failed", key, e)
            n = yahoo_news(key); n["error"] = str(e)[:200]
    else:
        n = yahoo_news(key)
    n["fetched"] = str(datetime.now(BKK).date())
    cache[key] = n
    return n


# ------------------------------------------------------------------ 4. presenter
RANK = {"น่าสนใจ": 0, "รอจังหวะ": 1, "ระวัง": 2}


def presenter(r, fin, news):
    score, why = 0, []
    if r["tag"] in ("BREAKOUT", "BOUNCE"):
        score += 2; why.append("มีสัญญาณเข้า " + ("เบรกแนวต้าน" if r["tag"] == "BREAKOUT" else "ชนแนวรับแล้วยืนได้"))
    elif r["tag"] == "NEAR_SUP":
        score += 1; why.append("ใกล้แนวรับ รอดูการยืน")
    elif r["tag"] == "NEAR_RES":
        why.append("ใกล้แนวต้าน รอเบรก")
    else:
        score -= 1; why.append("ราคาอยู่กลางทาง ไม่มีจุดเข้าชัด")
    if r["trend_ok"]:
        score += 1
    if r["rr"] and r["rr"] >= 2:
        score += 1; why.append(f"R:R {r['rr']:.1f}")
    fl = fin.get("flags", [])
    bad = sum(1 for x in fl if x[0] == "bad"); good = sum(1 for x in fl if x[0] == "good")
    score += min(good, 3) * 0.5 - bad * 1.5
    if bad:
        why.append("งบมีจุดน่ากังวล " + "; ".join(x[1] for x in fl if x[0] == "bad"))
    if r["div"]["special"]:
        score -= 1
    if news.get("sentiment") == "negative":
        score -= 1.5; why.append("ข่าวช่วงนี้เป็นลบ")
    elif news.get("sentiment") == "positive":
        score += 0.5
    if not r["pass_screen"]:
        score -= 2
    verdict = "น่าสนใจ" if score >= 2.5 else "รอจังหวะ" if score >= 0.5 else "ระวัง"
    llm = news.get("verdict")
    final = verdict
    if llm in RANK and RANK[llm] > RANK[verdict]:
        final = llm; why.append(f"agent ข่าวให้ {llm}")
    plan = {}
    if r["resistance"]:
        plan["breakout"] = r["resistance"][0]
    if r["support"]:
        plan["bounce"] = r["support"][0]
    plan["stop"] = r["stop"]
    return dict(score=round(score, 2), rule_verdict=verdict, verdict=final, why=why, plan=plan)


# ------------------------------------------------------------------ run
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p = OUT / "news_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    stocks, market = screener()
    log = [f"screener: วิเคราะห์ {len(stocks)} ตัว ผ่านเกณฑ์ {sum(s['pass_screen'] for s in stocks)} ตัว"]
    deep = [s for s in stocks if s["pass_screen"] or s["watch"] or
            (s["tag"] in ("BREAKOUT", "BOUNCE") and (s["div"]["dy_normal"] or 0) >= CFG["min_div_yield"] - 0.01)]
    deep.sort(key=lambda s: (not s["pass_screen"], -(s["div"]["dy_normal"] or 0)))
    deep = deep[:CFG["max_deep"]]
    for s in deep:
        steps = [dict(agent="คัดกรอง", ok=s["pass_screen"], note="; ".join(s["screen_notes"]))]
        try:
            fin = financials(s)
        except Exception as e:
            traceback.print_exc(); fin = {"flags": [["warn", f"ดึงงบไม่ได้ {e}"]]}
        nb = sum(1 for x in fin.get("flags", []) if x[0] == "bad")
        steps.append(dict(agent="เช็คงบ", ok=nb == 0, note=f"{len(fin.get('flags', []))} ประเด็น, น่ากังวล {nb}"))
        news = news_agent(s, fin, cache)
        steps.append(dict(agent="เช็คข่าว", ok=news.get("sentiment") != "negative",
                          note=f"{len(news.get('news', []))} ข่าว ({'Claude + web search' if news.get('engine') == 'claude' else 'Yahoo หัวข้อข่าว'})"
                               + (" แคช" if news.get("cached") else "")))
        pres = presenter(s, fin, news)
        steps.append(dict(agent="สรุป", ok=pres["verdict"] != "ระวัง", note=pres["verdict"]))
        s.update(fin=fin, news=news, view=pres, steps=steps, deep=True)
        print(s["sym"], pres["verdict"], pres["score"])
    log.append(f"เจาะลึก {len(deep)} ตัว")
    for s in stocks:
        if not s.get("deep"):
            s.pop("hist", None)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False))
    now = datetime.now(BKK)
    data = dict(generated=now.isoformat(timespec="minutes"), price_date=max(s["date"] for s in stocks),
                market=market, settings={k: CFG[k] for k in ("min_div_yield", "require_ema_stack", "universe_top_value")},
                news_engine="claude" if os.getenv("ANTHROPIC_API_KEY") else "yahoo",
                log=log, runtime_s=int(time.time() - t0), stocks=stocks)
    body = json.dumps(data, ensure_ascii=False, default=str)
    (OUT / "latest.json").write_text(body)
    (OUT / "history").mkdir(exist_ok=True)
    (OUT / "history" / f"{data['price_date']}.json").write_text(body)
    dates = sorted((p.stem for p in (OUT / "history").glob("*.json")), reverse=True)
    (OUT / "dates.json").write_text(json.dumps(dates))
    print("\n".join(log), "runtime", data["runtime_s"], "s")


if __name__ == "__main__":
    main()
