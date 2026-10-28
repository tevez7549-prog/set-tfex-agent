"""Runs on the VPS runner: can the Settrade sandbox give historical candles? Writes results/vps/history_*."""
import csv, json, os
from datetime import datetime
from settrade_v2 import Investor

inv = Investor(app_id=os.environ["SETTRADE_APP_ID"], app_secret=os.environ["SETTRADE_APP_SECRET"],
               broker_id="SANDBOX", app_code="SANDBOX", is_auto_queue=False)
md = inv.MarketData()
out_dir = os.path.join("out")
os.makedirs(out_dir, exist_ok=True)
TESTS = [("PTT", "1d"), ("KBANK", "1d"), ("SET50", "1d"), ("SET", "1d"),
         ("S50Z26", "1d"), ("S50H27", "1d"), ("GOZ26", "1d"), ("GF10Z26", "1d"), ("USDZ26", "1d"),
         ("PTT", "1h"), ("S50Z26", "1h")]
summary = []
for sym, iv in TESTS:
    rec = dict(symbol=sym, interval=iv)
    for kw in ({"start": "2010-01-01T00:00", "limit": 5000}, {"limit": 5000}, {"limit": 1000}):
        try:
            r = md.get_candlestick(symbol=sym, interval=iv, **kw)
        except Exception as e:
            rec[f"err_{list(kw)}"] = str(e)[:200]
            continue
        t = r.get("time") or []
        rec["keys"] = list(r)[:12]
        if not t:
            continue
        rec.update(params=kw, bars=len(t), first=datetime.utcfromtimestamp(t[0]).isoformat(),
                   last=datetime.utcfromtimestamp(t[-1]).isoformat(), last_close=r["close"][-1])
        cols = [k for k in ("time", "open", "high", "low", "close", "volume", "value") if k in r]
        with open(os.path.join(out_dir, f"{sym}_{iv}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for i in range(len(t)):
                w.writerow([r[c][i] for c in cols])
        break
    summary.append(rec)
    print(json.dumps(rec, default=str))
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
