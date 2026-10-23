"""Open TFEX series pages in a headless browser and log the data APIs they call."""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [
    "https://www.tfex.co.th/en/products/equity/set50-index-futures/S50U25/historical-trading",
    "https://www.tfex.co.th/en/products/equity/set50-index-futures/S50Z25/historical-trading",
    "https://www.tfex.co.th/en/products/precious-metal/gold-online-futures/GOZ25/historical-trading",
]
out = Path("results/tfex_probe"); out.mkdir(parents=True, exist_ok=True)
log = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()

    def on_resp(r):
        if r.request.resource_type in ("xhr", "fetch"):
            item = {"url": r.url, "status": r.status}
            try:
                body = r.text()
                item["len"] = len(body)
                if body[:1] in "[{" and len(log) < 80:
                    (out / f"resp_{len(log)}.json").write_text(body[:200000])
                    item["saved"] = f"resp_{len(log)}.json"
            except Exception as e:
                item["err"] = str(e)
            log.append(item)
            print(item, flush=True)
    pg.on("response", on_resp)
    for u in PAGES:
        print("OPEN", u, flush=True)
        try:
            pg.goto(u, wait_until="networkidle", timeout=90000)
            pg.wait_for_timeout(4000)
            print("TITLE", pg.title(), flush=True)
            (out / (u.rstrip("/").split("/")[-2] + ".txt")).write_text(pg.inner_text("body")[:20000])
        except Exception as e:
            print("FAIL", u, e, flush=True)
    b.close()
(out / "requests.json").write_text(json.dumps(log, indent=1))
