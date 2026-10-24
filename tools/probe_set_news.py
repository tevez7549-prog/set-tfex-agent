"""Plain HTTP check: can SET news be read without a bot challenge? No bypass attempts."""
import urllib.request, datetime as dt
td = dt.date.today(); fd = td - dt.timedelta(days=30)
urls = [
 f"https://www.set.or.th/api/set/news/search?symbol=PTT&fromDate={fd:%d/%m/%Y}&toDate={td:%d/%m/%Y}&keyword=&lang=th",
 "https://www.set.or.th/th/market/product/stock/quote/PTT/news",
 "https://www.set.or.th/api/set/stock/PTT/info?lang=th",
]
for u in urls:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (personal stock scanner)", "Accept": "application/json,text/html"}), timeout=30)
        b = r.read().decode("utf-8", "replace")
        print("==", r.status, u, len(b)); print(b[:800].replace("\n", " "))
    except Exception as e:
        print("==", "ERR", u, e)
