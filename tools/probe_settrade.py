import urllib.request, os
for u in ["https://open-api.settrade.com/sdk-open-api/sdk-version.json", "https://open-api.settrade.com/"]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "python-requests/2.32"}), timeout=30)
        b = r.read().decode("utf-8", "replace")
        print("==", r.status, u, r.headers.get("Content-Type"), len(b)); print(b[:500].replace("\n", " "))
    except Exception as e:
        print("==", "ERR", u, e)
