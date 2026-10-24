"""Plain HTTP check of SET's public news page (server-rendered). No bypass attempts."""
import re, urllib.request
u = "https://www.set.or.th/th/market/product/stock/quote/PTT/news"
b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (personal stock scanner)"}), timeout=30).read().decode("utf-8", "replace")
print("len", len(b))
for key in ["headline", "newsId", "datetime", "__NUXT__", "ประกาศ", "งบการเงิน"]:
    print(key, b.count(key))
for m in list(re.finditer(r"headline", b))[:5]:
    print("...", b[max(0, m.start() - 200): m.start() + 300].replace("\n", " "))
i = b.find("__NUXT__")
print("NUXT", b[i:i + 1500] if i >= 0 else "none")
