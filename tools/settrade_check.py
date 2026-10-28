"""Read-only check of the Settrade Open API sandbox: login, account info, portfolio.
No orders are placed. Keys come from GitHub Secrets (never from files in the repo)."""
import json, os
from settrade_v2 import Investor

import traceback
os.makedirs("results", exist_ok=True)
present = {k: bool(os.getenv(k)) for k in ["SETTRADE_APP_ID", "SETTRADE_APP_SECRET", "SETTRADE_BROKER_ID",
                                           "SETTRADE_APP_CODE", "SETTRADE_ACCOUNT_EQ", "SETTRADE_ACCOUNT_DV", "SETTRADE_PIN"]}
try:
    inv = Investor(app_id=os.environ["SETTRADE_APP_ID"], app_secret=os.environ["SETTRADE_APP_SECRET"],
               broker_id=os.getenv("SETTRADE_BROKER_ID") or "SANDBOX",
               app_code=os.getenv("SETTRADE_APP_CODE") or "SANDBOX", is_auto_queue=False)
except Exception as e:
    msg = traceback.format_exc()
    for k in ("SETTRADE_APP_ID", "SETTRADE_APP_SECRET"):
        if os.getenv(k):
            msg = msg.replace(os.environ[k], "***")
    json.dump({"secrets_present": present, "login_error": msg[-1500:]}, open("results/settrade_check.json", "w"), indent=1)
    print(msg); raise SystemExit(1)
out = {"secrets_present": present}
eq_acc = os.getenv("SETTRADE_ACCOUNT_EQ")
dv_acc = os.getenv("SETTRADE_ACCOUNT_DV")
if eq_acc:
    eq = inv.Equity(account_no=eq_acc)
    for name in ("get_account_info", "get_portfolios"):
        try:
            out[f"equity.{name}"] = getattr(eq, name)()
        except Exception as e:
            out[f"equity.{name}"] = f"error: {e}"
if dv_acc:
    dv = inv.Derivatives(account_no=dv_acc)
    for name in ("get_account_info", "get_portfolios"):
        try:
            out[f"derivatives.{name}"] = getattr(dv, name)()
        except Exception as e:
            out[f"derivatives.{name}"] = f"error: {e}"
try:
    out["quote.PTT"] = inv.MarketData().get_quote_symbol("PTT")
except Exception as e:
    out["quote.PTT"] = f"error: {e}"
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/settrade_check.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str)[:3000])
