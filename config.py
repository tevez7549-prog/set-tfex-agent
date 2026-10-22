"""Shared settings for the SET + TFEX agent.

Everything uses paths relative to this folder, so the same code runs in
GitHub Actions now and on the Windows VPS later (no code changes).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# ---- Test periods (decided BEFORE looking at any result) ----
DATA_START = "2005-01-01"
SEARCH = ("2008-01-01", "2017-12-31")
VALIDATE = ("2018-01-01", "2025-10-21")
HOLDOUT_START = "2025-10-22"   # last ~12 months, opened once only

# ---- Yahoo symbols ----
# SET50 index is the proxy for S50 futures (TFEX).
SET50_CANDIDATES = ["^SET50.BK", "^SET50"]
SET_INDEX_CANDIDATES = ["^SET.BK", "^SETI"]
GOLD_USD = "GC=F"      # COMEX gold, continuous front month
USDTHB = "THB=X"       # baht per 1 USD

# Large / liquid Thai stocks (roughly SET100). Current names only, so there is
# survivorship bias: companies that dropped out or were delisted are missing.
STOCKS = """
ADVANC AOT AWC BANPU BBL BDMS BEM BH BJC BTS CBG CCET CENTEL COM7 CPALL CPF CPN
CRC DELTA EGCO GPSC GULF HMPRO IVL KBANK KKP KTB KTC LH MINT MTC OR OSP PTT
PTTEP PTTGC RATCH SCB SCC SCGP TCAP TIDLOR TISCO TLI TOP TRUE TTB TU WHA
AMATA AP BAM BCH BCP BCPG BGRIM BLA BTG CHG CK CKP DOHOME EA ERW GLOBAL GUNKUL
HANA IRPC ITC JMART JMT KCE M MEGA ORI PLANB PR9 QH SAWAD SIRI SJWD SPALI SPRC
STGT THANI TOA VGI WHAUP
""".split()

# ---- Costs (per side, as a fraction of notional) ----
# S50 futures: ~50 THB commission per contract + 1 tick slippage on ~180k notional
COST_S50 = 0.0004
# Gold futures (THB): commission + spread
COST_GOLD = 0.0006
# Thai stocks: ~0.157% commission + VAT + ~0.08% slippage
COST_STOCK = 0.0025

# Pass rules
MIN_TRADES = 30
MIN_PF = 1.15
MAX_P = 0.10   # must beat >= 90% of random-entry runs
