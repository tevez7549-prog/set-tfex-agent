"""Pull financial statements and dividend history for a list of Thai stocks.

    python tools/deep_dive.py BH PTT ...   ->  results/deep_dive.json
"""
import json
import sys

import numpy as np
import pandas as pd
import yfinance as yf

ROWS_IS = ["Total Revenue", "Operating Revenue", "Net Interest Income", "Gross Profit",
           "Operating Income", "Net Income", "Net Income Common Stockholders", "Diluted EPS", "Basic EPS"]
ROWS_BS = ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity",
           "Total Debt", "Net Debt", "Cash And Cash Equivalents"]
ROWS_CF = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Cash Dividends Paid"]


def table(df, rows):
    out = {}
    if df is None or df.empty:
        return out
    for r in rows:
        if r in df.index:
            s = df.loc[r]
            out[r] = {str(pd.Timestamp(k).date()): (None if pd.isna(v) else float(v)) for k, v in s.items()}
    return out


def main():
    res = {}
    for sym in sys.argv[1:]:
        t = yf.Ticker(f"{sym}.BK")
        d = {}
        try:
            info = t.info
            keep = ["shortName", "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                    "priceToBook", "returnOnEquity", "returnOnAssets", "profitMargins", "debtToEquity",
                    "payoutRatio", "dividendRate", "dividendYield", "trailingEps", "bookValue",
                    "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "revenueGrowth",
                    "earningsGrowth", "earningsQuarterlyGrowth", "longBusinessSummary"]
            d["info"] = {k: info.get(k) for k in keep}
        except Exception as e:
            d["info_err"] = str(e)
        for name, attr, rows in [("is", "income_stmt", ROWS_IS), ("bs", "balance_sheet", ROWS_BS),
                                 ("cf", "cashflow", ROWS_CF), ("qis", "quarterly_income_stmt", ROWS_IS)]:
            try:
                d[name] = table(getattr(t, attr), rows)
            except Exception as e:
                d[name + "_err"] = str(e)
        try:
            dv = t.dividends
            dv.index = pd.to_datetime(dv.index).tz_localize(None)
            d["div_by_year"] = {str(y): float(v) for y, v in dv.groupby(dv.index.year).sum().items() if y >= 2016}
            d["div_last"] = [(str(i.date()), float(v)) for i, v in dv.tail(6).items()]
        except Exception as e:
            d["div_err"] = str(e)
        try:
            h = t.history(period="5y", auto_adjust=False)
            h.index = pd.to_datetime(h.index).tz_localize(None)
            c = h["Close"]
            d["px"] = {"last": float(c.iloc[-1]), "date": str(c.index[-1].date()),
                       "chg_1y": float(c.iloc[-1] / c.iloc[-250] - 1) if len(c) > 250 else None,
                       "chg_3y": float(c.iloc[-1] / c.iloc[-750] - 1) if len(c) > 750 else None,
                       "hi_5y": float(c.max()), "lo_5y": float(c.min())}
        except Exception as e:
            d["px_err"] = str(e)
        res[sym] = d
        print(sym, "ok", list(d.keys()))
    with open("results/deep_dive.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
