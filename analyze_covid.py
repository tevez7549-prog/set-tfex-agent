"""Look-only analysis: how much of the round-4 Donchian drawdown came from COVID?

This does NOT select a strategy (you cannot skip crashes when trading).
It re-runs the 12 round-4 variants and reports, for 2018-01-01..2025-10-21:
  full       all days
  ex-covid   COVID window removed (returns on those days set to 0, as if
             the account had been frozen)
  covid      only the COVID window
Also the worst drawdown per calendar year. Holdout is not touched.

    python analyze_covid.py   ->  results/covid_r4.csv, results/covid_r4_years.csv
"""
import numpy as np
import pandas as pd

import backtest as B
import backtest_r4 as R4
import config as C

COVID = ("2020-01-20", "2020-06-30")
VAL = C.VALIDATE


def dd(daily):
    eq = (1 + daily.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def stats(daily):
    s = B.curve_stats(daily)
    return dict(cagr=s["cagr"], maxdd=dd(daily), sharpe=s["sharpe"],
                total=float((1 + daily.fillna(0)).prod() - 1))


def main():
    px, val, div = R4.load()
    D = R4.prepare(px, val, div, B.load("SET"))
    series = {"benchmark_50": R4.bench(D)}
    for n, m, f, reg in R4.GRID:
        d, _, _ = R4.simulate(D, n, m, f, reg)
        series[f"don_{n}_{m}_{f}_{'regime' if reg else 'always'}"] = d
    rows, years = [], []
    for name, d in series.items():
        v = d[VAL[0]:VAL[1]]
        in_c = (v.index >= COVID[0]) & (v.index <= COVID[1])
        ex = v.where(~in_c, 0.0)
        full, exs, cv = stats(v), stats(ex), stats(v[in_c])
        rows.append(dict(variant=name,
                         full_cagr=full["cagr"], full_dd=full["maxdd"],
                         excovid_cagr=exs["cagr"], excovid_dd=exs["maxdd"], excovid_sharpe=exs["sharpe"],
                         covid_return=cv["total"], covid_dd=cv["maxdd"]))
        for y, g in d[:VAL[1]].groupby(d[:VAL[1]].index.year):
            years.append(dict(variant=name, year=y, ret=float((1 + g).prod() - 1), dd=dd(g)))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "covid_r4.csv", index=False, float_format="%.4f")
    yr = pd.DataFrame(years)
    yr.to_csv(C.RESULTS / "covid_r4_years.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    keep = ["benchmark_50", "don_100_50_div_always", "don_100_50_div_regime", "don_55_20_div_regime"]
    print("\nyearly return / worst drawdown inside the year")
    print(yr[yr.variant.isin(keep)].pivot(index="year", columns="variant", values=["ret", "dd"]).round(3).to_string())


if __name__ == "__main__":
    main()
