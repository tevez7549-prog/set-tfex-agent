"""Round 2: momentum with a point-in-time universe (no 'today's winners' list).

At each month-end the universe is the 100 most-traded SET stocks
(60-day average traded value) that have enough history. That mimics SET100
as it was at the time, instead of using today's list.
Same 8 variants, same periods, same pass rules as round 1.

    python backtest_pit.py   ->  results/grid_pit.csv, passed_pit.json
    python holdout.py _pit
"""
import pandas as pd

import backtest as B
import config as C


def main():
    C.RESULTS.mkdir(exist_ok=True)
    px = pd.read_parquet(C.DATA / "panel_close.parquet").sort_index()
    val = pd.read_parquet(C.DATA / "panel_value.parquet").reindex(px.index)
    setidx = B.load("SET")
    B.MOM_UNIV.update(top_k=100, min_val=0, market="SET_pit100")
    grids, holds, passed = [], [], []
    B.run_c(px, val, setidx, grids, holds, passed)
    B.finish(grids, holds, passed, tag="_pit")


if __name__ == "__main__":
    main()
