"""Open the holdout ONCE, only for variants that passed search + validate.

    python holdout.py
"""
import json

import pandas as pd

import config as C


def main():
    passed = json.loads((C.RESULTS / "passed.json").read_text())
    if not passed:
        print("No variant passed search + validate; holdout stays sealed.")
        (C.RESULTS / "holdout_opened.csv").write_text("none passed\n")
        return
    h = pd.read_csv(C.RESULTS / "_sealed_holdout.csv")
    keys = {(p["market"], p["variant"]) for p in passed}
    out = h[[(m, v) in keys for m, v in zip(h.market, h.variant)]]
    out.to_csv(C.RESULTS / "holdout_opened.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
