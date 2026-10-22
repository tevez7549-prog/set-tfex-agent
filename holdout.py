"""Open the holdout ONCE, only for variants that passed search + validate.

    python holdout.py
"""
import json
import sys

import pandas as pd

import config as C


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    passed = json.loads((C.RESULTS / f"passed{tag}.json").read_text())
    if not passed:
        print("No variant passed search + validate; holdout stays sealed.")
        (C.RESULTS / f"holdout_opened{tag}.csv").write_text("none passed\n")
        return
    h = pd.read_csv(C.RESULTS / f"_sealed_holdout{tag}.csv")
    keys = {(p["market"], p["variant"]) for p in passed}
    out = h[[(m, v) in keys for m, v in zip(h.market, h.variant)]]
    out.to_csv(C.RESULTS / f"holdout_opened{tag}.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
