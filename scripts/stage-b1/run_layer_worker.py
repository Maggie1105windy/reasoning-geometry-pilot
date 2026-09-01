#!/usr/bin/env python3
"""Run one resumable Stage B1 layer checkpoint."""

import sys

import pandas as pd

from probe_b1 import OUT, run_layer


def main() -> None:
    layer = int(sys.argv[1])
    union = pd.read_csv(OUT / "union_manifest.csv")
    union["computed_hop"] = union["computed_hop"].astype(int)
    run_layer(layer, union)


if __name__ == "__main__":
    main()

