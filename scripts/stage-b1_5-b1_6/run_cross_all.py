#!/usr/bin/env python3
"""Parallel, resumable Stage B1.6 layer orchestrator."""

from joblib import Parallel, delayed

from cross_hop_b156 import LAYERS, consolidate, run_layer


if __name__ == "__main__":
    Parallel(n_jobs=8, backend="loky", verbose=10)(delayed(run_layer)(layer) for layer in LAYERS)
    consolidate()
