#!/usr/bin/env python3
"""Small known-DAG sanity check; not a benchmark of real-world causality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecommerce_sim.nts_training import fit_nts_notears


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/nts-known-dag"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sequences = []
    for _ in range(50):
        sequence = np.zeros((50, 3))
        for t in range(1, len(sequence)):
            sequence[t, 0] = 0.6 * sequence[t - 1, 0] + rng.normal(scale=0.5)
            sequence[t, 1] = 1.4 * np.tanh(sequence[t - 1, 0]) + rng.normal(scale=0.4)
            sequence[t, 2] = 0.8 * sequence[t - 1, 1] + rng.normal(scale=0.4)
        sequences.append(sequence)
    path = args.output / "synthetic_sequences.npz"
    np.savez_compressed(path, values=np.concatenate(sequences), offsets=np.arange(51) * 50, features=np.array(["x0", "x1", "x2"]))
    result = fit_nts_notears(path, args.output, "external/NTS-NOTEARS", max_lag=1, hidden=4, lambda1=0.03, threshold=0.15, max_iter=12, seed=args.seed)
    truth = {("x0", "x0", 1), ("x0", "x1", 1), ("x1", "x2", 1)}
    estimated = {(edge["source"], edge["target"], edge["lag"]) for edge in result["edges"]}
    tp = len(truth & estimated)
    precision = tp / len(estimated) if estimated else 0.0
    recall = tp / len(truth)
    summary = {"truth": sorted(truth), "estimated": sorted(estimated), "true_positives": tp, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "converged": result["converged"], "instantaneous_is_dag": result["instantaneous_is_dag"], "seed": args.seed}
    (args.output / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
