#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_sim.models.nts_training import fit_nts_notears


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit NTS-NOTEARS to boundary-safe AmExpert windows")
    parser.add_argument("--sequences", type=Path, default=Path("artifacts/amexpert-panel/customer_sequences.npz"))
    parser.add_argument("--source", type=Path, default=Path("external/NTS-NOTEARS"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/amexpert-nts"))
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--lambda1", type=float, default=0.005)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--drop-features", nargs="+", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preserve-binary", action="store_true")
    parser.add_argument("--forbid-instantaneous", action="store_true")
    parser.add_argument("--bootstrap-users", type=int)
    args = parser.parse_args()
    result = fit_nts_notears(
        args.sequences,
        args.output,
        args.source,
        max_lag=args.max_lag,
        hidden=args.hidden,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        threshold=args.threshold,
        max_iter=args.max_iter,
        max_sequences=args.max_sequences,
        drop_features=args.drop_features,
        seed=args.seed,
        preserve_binary=args.preserve_binary,
        forbid_instantaneous=args.forbid_instantaneous,
        bootstrap_users=args.bootstrap_users,
    )
    print(f"saved {len(result['edges'])} NTS-NOTEARS edges to {args.output}")


if __name__ == "__main__":
    main()
