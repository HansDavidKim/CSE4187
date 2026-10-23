#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_sim.causal_features import save_feature_sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract leakage-safe feature sets from an NTS DAG")
    parser.add_argument("--dag", type=Path, default=Path("artifacts/amexpert-nts/nts_dag.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/amexpert-nts/feature_sets.json"))
    parser.add_argument("--targets", nargs="+", default=["transactions", "spend", "redemptions_reported"])
    parser.add_argument(
        "--known-at-decision",
        nargs="*",
        default=["offers_started", "unique_coupons_started", "active_offers"],
    )
    args = parser.parse_args()
    result = save_feature_sets(args.dag, args.output, args.targets, set(args.known_at_decision))
    print(f"saved feature sets for {len(result['targets'])} targets to {args.output}")


if __name__ == "__main__":
    main()
