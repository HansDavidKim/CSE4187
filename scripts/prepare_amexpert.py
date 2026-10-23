#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_sim.preprocessing.amexpert import save_amexpert_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local AmExpert files and build user-week sequences")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/amexpert-panel"))
    args = parser.parse_args()
    save_amexpert_panel(args.data, args.output)
    print(f"saved boundary-safe AmExpert sequences to {args.output}")


if __name__ == "__main__":
    main()

