#!/usr/bin/env python3
"""Create a training-only sequence archive for causal discovery."""
import argparse
from pathlib import Path

import numpy as np
from ecommerce_sim.nts_adapter import load_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-train-only/customer_sequences.npz'))
    parser.add_argument('--train-weeks', type=int, default=52)
    args = parser.parse_args()
    sequences, features = load_sequences(args.input)
    if args.train_weeks < 3 or any(len(s) < args.train_weeks for s in sequences):
        parser.error('invalid training window length')
    train = [s[:args.train_weeks] for s in sequences]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, values=np.concatenate(train),
                        offsets=np.cumsum([0] + [len(s) for s in train]), features=features)
    print(f'saved {len(train)} customer training sequences of {args.train_weeks} weeks')


if __name__ == '__main__':
    main()
