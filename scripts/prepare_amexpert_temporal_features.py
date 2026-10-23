#!/usr/bin/env python3
"""Prepare role-separated AmExpert weekly features without running causal discovery."""
import argparse
import json
from pathlib import Path

import pandas as pd

from ecommerce_sim.amexpert_temporal_features import build_temporal_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-temporal-features-v2'))
    args = parser.parse_args()
    tx = pd.read_csv(args.data / 'customer_transaction_data.csv')
    campaigns = pd.read_csv(args.data / 'campaign_data.csv')
    assignments = pd.read_csv(args.data / 'train.csv')
    panel, report = build_temporal_features(tx, campaigns, assignments)
    args.output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output / 'local_customer_week_features.csv.gz', index=False)
    (args.output / 'feature_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
