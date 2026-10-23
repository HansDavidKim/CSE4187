#!/usr/bin/env python3
"""Reconstruct and audit customer-by-campaign-validity-period data."""
import argparse
import json
from pathlib import Path

from ecommerce_sim.analysis.campaign_audit import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-campaign-audit'))
    args = parser.parse_args()
    periods, panel, report = reconstruct(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    periods.assign(campaign_ids=periods.campaign_ids.map(lambda ids: ','.join(map(str, ids)))).to_csv(args.output / 'periods.csv', index=False)
    panel.to_csv(args.output / 'local_customer_periods.csv.gz', index=False)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
