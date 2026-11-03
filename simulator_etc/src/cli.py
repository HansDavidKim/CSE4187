from __future__ import annotations

import argparse
from pathlib import Path

from simulate.pipeline import EcommercePipeline, load_starbucks_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Intervention-aware e-commerce simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download the selected Kaggle dataset")
    fetch.add_argument("--dataset", default="blacktile/starbucks-app-customer-reward-program-data")

    uplift = sub.add_parser("uplift", help="fit the official Udacity promotion-response simulator")
    uplift.add_argument("--data", type=Path, default=Path("data/raw/starbucks_uplift"))
    uplift.add_argument("--output", type=Path, default=Path("artifacts/starbucks-uplift"))
    uplift.add_argument("--segments", type=int, default=6)
    uplift.add_argument("--synthetic-rows", type=int, default=100_000)

    run = sub.add_parser("run", help="fit discovery/GMM/IRL and generate a synthetic panel")
    run.add_argument("--data", type=Path, default=Path("data/raw/starbucks"))
    run.add_argument("--output", type=Path, default=Path("artifacts/starbucks-run"))
    run.add_argument("--max-users", type=int, default=2000)
    run.add_argument("--max-rows", type=int, default=5000)
    run.add_argument("--max-iter", type=int, default=10)
    run.add_argument("--method", choices=("notears-mlp", "correlation-smoke"), default="notears-mlp")
    run.add_argument("--synthetic-users", type=int, default=500)
    run.add_argument("--periods", type=int, default=30)

    args = parser.parse_args()
    if args.command == "fetch":
        import kagglehub

        print(kagglehub.dataset_download(args.dataset))
        return

    if args.command == "uplift":
        from simulate.uplift import StarbucksUpliftSimulator, load_uplift_data

        train, test = load_uplift_data(args.data)
        simulator = StarbucksUpliftSimulator(segments=args.segments).fit(train)
        simulator.save(args.output, test, args.synthetic_rows)
        print(f"saved uplift calibration and {args.synthetic_rows * 2} synthetic rows to {args.output}")
        return

    panel = load_starbucks_panel(args.data, max_users=args.max_users)
    pipeline = EcommercePipeline().fit(
        panel,
        discovery_method=args.method,
        max_iter=args.max_iter,
        max_rows=args.max_rows,
    )
    pipeline.save(args.output, args.synthetic_users, args.periods)
    print(f"saved {len(pipeline.discovery.edges)} DAG edges and synthetic panel to {args.output}")


if __name__ == "__main__":
    main()

