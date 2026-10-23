#!/usr/bin/env python3
"""Download the pinned Udacity Starbucks promotion files without redistributing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


COMMIT = "6cc091fff66b8f441cbfeb5bb9e7ba70a7d74117"
BASE = f"https://raw.githubusercontent.com/udacity/DSND_Term2/{COMMIT}/portfolio_exercises/Experiments/data"
EXPECTED_COLUMNS = ["ID", "Promotion", "purchase", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]
FILES = {
    "training.csv": {
        "rows": 84_534,
        "bytes": 3_747_906,
        "sha256": "4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f",
    },
    "Test.csv": {
        "rows": 41_650,
        "bytes": 1_846_568,
        "sha256": "f5607ca0b2f6a6580ee2085659387b45735e81b41cad97c756ba068fcc06a625",
    },
}


def download(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        url = f"{BASE}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(f"source integrity check failed for {name}")
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        frame = pd.read_csv(temporary)
        if list(frame.columns) != EXPECTED_COLUMNS or len(frame) != expected["rows"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"schema/row check failed for {name}")
        os.replace(temporary, output / name)
        records.append({"name": name, "url": url, **expected})

    manifest = {
        "source_repository": "https://github.com/udacity/DSND_Term2",
        "source_commit": COMMIT,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "columns": EXPECTED_COLUMNS,
        "license_status": "Repository has MIT LICENSE and README states CC BY-NC-ND 4.0; dataset-specific redistribution grant not verified.",
        "redistribution": "Do not commit or redistribute raw or modified CSV files.",
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/starbucks_uplift"))
    parser.add_argument("--acknowledge-license-uncertainty", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_license_uncertainty:
        parser.error("review research/DATA_POLICY.md, then pass --acknowledge-license-uncertainty")
    download(args.output)
    print(f"verified Starbucks uplift files in {args.output}")


if __name__ == "__main__":
    main()

