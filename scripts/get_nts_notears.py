#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


URL = "https://github.com/xiangyu-sun-789/NTS-NOTEARS.git"
COMMIT = "e1ba34b784f9eec3d2139904589d2fd1ed4ea7ad"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the official pinned NTS-NOTEARS source")
    parser.add_argument("--output", type=Path, default=Path("external/NTS-NOTEARS"))
    args = parser.parse_args()
    if args.output.exists():
        head = subprocess.check_output(["git", "-C", str(args.output), "rev-parse", "HEAD"], text=True).strip()
        if head != COMMIT:
            raise RuntimeError(f"existing source is at {head}, expected {COMMIT}")
        print(f"official NTS-NOTEARS already pinned at {COMMIT}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", URL, str(args.output)], check=True)
    subprocess.run(["git", "-C", str(args.output), "checkout", COMMIT], check=True)
    print(f"checked out official NTS-NOTEARS at {COMMIT}")


if __name__ == "__main__":
    main()
