"""Read shared point-in-time predictors from the simulator's features.sqlite."""

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pandas as pd


SNAPSHOT_KEYS = ["dataset_id", "customer_id", "as_of_at", "snapshot_id"]


@contextmanager
def open_feature_store(path):
    """Open a materialized directory or SQLite file, checking its manifest when present."""
    path = Path(path)
    database = path / "features.sqlite" if path.is_dir() else path
    manifest = database.parent / "manifest.json"
    if manifest.is_file():
        expected = json.loads(manifest.read_text(encoding="utf-8"))["file_sha256"][database.name]
        digest = hashlib.sha256()
        with database.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SQLite checksum differs from {manifest}")
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        yield connection


def read_kit_features(connection, names):
    """Return snapshot keys and requested KIT39 v3 predictors in registry order."""
    registry = connection.execute(
        "SELECT feature_name, feature_set_version FROM feature_registry ORDER BY ordinal"
    ).fetchall()
    if len(registry) != 39 or any(version != "kit39-review-cart-v3" for _, version in registry):
        raise ValueError("Expected the simulator's KIT39 v3 feature registry")
    requested = set(names)
    selected = [name for name, _ in registry if name in requested]
    if set(selected) != requested or not all(name.isidentifier() for name in selected):
        raise ValueError("Requested predictors must be registered KIT39 columns")
    columns = ", ".join(f'f."{name}"' for name in [*SNAPSHOT_KEYS, *selected])
    return pd.read_sql_query(f"SELECT {columns} FROM features AS f", connection)
