from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from retention.data import load_canonical
from retention.pipeline import read_config, local_output


def test_csv_contract_projection_and_leading_zero_ids(tmp_path, records):
    for name, source in records.tables.items():
        frame = source.copy()
        frame["internal_future_gap"] = 12345
        # Both CSV and Parquet preserve string IDs; test the CSV spelling explicitly.
        for col in frame:
            if col.endswith("_id"):
                frame[col] = "000" + frame[col]
        frame.to_csv(tmp_path / (name + ".csv"), index=False)
    records = load_canonical(tmp_path)
    assert records["customers"].customer_id.str.startswith("000").all()
    assert all("internal_future_gap" not in t for t in records.tables.values())
    assert records.metadata["validation"]["counts"]["orders"] > 0


def test_training_requires_collection_assumption():
    with pytest.raises(ValueError, match="observation_assumption"):
        read_config(Path(__file__).resolve().parents[1] / "config.yaml")


def test_output_cannot_escape_retention(tmp_path):
    with pytest.raises(ValueError, match="inside retention"):
        local_output(tmp_path / "output")
