"""Prepared caches bind fitted provenance to actual unchanged source rows."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from simulate.v4.irl import fit_donor_prior as prior
from simulate.v4.irl.fit_shopping import SOURCE_COLUMNS, load_source


@pytest.fixture
def recorded_source(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    path = folder / "2019-10.csv"
    rows = []
    for uid in (100, 200):
        for step, kind in enumerate(("view", "cart", "purchase")):
            rows.append(dict(event_time=f"2019-10-01T10:00:0{step}Z", event_type=kind,
                             product_id="p1", category_id="c1", category_code="", brand="", price="1.05",
                             user_id=str(uid), user_session=f"s{uid}"))
    pd.DataFrame(rows, columns=SOURCE_COLUMNS).to_csv(path, index=False)
    _, source = load_source(folder)
    return {"source": source}, path


def test_missing_cache_is_published_and_reused_from_exact_recorded_source(recorded_source, tmp_path):
    model, _ = recorded_source
    path = tmp_path / "prepared" / "source.parquet"
    rows, descriptor = prior.load_prior_source(model, path)
    assert path.is_file() and descriptor["cache_hit"] is False
    metadata = json.loads(pq.read_schema(path).metadata[b"quality_source_cache"])
    assert metadata["fingerprint"] == descriptor["fingerprint"]
    assert metadata["source_provenance"] == model["source"]
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    again, cached = prior.load_prior_source(model, path)
    pd.testing.assert_frame_equal(rows, again)
    assert cached["cache_hit"] is True
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_mismatched_cache_is_never_overwritten_or_hashed_as_the_new_source(recorded_source, tmp_path):
    model, _ = recorded_source
    path = tmp_path / "source.parquet"
    prior.load_prior_source(model, path)
    before = path.read_bytes()
    changed = deepcopy(model)
    changed["source"]["sample_modulus"] += 1
    with pytest.raises(ValueError, match="unused cache path"):
        prior.load_prior_source(changed, path)
    assert path.read_bytes() == before


def test_changed_raw_source_cannot_reuse_cache(recorded_source, tmp_path):
    model, raw = recorded_source
    path = tmp_path / "source.parquet"
    prior.load_prior_source(model, path)
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="original source file changed"):
        prior.load_prior_source(model, path)


def test_racing_cache_publication_preserves_other_writer_and_cleans_temporary(recorded_source, tmp_path, monkeypatch):
    model, _ = recorded_source
    path = tmp_path / "source.parquet"
    real_link = prior.os.link
    def race(source, target):
        Path(target).write_bytes(b"another writer")
        return real_link(source, target)
    monkeypatch.setattr(prior.os, "link", race)
    with pytest.raises(FileExistsError):
        prior.load_prior_source(model, path)
    assert path.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".source.parquet-*"))
