import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl.source_contract import file_sha
from simulate.v4.irl.source_personas import EMBEDDING, attach_frozen_assignments, load_frozen_assignments


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "persona"
    folder = root / "cosmetics"
    folder.mkdir(parents=True)
    cutoff = pd.Timestamp("2024-02-01T00:00:00Z")
    split = "user_holdout" if hashlib.sha256(b"42:100").digest()[0] >= 205 else (
        "inner_holdout" if hashlib.sha256(b"42:inner:100").digest()[0] >= 205 else "train")
    frame = pd.DataFrame([{"user_id": 100, "cluster": 0, "persona": "P0", "split": split,
                          "assignment_cutoff": cutoff, "representation_end_at": cutoff - pd.Timedelta(days=1),
                          "prefix_last_at": cutoff - pd.Timedelta(hours=1), "prob_0": 1.,
                          **{name: index / 100 for index, name in enumerate(EMBEDDING)}}])
    path = folder / "assignments_outer.parquet"
    frame.to_parquet(path, index=False)
    model_path = folder / "model.json"
    manifest_path = root / "manifest.json"

    def bind(updated=frame):
        updated.to_parquet(path, index=False)
        model_path.write_text(json.dumps({"files": {path.name: file_sha(path)}, "assignment_cutoffs": {"outer": cutoff.isoformat()}}))
        manifest_path.write_text(json.dumps({"status": "complete", "seed": 42, "source_sha256": {"cosmetics": "a" * 64},
            "models": {"cosmetics": {"path": "cosmetics/model.json", "sha256": file_sha(model_path)}}}))
        return dict(domain="cosmetics", cutoff_at=cutoff, expected_sha256=file_sha(path), source_sha256="a" * 64,
                    persona_manifest_path=manifest_path)
    return path, frame, bind(), bind


def test_frozen_assignment_preserves_embeddings_roles_ids_and_fit_row_order(frozen):
    path, expected, kwargs, _ = frozen
    frame, provenance = load_frozen_assignments(path, **kwargs)
    assert frame.user_id.tolist() == ["100"]
    np.testing.assert_array_equal(frame[list(EMBEDDING)], expected[list(EMBEDDING)])
    assert frame.persona.tolist() == ["P0"] and provenance["embedding_dimensions"] == 48
    records = pd.DataFrame({"user_id": ["100", "100"], "at": [kwargs["cutoff_at"], kwargs["cutoff_at"] + pd.Timedelta(days=1)]}, index=[8, 2])
    attached = attach_frozen_assignments(records, frame, forecast=True)
    assert attached.index.tolist() == [8, 2]
    pd.testing.assert_frame_equal(attached[records.columns], records)
    assert attached.persona_cluster.tolist() == [0, 0]
    assert attached.persona_role.tolist() == ["P0", "P0"]
    assert not any(name.startswith("z_") for name in attached)
    with pytest.raises(ValueError, match="unavailable before"):
        attach_frozen_assignments(records.assign(at=kwargs["cutoff_at"] - pd.Timedelta(nanoseconds=1)), frame, forecast=True)
    with pytest.raises(ValueError, match="unavailable before"):
        attach_frozen_assignments(records.assign(at=pd.Series(pd.NaT, index=records.index, dtype="datetime64[ns, UTC]")), frame, forecast=True)
    with pytest.raises(ValueError, match="explicit prior fallback"):
        attach_frozen_assignments(records.assign(user_id="200"), frame)


@pytest.mark.parametrize("mutation", ["future_representation", "naive_cutoff", "float_uid", "wrong_split", "embedding_nan"])
def test_assignment_validation_rejects_leakage_or_lossy_namespace(frozen, mutation):
    path, frame, kwargs, bind = frozen
    frame = frame.copy()
    if mutation == "future_representation":
        frame["representation_end_at"] = kwargs["cutoff_at"]
    elif mutation == "naive_cutoff":
        frame["assignment_cutoff"] = frame.assignment_cutoff.dt.tz_localize(None)
    elif mutation == "float_uid":
        frame["user_id"] = frame.user_id.astype(float)
    elif mutation == "wrong_split":
        frame["split"] = "wrong"
    else:
        frame["z_47"] = np.nan
    kwargs = bind(frame)
    with pytest.raises(ValueError):
        load_frozen_assignments(path, **kwargs)


def test_assignment_and_source_sha_are_bound_before_loading(frozen):
    path, _, kwargs, _ = frozen
    with pytest.raises(ValueError, match="SHA256 changed"):
        load_frozen_assignments(path, **{**kwargs, "expected_sha256": "0" * 64})
    with pytest.raises(ValueError, match="prepared source"):
        load_frozen_assignments(path, **{**kwargs, "source_sha256": "b" * 64})
