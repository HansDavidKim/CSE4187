"""Bind frozen BiGRU prefix assignments to fitting rows without changing source IDs."""
from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

from .source_contract import DOMAINS, file_sha, utc_cutoff

EMBEDDING = tuple(f"z_{index}" for index in range(48))


def _user_text(values):
    def convert(value):
        if isinstance(value, str) and value.isdigit():
            return value
        if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0:
            return str(value)
        raise ValueError("frozen persona IDs must retain exact source integer/text IDs; floating IDs are unsafe")
    return values.map(convert)


def load_frozen_assignments(path, *, domain, cutoff_at, expected_sha256, source_sha256, persona_manifest_path):
    path, manifest_path = Path(path).resolve(), Path(persona_manifest_path).resolve()
    if domain not in DOMAINS or path.name not in {"assignments_inner.parquet", "assignments_outer.parquet"}:
        raise ValueError("explicit frozen persona domain and inner/outer assignment file are required")
    if file_sha(path) != expected_sha256:
        raise ValueError("frozen persona assignment SHA256 changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest["source_sha256"].get(domain) != source_sha256:
        raise ValueError("frozen persona manifest does not bind the prepared source")
    recorded_model = manifest["models"][domain]
    model_path = (manifest_path.parent / recorded_model["path"]).resolve()
    if file_sha(model_path) != recorded_model["sha256"]:
        raise ValueError("frozen persona model SHA256 changed")
    model = json.loads(model_path.read_text())
    frame = pd.read_parquet(path)
    required = {"user_id", "cluster", "persona", "split", "assignment_cutoff", "representation_end_at", *EMBEDDING}
    if frame.empty or not required <= set(frame):
        raise ValueError("frozen persona assignments require roles, strict prefix clocks and all 48 embedding coordinates")
    frame = frame.copy()
    frame["user_id"] = _user_text(frame.user_id)
    if frame.user_id.duplicated().any() or frame.cluster.isna().any() or frame.persona.isna().any():
        raise ValueError("frozen persona assignments require unique known source users and labels")
    cluster = pd.to_numeric(frame.cluster, errors="raise")
    if ((cluster < 0).any() or not cluster.eq(np.floor(cluster)).all()
            or not frame.persona.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError("frozen persona cluster/role labels changed")
    if "domain" in frame and not frame.domain.eq(domain).all():
        raise ValueError("frozen persona source domain changed")
    if "original_user_id" in frame and not _user_text(frame.original_user_id).eq(frame.user_id).all():
        raise ValueError("joint persona namespace must not replace original source IDs")
    cutoff = utc_cutoff(cutoff_at)
    for name in ("assignment_cutoff", "representation_end_at", "prefix_last_at"):
        if name in frame:
            if any(pd.Timestamp(value).tzinfo is None for value in frame[name]):
                raise ValueError("persona availability clocks must be timezone-aware")
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
            if frame[name].isna().any() or (name != "assignment_cutoff" and frame[name].ge(cutoff).any()):
                raise ValueError("persona representation must be strictly before its assignment cutoff")
    if not frame.assignment_cutoff.eq(cutoff).all() or not np.isfinite(frame[list(EMBEDDING)].to_numpy(float)).all():
        raise ValueError("persona cutoff or embedding coordinates changed")
    phase = "inner" if path.name == "assignments_inner.parquet" else "outer"
    seed = int(manifest["seed"])
    split = frame.user_id.map(lambda uid: "user_holdout" if hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] >= 205
                             else "inner_holdout" if hashlib.sha256(f"{seed}:inner:{uid}".encode()).digest()[0] >= 205 else "train")
    if not split.eq(frame.split).all():
        raise ValueError("frozen persona source user split/namespace changed")
    probability_columns = sorted(column for column in frame if column.startswith("prob_"))
    if probability_columns:
        probability = frame[probability_columns].to_numpy(float)
        if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1):
            raise ValueError("frozen persona membership probabilities changed")
    bindings = {str(path): expected_sha256, str(manifest_path): file_sha(manifest_path), str(model_path): recorded_model["sha256"]}
    if model.get("type") == "joint_retail_persona_alias":
        if model["source_domain"] != domain or model["assignment_user_id"] != "original source user ID":
            raise ValueError("joint persona alias source namespace changed")
        if utc_cutoff(model[f"{phase}_cutoff"]) != cutoff:
            raise ValueError("joint persona alias cutoff changed")
        shared_path = (model_path.parent / model["model"]).resolve()
        if file_sha(shared_path) != model["model_sha256"]:
            raise ValueError("shared frozen retail persona model changed")
        shared = json.loads(shared_path.read_text())
        shared_assignment = shared_path.parent / path.name
        if file_sha(shared_assignment) != shared["files"][path.name]:
            raise ValueError("shared frozen retail persona assignments changed")
        parent = pd.read_parquet(shared_assignment)
        parent = parent.loc[parent.domain.eq(domain)].copy()
        parent["original_user_id"] = _user_text(parent.original_user_id)
        parent = parent.set_index("original_user_id")
        if parent.index.duplicated().any() or set(parent.index) != set(frame.user_id):
            raise ValueError("joint persona alias does not retain the same original users")
        aligned = parent.loc[frame.user_id]
        columns = ["cluster", "persona", "split", "assignment_cutoff", "representation_end_at", *EMBEDDING, *probability_columns]
        try:
            pd.testing.assert_frame_equal(frame[columns].reset_index(drop=True), aligned[columns].reset_index(drop=True), check_dtype=False)
        except AssertionError as exc:
            raise ValueError("joint persona alias labels, availability or embeddings changed") from exc
        if "joint_namespace_user_id" not in frame or not _user_text(frame.joint_namespace_user_id).to_numpy().tolist() == _user_text(aligned.user_id).to_numpy().tolist():
            raise ValueError("joint persona alias namespace identity changed")
        bindings.update({str(shared_path): model["model_sha256"], str(shared_assignment): shared["files"][path.name]})
    elif model["files"].get(path.name) != expected_sha256 or utc_cutoff(model["assignment_cutoffs"][phase]) != cutoff:
        raise ValueError("frozen persona model does not bind this assignment file/cutoff")
    if any(file_sha(bound) != expected for bound, expected in bindings.items()):
        raise ValueError("frozen persona inputs changed during loading")
    provenance = {"domain": domain, "cutoff_at": cutoff.isoformat(), "phase": phase,
                  "prepared_source_sha256": source_sha256, "input_sha256": bindings,
                  "embedding_dimensions": 48, "user_id_rule": "original source user ID text; joint namespace retained only as a diagnostic",
                  "availability_rule": "representation and prefix observations strictly before cutoff; forecast actions at or after cutoff"}
    frame.attrs["frozen_persona_contract"] = provenance
    return frame, provenance


def attach_frozen_assignments(records, assignments, *, forecast=False):
    """Attach private labels to fit rows; unknown customers need an explicit fallback."""
    contract = assignments.attrs.get("frozen_persona_contract")
    if contract is None or "user_id" not in records:
        raise ValueError("use SHA-bound frozen assignments and source-keyed records")
    users = _user_text(records.user_id)
    labels = assignments.set_index("user_id")
    if not users.isin(labels.index).all():
        raise ValueError("source customers lack frozen prefix assignments; choose an explicit prior fallback")
    cutoff = utc_cutoff(contract["cutoff_at"])
    if forecast:
        if ("at" not in records or not isinstance(records["at"].dtype, pd.DatetimeTZDtype)
                or records["at"].isna().any() or records["at"].lt(cutoff).any()):
            raise ValueError("forecast persona assignments are unavailable before their explicit cutoff")
    result = records.copy()
    for original, attached in (("cluster", "persona_cluster"), ("persona", "persona_role"),
                               ("assignment_cutoff", "persona_assignment_cutoff"),
                               ("representation_end_at", "persona_representation_end_at")):
        if attached in result:
            raise ValueError("fit records already contain persona assignment columns")
        result[attached] = users.map(labels[original])
    result.attrs["frozen_persona_contract"] = {**contract, "usage": "forecast" if forecast else "offline fitting strata"}
    return result
