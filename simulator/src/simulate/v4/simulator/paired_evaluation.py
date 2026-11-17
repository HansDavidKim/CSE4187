"""Evaluation-only paired draws of the generator's conditional return marginals."""
from __future__ import annotations

import hashlib
import json
import numbers

import numpy as np
import pandas as pd


PAIRED_METHOD_VERSION = "paired-retention-equation-v1"
OUTCOME_DEFINITION = "future_no_session_30d_active30d"
_HORIZONS = {OUTCOME_DEFINITION: 30, "future_no_session_28d_active30d": 28}
_OUTCOME_TYPES = {(1, 1): "ST", (0, 1): "P", (1, 0): "SD", (0, 0): "LC"}


def _integer(value, name, upper):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value < upper:
        raise ValueError(f"{name} must be a nonnegative integer below {upper}")
    return int(value)


def paired_retention_rollouts(diagnostics: pd.DataFrame, metadata: dict,
                              repetitions: int = 32) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share new SHA-keyed uniforms between two conditional return marginals.

    This evaluates a latent generator equation. It does not replay the factual
    global RNG, regenerate shopping trajectories, or estimate a real coupon CATE.
    Both returned frames belong only in an ``evaluation_only`` destination.
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    if (isinstance(metadata.get("schema_version"), bool) or
            not isinstance(metadata.get("schema_version"), numbers.Integral) or metadata["schema_version"] != 7 or
            metadata.get("simulator_version") not in {f"source-shopping-v{version}" for version in range(2, 13)}):
        raise ValueError("paired equation evaluation requires schema 7 and source-shopping-v2 through v12")
    definition = metadata.get("definition_id")
    if not isinstance(definition, str) or definition not in _HORIZONS:
        raise ValueError("unsupported return outcome definition")
    days = _HORIZONS[definition]
    for name, expected in (("outcome_window_days", days), ("generation_window_days", 30), ("eligibility_window_days", 30)):
        declared = metadata.get(name, expected if days == 30 else None)
        if isinstance(declared, bool) or not isinstance(declared, numbers.Integral) or declared != expected:
            raise ValueError(f"{name} must explicitly match {expected}")
    p0_name, p1_name = f"no_coupon_return_probability_{days}d", f"coupon_return_probability_{days}d"
    scope = ("first_30_day_activity_equation" if days == 30 else
             "first_28_day_return_marginal_of_30d_activity_window")
    if days == 28 and (metadata.get("generation_definition_id") != OUTCOME_DEFINITION or
                       not isinstance(metadata.get("parent_dataset_id"), str) or not metadata["parent_dataset_id"].strip()):
        raise ValueError("28-day evaluation requires the original 30-day definition and parent dataset")
    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be explicit")
    seed = _integer(metadata.get("seed"), "seed", 2 ** 64)
    repetitions = _integer(repetitions, "repetitions", 2 ** 32)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    try:
        snapshot = pd.Timestamp(metadata["as_of_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("as_of_at must be timezone-aware") from exc
    if pd.isna(snapshot) or snapshot.tzinfo is None:
        raise ValueError("as_of_at must be timezone-aware")
    snapshot = snapshot.tz_convert("UTC")
    if not isinstance(diagnostics, pd.DataFrame) or diagnostics.empty or not {"customer_id", p0_name, p1_name} <= set(diagnostics):
        raise ValueError("nonempty diagnostics with customer_id and both return probabilities are required")
    if (not diagnostics.customer_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all() or
            diagnostics.customer_id.duplicated().any()):
        raise ValueError("customer_id must be a unique, nonempty string")
    for name in (p0_name, p1_name):
        if not diagnostics[name].map(lambda value: not isinstance(value, bool) and
                                    isinstance(value, numbers.Real) and np.isfinite(value) and 0 <= value <= 1).all():
            raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    if "response_type" in diagnostics and not diagnostics.response_type.isin(["ST", "P", "SD", "LC"]).all():
        raise ValueError("response_type must be ST, P, SD or LC when supplied")
    provenance = {}
    for key in ("model_sha256", "return_model_sha256", "simulator_code_sha256", "raw_database_sha256", "source_raw_database_sha256"):
        value = metadata.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64 or
                                  any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"{key} must be a lowercase SHA256 when supplied")
        provenance[key] = value
    pair_rows, truth_rows = [], []
    for row in diagnostics.sort_values("customer_id").to_dict("records"):
        customer_id = row["customer_id"]
        key = json.dumps([PAIRED_METHOD_VERSION, dataset_id, snapshot.isoformat(), seed, customer_id],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        common = {"customer_id": customer_id, "dataset_id": dataset_id, "as_of_at": snapshot,
                  "snapshot_id": f"{customer_id}:{snapshot.isoformat()}", "actual_experiment_seed": seed,
                  "paired_derived_seed": derived_seed, "paired_method_version": PAIRED_METHOD_VERSION,
                  "simulator_version": metadata["simulator_version"], "outcome_definition_id": definition,
                  "outcome_window_days": days, "generation_window_days": 30, "eligibility_window_days": 30,
                  "parent_dataset_id": metadata.get("parent_dataset_id"),
                  "generation_definition_id": metadata.get("generation_definition_id", OUTCOME_DEFINITION),
                  "outcome_scope": scope, "is_factual_replay": False,
                  "evaluation_only": True}
        p0, p1 = float(row[p0_name]), float(row[p1_name])
        effects = []
        for replicate in range(repetitions):
            digest = hashlib.sha256(key + b"\x00" + replicate.to_bytes(4, "big")).digest()
            uniform = (int.from_bytes(digest[:8], "big") >> 11) / 2 ** 53
            control, coupon = int(uniform < p0), int(uniform < p1)
            effect = coupon - control
            effects.append(effect)
            pair_rows.append({**common, "replicate": replicate, "shared_uniform": uniform,
                              "control_retained": control, "coupon_retained": coupon,
                              "control_churn": 1-control, "coupon_churn": 1-coupon,
                              "realized_retention_difference": effect,
                              "paired_outcome_type": _OUTCOME_TYPES[control, coupon]})
        truth_rows.append({**common, **provenance, p0_name: p0, p1_name: p1,
                           "true_retention_uplift": p1-p0,
                           "mean_paired_retention_difference": float(np.mean(effects)),
                           "repetitions": repetitions, "latent_response_type": row.get("response_type")})
    pairs, truth = pd.DataFrame(pair_rows), pd.DataFrame(truth_rows)
    for frame in (pairs, truth):
        for name in ("actual_experiment_seed", "paired_derived_seed"):
            frame[name] = frame[name].astype("uint64")
        frame.attrs = {"evaluation_only": True, "recommended_directory": "evaluation_only",
                       "is_factual_replay": False, "method_version": PAIRED_METHOD_VERSION,
                       "meaning": "New common-noise draws of a latent return marginal, not factual RNG or full shopping replay.",
                       "outcome_view": {key: metadata.get(key) for key in (
                           "parent_dataset_id", "generation_definition_id", "generation_policy_id", "analytic_outcome_view")}}
    return pairs, truth
