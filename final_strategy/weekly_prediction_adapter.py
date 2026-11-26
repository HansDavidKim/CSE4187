"""Bridge official weekly producer handoffs to the existing strategy engine."""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from simulate.v4.storage import FEATURE_KEYS
from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, weekly_metadata
from .inputs import IntegratedInputs, PredictionBatch, integrate_inputs, load_feature_store
from .retention_prediction_adapter import DEFINITION, STATUSES
from .uplift_prediction_adapter import UPLIFT_FEATURES
from .validation import (CONTEXT_COLUMNS, JOIN_KEYS, PredictionMetadata, InputValidationError,
                         _context, _numeric, _times, validate_feature_frame)
from .weekly_validation import PredictionMetadata as WeeklyMetadata


@dataclass(frozen=True)
class _WeeklyContents:
    integrated: IntegratedInputs | None
    roster: pd.DataFrame
    excluded: pd.DataFrame
    producer_metadata: dict
    prediction_metadata: dict
    evidence: dict


@dataclass(frozen=True)
class WeeklyStrategyInputs:
    """A validated producer snapshot. Each public view is independently editable."""

    _contents: _WeeklyContents

    def snapshot(self):
        return deepcopy(self._contents)

    @property
    def integrated(self):
        return deepcopy(self._contents.integrated)

    @property
    def roster(self):
        return self._contents.roster.copy(deep=True)

    @property
    def excluded(self):
        return self._contents.excluded.copy(deep=True)

    @property
    def producer_metadata(self):
        return deepcopy(self._contents.producer_metadata)

    @property
    def prediction_metadata(self):
        return deepcopy(self._contents.prediction_metadata)

    @property
    def evidence(self):
        return deepcopy(self._contents.evidence)


def _require(condition, message):
    if not condition:
        raise InputValidationError(message)


def _align(source, frame, role, weeks):
    """Normalize identity before aligning. Reindex only after proving exact coverage."""
    _require(isinstance(frame, pd.DataFrame) and not frame.columns.duplicated().any()
             and set(CONTEXT_COLUMNS) <= set(frame), f"{role}: incomplete producer identity")
    _require(frame.attrs.get("evaluation_only") is None, f"{role}: evaluation_only")
    checked = _context(frame.copy(deep=True).reset_index(drop=True), role)
    index = pd.MultiIndex.from_frame(source[list(JOIN_KEYS)])
    keyed = checked.set_index(list(JOIN_KEYS))
    _require(len(keyed) == len(index) and index.isin(keyed.index).all(), f"{role}: customer coverage")
    checked = keyed.reindex(index).reset_index()
    for key in CONTEXT_COLUMNS:
        _require(checked[key].eq(source[key]).all(), f"{role}: {key} mismatch")
    for name, end in (("input_cutoff_at", source.as_of_at),
                      ("prediction_end_at", source.as_of_at + pd.Timedelta(weeks=weeks))):
        _require(name in checked and _times(checked[name], name).eq(end).all(), f"{role}: {name} mismatch")
    return checked


def _declaration(raw, role, target, semantics):
    try:
        meta = WeeklyMetadata(**raw)
    except (ValueError, TypeError) as error:
        raise InputValidationError(f"{role}: malformed prediction metadata") from error
    meta.validate(role)
    _require(isinstance(meta.model_version, str) and bool(meta.model_version.strip())
             and meta.target_definition_id == target and meta.value_semantics == semantics,
             f"{role}: model version, definition or value semantics mismatch")
    return PredictionMetadata(**vars(meta))


def _values(frame, name, *, probability=False, nullable=False):
    _require(name in frame, f"missing prediction: {name}")
    values = _numeric(frame[name], name, allow_null=nullable)
    known = values.dropna()
    _require(known.ge(0).all() and (not probability or known.le(1).all()), f"{name}: out of range")
    return values


def adapt_weekly_predictions(features, *, retention, uplift, clv,
                             expected_retention_policy="observed_experiment_mix",
                             allow_mixed_clv_for_development=False):
    """Adapt previously computed batches. Only explicit Retention statuses exclude rows."""
    from uplift.common.schema import FEATURES

    source = validate_feature_frame(features)
    _require(source.expanded_feature_set_version.eq(WEEKLY_FEATURE_VERSION).all()
             and source.as_of_at.nunique() == 1, "weekly adapter requires one weekly cutoff")
    _require(tuple(FEATURES) == UPLIFT_FEATURES and len(FEATURES) == 36, "KIT36 registry mismatch")
    objects = {"churn": retention, "uplift": uplift, "clv": clv}
    producers = {role: deepcopy(obj.producer_metadata) for role, obj in objects.items()}
    declarations = {
        "churn": _declaration(retention.metadata, "churn", DEFINITION, "probability"),
        "uplift": _declaration(uplift.metadata, "uplift", "purchase_28d_randomized_coupon_v1", "probability_pair"),
        "clv": _declaration(clv.metadata, "clv", "clv_paid_amount_52w_v1", "nonnegative_amount"),
    }
    rows = {role: _align(source, obj.frame, role, declarations[role].horizon_value)
            for role, obj in objects.items()}
    roster = source[[*FEATURE_KEYS, "observed_tenure_days"]].copy()
    r = rows["churn"]
    rp = producers["churn"]
    clock = weekly_metadata()
    _require(rp.get("component") == "retention" and rp.get("time_spec") == clock["time_spec"]
             and rp.get("time_windows_days") == clock["time_windows_days"]
             and "expanded_feature_set_version" in r
             and r.expanded_feature_set_version.notna().all()
             and r.expanded_feature_set_version.eq(WEEKLY_FEATURE_VERSION).all(), "Retention feature contract")
    _require(expected_retention_policy in ("no_new_coupon", "observed_experiment_mix")
             and rp.get("policy_condition") == expected_retention_policy, "Retention policy condition")
    _require("status" in r and r.status.isin(STATUSES).all(), "Retention status")
    scored = r.status.eq("scored")
    _require(rp.get("status_counts") == r.status.value_counts().to_dict()
             and rp.get("excluded_rows") == int((~scored).sum()), "Retention exclusion counts")
    probability_names = [name for name in ("churn_prob", "churn_probability") if name in r]
    _require(bool(probability_names), "Retention probability missing")
    churn = _values(r, probability_names[0], probability=True, nullable=True)
    for name in probability_names[1:]:
        _require(churn.equals(_values(r, name, probability=True, nullable=True)), "conflicting churn aliases")
    _require(churn.notna().eq(scored).all(), "Retention probability/status mismatch")
    roster["status"] = r.status
    roster["churn_probability"] = churn

    u = rows["uplift"]
    up = producers["uplift"]
    _require(up.get("component") == "uplift" and up.get("input_contract") == clock, "Uplift weekly KIT36 contract")
    training = up.get("training", {})
    for field in ("training_customers", "selection_customers"):
        _require(isinstance(training.get(field), list), f"Uplift training declaration lacks {field}")
    _require(isinstance(training.get("dataset_id"), str) and bool(training["dataset_id"]), "Uplift training dataset")
    same_training_data = training["dataset_id"] == source.dataset_id.iloc[0]
    if same_training_data:
        _require(training.get("raw_database_sha256") in (None, source.raw_database_sha256.iloc[0]),
                 "Uplift training ledger mismatch")
        seen = set(training["training_customers"]) | set(training["selection_customers"])
        _require(not set(source.customer_id).intersection(seen), "Uplift training/selection customer overlap")
    _require(up.get("development_overlap_allowed") is False and up.get("training_overlap_customers") == [],
             "Uplift development overlap override or missing check")
    for name in ("p0", "p1"):
        roster[name] = _values(u, name, probability=True)
    _require("uplift_score" in u, "missing uplift_score")
    score = _numeric(u.uplift_score, "uplift_score")
    _require(score.between(-1, 1).all() and np.allclose(score, roster.p1-roster.p0, rtol=0, atol=1e-12),
             "Uplift score differs from p1-p0")
    roster["uplift_score"] = score

    cp = producers["clv"]
    allowed_policies = {"control_no_coupon"}
    if allow_mixed_clv_for_development is True:
        allowed_policies.add("observed_experiment_mix")
    _require(cp.get("producer") == "clv" and cp.get("policy_condition") in allowed_policies,
             "CLV producer or policy condition")
    _require(cp.get("model_version") == declarations["clv"].model_version, "CLV model version")
    origin = cp.get("inputs", {})
    for key in ("dataset_id", "raw_database_sha256"):
        _require(origin.get(key) == source[key].iloc[0], f"CLV producer {key} mismatch")
    audit = clv.audit
    _require(isinstance(audit, pd.DataFrame) and not audit.columns.duplicated().any()
             and {*JOIN_KEYS, "clv_training_role"} <= set(audit), "CLV training audit missing")
    audit = audit.copy(deep=True)
    audit["as_of_at"] = _times(audit.as_of_at, "clv.audit.as_of_at")
    idx = pd.MultiIndex.from_frame(source[list(JOIN_KEYS)])
    roles = audit.set_index(list(JOIN_KEYS)).clv_training_role
    _require(roles.index.is_unique and len(roles) == len(idx) and idx.isin(roles.index).all(),
             "CLV audit customer coverage")
    roles = roles.reindex(idx).reset_index(drop=True)
    overlap = cp.get("training_overlap", {})
    counts = roles.value_counts().reindex(["train","validation","test","unused"], fill_value=0).to_dict()
    _require(roles.isin(["test", "unused"]).all() and overlap.get("allowed") is False
             and counts == overlap.get("roles"), "CLV training/selection customer overlap or inconsistent audit")
    roster["clv_training_role"] = roles
    roster["predicted_clv_12m"] = _values(rows["clv"], "predicted_clv_12m")
    gross_status = clv.gross_4w_status
    gross_info = cp.get("gross_4w", {})
    _require(isinstance(gross_status, str) and bool(gross_status) and gross_info.get("status") == gross_status,
             "CLV gross status mismatch")
    if clv.gross_4w is None:
        _require(gross_status != "trained" and clv.gross_4w_metadata is None, "missing trained gross batch")
        roster["predicted_gross_4w_if_purchase"] = np.nan
    else:
        _require(gross_status == "trained", "unexpected gross batch")
        gross_meta = _declaration(clv.gross_4w_metadata, "gross_4w", "gross_amount_4w_if_purchase_v1", "nonnegative_amount")
        _require(gross_info.get("model_version") == gross_meta.model_version, "gross model version mismatch")
        declarations["gross_4w"] = gross_meta
        gross = _align(source, clv.gross_4w, "gross_4w", 4)
        roster["predicted_gross_4w_if_purchase"] = _values(gross, "predicted_gross_4w_if_purchase")

    batches = {}
    for role, name in (("churn", "churn_probability"), ("uplift", "uplift_score"), ("clv", "predicted_clv_12m")):
        frame = rows[role].loc[scored, [*CONTEXT_COLUMNS, "input_cutoff_at", "prediction_end_at"]].copy()
        frame[name] = roster.loc[scored, name]
        frame.attrs["prediction_feature_version"] = WEEKLY_FEATURE_VERSION
        declaration = declarations[role]
        if role == "uplift":
            declaration = replace(declaration, value_semantics="probability_difference")
        batches[role] = PredictionBatch(frame, declaration)
    integrated = integrate_inputs(source.loc[scored], batches) if scored.any() else None
    excluded = roster.loc[~scored, [*CONTEXT_COLUMNS, "status"]].copy()
    excluded["exclusion_stage"] = "retention_prediction"
    excluded["exclusion_reason"] = excluded.status
    evidence = {"feature_contract": deepcopy(clock), "feature_store": deepcopy(source.attrs.get("feature_store_evidence")),
                "retention_training_overlap": "not_reported", "overlap_checks": "producer declarations and supplied customer keys",
                "gross_4w_status": gross_status, "dispatch_authorized": False}
    return WeeklyStrategyInputs(_WeeklyContents(integrated, roster, excluded, producers, declarations, evidence))


def predict_weekly_inputs(feature_directory, *, retention_bundle, raw_dataset, uplift_model, clv_model_dir,
                          expected_retention_policy="observed_experiment_mix"):
    """Invoke the three public producers once on one cohort. Model imports stay optional."""
    from retention.api import predict_for_strategy as predict_retention
    from uplift.api import predict_for_strategy as predict_uplift
    from clv.api import predict_for_strategy as predict_clv

    features = load_feature_store(feature_directory)
    _require(features.expanded_feature_set_version.eq(WEEKLY_FEATURE_VERSION).all()
             and features.as_of_at.nunique() == 1, "expected a single weekly Feature Store")
    retention = predict_retention(retention_bundle, raw_dataset, features.as_of_at.iloc[0])
    uplift = predict_uplift(uplift_model, feature_directory, allow_training_customers=False)
    clv = predict_clv(feature_directory, model_dir=clv_model_dir, ledger=raw_dataset,
                      allow_training_customers=False, require_control_only=True, require_gross_4w=False)
    return adapt_weekly_predictions(features, retention=retention, uplift=uplift, clv=clv,
                                    expected_retention_policy=expected_retention_policy)
