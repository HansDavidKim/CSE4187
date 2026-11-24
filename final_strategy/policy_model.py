"""Exploratory 28-day purchase calibration on a fully observed fixed assignment log."""
from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
from itertools import product
import json
import math
from numbers import Real

import numpy as np
import pandas as pd

from .inputs import IntegratedInputs
from .policy_value import ExperimentEvidence, _prepare_log
from .segmentation import _revalidate, segment_customers
from .validation import EXPANDED_FEATURES, JOIN_KEYS, InputValidationError, PREDICTION_COLUMNS


BASE_FEATURES = tuple(PREDICTION_COLUMNS.values())
ROLES = ("train", "discovery", "confirmation", "selection", "final")


def _check(test, message):
    if not test:
        raise InputValidationError(f"policy_model: {message}")


def _integer(x, minimum):
    return type(x) is int and x >= minimum


def _finite(x):
    return isinstance(x, Real) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def _sequence(values, name):
    _check(not isinstance(values, (str, bytes)), f"{name} requires a sequence of values")
    try:
        return tuple(values)
    except TypeError as exc:
        raise InputValidationError(f"policy_model: {name} requires an iterable") from exc


@dataclass(frozen=True)
class SplitEvidence:
    """Caller declarations. Customer IDs in the split must use this namespace."""

    namespace: str
    source: str
    seed: int
    upstream_population_source: str
    upstream_customer_ids: tuple[str, ...] | None = None
    oof_source: str | None = None


@dataclass(frozen=True)
class ResidualConfig:
    random_state: int = 0
    max_depth: int = 2
    n_estimators: int = 32
    learning_rate: float = 0.1
    min_child_weight: float = 1
    subsample: float = 1
    colsample_bytree: float = 1
    reg_lambda: float = 1
    reg_alpha: float = 0
    n_jobs: int = 1

    def __post_init__(self):
        for key in ("max_depth", "n_estimators", "n_jobs", "random_state"):
            _check(_integer(getattr(self, key), 0 if key == "random_state" else 1), key)
        for key in ("learning_rate", "subsample", "colsample_bytree"):
            x = getattr(self, key)
            _check(_finite(x) and 0 < x <= 1, key)
            object.__setattr__(self, key, float(x))
        for key in ("min_child_weight", "reg_lambda", "reg_alpha"):
            x = getattr(self, key)
            _check(_finite(x) and x >= 0, key)
            object.__setattr__(self, key, float(x))


def _report_bins(frame, cuts, min_arm, min_events):
    # right-closed intervals match the strict greater-than retention threshold.
    bins = np.searchsorted(cuts, frame.uplift_score, side="left")
    reports = []
    for risk, value in product((None, "high", "low"), (None, "high_value", "standard_value")):
        group = np.ones(len(frame), dtype=bool)
        if risk:
            group &= frame.churn_band.eq(risk)
        if value:
            group &= frame.clv_band.eq(value)
        for i in range(len(cuts) + 1):
            rows = frame.loc[group & (bins == i)]
            arms = [rows.loc[rows.treatment.eq(action), "purchase"] for action in (0, 1)]
            counts = [len(a) for a in arms]
            events = [int(a.sum()) for a in arms]
            rates = [s / n if n else None for s, n in zip(events, counts)]
            effect = rates[1] - rates[0] if min(counts) else None
            mean = float(rows.uplift_score.mean()) if len(rows) else None
            adjusted = (float(rows.adjusted.mean()) if len(rows) and "adjusted" in rows else None)
            reports.append(dict(churn=risk, clv=value, lower=cuts[i-1] if i else None,
                                upper=cuts[i] if i < len(cuts) else None, n=len(rows),
                                control_n=counts[0], treatment_n=counts[1], control_events=events[0],
                                treatment_events=events[1], control_rate=rates[0], treatment_rate=rates[1],
                                observed_difference=effect, mean_uplift=mean,
                                observed_minus_uplift=effect-mean if effect is not None else None,
                                mean_residual=float(rows.residual.mean()) if len(rows) else None,
                                mean_adjusted=adjusted, observed_minus_adjusted=(
                                    effect-adjusted if effect is not None and adjusted is not None else None),
                                supported=min(counts) >= min_arm and min(events) >= min_events))
    return reports


def _fit(frame, features, config):
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError("Phase C residual training requires the optional xgboost package") from exc
    train = frame.loc[frame.role.eq("train")]
    _check(train.treatment.nunique() == 2, "training split requires both assignments")
    model = xgboost.XGBRegressor(objective="reg:squarederror", tree_method="hist", **asdict(config))
    model.fit(train[features], train.residual)
    trees = model.get_booster().trees_to_dataframe()
    splits = []
    for (name, cut), nodes in trees.loc[trees.Feature.ne("Leaf")].groupby(["Feature", "Split"]):
        splits.append(dict(feature=name, threshold=float(cut), occurrences=len(nodes), gain=float(nodes.Gain.sum())))
    booster_hash = sha256(model.get_booster().save_raw(raw_format="ubj")).hexdigest()
    return model, dict(version=booster_hash, library=xgboost.__version__, config=asdict(config), splits=splits)


def analyze_uplift(inputs: IntegratedInputs, experiment_log, *, evidence: ExperimentEvidence,
                   roles: pd.DataFrame, split: SplitEvidence, cuts=(), quantiles=(), churn_cuts=(), min_arm: int,
                   min_events: int, extra_features=(), model_config: ResidualConfig | None = None,
                   allow_unverified_log=False) -> dict:
    """Analyze train/discovery/confirmation only; reserve selection/final outcomes.

    roles supplies exactly customer_id and role for every member of one cohort.
    Upstream lists and OOF provenance are declarations, not authenticated evidence.
    """
    _check(_integer(min_arm, 1) and _integer(min_events, 0), "invalid minimum support")
    _check(isinstance(split, SplitEvidence) and all(isinstance(s, str) and s.strip() for s in
           (split.namespace, split.source, split.upstream_population_source)) and _integer(split.seed, 0),
           "split evidence required")
    _check(split.oof_source is None or isinstance(split.oof_source, str) and split.oof_source.strip(),
           "OOF source must be a nonempty string")
    extra_features = _sequence(extra_features, "extra_features")
    cuts = _sequence(cuts, "cuts")
    quantiles = _sequence(quantiles, "quantiles")
    churn_cuts = _sequence(churn_cuts, "churn_cuts")
    _check(all(isinstance(f, str) for f in extra_features), "KIT39 feature names must be strings")
    features = [*BASE_FEATURES, *extra_features]
    _check(len(set(features)) == len(features) and set(extra_features) <= set(EXPANDED_FEATURES),
           "extra_features must be unique pre-assignment KIT39 features")
    _check(model_config is None or isinstance(model_config, ResidualConfig), "invalid residual model config")
    validated = _revalidate(inputs)
    for key in ("churn", "uplift"):
        meta = inputs.prediction_metadata[key]
        _check((meta.horizon_value, meta.horizon_unit) in ((28, "days"), (4, "weeks")), "28-day prediction required")
    table, records, _, provenance, block = _prepare_log(experiment_log, evidence, allow_unverified_log)
    _check(block is None, block)
    _check(all(observed and rewards[0] is not None for _, observed, rewards in records),
           "complete 28-day purchase observations required for every ITT customer")
    frame = validated.data.sort_values(list(JOIN_KEYS)).reset_index(drop=True)
    _check(len(frame) == len(table), "log roster size mismatch")
    for key in (*JOIN_KEYS, "snapshot_id", "currency"):
        _check(frame[key].eq(table[key]).fillna(False).all(), f"log {key} mismatch")
    _check(frame.amount_unit.map(Decimal).eq(table.amount_unit).all(), "log amount_unit mismatch")
    _check(frame.raw_database_sha256.eq(evidence.raw_database_sha256).all(), "raw source mismatch")
    _check(isinstance(roles, pd.DataFrame) and list(sorted(roles.columns)) == ["customer_id", "role"]
           and not roles.customer_id.duplicated().any() and set(roles.customer_id) == set(frame.customer_id)
           and roles.role.isin(ROLES).all(), "invalid split roster or overlapping customer roles")
    frame["role"] = frame.customer_id.map(roles.set_index("customer_id").role)
    _check({"discovery", "confirmation"} <= set(frame.role), "discovery and confirmation roles required")
    upstream = split.upstream_customer_ids
    if upstream is not None:
        _check(isinstance(upstream, tuple) and all(isinstance(k, str) and k.strip() for k in upstream),
               "invalid upstream roster")
        overlap = frame.loc[frame.customer_id.isin(upstream)]
        _check(not overlap.role.isin(["selection", "final"]).any(), "upstream overlap with reserved customers")
        _check(overlap.empty or split.oof_source is not None,
               "upstream overlap requires OOF prediction provenance")
    counts = {r: int(frame.role.eq(r).sum()) for r in ROLES}
    roster_hash = sha256(json.dumps(sorted(zip(frame.customer_id, frame.role)),
                                       ensure_ascii=False).encode()).hexdigest()
    development_roles = ("train", "discovery", "confirmation")
    development_ids = frame.loc[frame.role.isin(development_roles), "customer_id"]
    cohort = segment_customers(IntegratedInputs(
        validated.data.loc[validated.data.customer_id.isin(development_ids)], validated.prediction_metadata,
        {k: context.loc[context.customer_id.isin(development_ids)] for k, context in validated.prediction_context.items()},
    ))
    outcomes = frame[[*JOIN_KEYS, "role"]].assign(treatment=table.treatment, purchase=[r[2][0] for r in records])
    frame = cohort.customers.merge(outcomes, on=list(JOIN_KEYS), validate="one_to_one").sort_values(list(JOIN_KEYS))
    frame["z"] = np.where(frame.treatment.eq(1), 2, -2) * frame.purchase
    frame["residual"] = frame.z - frame.uplift_score
    discovery = frame.loc[frame.role.eq("discovery")]
    _check(all(_finite(x) and -1 <= x <= 1 for x in cuts), "invalid uplift cut")
    _check(all(_finite(q) and 0 < q < 1 for q in quantiles), "invalid discovery quantile")
    origins = [{"feature": "uplift_score", "threshold": float(x), "source": "declared_bin"} for x in cuts]
    origins.extend(dict(feature="uplift_score", threshold=float(discovery.uplift_score.quantile(q)),
                        source="discovery_quantile", quantile=float(q)) for q in quantiles)
    boundaries = sorted({o["threshold"] for o in origins})
    _check(all(_finite(x) and 0 <= x <= 1 for x in churn_cuts), "invalid churn cut")
    origins.extend(dict(feature="churn_probability", threshold=float(x), source="declared_bin") for x in churn_cuts)
    model_info = None
    scores = {}
    if model_config is not None:
        model, model_info = _fit(frame, features, model_config)
        for item in model_info["splits"]:
            # XGBoost compares float32 features with '<'. Translate the boundary
            # to an observed discovery score for the strategy's strict '>' / '>='.
            feature = item["feature"]
            cut = item["threshold"]
            values = discovery[feature].dropna()
            left = values.astype("float32").lt(np.float32(cut))
            if feature in ("uplift_score", "churn_probability") and left.any() and (~left).any():
                cut = float(values[left].max() if feature == "uplift_score" else values[~left].min())
            origins.append(dict(item, source="xgboost_split", tree_threshold=item["threshold"], threshold=cut))
        frame["adjusted"] = frame.uplift_score + model.predict(frame[features])
        _check(np.isfinite(frame.adjusted).all(), "non-finite diagnostic prediction")
    bins = {}
    for role in ("discovery", "confirmation"):
        rows = frame.loc[frame.role.eq(role)]
        bins[role] = _report_bins(rows, boundaries, min_arm, min_events)
        scores[role] = dict(mean_z=float(rows.z.mean()), mean_residual=float(rows.residual.mean()),
                            baseline_mse=float(rows.residual.pow(2).mean()))
        if model_info is not None:
            scores[role].update(adjusted_mse=float((rows.z-rows.adjusted).pow(2).mean()),
                                outside_probability_difference=int(rows.adjusted.abs().gt(1).sum()))
    # Record local support on each side of each proposed threshold in both samples.
    for item in origins:
        item["support"] = {}
        for role in bins:
            rows = frame.loc[frame.role.eq(role)]
            v = rows[item["feature"]]
            lower = v.lt(item["threshold"]) if item["feature"] == "churn_probability" else v.le(item["threshold"])
            sides = []
            for mask in (lower, ~lower & v.notna()):
                group = rows.loc[mask]
                ns = [group.loc[group.treatment.eq(a), "purchase"] for a in (0, 1)]
                sides.append(dict(count=[len(n) for n in ns], purchases=[int(n.sum()) for n in ns],
                                  supported=all(len(n) >= min_arm and n.sum() >= min_events for n in ns)))
            item["support"][role] = sides
        item["supported"] = all(bool(s["supported"]) for sides in item["support"].values() for s in sides)
    # Counts suffice to check support for a proposed conjunction without passing
    # customer identifiers or reserved outcomes to the candidate generator.
    support_keys = ["role", "churn_probability", "clv_band", "uplift_score", "treatment"]
    support = (frame.loc[frame.role.isin(bins)].groupby(support_keys).purchase
               .agg(n="size", purchases="sum").reset_index().to_dict("records"))
    return dict(provenance=provenance, split=dict(namespace=split.namespace, source=split.source, seed=split.seed,
                roster_sha256=roster_hash, counts=counts, upstream_source=split.upstream_population_source,
                development_roles=list(development_roles), development_count=len(frame),
                development_roster_sha256=sha256(json.dumps(sorted(zip(frame.customer_id, frame.role)),
                    ensure_ascii=False).encode()).hexdigest(), clv_rank_population="pooled_development_roles",
                upstream_roster_sha256=(None if upstream is None else
                    sha256(json.dumps(sorted(upstream), ensure_ascii=False).encode()).hexdigest()),
                upstream_roster_supplied=upstream is not None, oof_source=split.oof_source,
                independence="not_independently_verified"), features=features, models=cohort.metadata["models"],
                feature_versions={k: str(frame[k].iloc[0]) for k in ("feature_set_version", "expanded_feature_set_version")},
                residual_model=model_info, diagnostics=scores, bins=bins, thresholds=origins, rule_support=support,
                min_arm=min_arm, min_events=min_events, stability="not_verified_across_repeated_splits",
                interpretation="exploratory; transformed outcomes have high individual variance",
                policy_effect_verified=False)
