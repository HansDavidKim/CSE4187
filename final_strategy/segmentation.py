"""Snapshot-specific segmentation and priority proxies from validated predictions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
from itertools import product
import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.storage import FEATURE_KEYS

from .inputs import IntegratedInputs, PredictionBatch, integrate_inputs
from .validation import (
    CONTEXT_COLUMNS, JOIN_KEYS, PREDICTION_COLUMNS, InputValidationError,
    PredictionMetadata,
)


COHORT_KEYS = ("dataset_id", "as_of_at")
INPUT_COLUMNS = (*FEATURE_KEYS, *PREDICTION_COLUMNS.values(), *EXPANDED_FEATURES)
CHURN_BANDS = ("high", "low")
UPLIFT_BANDS = ("negative", "neutral_low", "positive")
CLV_BANDS = ("high_value", "standard_value")
SEGMENT_COLUMNS = ("churn_band", "uplift_band", "clv_band", "segment_id", "segment_name")
DERIVED_COLUMNS = (*SEGMENT_COLUMNS, "is_high_clv", "priority_score", "priority_rank")
MEAN_COLUMNS = (*PREDICTION_COLUMNS.values(), "priority_score")
SUMMARY_COLUMNS = (
    *COHORT_KEYS, *SEGMENT_COLUMNS, "population_count", "customer_count", "customer_ratio",
    *[f"mean_{column}" for column in MEAN_COLUMNS], "priority_score_sum", "currency", "amount_unit",
)


@dataclass(frozen=True)
class SegmentationConfig:
    """Development defaults, not validated marketing policy thresholds."""

    churn_threshold: float = 0.5
    uplift_threshold: float = 0.0
    high_clv_fraction: float = 0.2
    segmentation_version: str = "segmentation-12-v1"

    def __post_init__(self):
        for name in ("churn_threshold", "uplift_threshold", "high_clv_fraction"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise InputValidationError(f"segmentation.{name}: must be finite and in [0,1]")
        if not isinstance(self.segmentation_version, str) or not self.segmentation_version.strip():
            raise InputValidationError("segmentation.segmentation_version: requires a nonempty string")


@dataclass(frozen=True)
class SegmentationResult:
    """Analyzed and excluded rows, twelve-bin summaries, and separate provenance."""

    customers: pd.DataFrame
    excluded: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict


def _columns(frame, expected, name):
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError(f"{name}: expected DataFrame")
    if frame.columns.duplicated().any() or set(frame) != set(expected):
        raise InputValidationError(f"{name}: requires exactly {list(expected)} without duplicate columns")
    if frame.attrs.get("evaluation_only") is not None:
        raise InputValidationError(f"{name}: evaluation_only data is forbidden")
    return frame.copy(deep=True).reset_index(drop=True)


def _clock(values, name):
    points = []
    for value in values:
        try:
            point = pd.Timestamp(value)
            if pd.isna(point) or point.tzinfo is None:
                raise ValueError("unknown or naive clock")
            points.append(point.tz_convert("UTC"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(f"{name}: requires timezone-aware clocks") from exc
    return pd.Series(points, index=values.index, dtype="datetime64[ns, UTC]")


def _model_metadata(metadata):
    # Empty inputs have no value rows for the existing validators to inspect.
    # Check their declarations as well; nonempty inputs also reuse integrate_inputs.
    if not isinstance(metadata, dict) or set(metadata) != set(PREDICTION_COLUMNS):
        raise InputValidationError("segmentation: requires churn/uplift/clv metadata")
    for model, item in metadata.items():
        if not isinstance(item, PredictionMetadata):
            raise InputValidationError(f"segmentation.{model}: requires PredictionMetadata")
        for field in ("model_version", "target_definition_id"):
            if not isinstance(getattr(item, field), str) or not getattr(item, field).strip():
                raise InputValidationError(f"segmentation.{model}.{field}: requires a declaration")
        if (isinstance(item.horizon_value, (bool, np.bool_))
                or not isinstance(item.horizon_value, numbers.Integral) or item.horizon_value <= 0
                or item.horizon_unit not in ("days", "weeks", "calendar_months")):
            raise InputValidationError(f"segmentation.{model}: invalid horizon declaration")
        semantics = {"churn": ("probability",), "uplift": ("probability_difference",),
                     "clv": ("amount", "nonnegative_amount")}[model]
        if item.value_semantics not in semantics:
            raise InputValidationError(f"segmentation.{model}: unsupported value semantics")
        if model == "clv" and (item.horizon_value != 52 or item.horizon_unit != "weeks"):
            raise InputValidationError("segmentation.clv: requires exactly 52 weeks")


def _same_keys(left, right, name):
    for frame in (left, right):
        if frame[list(JOIN_KEYS)].isna().any().any() or frame.duplicated(list(JOIN_KEYS)).any():
            raise InputValidationError(f"{name}: missing or duplicate snapshot keys")
        for column in ("customer_id", "dataset_id"):
            if not frame[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise InputValidationError(f"{name}.{column}: requires nonempty string IDs")
    lhs, rhs = (pd.MultiIndex.from_frame(frame[list(JOIN_KEYS)]) for frame in (left, right))
    if len(lhs.difference(rhs)) or len(rhs.difference(lhs)):
        raise InputValidationError(f"{name}: snapshot key coverage mismatch")


def _revalidate(inputs):
    if not isinstance(inputs, IntegratedInputs):
        raise InputValidationError("segmentation: requires IntegratedInputs")
    _model_metadata(inputs.prediction_metadata)
    frame = _columns(inputs.data, INPUT_COLUMNS, "segmentation.data")
    frame["as_of_at"] = _clock(frame.as_of_at, "segmentation.data.as_of_at")
    if not isinstance(inputs.prediction_context, dict) or set(inputs.prediction_context) != set(PREDICTION_COLUMNS):
        raise InputValidationError("segmentation: requires context for all three models")
    batches, contexts = {}, {}
    for model, value in PREDICTION_COLUMNS.items():
        context = _columns(inputs.prediction_context[model],
                           (*JOIN_KEYS, "input_cutoff_at", "prediction_end_at"), f"segmentation.{model}.context")
        for column in ("as_of_at", "input_cutoff_at", "prediction_end_at"):
            context[column] = _clock(context[column], f"segmentation.{model}.{column}")
        _same_keys(frame, context, f"segmentation.{model}.context")
        contexts[model] = context
        merged = frame[[*CONTEXT_COLUMNS, value]].merge(context, on=list(JOIN_KEYS), validate="one_to_one")
        # pandas merge can drop the batch clock during input revalidation.
        if "prediction_feature_version" in context.attrs:
            merged.attrs["prediction_feature_version"] = context.attrs["prediction_feature_version"]
        batches[model] = PredictionBatch(merged, inputs.prediction_metadata[model])
    if frame.empty:
        # Empty slices retain the schema/metadata but generic input validation
        # deliberately rejects empty frames. No numeric values exist to validate.
        return IntegratedInputs(frame.loc[:, [*FEATURE_KEYS, *EXPANDED_FEATURES, *PREDICTION_COLUMNS.values()]],
                                deepcopy(inputs.prediction_metadata), contexts)
    validated = integrate_inputs(frame[[*FEATURE_KEYS, *EXPANDED_FEATURES]], batches)
    if validated.data.predicted_clv_12m.lt(0).any():
        raise InputValidationError("segmentation.predicted_clv_12m: future paid-revenue CLV must be nonnegative")
    return validated


def _eligibility(frame, eligibility):
    if eligibility is None:
        return pd.Series(True, index=frame.index), pd.Series(pd.NA, index=frame.index, dtype="object")
    decisions = _columns(eligibility, (*JOIN_KEYS, "eligible", "exclusion_reason"), "segmentation.eligibility")
    decisions["as_of_at"] = _clock(decisions.as_of_at, "segmentation.eligibility.as_of_at")
    _same_keys(frame, decisions, "segmentation.eligibility")
    if not decisions.eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise InputValidationError("segmentation.eligibility.eligible: requires explicit booleans")
    excluded = ~decisions.eligible.astype(bool)
    if not decisions.loc[excluded, "exclusion_reason"].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise InputValidationError("segmentation.eligibility.exclusion_reason: required for excluded customers")
    if decisions.loc[~excluded, "exclusion_reason"].notna().any():
        raise InputValidationError("segmentation.eligibility.exclusion_reason: eligible customers cannot have an exclusion reason")
    aligned = frame[list(JOIN_KEYS)].merge(decisions, on=list(JOIN_KEYS), validate="one_to_one")
    return aligned.eligible.astype(bool), aligned.exclusion_reason


def _segment(churn, uplift, clv):
    identifier = f"{churn}__{uplift}__{clv}"
    return identifier, f"{churn.title()} risk / {uplift.replace('_', ' ').title()} uplift / {clv.replace('_', ' ').title()}"


def _records(frame):
    return [{key: value.isoformat() if isinstance(value, pd.Timestamp) else value
             for key, value in row.items()} for row in frame.to_dict("records")]


def segment_customers(inputs: IntegratedInputs, config: SegmentationConfig | None = None,
                      eligibility: pd.DataFrame | None = None) -> SegmentationResult:
    """Assign each analyzed snapshot exactly once; never infer coupon eligibility.

    eligibility, when supplied, covers every input key with an explicit eligible
    boolean and an exclusion_reason for excluded rows. No file/publisher is read.
    """
    config = SegmentationConfig() if config is None else config
    if not isinstance(config, SegmentationConfig):
        raise InputValidationError("segmentation.config: requires SegmentationConfig")
    validated = _revalidate(inputs)
    frame = validated.data.sort_values(list(JOIN_KEYS), kind="stable").reset_index(drop=True)
    eligible, reasons = _eligibility(frame, eligibility)
    customers, excluded = frame.loc[eligible].copy(), frame.loc[~eligible].copy()
    excluded["exclusion_reason"] = reasons.loc[~eligible].to_numpy()
    customers["churn_band"] = np.where(customers.churn_probability.ge(config.churn_threshold), "high", "low")
    customers["uplift_band"] = np.select(
        [customers.uplift_score.lt(0), customers.uplift_score.gt(config.uplift_threshold)],
        ["negative", "positive"], default="neutral_low")
    customers["is_high_clv"] = False
    customers["priority_score"] = customers.uplift_score.astype(float) * customers.predicted_clv_12m.astype(float)
    customers["priority_rank"] = pd.Series(pd.NA, index=customers.index, dtype="Int64")
    summaries, cohorts = [], []
    for (dataset_id, as_of_at), all_rows in frame.groupby(list(COHORT_KEYS), sort=True):
        group = customers.loc[customers.dataset_id.eq(dataset_id) & customers.as_of_at.eq(as_of_at)]
        count = len(group)
        top_count = int((Decimal(count) * Decimal(str(config.high_clv_fraction))).to_integral_value(rounding=ROUND_CEILING))
        high = group.sort_values(["predicted_clv_12m", "customer_id"], ascending=[False, True], kind="stable").head(top_count)
        customers.loc[high.index, "is_high_clv"] = True
        ranked = group.sort_values(["priority_score", "customer_id"], ascending=[False, True], kind="stable")
        customers.loc[ranked.index, "priority_rank"] = range(1, count + 1)
        cohorts.append({"dataset_id": dataset_id, "as_of_at": as_of_at.isoformat(),
                        "input_count": len(all_rows), "analyzed_count": count, "excluded_count": len(all_rows) - count,
                        "high_clv_count": top_count, "high_clv_ratio": top_count / count if count else 0.0,
                        "currency": all_rows.currency.iloc[0], "amount_unit": all_rows.amount_unit.iloc[0],
                        "raw_database_sha256": all_rows.raw_database_sha256.iloc[0]})
    customers["clv_band"] = np.where(customers.is_high_clv, "high_value", "standard_value")
    segments = [_segment(*values) for values in customers[["churn_band", "uplift_band", "clv_band"]].itertuples(index=False, name=None)]
    customers["segment_id"] = [values[0] for values in segments]
    customers["segment_name"] = [values[1] for values in segments]
    for cohort in cohorts:
        as_of_at = pd.Timestamp(cohort["as_of_at"])
        group = customers.loc[customers.dataset_id.eq(cohort["dataset_id"]) & customers.as_of_at.eq(as_of_at)]
        for churn, uplift, clv in product(CHURN_BANDS, UPLIFT_BANDS, CLV_BANDS):
            identifier, name = _segment(churn, uplift, clv)
            members = group.loc[group.segment_id.eq(identifier)]
            row = {"dataset_id": cohort["dataset_id"], "as_of_at": as_of_at,
                   "churn_band": churn, "uplift_band": uplift, "clv_band": clv,
                   "segment_id": identifier, "segment_name": name,
                   "population_count": len(group), "customer_count": len(members),
                   "customer_ratio": len(members) / len(group) if len(group) else 0.0,
                   "currency": cohort["currency"], "amount_unit": cohort["amount_unit"]}
            row.update({f"mean_{column}": members[column].mean() for column in MEAN_COLUMNS})
            row["priority_score_sum"] = members.priority_score.sum() if len(members) else 0.0
            if any(not pd.isna(v) and not np.isfinite(v) for k, v in row.items() if k.startswith("mean_") or k == "priority_score_sum"):
                raise InputValidationError("segmentation.summary: non-finite score or mean aggregate")
            summaries.append(row)
        if sum(row["customer_count"] for row in summaries[-12:]) != len(group):
            raise InputValidationError("segmentation.summary: customer count reconciliation failed")
    customers = customers.sort_values([*COHORT_KEYS, "priority_rank"], kind="stable").reset_index(drop=True)
    excluded = excluded.reset_index(drop=True)
    metadata = {
        "config": asdict(config), "population_scope": "validated_inputs" if eligibility is None else "explicit_eligibility",
        "coupon_eligibility_inferred": False,
        "input_count": len(frame), "analyzed_count": len(customers), "excluded_count": len(excluded),
        "models": {name: asdict(item) for name, item in deepcopy(validated.prediction_metadata).items()},
        "input_snapshots": _records(frame[[*JOIN_KEYS, "reference_week", "snapshot_id", "raw_database_sha256",
                                           "source_model_sha256", "expanded_feature_set_version"]]),
        "prediction_context": {name: _records(context.sort_values(list(JOIN_KEYS), kind="stable"))
                               for name, context in deepcopy(validated.prediction_context).items()},
        "cohorts": cohorts,
        "high_clv_policy": "ceil(N*fraction); CLV descending, customer_id ascending",
        "priority_policy": "uplift*CLV; score descending, customer_id ascending; sequential ranks per dataset/as_of_at",
        "priority_interpretation": "ranking proxy; not actual incremental revenue or profit; original horizons retained",
    }
    return SegmentationResult(customers, excluded, pd.DataFrame(summaries, columns=SUMMARY_COLUMNS), metadata)


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def export_segments(result: SegmentationResult, csv_path: str | Path) -> dict[str, Path]:
    """Write caller-selected CSV plus summary/exclusions/metadata; never overwrite."""
    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        raise InputValidationError("segmentation export: expected a .csv destination")
    paths = {"customers": path, "summary": path.with_name(path.stem + "_summary.csv"),
             "excluded": path.with_name(path.stem + "_excluded.csv"),
             "metadata": path.with_name(path.stem + "_metadata.json")}
    if any(item.exists() for item in paths.values()):
        raise FileExistsError("segmentation export preserves existing output files")
    text = json.dumps(result.metadata, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.customers[[*CONTEXT_COLUMNS, *PREDICTION_COLUMNS.values(), *DERIVED_COLUMNS]].to_csv(path, index=False, mode="x")
    result.summary.to_csv(paths["summary"], index=False, mode="x")
    result.excluded[[*CONTEXT_COLUMNS, *PREDICTION_COLUMNS.values(), "exclusion_reason"]].to_csv(paths["excluded"], index=False, mode="x")
    with paths["metadata"].open("x", encoding="utf-8") as stream:
        stream.write(text)
    return paths


def plot_segments(result: SegmentationResult, dataset_id: str, as_of_at):
    """Return a standalone Matplotlib figure for exactly one snapshot cohort."""
    from matplotlib.figure import Figure

    clock = _clock(pd.Series([as_of_at]), "segmentation.plot.as_of_at").iloc[0]
    rows = result.summary.loc[result.summary.dataset_id.eq(dataset_id) & result.summary.as_of_at.eq(clock)]
    if rows.empty:
        raise InputValidationError("segmentation.plot: requested dataset/as_of_at cohort does not exist")
    figure = Figure(figsize=(16, 9), layout="constrained")
    axes = figure.subplots(2, 2)
    labels = [f"{r.churn_band}/{r.uplift_band}/{r.clv_band}" for r in rows.itertuples()]
    unit = f"{rows.currency.iloc[0]}; amount unit {rows.amount_unit.iloc[0]}"
    metrics = (("customer_count", "Customers", "Count"),
               ("customer_ratio", "Population share", "Fraction"),
               ("mean_predicted_clv_12m", "Mean predicted 52-week CLV", unit),
               ("mean_priority_score", "Mean priority proxy", unit + " (proxy)"))
    for axis, (column, title, ylabel) in zip(axes.flat, metrics):
        values = rows[column].to_numpy(dtype=float)
        known = np.isfinite(values)
        axis.bar(np.arange(len(rows))[known], values[known], color="#237B9B")
        axis.set_xticks(range(len(rows)), labels, rotation=55, ha="right", fontsize=8)
        axis.set(title=title, ylabel=ylabel)
        axis.axhline(0, color="#555555", linewidth=.7)
        if not known.all():
            axis.text(.02, .98, "Empty segments: mean unavailable", transform=axis.transAxes, va="top", fontsize=8)
    figure.suptitle(f"{dataset_id} | {clock.isoformat()} | analyzed N={int(rows.population_count.iloc[0])}")
    return figure
