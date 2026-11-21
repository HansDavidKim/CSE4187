"""Export already-loaded canonical facts and isolated paired evaluation."""
from __future__ import annotations

import hashlib
import json
import numbers
import os
from pathlib import Path
import tempfile

import pandas as pd

from simulate.v4.data.weekly import build_experiment_log, build_weekly_panel
from simulate.v4.simulator.outcomes import experiment_window_days
from simulate.v4.simulator.paired_evaluation import PAIRED_METHOD_VERSION, paired_retention_rollouts


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _horizon(tables, metadata):
    labels = tables["labels"]
    durations = ((pd.to_datetime(labels.outcome_end_at, utc=True) - pd.to_datetime(labels.as_of_at, utc=True))
                 .dt.total_seconds() / 86400).dropna().unique()
    if len(durations) != 1 or durations[0] not in (28, 30):
        raise ValueError("exports require one explicit 28- or 30-day label window")
    days = int(durations[0])
    if metadata.get("outcome_window_days", days) != days:
        raise ValueError("label and metadata outcome windows differ")
    eligibility_days = experiment_window_days(metadata, "eligibility_window_days")
    if metadata.get("definition_id") != f"future_no_session_{days}d_active{eligibility_days}d":
        raise ValueError("paired retention requires the matching no-session definition")
    return days, eligibility_days


def build_no_purchase_labels(tables, metadata, experiment):
    """A separate order-based outcome using the same explicit observation window."""
    days, eligibility_days = _horizon(tables, metadata)
    columns = ["customer_id", "as_of_at", "snapshot_id", "outcome_end_at", "eligible_at_assignment"]
    result = experiment[columns].copy()
    count_column = f"order_count_{days}d"
    result[count_column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    observation_end = pd.Timestamp(metadata["observation_end_at"])
    if observation_end.tzinfo is None:
        raise ValueError("observation_end_at must be timezone-aware")
    purchases = tables["events"].loc[tables["events"].event_type.eq("purchase"),
                                      ["event_id", "customer_id", "event_at"]]
    orders = tables["orders"][["order_id", "purchase_event_id"]].merge(
        purchases, left_on="purchase_event_id", right_on="event_id", validate="one_to_one")
    for (t0, until), indices in result.groupby(["as_of_at", "outcome_end_at"]).groups.items():
        if until <= observation_end:
            counts = orders.loc[orders.event_at.ge(t0) & orders.event_at.lt(until)].groupby("customer_id").order_id.nunique()
            result.loc[indices, count_column] = result.loc[indices, "customer_id"].map(counts).fillna(0)
    result["no_purchase_churn"] = result[count_column].eq(0).astype("Int8")
    result["is_censored"] = result[count_column].isna()
    result["observation_status"] = result.is_censored.map({False: "complete", True: "immature"})
    result["definition_id"] = f"future_no_order_{days}d_active{eligibility_days}d"
    result["outcome_window_days"], result["eligibility_window_days"] = days, eligibility_days
    for column in ("dataset_id", "seed", "currency", "amount_unit"):
        result[column] = metadata[column]
    return result


def export_tables(tables, metadata, output, evaluation_output, repetitions=32):
    """Write already-loaded facts and private pairs; never reload or regenerate raw data."""
    output, evaluation_output = Path(output), Path(evaluation_output)
    if "evaluation_only" not in evaluation_output.parts or output.resolve() == evaluation_output.resolve():
        raise ValueError("paired evaluation must have a separate evaluation_only destination")
    if isinstance(repetitions, bool) or not isinstance(repetitions, numbers.Integral) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    days, eligibility_days = _horizon(tables, metadata)
    coupon_days = experiment_window_days(metadata, "coupon_validity_days")
    history_days = metadata.get("min_customer_history_days", 0)
    if isinstance(history_days, bool) or not isinstance(history_days, numbers.Integral) or history_days < 0:
        raise ValueError("metadata.min_customer_history_days must be a nonnegative integer")
    fingerprint = {key: metadata.get(key) for key in (
        "dataset_id", "seed", "definition_id", "model_sha256", "return_model_sha256", "simulator_code_sha256",
        "raw_database_sha256", "source_raw_database_sha256", "simulator_version", "policy_id",
        "observation_start_at", "observation_end_at", "as_of_at", "currency", "amount_unit")}
    if history_days:
        fingerprint["min_customer_history_days"] = int(history_days)
    fingerprint["sidecar_sha256"] = {
        name: hashlib.sha256(pd.util.hash_pandas_object(tables[name], index=True).to_numpy().tobytes()).hexdigest()
        for name in ("labels", "treatments", "diagnostics")}
    fingerprint.update(outcome_window_days=days, repetitions=int(repetitions),
                       paired_method_version=PAIRED_METHOD_VERSION, export_schema_version=2,
                       event_scope_version="shopping_coupon_review_v1")
    if (eligibility_days, coupon_days) != (30, 30):
        fingerprint.update(eligibility_window_days=eligibility_days, coupon_validity_days=coupon_days)
    destinations = {
        "weekly_panel": output / "weekly_panel.parquet", "experiment_log": output / "experiment_log.parquet",
        "no_purchase_labels": output / "no_purchase_labels.parquet",
        "paired_rollouts": evaluation_output / "paired_rollouts.parquet", "latent_truth": evaluation_output / "latent_truth.parquet",
        "manifest": output / "experiment_export_manifest.json",
    }
    paths = {key: str(path.resolve()) for key, path in destinations.items()}
    manifest_path = destinations["manifest"]
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("fingerprint") != fingerprint or previous.get("files") != paths:
            raise FileExistsError("existing exports describe another dataset, definition or destination")
        for name, path in destinations.items():
            if name != "manifest" and (not path.is_file() or _checksum(path) != previous.get("file_sha256", {}).get(name)):
                raise ValueError(f"existing experiment export is missing or changed: {path}")
        return {**previous, "reused": True}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(f"incomplete existing export is preserved: {path}")
    if "diagnostics" not in tables:
        raise ValueError("load diagnostics privately at export time for paired evaluation")
    output.mkdir(parents=True, exist_ok=True)
    evaluation_output.mkdir(parents=True, exist_ok=True)
    rows, published = {}, []
    with tempfile.TemporaryDirectory(prefix=".experiment-export-", dir=output.parent) as temporary:
        staging = Path(temporary)
        weekly = build_weekly_panel(tables, metadata)
        rows["weekly_panel"] = len(weekly)
        weekly.to_parquet(staging / destinations["weekly_panel"].name, index=False)
        del weekly
        experiment = build_experiment_log(tables, metadata)
        rows["experiment_log"] = len(experiment)
        experiment.to_parquet(staging / destinations["experiment_log"].name, index=False)
        no_purchase = build_no_purchase_labels(tables, metadata, experiment)
        rows["no_purchase_labels"] = len(no_purchase)
        no_purchase.to_parquet(staging / destinations["no_purchase_labels"].name, index=False)
        del no_purchase, experiment
        pairs, truth = paired_retention_rollouts(tables["diagnostics"], metadata, repetitions=int(repetitions))
        for name, frame in (("paired_rollouts", pairs), ("latent_truth", truth)):
            if not frame.evaluation_only.all():
                raise ValueError("paired outputs must be explicitly evaluation-only")
            frame["seed"] = metadata["seed"]
            rows[name] = len(frame)
            frame.to_parquet(staging / destinations[name].name, index=False)
        del pairs, truth
        checksums = {name: _checksum(staging / path.name) for name, path in destinations.items() if name != "manifest"}
        report = {
            "dataset_id": metadata["dataset_id"], "parent_dataset_id": metadata.get("parent_dataset_id"), "seed": metadata["seed"],
            "fingerprint": fingerprint, "files": paths, "file_sha256": checksums, "rows": rows, "reused": False,
            "outcome_window_days": days, "eligibility_window_days": eligibility_days, "coupon_validity_days": coupon_days,
            "repetitions": int(repetitions), "analysis_metadata": metadata,
            "paired_scope": "evaluation-only conditional return equation; not factual RNG or full-shopping replay",
            "no_purchase_definition_id": f"future_no_order_{days}d_active{eligibility_days}d",
            "factual_churn_definition_id": metadata["definition_id"],
        }
        (staging / manifest_path.name).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        try:
            for name, path in destinations.items():
                os.link(staging / path.name, path)
                published.append(path)
        except BaseException:
            for path in published:
                path.unlink()
            raise
    return report
