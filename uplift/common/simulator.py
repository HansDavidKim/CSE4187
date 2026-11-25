"""Assemble an uplift cohort from factual simulator assignments and precomputed KIT39."""

import re

import pandas as pd

from common.simulator_features import SNAPSHOT_KEYS, open_feature_store, read_kit_features
from . import schema as S

_ELIGIBILITY_DEFINITION = (
    re.escape(f"session start in [as_of_at-{S.ELIGIBILITY_DAYS}d,as_of_at)")
    + r"(?: and first observed shopping event <= as_of_at-[1-9][0-9]*d "
    r"\(view/cart/remove_from_cart/purchase before as_of_at\))?"
)


def load_feature_dataset(path, *, for_training=True, ledger=None):
    """One row per eligible customer. An incomplete 28-day outcome stays missing."""
    from .weekly import manifest, load_dataset
    metadata = manifest(path)
    if metadata.get("time_spec") is not None or metadata.get("feature_version") == "kit39-weekly-4-8-13w-v1":
        return load_dataset(path, for_training=for_training, ledger=ledger)
    with open_feature_store(path) as connection:
        features = read_kit_features(connection, S.KIT39)
        experiment = pd.read_sql_query("""
            SELECT dataset_id, customer_id, as_of_at, snapshot_id, assigned_at,
                   treatment, assignment_probability, eligibility_definition, eligible_at_assignment,
                   outcome_horizon_days, outcome_observed, factual_no_order
            FROM experiment_log
        """, connection)
    if (experiment.empty or experiment[SNAPSHOT_KEYS].isna().any().any()
            or experiment.customer_id.duplicated().any() or experiment.dataset_id.nunique() != 1):
        raise ValueError("Expected one experiment with one assignment per customer")
    eligibility = experiment.eligibility_definition.astype("string")
    if (not experiment.assigned_at.eq(experiment.as_of_at).all()
            or eligibility.nunique() != 1
            or not eligibility.str.fullmatch(_ELIGIBILITY_DEFINITION, na=False).all()
            or not experiment.outcome_horizon_days.eq(S.OUTCOME_DAYS).all()
            or (for_training and not experiment.assignment_probability.eq(0.5).all())):
        raise ValueError("Expected a 1:1 assignment with 28-day eligibility and outcomes")
    flags = ["eligible_at_assignment", "outcome_observed", S.TREATMENT]
    if not experiment[flags].isin([0, 1]).all().all():
        raise ValueError("Missing or invalid assignment or observation flags")
    eligible = experiment.loc[experiment.eligible_at_assignment.eq(1)]
    data = eligible.merge(features, on=SNAPSHOT_KEYS, how="left", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("Missing KIT39 features for an assignment snapshot")
    observed = data.outcome_observed.eq(1)
    if not data.loc[observed, "factual_no_order"].isin([0, 1]).all():
        raise ValueError("Observed outcomes require a binary factual_no_order")
    data[S.LABEL] = 1 - data.factual_no_order.where(observed)
    data[S.T0] = pd.to_datetime(data.as_of_at, utc=True)
    result = data[["dataset_id", S.CUSTOMER_ID, S.T0, S.TREATMENT, *S.KIT39, S.LABEL]]
    result.attrs = {"dataset_id": experiment.dataset_id.iloc[0], "assignment_probability": 0.5,
                    "input_contract": {"feature_version": "kit39-review-cart-v3", "time_spec": "legacy_calendar"}}
    return result
