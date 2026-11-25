"""Hand off maintained T/X/H purchase probabilities to the weekly final strategy."""
from dataclasses import dataclass
import hashlib
import pickle

import numpy as np
import pandas as pd

from .common import schema as S
from .common.base import BaseUpliftLearner
from .common.weekly import load_features


@dataclass(frozen=True)
class StrategyPredictions:
    frame: pd.DataFrame
    metadata: dict
    producer_metadata: dict


def bind_training_contract(model, training_data, *, selected_customers=()):
    """Record the actual input clock and development users; never relabel a legacy fit."""
    contract = training_data.attrs.get("input_contract")
    if contract is None:
        raise ValueError("Training data must declare its feature/time contract")
    model.uplift_input_contract_ = dict(contract)
    model.uplift_input_features_ = list(S.FEATURES)
    model.uplift_training_ = {
        "dataset_id": training_data.attrs.get("dataset_id"),
        "raw_database_sha256": (training_data.raw_database_sha256.iloc[0]
            if "raw_database_sha256" in training_data and training_data.raw_database_sha256.nunique() == 1 else None),
        "training_customers": training_data[S.CUSTOMER_ID].tolist(),
        "selection_customers": list(selected_customers),
    }
    return model


def predict_for_strategy(model, feature_directory, *, allow_training_customers=False):
    """Predict the exact decision cohort without reading assignment or future outcomes."""
    if not isinstance(model, BaseUpliftLearner):
        model = BaseUpliftLearner.load(model)
    features = load_features(feature_directory)
    if (getattr(model, "uplift_input_contract_", None) != features.attrs["input_contract"]
            or getattr(model, "uplift_input_features_", None) != S.FEATURES):
        raise ValueError("Model input contract differs from the weekly KIT36; refit through the maintained Uplift API")
    training = model.uplift_training_
    overlap = []
    if training["dataset_id"] == features.dataset_id.iloc[0]:
        recorded = training.get("raw_database_sha256")
        if recorded is not None and not features.raw_database_sha256.eq(recorded).all():
            raise ValueError("Model training dataset identity was reused with a different source ledger")
        development = set(training["training_customers"]) | set(training["selection_customers"])
        overlap = features.loc[features.customer_id.isin(development), "customer_id"].tolist()
    if overlap and not allow_training_customers:
        raise ValueError("Decision cohort includes Uplift training/selection customers; use a fresh cohort or explicitly allow development checks")
    prediction = model.predict(features[S.FEATURES])
    if not prediction.index.equals(features.index) or list(prediction) != ["p0", "p1", "uplift_score"]:
        raise ValueError("Uplift predictions changed customer row order or output schema")
    values = prediction[["p0", "p1"]].to_numpy(float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("Uplift purchase probabilities must be finite and in [0,1]")
    if not np.allclose(prediction.uplift_score, prediction.p1-prediction.p0, rtol=0, atol=1e-12):
        raise ValueError("Uplift score must equal purchase p1-p0")
    identity = [name for name in ["dataset_id", "customer_id", "as_of_at", "snapshot_id", "reference_week",
        "raw_database_sha256", "currency", "amount_unit"] if name in features]
    frame = features[identity].join(prediction)
    frame["input_cutoff_at"] = frame.as_of_at
    frame["prediction_end_at"] = frame.as_of_at + pd.Timedelta(weeks=4)
    version = "kit36-purchase28d:" + hashlib.sha256(pickle.dumps(model)).hexdigest()[:16]
    metadata = {"model_version": version, "target_definition_id": "purchase_28d_randomized_coupon_v1",
                "horizon_value": 4, "horizon_unit": "weeks", "value_semantics": "probability_pair"}
    producer = {"component": "uplift", "model": model.name, "training": training,
                "input_contract": features.attrs["input_contract"], "training_overlap_customers": overlap,
                "development_overlap_allowed": bool(allow_training_customers),
                "source_ledger_verified": features.attrs["raw_database_sha256_verified"],
                "scope": "Fitted purchase probability pair; no individual canonical type or real-world causal effect is identified"}
    return StrategyPredictions(frame, metadata, producer)
