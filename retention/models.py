"""Small sklearn-compatible outcome models; transformations fit on train only."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Learn finite per-feature limits from the fitting partition only."""

    def __init__(self, lower=.005, upper=.995):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        frame = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.lower_ = frame.quantile(self.lower)
        self.upper_ = frame.quantile(self.upper)
        return self

    def transform(self, x):
        frame = pd.DataFrame(x, columns=self.feature_names_in_).replace([np.inf, -np.inf], np.nan)
        return frame.clip(self.lower_, self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)


def make_model(name, params=None, seed=42):
    params = params or {}
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if name == "constant":
        return make_pipeline(QuantileClipper(), imputer, DummyClassifier(strategy="prior"))
    if name == "recency":
        return make_pipeline(QuantileClipper(), imputer, StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    if name == "lightgbm":
        learner = LGBMClassifier(n_estimators=params.get("iterations", 120), learning_rate=params.get("learning_rate", .05),
            num_leaves=params.get("num_leaves", 15), min_child_samples=params.get("min_child_samples", 30),
            reg_lambda=1., class_weight="balanced", random_state=seed, n_jobs=2, verbosity=-1,
            deterministic=True, force_col_wise=True)
    elif name == "catboost":
        learner = CatBoostClassifier(iterations=params.get("iterations", 120), learning_rate=params.get("learning_rate", .05),
            depth=params.get("depth", 5), auto_class_weights="Balanced", random_seed=seed,
            thread_count=2, verbose=False, allow_writing_files=False)
    elif name == "xgboost":
        # Sample weights are supplied by fit_model using THIS training subset only.
        learner = XGBClassifier(n_estimators=params.get("iterations", 120), learning_rate=params.get("learning_rate", .05),
            max_depth=params.get("depth", 4), min_child_weight=2., reg_lambda=1.,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=2)
    else:
        raise ValueError(name)
    return make_pipeline(QuantileClipper(), imputer, learner)


def fit_model(model, x, y, name):
    if name == "xgboost":
        y = np.asarray(y)
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        if min(n0, n1) == 0:
            raise ValueError("XGBoost requires both classes")
        weights = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))
        model.fit(x, y, xgbclassifier__sample_weight=weights)
    else:
        model.fit(x, y)
    return model


def input_columns(name, columns):
    return ["visit_recency_days"] if name == "recency" else columns


def predict(model, x):
    return np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)


def probability_input(p):
    """Finite log odds for sigmoid calibration."""
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    return np.log(p / (1-p)).reshape(-1, 1)


def calibrated(p, calibrator):
    return np.asarray(p) if calibrator is None else calibrator.predict_proba(probability_input(p))[:, 1]
