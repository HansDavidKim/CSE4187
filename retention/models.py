"""Small sklearn-compatible outcome models; transformations fit on train only."""
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


def make_model(name, params=None, seed=42):
    params = params or {}
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if name == "constant":
        return make_pipeline(imputer, DummyClassifier(strategy="prior"))
    if name == "recency":
        return make_pipeline(imputer, StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
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
    return make_pipeline(imputer, learner)


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
