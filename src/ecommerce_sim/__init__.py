"""Intervention-aware e-commerce synthetic panel generator."""

from ecommerce_sim.simulate.pipeline import EcommercePipeline, load_starbucks_panel
from ecommerce_sim.simulate.uplift import StarbucksUpliftSimulator, load_uplift_data
from ecommerce_sim.preprocessing.amexpert import build_customer_week_panel, save_amexpert_panel
from ecommerce_sim.preprocessing.nts_adapter import load_sequences, make_lag_windows
from ecommerce_sim.models.nts_training import fit_nts_notears
from ecommerce_sim.analysis.causal_features import extract_markov_boundary, save_feature_sets
from ecommerce_sim.models.choice import CalibratedMultinomialLogit

__all__ = [
    "EcommercePipeline",
    "StarbucksUpliftSimulator",
    "load_starbucks_panel",
    "load_uplift_data",
    "build_customer_week_panel",
    "save_amexpert_panel",
    "load_sequences",
    "make_lag_windows",
    "fit_nts_notears",
    "extract_markov_boundary",
    "save_feature_sets",
    "CalibratedMultinomialLogit",
]


# Keep existing imports and serialized class paths working after the module moves.
from importlib import import_module as _import_module
import sys as _sys

_COMPAT_MODULES = {
    "amexpert": "preprocessing.amexpert",
    "amexpert_temporal_features": "preprocessing.amexpert_temporal_features",
    "backend_ts": "preprocessing.backend_ts",
    "backend_ts_v2": "preprocessing.backend_ts_v2",
    "calendar_day_features": "preprocessing.calendar_day_features",
    "campaign_audit": "analysis.campaign_audit",
    "causal_features": "analysis.causal_features",
    "choice": "models.choice",
    "common_behavior_irl": "models.common_behavior_irl",
    "conditional_trajectory_irl": "models.conditional_trajectory_irl",
    "extended_session_irl": "models.extended_session_irl",
    "history_trajectory_irl": "models.history_trajectory_irl",
    "inactivity": "analysis.inactivity",
    "latent_session_model": "simulate.latent_session_model",
    "lightgcn": "models.lightgcn",
    "nonlinear_trajectory_irl": "models.nonlinear_trajectory_irl",
    "nts_adapter": "preprocessing.nts_adapter",
    "nts_training": "models.nts_training",
    "observed_day_features": "preprocessing.observed_day_features",
    "persona_evolution": "analysis.persona_evolution",
    "pipeline": "simulate.pipeline",
    "recent_history_features": "preprocessing.recent_history_features",
    "session_return": "analysis.session_return",
    "trajectory_irl": "models.trajectory_irl",
    "uplift": "simulate.uplift",
    "user_choice": "models.user_choice",
}

for _name, _target in _COMPAT_MODULES.items():
    _module = _import_module(f".{_target}", __name__)
    _sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
