"""Intervention-aware e-commerce synthetic panel generator."""

from .pipeline import EcommercePipeline, load_starbucks_panel
from .uplift import StarbucksUpliftSimulator, load_uplift_data
from .amexpert import build_customer_week_panel, save_amexpert_panel
from .nts_adapter import load_sequences, make_lag_windows
from .nts_training import fit_nts_notears
from .causal_features import extract_markov_boundary, save_feature_sets
from .choice import CalibratedMultinomialLogit

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

