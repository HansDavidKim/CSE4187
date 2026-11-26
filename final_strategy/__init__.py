"""Validated model handoff and coupon decisions at one weekly cutoff."""
from .weekly_inputs import PredictionBatch, join_weekly_predictions, load_feature_store
from .weekly_validation import PredictionMetadata
from .pipeline import DecisionConfig, DecisionOutputs, build_decisions, write_outputs
from .strategy_profiles import build_strategy_profiles

# Historical CLI-facing alias. Engine clients use final_strategy.inputs.integrate_inputs.
integrate_inputs = join_weekly_predictions

__all__ = ["PredictionBatch", "PredictionMetadata", "integrate_inputs", "join_weekly_predictions", "load_feature_store",
           "DecisionConfig", "DecisionOutputs", "build_decisions", "write_outputs", "build_strategy_profiles"]
