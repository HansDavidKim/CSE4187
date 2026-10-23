# Script index

Run from the repository root after installing the project. Existing filenames,
arguments and output names are retained. Use `python -m scripts.<name>` for
commands that import other scripts; direct `python scripts/<name>.py` commands
may also require the repository root on `PYTHONPATH`, as in the original setup.

Argparse commands support `--help`; scripts with positional dataset names use
their final execution block instead. Examples:

```bash
python -m scripts.run_latent_session_pilot electronics cosmetics
python -m scripts.analyze_visit_gaps electronics cosmetics
python -m scripts.compare_masked_gru_clusters electronics cosmetics
python -m scripts.canonical_type_bounds
```

Raw data, saved embeddings and previous artifacts must be supplied separately.
The external-code column marks direct dependencies on the separate capstone
`src.irl` or `src.analysis` modules. Put that project on `PYTHONPATH` and set
`ECOMMERCE_CAPSTONE_ROOT` to its data root. Analytical and causal limitations
remain in [analysis.md](../analysis.md) and the experiment reports.

## Download

| Script | Purpose | External code |
|---|---|---|
| [get_nts_notears.py](get_nts_notears.py) | get_nts_notears | No |
| [get_starbucks_uplift.py](get_starbucks_uplift.py) | Download the pinned Udacity Starbucks promotion files without redistributing them. | No |

## Preprocessing and features

| Script | Purpose | External code |
|---|---|---|
| [extract_causal_features.py](extract_causal_features.py) | extract_causal_features | No |
| [prepare_amexpert.py](prepare_amexpert.py) | prepare_amexpert | No |
| [prepare_amexpert_temporal_features.py](prepare_amexpert_temporal_features.py) | Prepare role-separated AmExpert weekly features without running causal discovery. | No |
| [prepare_amexpert_train_sequences.py](prepare_amexpert_train_sequences.py) | Create a training-only sequence archive for causal discovery. | No |
| [prepare_backend_ts.py](prepare_backend_ts.py) | Create Backend-TS hourly history sequences and separately timed episode labels. | No |
| [prepare_backend_ts_v2.py](prepare_backend_ts_v2.py) | Build user-disjoint Backend-TS feature arrays from validated episode histories. | No |
| [prepare_kuairand.py](prepare_kuairand.py) | Prepare user-session exposure sequences; Pure is diagnostic only. | No |

## Behavior and data analysis

| Script | Purpose | External code |
|---|---|---|
| [analyze_electronics_inactivity.py](analyze_electronics_inactivity.py) | Describe observed post-second-session inactivity with a fixed follow-up window. | Yes |
| [analyze_late_inactivity.py](analyze_late_inactivity.py) | Compare observed third sessions by 30-day subsequent inactivity. | Yes |
| [analyze_long_inactivity.py](analyze_long_inactivity.py) | Anchor users at a common date and examine subsequent 60-day inactivity. | No |
| [analyze_other_inactivity.py](analyze_other_inactivity.py) | Evaluate 30-day inactivity in Retailrocket and sampled Cosmetics logs. | Yes |
| [analyze_persona_delays.py](analyze_persona_delays.py) | Retrospective persona-labelled second-session gaps; labels use future prefix. | Yes |
| [analyze_persona_evolution.py](analyze_persona_evolution.py) | Compare early and later session windows with saved masked-GRU checkpoints. | Yes |
| [analyze_persona_heterogeneity.py](analyze_persona_heterogeneity.py) | Within-cluster variation in observed prefixes, sessions and return gaps. | Yes |
| [analyze_visit_gaps.py](analyze_visit_gaps.py) | Compare first three observed UTC visit-day gaps; writes aggregate results only. | No |
| [audit_amexpert_campaigns.py](audit_amexpert_campaigns.py) | Reconstruct and audit customer-by-campaign-validity-period data. | No |
| [audit_backend_episode_splits.py](audit_backend_episode_splits.py) | Audit user boundaries and cross-episode label/history overlap in Backend-TS. | No |
| [audit_backend_overlap.py](audit_backend_overlap.py) | Cross-fitted, user-grouped propensity overlap diagnostic; not causal validation. | No |
| [audit_calendar_days.py](audit_calendar_days.py) | Audit calendar-day features on a deterministic user subsample; no training. | No |
| [audit_observed_days.py](audit_observed_days.py) | Audit observed event-days without inferring user inactivity or training IRL. | No |
| [audit_persona_stability.py](audit_persona_stability.py) | Cluster stability under train-user subsampling and KMeans seed changes. | No |
| [audit_recent_features.py](audit_recent_features.py) | Audit recent-state coverage, without training IRL or storing user records. | Yes |
| [audit_sequential_datasets.py](audit_sequential_datasets.py) | Audit sequential intervention data; absence of a campaign record is not a control. | No |
| [canonical_type_bounds.py](canonical_type_bounds.py) | Exploratory risk differences and assumption-dependent joint potential-outcome bounds. | No |
| [coupon_before_after.py](coupon_before_after.py) | Descriptive paired windows; intervention and redemption are not equivalent. | No |
| [coupon_long_term_pilot.py](coupon_long_term_pilot.py) | Observed 4-week outcomes by receipt, not a causal intervention estimate. | No |
| [coupon_persistence_patterns.py](coupon_persistence_patterns.py) | Six-week observed event patterns, not causal effect durations. | No |
| [coupon_persona_patterns.py](coupon_persona_patterns.py) | Compare observed pre/post activity, without assigning cross-domain GRU clusters. | No |
| [coupon_reactivation.py](coupon_reactivation.py) | Explore logged activity after a quiet period; not a coupon causal effect. | No |
| [evaluate_session_return.py](evaluate_session_return.py) | Temporal, administrative-censoring-aware evaluation of next-session hazards. | No |
| [explore_amexpert_uplift.py](explore_amexpert_uplift.py) | Sensitivity-only AIPW: assumes missing offer records mean no offer. | No |
| [tsboss_lingam_diagnostic.py](tsboss_lingam_diagnostic.py) | Exploratory lag-residual DirectLiNGAM orientation of a TS-BOSS skeleton. | No |
| [uplift_sequence_pilot.py](uplift_sequence_pilot.py) | Predict next-day recorded activity; coupon receipt is not randomized treatment. | No |

## Model and simulator pilots

| Script | Purpose | External code |
|---|---|---|
| [run_amexpert_user_sim.py](run_amexpert_user_sim.py) | Train user-wise one-step MaxEnt and evaluate observational choice simulation. | No |
| [run_backend_outcome_baseline.py](run_backend_outcome_baseline.py) | Predict observed 3h login from prehistory and episode treatment; not uplift. | No |
| [run_common_behavior_irl.py](run_common_behavior_irl.py) | Fit per-cluster finite-horizon MaxEnt IRL on common observed event windows. | Yes |
| [run_conditional_trajectory_irl.py](run_conditional_trajectory_irl.py) | Fit purchase trajectories conditional on a separately recorded offer proxy. | No |
| [run_electronics_personas.py](run_electronics_personas.py) | Reproduce the existing masked-GRU pilot and profile held-out input prefixes. | Yes |
| [run_group_trajectory_irl.py](run_group_trajectory_irl.py) | GRU masked sequence representation, GMM segments and finite-horizon MaxEnt IRL. | No |
| [run_history_trajectory_irl.py](run_history_trajectory_irl.py) | Train common-history MaxEnt IRL; outputs contain only aggregate statistics. | Yes |
| [run_latent_session_pilot.py](run_latent_session_pilot.py) | Fit/evaluate prospective-prefix session IRL + return hazard on two sources. | No |
| [run_nts_notears.py](run_nts_notears.py) | run_nts_notears | No |
| [run_ts_boss.py](run_ts_boss.py) | Run the authors' TS-BOSS on boundary-safe, pooled customer windows. | No |
| [train_lightgcn_embeddings.py](train_lightgcn_embeddings.py) | Train dataset-specific LightGCN on time-split binary user/item interactions. | No |

## Model comparisons

| Script | Purpose | External code |
|---|---|---|
| [benchmark_nts_synthetic.py](benchmark_nts_synthetic.py) | Small known-DAG sanity check; not a benchmark of real-world causality. | No |
| [compare_early_later_behavior.py](compare_early_later_behavior.py) | Compare sessions 1-2 with 3-4 for held-out users, within each user. | Yes |
| [compare_l1_history_irl.py](compare_l1_history_irl.py) | Compare quadratic L2 and quadratic elastic-net on the same users. | Yes |
| [compare_masked_gru_clusters.py](compare_masked_gru_clusters.py) | Compare clusterers on saved dataset-specific masked-GRU representations. | Yes |
| [compare_nonlinear_trajectory_irl.py](compare_nonlinear_trajectory_irl.py) | Compare linear and neural utilities under the same trajectory IRL MDP. | No |
| [compare_quadratic_history_irl.py](compare_quadratic_history_irl.py) | Compare linear and pairwise-quadratic trajectory rewards on the same users. | Yes |
| [sweep_electronics_personas.py](sweep_electronics_personas.py) | Profile held-out Electronics prefixes for multiple cluster counts. | Yes |
| [sweep_history_temperature.py](sweep_history_temperature.py) | Calibrate trajectory logits on held-out users, then evaluate once on test users. | Yes |
| [sweep_three_ecommerce_personas.py](sweep_three_ecommerce_personas.py) | Sweep K on three existing dataset-specific masked-GRU representations. | Yes |

## Reports and figures

| Script | Purpose | External code |
|---|---|---|
| [plot_analysis_summary.py](plot_analysis_summary.py) | Recreate figures in analysis.md from existing aggregate artifacts. | No |
| [summarize_backend_bootstrap.py](summarize_backend_bootstrap.py) | Summarize user-bootstrap lagged edge selection; exploratory, not a causal verdict. | No |
| [summarize_common_behavior_irl.py](summarize_common_behavior_irl.py) | Create one coefficient observation per fitted dataset/cluster, not per user. | No |

## R extraction

[coupon_journey_extract.R](coupon_journey_extract.R) reads the original RDS/RDA
files and writes daily transactions and coupon redemptions:

```bash
Rscript scripts/coupon_journey_extract.R <input-directory> <output-directory>
```

Similar names do not imply duplicate analyses: cohorts, session definitions and
estimation targets differ. Existing experiments and regression tests are retained.
