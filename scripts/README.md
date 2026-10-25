# Script index

Run from the repository root after installing the project. The listed entry points are retained for the current workflow. Use `python -m scripts.<name>` for
commands that import other scripts; direct `python scripts/<name>.py` commands
may also require the repository root on `PYTHONPATH`, as in the original setup.

Argparse commands support `--help`; scripts with positional dataset names use
their final execution block instead. Examples:

```bash
python -m scripts.run_latent_session_pilot electronics cosmetics
python -m analysis.eda visit-gaps electronics cosmetics
python -m analysis.feature_selection electronics --output artifacts/feature-selection-run
python -m analysis.persona.cluster compare electronics cosmetics
python -m analysis.uplift.canonical
python -m analysis.uplift.observed reactivation
```

Raw data, saved embeddings and previous artifacts must be supplied separately.
The external-code column marks direct dependencies on the separate capstone
`src.irl` or `src.analysis` modules. Put that project on `PYTHONPATH` and set
`ECOMMERCE_CAPSTONE_ROOT` to its data root. Analytical and causal limitations
remain in [analysis.md](../analysis.md) including historical reports in its appendix.

## Behavior and data analysis

| Script | Purpose | External code |
|---|---|---|
| [prepare_kuairand.py](prepare_kuairand.py) | Historical KuaiRand preparation; retained for regression tests. | No |
| [tsboss_lingam_diagnostic.py](tsboss_lingam_diagnostic.py) | Historical TS-BOSS diagnostic; retained for regression tests. | No |
| [eda.py](../src/analysis/eda.py) | Intervention audits, causal features, inactivity, visit gaps and session returns; `late-session` needs external loader. | Conditional |
| [feature_selection.py](../src/analysis/feature_selection.py) | Past-only action feature selection and exploratory session-level NTS; requires raw data and local NTS source. | No |
| [behavior.py](../src/analysis/persona/behavior.py) | Persona windows, dynamics, delays and within-cluster heterogeneity. | Yes |
| [audit_calendar_days.py](audit_calendar_days.py) | Audit calendar-day features on a deterministic user subsample; no training. | No |
| [audit_observed_days.py](audit_observed_days.py) | Audit observed event-days without inferring user inactivity or training IRL. | No |
| [history.py](../src/analysis/irl/history.py) | History IRL, reward comparisons, temperature sweep and feature audit. | Yes |
| [canonical.py](../src/analysis/uplift/canonical.py) | Assumption-dependent potential-outcome bounds. | No |
| [observed.py](../src/analysis/uplift/observed.py) | Descriptive paired, persona, reactivation and persistence analyses; not causal effects. | No |
| [sequence.py](../src/analysis/uplift/sequence.py) | Predict recorded activity; receipt is not randomized treatment. | No |

## Model and simulator pilots

| Script | Purpose | External code |
|---|---|---|
| [run_backend_outcome_baseline.py](run_backend_outcome_baseline.py) | Historical Backend-TS baseline; retained for regression tests. | No |
| [run_ts_boss.py](run_ts_boss.py) | Historical TS-BOSS adapter; retained for regression tests. | No |
| [run_common_behavior_irl.py](run_common_behavior_irl.py) | Fit per-cluster finite-horizon MaxEnt IRL on common observed event windows. | Yes |
| [run_latent_session_pilot.py](run_latent_session_pilot.py) | Fit/evaluate prospective-prefix session IRL + return hazard on two sources. | No |
| [run_agent_benchmark.py](run_agent_benchmark.py) | Train six weekly coupon policies on 1,000 users for 13 weeks, then freeze and test on 5,000 users for 52 weeks. | No |
| [train_lightgcn_embeddings.py](train_lightgcn_embeddings.py) | Train dataset-specific LightGCN on time-split binary user/item interactions. | No |

## Model comparisons

| Script | Purpose | External code |
|---|---|---|
| [cluster.py](../src/analysis/persona/cluster.py) | Compare, sweep and assess stability of embedding clusters. | Yes |

## Reports and figures

| Script | Purpose | External code |
|---|---|---|
| [plot_analysis_summary.py](plot_analysis_summary.py) | Recreate figures in analysis.md from existing aggregate artifacts. | No |
| [summarize_common_behavior_irl.py](summarize_common_behavior_irl.py) | Create one coefficient observation per fitted dataset/cluster, not per user. | No |

## R extraction

[coupon_journey_extract.R](coupon_journey_extract.R) reads the original RDS/RDA
files and writes daily transactions and coupon redemptions:

```bash
Rscript scripts/coupon_journey_extract.R <input-directory> <output-directory>
```

Some retained analyses use different cohorts or session definitions.
