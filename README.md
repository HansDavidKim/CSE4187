# eCommerce-Sim

## Role-separated AmExpert features

The separate preprocessing pilot in `scripts/prepare_amexpert_temporal_features.py` builds
pre-week history, offer-record proxies, binary purchase behavior, and post-decision outcomes.
It preserves unknown offer exposure and first-week missing history; these are not validated causal
interventions. See [`research/AMEXPERT_TEMPORAL_FEATURES_V2.md`](research/AMEXPERT_TEMPORAL_FEATURES_V2.md).
This feature panel is not yet an input-ready NTS-NOTEARS archive; graph constraints and missing-data
handling must be implemented and validated first.

## Corrected AmExpert conditional trajectory pilot

`scripts/run_conditional_trajectory_irl.py` separates the user's binary purchase action from the
current-week recorded-offer proxy. The older `run_group_trajectory_irl.py` and
`compare_nonlinear_trajectory_irl.py` categorize coupon-discount usage as part of the action and
do **not** implement the requested current-intervention-conditional model; their results are legacy
behavioral pilots, not intervention effects. See
[`research/AMEXPERT_CONDITIONAL_IRL.md`](research/AMEXPERT_CONDITIONAL_IRL.md).

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_conditional_trajectory_irl.py
```

This pilot conditions on an offer *record* rather than a verified randomized or pre-purchase
intervention. Missing records do not establish no exposure; its outputs are not causal uplift.

An intervention-aware e-commerce synthetic panel generator. It turns timestamped user actions into
a lagged panel, learns a nonlinear DAG, adds conservative intervention evidence, learns user
segments with a Gaussian mixture, estimates one-step maximum-entropy IRL reward features, and
rolls out a fitted structural transition model.

## Included development data

The compact [Starbucks rewards dataset](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data)
has been downloaded to `data/raw/starbucks/` for local smoke tests. Its Kaggle metadata does not
contain a clear reuse license, so it is **not approved as a headline KSC paper dataset**. Raw data
is ignored by git. The paper plan uses licensed Criteo Uplift v2.1 for randomized intervention
validation and CC BY 4.0 UCI Online Retail for behavioral trajectories. See
[`research/DATA_POLICY.md`](research/DATA_POLICY.md) and
[`research/DATASET_SHORTLIST.md`](research/DATASET_SHORTLIST.md).

## Run

```bash
python -m pip install -e .
python -m ecommerce_sim.cli run \
  --data data/raw/starbucks \
  --output artifacts/starbucks-run \
  --max-users 2000 \
  --max-rows 5000 \
  --max-iter 10
```

Fast wiring check:

```bash
python -m ecommerce_sim.cli run --method correlation-smoke --max-users 200 --max-rows 1000 \
  --synthetic-users 50 --periods 14 --output artifacts/smoke
pytest -q
```

Official Udacity promotion-response calibration:

```bash
python scripts/get_starbucks_uplift.py --acknowledge-license-uncertainty
python -m ecommerce_sim.cli uplift \
  --data data/raw/starbucks_uplift \
  --output artifacts/starbucks-uplift \
  --synthetic-rows 100000
```

The downloader pins the Udacity source commit, SHA-256, byte size, row count and schema. It also
writes `source_manifest.json`. Synthetic outputs contain no original rows; row-level model
representations remain local-only derived artifacts and must not be redistributed without a rights
review. This module is a one-step promotion contextual bandit; it does not create multi-step
recommendation dynamics.

Outputs:

- `causal_dag.json`: lag-aware DAG and action/outcome evidence
- `user_features.csv`: GMM responsibilities and IRL reward representation by user
- `synthetic_panel.csv`: generated trajectories under sampled actions
- `summary.json`: run size and identification warning

The uplift path writes `uplift_metrics.json`, `test_user_representations.csv`, randomized-policy
synthetic data and targeted-policy synthetic data.

## AmExpert sequential path

Place the six user-obtained AmExpert CSV files in a local directory and run:

```bash
PYTHONPATH=src python scripts/prepare_amexpert.py \
  --data data/raw/amexpert \
  --output artifacts/amexpert-panel
```

This validates the relational archive and builds weekly customer sequences plus explicit offsets,
so lag windows never cross user boundaries. See
[`research/AMEXPERT_NTS_PLAN.md`](research/AMEXPERT_NTS_PLAN.md) for the NTS-NOTEARS design,
identification limits and required baselines.

Only complete weeks within the transaction observation period are retained; partial boundary
weeks and unobserved later campaign dates are not treated as zero purchases. The local pilot and
its data-quality limitations are recorded in [`research/EXPERIMENT_2026-10-20.md`](research/EXPERIMENT_2026-10-20.md).

The `ecommerce_sim.nts_adapter` loader converts that NPZ into `[window, feature, lag+target]`
arrays and preserves a customer-sequence index for grouped validation and bootstrap stability.

Fetch the official Apache-2.0 implementation at the pinned paper-code commit, then fit the
multi-sequence adaptation:

```bash
python scripts/get_nts_notears.py
PYTHONPATH=src python scripts/run_nts_notears.py --max-lag 2
```

The runner applies the official CNN and bounded L-BFGS-B components to independent customer
windows, rather than concatenating customers into a false continuous series. Its default prior
forbids same-week transaction outcomes from pointing into campaign-start assignment variables.
The resulting `nts_dag.json` records edge lags, weights, constraints and acyclicity history.
When `--max-sequences` is set, customers are sampled without replacement using `--seed`.
Use `--drop-features` for sensitivity analyses of redundant variables.

Extract temporal Markov-boundary hypotheses and a leakage-safe state feature set:

```bash
PYTHONPATH=src python scripts/extract_causal_features.py
```

The full boundary is retained for analysis, while the default state set only admits lagged
parents and explicitly pre-decision contemporaneous variables. Children and spouses are not
silently fed into IRL because they can be measured after the decision.
The feature extractor refuses graphs that did not satisfy both the optimization convergence
criterion and a discrete acyclicity check.

`CalibratedMultinomialLogit` provides the downstream economic choice layer. It fixes centered
utility variance to one before bounded temperature estimation, avoiding the unidentified
utility-scale/temperature tradeoff. Diagnostics expose normalized choice entropy and mean top-1
probability so generated behavior can be checked for both near-random and near-deterministic collapse.

### AmExpert user-specific choice pilot

The local end-to-end pilot uses a training-only NTS graph, per-user regularized one-step
contextual MaxEnt rewards, validation-calibrated multinomial logit, and a held-out rollout.
Full details, results, split and caveats: [`research/AMEXPERT_USER_SIM_PILOT.md`](research/AMEXPERT_USER_SIM_PILOT.md).

```bash
PYTHONPATH=src python scripts/prepare_amexpert_train_sequences.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_nts_notears.py \
  --sequences artifacts/amexpert-train-only/customer_sequences.npz --max-lag 2 \
  --hidden 4 --lambda1 0.03 --lambda2 0.01 --threshold 0.15 --max-iter 12 \
  --max-sequences 100 --drop-features unique_coupons_started --seed 2026 \
  --output artifacts/amexpert-nts-train-2026
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_amexpert_user_sim.py
```

The output arrays are local user-linked representations and simulated sequences; do not
redistribute them until the competition data and derivative rights are checked. Only weekly
purchase-status proxies are modeled: no observed recommendation choice sets are available.

For the masked GRU + GMM + group trajectory MaxEnt IRL pilot, run
`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_group_trajectory_irl.py`.
See [`research/AMEXPERT_GROUP_TRAJECTORY_IRL.md`](research/AMEXPERT_GROUP_TRAJECTORY_IRL.md)
for its small finite-state MDP, test results, and limitations.

For the expanded-state linear-versus-neural trajectory reward and greedy sensitivity,
see [`research/AMEXPERT_NONLINEAR_TRAJECTORY_IRL.md`](research/AMEXPERT_NONLINEAR_TRAJECTORY_IRL.md).

## Interpretation boundary

The Starbucks log records offers but the Kaggle card does not establish randomized assignment or
a clear reuse license.
The DAG is a hypothesis generator, not an identified causal truth. The IRL component is explicitly
one-step contextual MaxEnt IRL; long-horizon reward recovery requires transition-rich trajectories,
policy support/overlap checks, and preferably randomized or otherwise identified interventions.
