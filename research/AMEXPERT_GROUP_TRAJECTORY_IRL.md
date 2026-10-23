# Masked GRU → GMM → group trajectory MaxEnt IRL pilot

## Procedure

We use the same 1,582 existing users and 78 complete weekly bins as the earlier
pilot, with weeks 0–51 as training, 52–64 validation, 65–77 test. The graph is
the earlier NTS-NOTEARS fit using only weeks 0–51 and 100 sampled customers.
Graph-selected input sources are active offers and quantity; past transactions
are an explicit control and coupon discount is added as a label-related feature.
The graph does not constrain the subsequent IRL transition model.

A bidirectional GRU consumes four log1p-standardized weekly features. Randomly
mask 15% of individual values and predict them with a linear reconstruction
head, trained for 20 epochs. Group embeddings are mean-pooled hidden states on
unmasked *training* weeks only, standardized and fit to a four-component GMM.
This is a GRU masked autoencoder inspired by masked reconstruction, not BERT or
a transformer. Four groups are an unvalidated pilot choice. Training masked MSE
fell from 0.948 to 0.384; this is not an independent reconstruction test.

IRL actions are weekly purchase-status proxies (no purchase / purchase without
coupon discount / purchase with recorded coupon discount). A finite-state MDP
uses the **previous** week's category as state and deterministically transitions
to the current category. Each group has a 3-by-3 state/action reward table. The
entropy-regularized finite-horizon soft Bellman recursion produces a time-dependent
policy with terminal value zero; fit rewards by regularized likelihood of
demonstrated full 13-week trajectories. This is maximum causal entropy trajectory
IRL in a very small specified MDP, not the earlier one-step contextual model.
The reward at (0,0) is anchored to zero and L2 regularization selects one
representative among observationally non-identifiable rewards. Horizon and entropy
scale are fixed, not validated. The four training episodes are weeks 1–13, 14–26,
27–39, and 39–51; week 39 appears twice, and episodes within a user are dependent.

The GMM labels are learned from training weeks and fixed for validation and test;
test rollout uses the observed week-64 category as initial state. A separate pooled
trajectory reward is fit to the same demonstrations for comparison. No validation
data was used to select groups, hyperparameters, horizon, or regularization.

## Results

GMM group sizes: 258, 149, 460, 715. Group-model validation log loss: 0.8152.
On the test period (teacher forced, observed previous action):

| Metric | Group trajectory IRL | Pooled trajectory IRL |
|---|---:|---:|
| Log loss ↓ | 0.8344 | 0.8506 |
| Brier ↓ | 0.5094 | 0.5157 |
| Accuracy | 0.6206 | 0.6206 |

On 10 full test-horizon rollouts, actual vs generated outcome proportions were
0.319 vs 0.410 (no purchase), 0.596 vs 0.528 (purchase without recorded coupon
discount), and 0.085 vs 0.062 (purchase with recorded coupon discount). Actual vs
generated switch rate was 0.408 vs 0.349 and P(purchase | prior purchase) was
0.775 vs 0.762. Average per-user action-rate absolute errors were 0.238, 0.223,
and 0.079. The model **does not yet reproduce all behavior accurately**; it over-
predicts no purchase and produces too few switches. The earlier one-step personalized
model achieved test log loss 0.8191 on the same time split; models use different
state information and personalization, so this is not an isolated ablation of IRL.

Full group-wise distributions and metrics are in the local `summary.json`; reward
tables, embeddings, memberships, fitted GRU and rollouts are saved locally. They
are linked to customers and must not be redistributed without rights/privacy review.

## Limits and reproduction

One group-level reward is fitted per group, not individual trajectory rewards.
Previous-action state alone excludes quantity and offer dynamics; there is no action
availability mask, and a coupon-discount outcome may be simulated even when no
coupon was actually offered. This is observational behavioral reconstruction,
not item recommendation choice or causal intervention evaluation. Customer
coverage, shared calendar effects, train-test distribution shift, group stability,
cold start and reward identification remain unresolved. No GMM model-selection,
independent masked reconstruction test, or confidence intervals were performed.

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_group_trajectory_irl.py
```

It requires the prepared local panel and training-only NTS graph. Next: compare
seeds/group counts, expand the finite MDP with NTS-selected lagged features and
realistic learned transitions, constrain available offers, evaluate chronological
holdouts, and extend to individual hierarchical trajectory rewards. No policy
effect claims without intervention evidence.
