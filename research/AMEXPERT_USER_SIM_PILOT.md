# AmExpert NTS → user-wise MaxEnt choice pilot

## Setup

1,582 customer sequences have 78 complete observed weeks. Split each user chronologically:
weeks 0–51 training, 52–64 validation, 65–77 test. Lag features at the first usable training
week come only from earlier weeks. NTS-NOTEARS is refit solely on the first 52 weeks, sampling
100 customers with seed 2026 and dropping the almost-duplicate `unique_coupons_started`.
It converged to h=3.88e-9, with 17 edges and an acyclic thresholded same-week graph. The
caller must ensure a supplied graph was fit within the training period; the simulation script
does not verify graph provenance automatically.

The three mutually exclusive **observed weekly outcome proxies** are no transaction, purchase
with no recorded coupon discount, and purchase with recorded coupon discount. They are not
observed recommendation choices, and coupon discount is not a confirmed redemption. Zero-filled
weeks also assume observation coverage and can reflect customer inactivity or missingness.

States include lagged parents selected by the training-only graph for relevant outcome/offer
nodes, plus explicitly designated lag-1 transaction and active-offer control variables. In this
run the four inputs are lag-1 active offers, lag-1 quantity, lag-1 transactions, and lag-2
quantity. No same-week outcomes are used as decision state.

Each customer's linear reward vector is optimized separately through a joint objective with
L2 shrinkage toward pooled coefficients. The outside-option reward is zero to fix the additive
reference; the pooled model supplies the prior. A multinomial softmax produces a one-step
maximum-entropy conditional policy. This is contextual one-step MaxEnt IRL / regularized
choice imitation, **not** multi-step trajectory MaxEnt IRL or uniquely identified true reward.
Penalty is selected from 2, 10, 50 and temperature calibrated on validation, then final held-out
metrics use only test weeks. The fitted model is not refit on validation.

The 13-week rollout starts from observed history, then feeds back simulated outcome vectors
resampled from action-matched training weeks (same user when possible, otherwise pooled).
Previously recorded offer-count schedules are replayed, not intervened on. This approximate
resampling preserves some within-week relationships but is not an identified NTS structural
transition model and cannot establish a counterfactual policy effect. Ten simulations are
generated using fixed seeds and stored locally.

## Held-out results

| One-step model | Log loss ↓ | Brier ↓ | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| User-specific | 0.8191 | 0.4988 | 0.6121 | 0.4414 |
| Pooled | 0.8430 | 0.5116 | 0.6071 | 0.3962 |
| Smoothed user frequency | 0.8385 | 0.5166 | 0.5936 | 0.4434 |

Macro F1 is marginally better for the frequency baseline: personalization is not uniformly
superior across metrics. The validation-selected shrinkage is 10; normalized utility
temperature is 0.934 and normalized entropy is 0.730 on validation.

| Outcome | Observed | Generated (10 rollouts) | Per-user rate MAE |
|---|---:|---:|---:|
| No purchase | 0.319 | 0.362 | 0.200 |
| Purchase, no coupon discount | 0.596 | 0.547 | 0.195 |
| Purchase, recorded coupon discount | 0.085 | 0.090 | 0.084 |

Observed vs generated successive-week action switch rates are 0.408 vs 0.463; probability of
purchase following purchase is 0.775 vs 0.718. The generated trajectories are only partially
similar, and 13-week per-user rates are themselves noisy. The time split is within existing
customers, so this is not cold-start evaluation. The single-seed graph and this single holdout
do not establish robustness.

The per-user reward arrays and generated trajectories inherit data rights and privacy concerns;
they remain local, and neither the underlying dataset nor derivatives should be redistributed
without a rights review. Follow-up: compare no-graph and alternative-state ablations, more
temporal splits, better user-specific transition models, explicit observed item choice sets,
GMM for genuinely new synthetic users, and intervention/overlap validation before RL policy
comparison. References: [MaxEnt IRL](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html),
[reward identifiability](https://proceedings.mlr.press/v139/kim21c.html).
