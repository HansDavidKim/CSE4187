# History-feature trajectory MaxEnt IRL: exploratory pilot

## Literature and design

GRU4Rec models the sequence within a session; BERT4Rec learns masked sequence
representations; TiSASRec additionally models inter-event time. These papers do
not prescribe this pilot's hand-crafted features or validate its learned rewards.
They motivate preserving short-term and longer-term histories and, eventually, time.
See [GRU4Rec](https://arxiv.org/abs/1511.06939),
[BERT4Rec](https://arxiv.org/abs/1904.06690), and
[TiSASRec](https://jiachengli1995.github.io/files/wsdm20.pdf).
This pilot uses the globally normalized trajectory distribution of
[MaxEnt IRL](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html).

## Exact feature definition

At each decision, use only the actions before that decision:

| State | Definition |
| --- | --- |
| intercept | constant 1 |
| session_views | views already observed in the current session |
| session_carts | cart events already observed in the current session |
| session_purchases | purchase events already observed in the current session |
| previous_sessions | completed earlier observed sessions |
| previous_purchases | purchases observed in earlier sessions |

Transform counts with `log1p(count)/log(65)`. Reward is linear in these states
interacted with each of the cart and purchase actions; view has reward zero as
the reference. Thus there are 12 fitted coefficients per dataset. Candidate
three-action trajectories update current-session counts deterministically;
completed-session counts remain fixed within a window. Exact enumeration of
the 27 possible trajectories yields their normalization and the conditional
trajectory likelihood. L2 penalty is 1, temperature is fixed at 1.

The counts are summaries, not a GRU embedding, product choice, an inferred
causal DAG, or a complete user representation. Inter-event time and recency,
item identity/diversity, prices, cart inventory and session termination are
not modeled: the loader drops item IDs, Retailrocket price is not directly
available, and their counterfactual transition dynamics are not specified.
These omissions are not statements that the original data never contains them.
The same applies to Cosmetics' remove-from-cart event: it breaks windows, and
is not imputed as absent in other datasets.

## Data selection and validation

Reuse the earlier 12k-user sample and disjoint user split, but do not use
the earlier GRU cluster assignment. The sample is stratified based on the
number of sessions in a user's full observation history, so this is a
retrospectively selected population, not a prospective or representative cohort.
Use the first 64 observed events and nonoverlapping complete three-event blocks
within each session. Inferred session boundaries and unmodeled events split
blocks; shorter fragments are discarded. Tied timestamps preserve raw input
order, which may not be the actual behavior order. All three actions are
assumed available at every step; this ignores exposures and cart constraints.
We condition on a block of exactly three observed common events and do not
estimate the probability of having such a block.

| Dataset | Train users/windows | Test users/windows | Test trajectory NLL per action | Constant action-frequency baseline |
| --- | ---: | ---: | ---: | ---: |
| Electronics | 2,758 / 5,975 | 674 / 1,610 | 0.6570 | 0.7172 |
| Retailrocket | 2,298 / 5,628 | 591 / 1,435 | 0.2623 | 0.2953 |
| Cosmetics | 3,077 / 14,182 | 792 / 3,870 | 0.6114 | 0.8989 |

The baseline is a fixed independent-action distribution estimated from training
windows with one pseudo-count per action. Lower NLL is better; this is a weak
baseline, not a comparison to behavior cloning, transition-only IRL or GRU4Rec.
Three-step trajectory likelihoods are directly comparable to its product of
independent per-action probabilities. Window selection, logged-event semantics,
session boundaries, missing actions and censoring differ across datasets; do
not rank their underlying shoppers using these numbers.

First-action observed vs predicted shares for view/cart/purchase are
Electronics 88.6/7.4/4.0% vs 75.9/15.4/8.7%; Retailrocket 96.2/3.4/0.3%
vs 92.9/5.8/1.4%; Cosmetics 52.5/41.2/6.3% vs 44.0/49.2/6.8%. First action
is not similarly distributed to later actions because blocks start at selected
session positions. The time-homogeneous reward and restricted features do not
calibrate the resulting first-action marginal adequately.

## Interpretation and reproduction

The estimates are dataset-level behavioral parameters under an assumed MDP,
not user-level coefficients, causal effects, identified human utilities or
evidence for any uplift type. Reward shaping/scale and action availability are
not identified by observational demonstrations. The three datasets contain
different populations and logging processes. No GMM, GRU persona conditioning,
user-level IRL, full-session rollout, confidence intervals, or temporal holdout
is implemented. A realistic simulator needs model(s) for session termination,
return gaps, exposure/item availability and source-specific missing actions.

Results: `artifacts/history-trajectory-irl/{dataset}.json` (aggregate only).
Run from the project root, with the existing capstone data and persona user
sample files installed locally:

```sh
PYTHONPATH='.:src:/Users/kimdaewon/Desktop/캡스톤디자인' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
scripts/run_history_trajectory_irl.py electronics retailrocket cosmetics
```
