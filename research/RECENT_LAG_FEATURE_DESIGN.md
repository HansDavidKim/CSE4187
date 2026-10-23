# Recent-history feature engineering for planned MaxEnt IRL

## Literature basis

- [Li et al., Towards Learning Reward Functions from User Interactions](https://arxiv.org/abs/1708.04378):
  exploratory MaxEnt IRL using user situations; the study encodes categorical
  topic and object-order features with one-hot indicators, discretizes duration,
  and compares two groups. The museum task and its actions differ from ours.
- [SASRec](https://arxiv.org/abs/1808.09781): sequential recommendation using
  recent user actions for context; discusses last-action Markov models and
  longer RNN histories. It is not an IRL method and does not prescribe our basis.
- [GRU4Rec](https://arxiv.org/abs/1511.06939): models within-session behavior.
  Its item recommendation task is not identical to our action-type model.
- [MaxEnt IRL](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html):
  globally normalized trajectory distribution under an assumed transition model.

The design below is our proposal inspired by these ideas, not a feature set
validated by those papers.

## Exact state and target

At the moment of choosing the **next logged action**, the most recent already
observed common action defines `lag0`; actions before it define lag1, lag2 and
lag3. For a history `view, cart, view`, the next-action features are
`lag0_view=1`, `lag1_cart=1`, `lag2_view=1`, and lag3 unavailable. The action
being predicted is never inserted into its own input. This convention is
deliberate: lag0 means current state, not the target action at the same index.
The action categories are view, cart and purchase; Retailrocket maps addtocart
and transaction to the last two. The time step is an event index, not an
equal-length calendar period.

For maximum lag `L=0,1,2,3`, use these state-basis terms:

- intercept and two mutually exclusive indicators for an incomplete history
  arising from session start or an unknown boundary (unsupported event/tie);
- for every lag `0..L`, three binary action indicators; all three are zero
  when that lag does not exist, not evidence of no activity;
- pairwise products of action indicators at two distinct lag positions, all
  nine action combinations per pair. No products involving three positions.

There are respectively 6, 18, 39 and 69 basis terms (including intercept and
boundary indicators). For future behavior IRL, interact the basis with candidate
cart and purchase action indicators, using view as a reward-reference action;
then the corresponding sizes are 12, 36, 78 and 138 coefficients. Categorical
main effects are linear in the basis; the pairwise terms give the quadratic
part. Because each lag is one-hot, squares duplicate main indicators and
within-lag products are identically zero, so neither is included.

No cumulative counts, log transform, item price, product ID, future timestamp,
GRU embedding, user identifier or cluster label is used. Session boundaries are
source/inference dependent. Drop every group of events sharing a timestamp in a
given user's session and reset recent context; do not invent order. Cosmetics
remove events are not shared actions: skip them and reset context. These rules
change the sample, particularly purchase-heavy baskets, and their potential
selection bias must be evaluated before use. All three actions are not necessarily
available at every state; item/cart availability remains unknown.

## Preliminary coverage audit

The same retrospectively stratified persona sample and user train/test split is
used; retain first 64 **events**, not the earlier 64 token positions. These are
all eligible per-event decisions, not the prior IRL's complete 3-event windows.
No IRL has been trained for the new features, and no trainable user-level data
is exported. Counts below are training decisions; `lag3 present` means four
prior common events since the most recent session or unknown boundary.

| Source | Train decisions | lag0 present | lag3 present | Tied events skipped | Unsupported events skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electronics | 44,291 | 19,240 | 5,338 | 506 | 0 |
| Retailrocket | 43,004 | 18,869 | 6,683 | 116 | 0 |
| Cosmetics | 68,038 | 39,384 | 19,958 | 9,721 | 5,598 |

Cosmetics' retained test targets contain only 14 purchases. Tied purchase events
may disproportionately be removed; this audit does not yet quantify removed
actions by type. Consequently its evaluation, especially rare-purchase fit,
could be very unstable and not representative. The absence of a lag is an
explicit boundary situation, not a row of artificial zero behavior.

## Next stage, not yet run

Construct trajectory windows using the same eligible data and user split for all
L=0..3; simulate candidate histories by shifting in candidate actions, never
using future observed actions. Fit pooled dataset-level MaxEnt trajectory IRL
at fixed temperature 1, fixed L1 and L2=0. Select lag on a validation-user split
using trajectory likelihood plus action/sequence calibration diagnostics, then
evaluate once on untouched test users; report coverage and bootstrap uncertainty.
For fairness do not score different cohorts or compare raw parameter magnitudes
without accounting for dimension. Decide tie/basket handling before finalizing.
Reward is not identified by observational data and does not establish causal
effects, personalized utility or uplift. This design does not model item
exposure, session termination, purchase feasibility, or long-term returns.

Implementation: `src/ecommerce_sim/preprocessing/recent_history_features.py`; audited totals:
`artifacts/recent-feature-audit/{dataset}.json`. Reproduce from project root:

```sh
PYTHONPATH='.:src:/Users/kimdaewon/Desktop/캡스톤디자인' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
scripts/audit_recent_features.py electronics retailrocket cosmetics
```
