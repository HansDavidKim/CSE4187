# Nonlinear trajectory reward and greedy sensitivity

## What changed

The previous group trajectory experiment used a full 3×3 reward table for
previous behavior and current behavior; despite being linear in one-hot features,
that table already had an independent parameter for every state–action pair.
The earlier rollout sampled from the soft policy; it was not greedy. Simply
replacing that table with an MLP on the same discrete input cannot add meaningful
expressivity to the table.

Here the state is expanded to 12 combinations of previous action (3), previous
quantity above the training-period median positive quantity of 20 (2), and whether
previous active offers are positive (2). Quantity and active offers are related to
lagged NTS feature candidates; previous action is an explicit behavioral control.
Both utility specifications use the exact same four GRU/GMM user groups, data
splits, state space, common pooled transition model, horizon 13, entropy scale 1,
regularization strength 1 and single random seed:

- Linear: action-specific affine function of the state indicators.
- Nonlinear: 5 → 12 → 3 fully connected tanh MLP for state-dependent reward.

Both use finite-horizon maximum causal entropy IRL with stochastic transitions.
Rewards are anchored by subtracting a **single global reference reward**; subtracting
a different no-purchase reward for every state is not in general a harmless
normalization in a dynamic MDP, because it changes state-dependent future values.
One global anchor and L2 regularization still do not identify true utility uniquely.
The pooled observational transition matrix is estimated only on weeks 1–50,
with add-0.5 smoothing and the known constraint that the next state's previous
behavior is the chosen category. This is not an intervention-identified transition.

Training 13-week episodes start at weeks 1, 14, 27 and 39 (week 39 overlaps).
Validation is 52–64 and held-out test 65–77. Groups and NTS features use only the
training period. Validation results are reported, not used to tune the fixed
hyperparameters. Teacher-forced metrics use actual preceding state; rollout starts
at the observed test initial state and then generates transitions.

## Results

| Metric | Linear | Neural MLP |
|---|---:|---:|
| Training log loss ↓ | 0.7372 | 0.7364 |
| Validation log loss ↓ | 0.8205 | 0.8336 |
| Test log loss ↓ | 0.8356 | 0.8518 |
| Test Brier ↓ | 0.5089 | 0.5162 |
| Test accuracy | 0.6204 | 0.6185 |

The MLP slightly improves training fit but worsens validation and test likelihood;
this fixed setup does not show better out-of-sample prediction. It has 111
parameters per group versus 18 for the linear model; equal parameter-penalty
strength does not mean equal effective regularization. One seed, grouping and
architecture do not establish whether any nonlinear utility can help.

| 13-week rollout statistic | Observed | Linear sampled | Neural sampled | Neural greedy |
|---|---:|---:|---:|---:|
| No purchase | 0.319 | 0.364 | 0.338 | 0.269 |
| Purchase, no recorded coupon discount | 0.596 | 0.546 | 0.549 | 0.731 |
| Purchase, recorded coupon discount | 0.085 | 0.090 | 0.113 | 0.0006 |
| Action switching | 0.408 | 0.375 | 0.387 | 0.0006 |
| P(purchase after purchase) | 0.775 | 0.780 | 0.789 | 1.000 |

Ten sampled trajectories per user were used. The neural model comes closer on
some aggregate statistics but overpredicts coupon-discount outcomes, and per-user
rate MAE remains substantial (0.217, 0.217, 0.104). Greedy uses argmax of the same
policy but samples transitions, producing almost no changes and practically no
coupon-discount outcomes. This demonstrates greedy can damage diversity in this
model; it does not explain the earlier sampled rollout's mismatch. Differences
also reflect finite state representation, pooled transition misspecification,
time shift, incomplete observation and user heterogeneity.

## Economic interpretation and limitations

The softmax of the soft Q-values can be interpreted as a conditional random-utility
logit with extreme-value shocks under the appropriate assumptions (Train,
[Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html)).
Nonlinearity in systematic utility does not remove conditional logit's IIA
restriction. Group membership permits modeled heterogeneity, but this implementation
is not a validated mixed-logit estimation. The observational weekly labels are
not observed alternative sets, and coupon discount is not necessarily an available
choice or redemption. There is no item-specific price or income and no proven
monotonicity, diminishing returns, willingness-to-pay or welfare interpretation.
The unconstrained network can violate economic shape restrictions, and IRL reward
values are not identified monetary utilities.

The practical decision on this experiment is to keep both versions, using the
linear model as a predictive reference and the MLP as a trajectory-fidelity
sensitivity, not claim a winner on every metric. Next: compare multiple random
seeds and validation-selected regularization, test flexible but economically
constrained price/discount response if those choice-set data become available,
validate transition distributions and coupon availability, and add user-level
hierarchical preferences. Do not use fitted transitions for counterfactual policy
evaluation without identifying assumptions or experimental validation.

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/compare_nonlinear_trajectory_irl.py
```

Output files are local customer-linked derivatives, not approved for redistribution.
