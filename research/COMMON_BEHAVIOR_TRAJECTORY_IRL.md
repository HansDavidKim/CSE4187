# Common behavioral trajectory IRL pilot

## Scope

Three dataset-specific K=6 partitions use the previously fitted masked-GRU
representations and KMeans. Retain clusters with at least 50 held-out repeat users
and minimum held-out cluster stability Jaccard >=0.8 in the prior five-run audit.
This threshold was chosen post hoc and uses the test cohort; it is exploratory,
not unbiased selection or proof of stable natural populations. K=6 is a practical
pilot, not a claim that six clusters are optimal. Cluster IDs and semantics are
dataset-specific. User memberships are inferred from a whole input prefix and
therefore can contain the very events used for modeling or evaluation.

For each user, take the existing first 64 tokens and split at inferred session
ends or at any event outside the common vocabulary (including Cosmetics removal).
Within each resulting segment, make nonoverlapping blocks of four events, yielding
three transitions per block; discard remaining shorter fragments. Unknown events
are not mapped to zero or artificially joined across a boundary. Only observed
view, cart and purchase are modeled; the model is conditional on such blocks,
does not model session end, removal, gaps, item choices or action availability.

State = previous action; action = next event; next state = current action. Shared
reward features are one-hot indicators for the nine state/action transitions;
the `view_to_view` coefficient is anchored to zero, leaving eight coefficients.
The same horizon=3, temperature=1, L2 penalty=1 and fitted transition model are
used for all clusters. Existing `fit_trajectory_reward` uses entropy-regularized
finite-horizon Bellman recursion and minimizes full-trajectory negative log
likelihood plus L2. Reward gauge, scale/temperature and shaping ambiguities remain;
the fixed anchor and penalty select one numerical representative, not the true
psychological reward. This is population/cluster-level behavioral IRL under a
highly simplified MDP, not individual IRL, causal inference or uplift.

## Run and representativeness

Each dataset is split into disjoint training and held-out users as in the earlier
clustering experiment. Fit per-cluster rewards from training users only. A fixed
prefix and three-transition-window selection discards many people: the input group
labels use full prefixes and selection uses the full observed record's repeat-session
criterion. The post-hoc group stability filter also used held-out data. Thus the
reported held-out log loss is **descriptive only**, not prospective generalization;
neither test memberships nor its selection are independent of the scored events.
Comparisons across logging systems and data sets should not be interpreted as true
differences in intrinsic user preferences.

| Dataset | Fitted cluster vectors | Window-bearing training users, by cluster | Trajectory windows, by cluster |
|---|---:|---|---|
| Electronics | 5 | 377, 227, 508, 91, 334 | 575, 416, 1,025, 134, 610 |
| Retailrocket | 5 | 105, 171, 235, 231, 577 | 141, 177, 532, 397, 1,327 |
| Cosmetics | 3 | 351, 275, 216 | 2,860, 1,170, 1,000 |

For example, Electronics c3 has 91/3,092 eligible training users with a usable
window; Retailrocket c0 has 105/3,410. Estimates for those clusters characterize
this small selected subset, not all group members. Cosmetics segments split at
remove events; therefore missing removal in other datasets is not treated as a
measured zero, but conditioning and selection differ across sources. Duplicate
events, timestamp ties and differing measurement processes also remain unresolved.

Five cluster-internal user bootstraps sample only users with at least one window,
and refit the whole reward vector. Their coefficient standard deviation is a
rough model/sampling uncertainty diagnostic; five replicates are far too few for
confidence intervals. It is **not** the distribution of individual utility
coefficients. Some states may rarely occur; coefficients there can be driven by
regularization. A user or session-level hierarchical reward model and uncertainty
checks would be needed before fitting a GMM intended to represent heterogeneous
people. The 13 cluster vectors alone are neither individual samples nor a sound
basis for an eight-persona GMM; no GMM is fitted in this pilot.

## Outputs and reproduction

`artifacts/common-behavior-irl/{source}.json` contains fitted coefficient vectors,
reward matrices, window/user counts, diagnostics, descriptive likelihoods, and
five bootstrap fits. `cluster_coefficients.csv` is a GMM-input *schema* with one
row per dataset-specific cluster and separate bootstrap standard deviations. It
is not a user-level coefficient dataset. The bootstrap rows must not be treated as
independent people. User-linked memberships in the upstream pipeline are local;
data, derived model and export rights have not been verified.

```
for dataset in electronics retailrocket cosmetics; do
  PYTHONPATH='src:/Users/kimdaewon/Desktop/캡스톤디자인' \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
    scripts/run_common_behavior_irl.py "$dataset"
done
PYTHONPATH=src /opt/anaconda3/bin/python scripts/summarize_common_behavior_irl.py
```

Requires the separate capstone data and existing K6/stability outputs. Further
work: prospective prefix-to-suffix evaluation and pooled baseline; explicit
handling of censoring, termination and heterogeneous logging; more bootstrap
replicates and regularization sensitivity; sufficiently informative user-level or
hierarchical IRL before drawing a GMM over user reward vectors.
