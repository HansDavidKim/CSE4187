# TS-BOSS, NTS-NOTEARS and non-Gaussian orientation diagnostic

## Setup and limits

The same lag-2 customer-week panel and three 100-customer samples are used for both
methods. We exclude `unique_coupons_started` because it can be exactly collinear
with `offers_started`. TS-BOSS uses linear Gaussian BIC with no domain prior; the
nonlinear NTS-NOTEARS uses domain constraints, so model differences are not
isolated. A graph is a hypothesis, not an identified causal effect.

| Seed | NTS edges | TS-BOSS edges | Shared adjacency | Adjacency Jaccard | Shared directed edges |
|---|---:|---:|---:|---:|---:|
| 2026 | 16 | 67 | 13 | 0.186 | 7 |
| 2027 | 15 | 75 | 14 | 0.184 | 11 |
| 2028 | 17 | 70 | 15 | 0.208 | 11 |

All three NTS runs reached h < 1e-8, and the thresholded instantaneous graphs
were acyclic. A shared adjacency ignores direction within a week; a shared
directed edge requires matching orientation. The two method outputs are not a
head-to-head test of correctness; no held-out intervention ground truth is known.
For ongoing simulator development, use the sparse NTS graph only as a provisional
working hypothesis and retain TS-BOSS as a sensitivity comparison; do not claim
that NTS is established as more accurate. Its sparser graph and nonlinear model
are pragmatic considerations, not validation.

For direction sensitivity, first regress each standardized present-time variable
on all lag-1 and lag-2 variables by OLS using the TS-BOSS training customers. Fit
gcastle DirectLiNGAM (pwling) to the pooled resulting innovations. Reorient the
TS-BOSS contemporaneous skeleton according to its estimated global order; retain
lagged edges. This is not the official VARLiNGAM algorithm, nor constrained
LiNGAM fit on the TS-BOSS skeleton. The resulting graph is a sensitivity candidate,
not a validated or deployable DAG. No prior constraint was applied to LiNGAM.

The diagnostics use 200 disjoint customers, one random week per customer. The
training windows overlap, while held-out customers can share calendar shocks.
Normality tests and 199-permutation pairwise RBF-HSIC checks on held-out fitted
structural residuals are exploratory and uncorrected for multiple tests; a p-value
is not proof of independence. See JSON for standardized residual variance,
skewness, excess kurtosis, p-values and all hypothesis edges. Results depend on
lag regression, linearity, variable definitions, sampling, and fitted model.

| Seed | TS-BOSS instantaneous edges | Reversed by LiNGAM ordering | Normality p<0.05 | HSIC p<0.05 |
|---|---:|---:|---:|---:|
| 2026 | 19 | 12 | 10/10 | 43/45 |
| 2027 | 20 | 10 | 10/10 | 31/45 |
| 2028 | 20 | 4 | 10/10 | 35/45 |

These diagnostics suggest strong departures from Gaussian residuals, but also
substantial residual dependence, contrary to the independent-errors assumption.
The zero-inflated, discrete, mechanically coupled features further challenge the
continuous linear LiNGAM model. Large numbers of reorientations and dependence
mean non-Gaussianity alone does not justify accepting the inferred directions.
Estimated order sometimes places redemption before offer start; neither this
algorithmic order nor raw same-week aggregation establishes actual event timing.

References: original LiNGAM https://jmlr.csail.mit.edu/beta/papers/v7/shimizu06a.html;
time-series extension https://lingam.readthedocs.io/en/latest/tutorial/var.html;
TS-BOSS https://arxiv.org/abs/2603.05370. Equal noise-variance identification is a
distinct assumption, not established here: https://arxiv.org/abs/1205.2536.

## Reproduce direction diagnostic

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/tsboss_lingam_diagnostic.py \
  --graph artifacts/amexpert-tsboss-no-duplicate-2026/tsboss_graph.json \
  --output artifacts/amexpert-tsboss-lingam-2026
```

The input panel is prepared locally; raw and row-level derivatives are not
redistributed. A next step is pre-specified temporal validation, diagnostics with
customer/calendar dependence accounted for, domain-constraint parity, and actual
intervention evidence. Do not use this graph as causal truth for RL/IRL.
