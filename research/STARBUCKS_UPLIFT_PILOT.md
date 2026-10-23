# Starbucks promotion-response pilot

## Frozen source

- Repository: `udacity/DSND_Term2`
- Commit: `6cc091fff66b8f441cbfeb5bb9e7ba70a7d74117`
- Training: 84,534 rows, SHA-256
  `4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f`
- Test: 41,650 rows, SHA-256
  `f5607ca0b2f6a6580ee2085659387b45735e81b41cad97c756ba068fcc06a625`
- Schema: `ID`, `Promotion`, `purchase`, `V1`–`V7`

Raw files and row-level derivatives are local-only and excluded from version control. The
repository-level MIT/CC BY-NC-ND notices are not treated as a verified Starbucks
dataset-redistribution grant.

## First reproducible run

The response model is a logistic S-learner with treatment interactions. A six-component Gaussian
mixture models the anonymous user covariate distribution. The frozen test results are:

| Quantity | Result |
|---|---:|
| Training promotion share | 0.50115 |
| Test purchase rate, promotion | 0.01634 |
| Test purchase rate, control | 0.00675 |
| Raw test difference | +0.00959 |
| Constant-propensity IPW difference | +0.00945 |
| Mean model uplift | +0.00960 |
| Maximum absolute V1–V7 SMD in training | 0.01454 |
| DR value, top-30% policy | 0.01204 |
| DR value, all-control policy | 0.00674 |
| DR incremental policy value | +0.00530 |

The small covariate imbalance is consistent with a well-balanced experiment but does not prove
random assignment. Until an authoritative design source is cited, the paper must call this a
promotion experiment and state the exchangeability assumption behind IPW/DR interpretations.

## Simulator boundary

The calibrated generator implements

`X ~ fitted GMM` followed by empirical-support projection, `A ~ policy(X)`, and
`Y ~ Bernoulli(fitted P(purchase=1 | X,A))`.

This validates one-step promotion response and treatment heterogeneity. It does not identify item
preference, repeated user-state transitions or long-horizon reward. Those are separate synthetic
mechanisms to be evaluated against known ground truth before making an IRL or sequential
recommendation claim.
