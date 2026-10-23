# L1-regularized quadratic history IRL

No extra logarithm is applied. The original state is `z=log1p(count)/log(65)`;
the existing basis includes 1, each `z`, its square and pairwise interactions.
For each of cart and purchase there are 21 parameters; view is the reference.
We add an L1 penalty on all 42 parameters, including both intercepts, while
keeping the previous L2 penalty and temperature 1. The objective is summed
trajectory negative log-likelihood + `1*sum(abs(theta)) + 0.5*sum(theta^2)`.
The exact nonsmooth L1 objective is optimized by positive/negative coefficient
splitting and bound-constrained L-BFGS-B; L1 and L2 are fixed at 1, not tuned.

| Dataset | Existing L2-only test NLL/action | L1+L2 test NLL/action | L1+L2 coefficients at zero (of 42) |
| --- | ---: | ---: | ---: |
| Electronics | 0.5961 | 0.5981 | 15 |
| Retailrocket | 0.2570 | 0.2584 | 19 |
| Cosmetics | 0.5986 | 0.5987 | 5 |

Zero means absolute value below 1e-8. Both methods use the same observed user
split and complete 3-event windows. The test data has been used in previous
exploratory model comparisons, so it is not an untouched final evaluation.
The small test-NLL increases compare only this fixed penalty setting; they do not
establish that all L1 strengths are worse or that nonzero terms are causally
relevant. L1 sparsity is sensitive to feature scales and collinearity, and the
model is dataset-level rather than user-specific. Trajectory termination,
exposure, action availability, return and product choice are not modeled.

Aggregate results: `artifacts/l1-history-irl/{dataset}.json`. Reproduce from
the project root, using the existing local data and quadratic comparison output:

```sh
PYTHONPATH='.:src:/Users/kimdaewon/Desktop/캡스톤디자인' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
scripts/compare_l1_history_irl.py electronics retailrocket cosmetics
```
