# Quadratic reward with L1 only

Same five log-scaled history counts, their squares and pairwise interactions as
the quadratic pilot. No extra logarithms and no new lagged-action variables.
The objective is summed three-action-trajectory NLL + `1 * sum(abs(theta))`;
L2 strength is zero, temperature is one. The L1 term covers all 42 coefficients
including intercepts. Positive/negative splitting with bound-constrained L-BFGS-B
is used. Compare the existing L2-only model on the same user and window split:

| Dataset | L2-only test NLL/action | L1-only test NLL/action | Zero coefficients / 42 | Largest absolute coefficient |
| --- | ---: | ---: | ---: | ---: |
| Electronics | 0.5961 | 0.5791 | 14 | 51.0 |
| Retailrocket | 0.2570 | 0.2514 | 21 | 22.3 |
| Cosmetics | 0.5986 | 0.5935 | 7 | 33.9 |

The optimizer reports convergence; this is not evidence of coefficient stability.
The same repeatedly inspected test split, fixed penalty and lack of uncertainty
estimation make the comparison exploratory. Penalty size depends on the summed
objective and feature scales; zero weights are not causal evidence. All results
remain pooled within a dataset and conditional on complete three-event windows.
The state aggregates prior events rather than representing their precise order;
log scaling is an analyst-chosen magnitude transformation, not a replacement for
lags, and the raw source counts are not supplied log-scaled.

Results: `artifacts/l1-only-history-irl/{dataset}.json`. Reproduce with:

```sh
PYTHONPATH='.:src:/Users/kimdaewon/Desktop/캡스톤디자인' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
scripts/compare_l1_history_irl.py --l1-only electronics retailrocket cosmetics
```
