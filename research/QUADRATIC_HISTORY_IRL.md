# Pairwise-quadratic history reward: exploratory comparison

The five prior-history counts and selection/windowing are identical to
`HISTORY_FEATURE_MAXENT_IRL.md`. For each state variable `z_i = log1p(count_i) /
log(65)`, create a feature basis consisting of one constant, the five `z_i`,
the five `z_i^2`, and the ten distinct `z_i*z_j` for `i < j`: 21 terms total.
There are no cubic or higher-order terms. For each possible cart/purchase action,
the reward has its own 21 coefficients; view is the zero-reward reference. Thus
the model is nonlinear in the log-scaled history, but linear in its 42
coefficients. The history counts update deterministically along hypothetical
three-event paths. Same training windows, user split, L2=1 and temperature=1
are used for the linear and quadratic models. No model uses future history.

| Dataset | Test windows | Linear test NLL/action | Quadratic test NLL/action | First-action view observed / linear / quadratic |
| --- | ---: | ---: | ---: | --- |
| Electronics | 1,610 | 0.6570 | 0.5961 | 88.6% / 75.9% / 79.8% |
| Retailrocket | 1,435 | 0.2623 | 0.2570 | 96.2% / 92.9% / 93.4% |
| Cosmetics | 3,870 | 0.6114 | 0.5986 | 52.5% / 44.0% / 45.9% |

Observed and predicted shares above are on the same conditional, complete
three-event test windows; they are not full-session or population rates.
Lower NLL is better. The quadratic model is better on this split but remains
miscalibrated on the first action. The test set informed this exploratory
comparison; it is not a final untouched evaluation for subsequent model
selection. The same absolute L2 penalty is used, but regularization has a
different effect with 42 rather than 12 parameters. No hyperparameter tuning,
confidence intervals, temporal split or uncertainty analysis was performed.

Models estimate a pooled dataset-level behavioral reward, not user-specific
utility or an intervention effect. Reward is not identified from behavior and
logging/exposure differences across datasets preclude straightforward causal
interpretation of coefficients. Action availability, product choice, session
end and return times remain unmodeled. Output files contain aggregate fitted
coefficients and evaluation only, at `artifacts/quadratic-history-irl/`.

Reproduce from the project root with local data:

```sh
PYTHONPATH='.:src:/Users/kimdaewon/Desktop/캡스톤디자인' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
scripts/compare_quadratic_history_irl.py electronics retailrocket cosmetics
```
