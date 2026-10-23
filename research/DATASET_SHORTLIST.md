# Dataset shortlist and intervention audit

| Rank | Dataset | License | Longitudinal user ID | Logged action | Outcome | Paper role |
|---:|---|---|---:|---|---|---|
| 1 | [Criteo Uplift v2.1](https://huggingface.co/datasets/criteo/criteo-uplift) | CC BY-NC-SA 4.0 | No | Randomized ad treatment | Visit, conversion, exposure | Primary intervention calibration/validation. |
| 2 | [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) | CC BY 4.0 | Yes | No randomized action | Timestamped purchases/cancellations | Behavioral trajectory, GMM and IRL calibration only. |
| 3 | [Udacity Starbucks promotion exercise](https://github.com/udacity/DSND_Term2/tree/master/portfolio_exercises/Experiments) | Ambiguous: MIT file + README CC BY-NC-ND 4.0; dataset grant unverified | No | Promotion Yes/No | Purchase | One-step non-commercial calibration; never redistribute rows; assignment mechanism not documented in checked notebook. |
| 4 | [Starbucks app customer rewards](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data) | **Unclear:** Kaggle says “Other” but supplies no license text | Yes | Offer received/type/time | View, completion, transaction amount | Development smoke only; exclude from KSC headline experiments unless written permission is obtained. |
| 5 | [RetailRocket recommender events](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | Check before use | Yes | No recommendation exposure | View, cart, transaction | Optional time-series robustness source. |
| 6 | [Customer support tickets](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) | Check before use | Weak | Priority/channel/response | Resolution time, satisfaction | Ticket schema prototype only. |

The project prototype starts with Starbucks because it contains a timestamped, user-level action
and subsequent outcomes in one compact download. It must not be the legal or causal foundation of
the KSC paper: public download does not replace an explicit license, and a logged action is not
automatically a `do()` intervention. The paper design therefore uses Criteo for randomized effects
and UCI Online Retail for longitudinal behavior, while synthetic known-truth experiments join the
two roles. Raw data is never redistributed with this repository.

## Discovery choice

- Current implementation: lagged user-day panel + nonlinear NOTEARS-MLP, with temporal tiers that
  prohibit current outcomes from causing past state or current actions.
- BOSS in the existing literature is primarily a static permutation search. TS-BOSS is a very
  recent time-series extension, so it is recorded as an experimental follow-up rather than made a
  required dependency.
- For nonlinear dynamic discovery, NTS-NOTEARS is the closest direct model upgrade. This first
  version uses lag expansion plus NOTEARS-MLP because gCastle is already installable and testable.
- If verified interventional assignments become available, replace the evidence overlay with a
  joint interventional learner such as IGSP/GIES or an interventional differentiable method.

