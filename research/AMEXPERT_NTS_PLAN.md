# AmExpert 2019 → NTS-NOTEARS study plan

## Suitability

AmExpert is a relational longitudinal retail dataset, not a ready-made multivariate time series.
It contains dated transactions from January 2012 through July 2013, coupon/customer assignments,
18 partially overlapping labeled campaigns, campaign windows, item metadata and customer
demographics. A published causal-ML application reports 1–208 coupon types per campaign and 0–37
coupons per customer/campaign.

This is materially better than the one-row Starbucks uplift table for a sequential simulator. It
still does not document randomized coupon assignment, so `coupon offered → purchase/redemption`
must not be presented as an identified causal effect without adjustment assumptions.

## Data access and rights

The original source is the Analytics Vidhya/American Express competition. No explicit dataset
reuse or redistribution license was found in the accessible official materials. Public GitHub
mirrors exist but are not authoritative and commonly have no license. Obtain the archive through
the original competition/account, keep it under `data/raw/`, and publish code plus aggregate
results—not raw or transformed rows—unless written permission is obtained.

## Panel contract

`prepare_amexpert.py` creates one dense weekly sequence per customer. Dynamic variables are:

1. transaction count;
2. quantity;
3. spend;
4. non-coupon discount;
5. coupon discount;
6. unique items;
7. unique categories;
8. coupon offers starting that week;
9. unique coupons starting that week;
10. active offers from overlapping campaigns;
11. reported redemptions placed at campaign end week.

Demographics remain static context for GMM/cohort analysis and are not repeated as degenerate time
series columns. Exact coupon send and redemption timestamps are absent; campaign start/end are
explicit proxy times and must be reported as such.

## NTS-NOTEARS design

Use the official Apache-2.0 implementation pinned to commit
`e1ba34b784f9eec3d2139904589d2fd1ed4ea7ad`. Start with weekly lags 1–4 and compare against lags
1–2 as a sensitivity analysis. NTS-NOTEARS learns both lagged and instantaneous nonlinear edges.

Required prior constraints:

- forbid all edges from future to past by construction;
- forbid current transaction outcomes → current `offers_started` because assignment is fixed at
  campaign start;
- allow previous behavior → future offers because targeting is observational;
- do not force offer → redemption/purchase edges;
- keep overlapping campaign intensity as an observed policy variable.

The official implementation accepts one continuous `n × d` series. Naively concatenating customer
panels creates false transitions from the last week of one customer to the first week of another.
The paper implementation must therefore train from separate lag windows per customer or fit
prespecified cohorts separately. The saved NPZ contains concatenated values plus offsets so a
multi-sequence loader can construct windows without crossing boundaries. `nts_adapter.py` performs
that conversion and returns each window's customer-sequence index for grouped splitting and
customer bootstrap. This is an input contract for a multi-sequence patch; it must not be flattened
back into a single time series before calling the official model.

## Evidence standard

The real-data DAG is a hypothesis graph, not known truth. Report:

- edge stability across user bootstrap samples, lag orders and weekly/daily aggregation;
- contemporaneous DAG acyclicity and edge-selection frequency;
- held-out predictive likelihood;
- comparison with DYNOTEARS and PCMCI+;
- synthetic known-DAG recovery before interpreting AmExpert edges;
- sensitivity to campaign overlap, assignment propensity and unobserved targeting.

Only edges stable across resamples and baselines should enter the synthetic generator. Intervene in
the generated SCM to validate policies; do not claim that observational AmExpert alone identifies
the corresponding real-world intervention effect.
