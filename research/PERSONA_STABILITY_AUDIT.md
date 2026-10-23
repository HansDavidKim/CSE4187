# Persona candidates: sample-perturbation stability audit

## Method and boundary

For each dataset and K=2–8, fit the previous full-training KMeans as the reference.
Five times, sample 80% of the same eligible training users without replacement,
refit KMeans with a different seed, assign the unchanged held-out users, and match
clusters by maximizing total Jaccard overlap on those users. Both train sampling
and KMeans initialization change together. Report the matched cluster's minimum
Jaccard over the five runs, and overall adjusted Rand index per run.

This test fixes the encoder, scaler, sampled cohort, train/test split, K and held-out
users. It tests conditional cluster robustness only; it does not test retraining
the encoder, temporal drift, another dataset, time leakage, sessionization, or the
truth of a semantic persona. The matching uses the held-out memberships as a
descriptive stability diagnostic, not downstream out-of-sample validation.

As an **exploratory screening rule chosen after seeing the output**, require at
least 50 held-out users and minimum matched Jaccard >=0.8. This is a pragmatic
filter, not a statistical confidence guarantee. Small-but-consistent groups are
excluded, as are groups with high overall ARI but low individual overlap. No
formal hypothesis testing or uncertainty interval was performed.

## Supported descriptive candidates

These examples pass that screening rule, but the labels are analyst-created and
can overlap. They are **seven working behavioral descriptions**, not seven
mutually exclusive, independently established personas. The eight-persona goal
remains unvalidated rather than being filled with a weak eighth group.

| Candidate | Representative cluster | Held-out users | Min Jaccard | Main prefix evidence |
|---|---|---:|---:|---|
| Short view-dominant | Electronics K6 c3; Retailrocket K6 c0 | 747; 860 | 1.000; 0.999 | view 96.1%; 99.6% of events; median tokens 4; 4 |
| Longer view-dominant, some carts | Retailrocket K6 c5 | 191 | 1.000 | view 96.8% of events, cart-user share 14.7%, median tokens 13 |
| Mixed view/cart/purchase | Electronics K5 c0 | 308 | 0.990 | view/cart/purchase 80.3/11.8/8.0% of events; buyer share 32.8% |
| Cart-active, some observed purchases | Retailrocket K6 c3 | 97 | 0.980 | cart-user share 100%, buyer share 35.1% |
| Cart/remove active, fewer observed buyers | Cosmetics K6 c2 | 74 | 0.946 | cart-user share 100%, remove-user share 62.2%, buyer share 13.5% |
| Cart/purchase active | Electronics K6 c4 | 168 | 0.994 | cart-user share 98.2%, buyer share 67.9% |
| Cart/remove/purchase active | Cosmetics K6 c5 | 63 | 1.000 | cart-user share 100%, remove-user share 74.6%, buyer share 100% |

The Electronics K7 c5 direct-purchase candidate has only three held-out users
and minimum Jaccard 0.0; it is excluded. Retailrocket K8 c7 has 68 users but
minimum Jaccard 0.0; it is also excluded. Cosmetics K6 c4 has 16 users and
minimum Jaccard 1.0, illustrating why cluster overlap alone does not establish
adequate support. Another Electronics K4 cluster has minimum Jaccard 0.51,
showing that results depend on K. Numbers compare users within each individual
dataset and must not be treated as cross-dataset aligned latent clusters.

All profile figures use first 64 tokens, including inferred session-end tokens;
the same input determines both cluster and description. "Buyer share" is presence
of a recorded purchase in that prefix, not future propensity. Selection by full
history and stratified sample sizes make percentages non-representative and
unsuitable as online evaluation. Cosmetics cart removal is not recorded in the
other two sources; missing measurement is not zero. No price sensitivity,
loyalty, uplift, causation, nor stable latent psychological type is established.

## Reproduce

First reproduce or retain the prior K sweep and local capstone embeddings. Then:

```
for dataset in electronics retailrocket cosmetics; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python \
    scripts/audit_persona_stability.py "$dataset"
done
```

The three `artifacts/three-ecommerce-personas/*_stability.json` files contain
all K and replicate scores, including rejected groups. Next meaningful checks:
independently retrain encoder across seeds, temporally separated cohorts and
prefix-to-future evaluation, and explicit merge rules with uncertainty before
claiming a fixed or mutually exclusive eight-persona taxonomy.
