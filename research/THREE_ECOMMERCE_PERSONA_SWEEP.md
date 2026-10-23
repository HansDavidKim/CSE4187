# Three eCommerce datasets: behavioral persona K sweep

## Objective and method

Discover and describe behavioral patterns, not treatment response or causal types.
Electronics, Retailrocket and Cosmetics are fitted separately. For each source,
the sweep reuses its existing dataset-specific, 48-dimensional masked bidirectional
GRU embeddings from the capstone project's `masked_embedding_clusters.py`. No GRU
was retrained here; the saved embeddings and the source data are required.
The same workflow reconstructs the 12,000-user sample and checks exact user order,
then uses the original seed-42 random 80/20 user split and fits KMeans at K=2–8
only on training users with at least two observed sessions. It describes repeat
users in held-out data using their first 64 input tokens. K is not selected by
silhouette. Cluster numbers are arbitrary, not matched across K or datasets.

The 12,000-user sample is stratified by full-history session count, and Cosmetics
first filters user IDs divisible by 100. These are exploratory samples, not
population prevalence estimates. Repeat-user eligibility uses full history and
is not an online selection procedure. Profiles summarize the same prefix as used
to assign clusters, not future behavior. Each dataset has a separate embedding
space; cohort, time periods, event logging and sessions differ. The first 64
tokens include inferred session-end tokens, and ties have no known causal order.
An immediate transition includes a possible inferred session-end as next token;
prefix-final events have no next-token denominator.

## Source and data quality

| Data | Observed events | Dataset scope and caveats |
|---|---|---|
| Electronics | view, cart, purchase | 885,129 raw rows, 165 missing-session rows excluded, 655 exact duplicate rows retained, 214 session IDs shared across users; 2020-09-24 to 2021-02-28 UTC. |
| Retailrocket | view, add-to-cart, transaction | 2,756,101 raw rows; 460 exact duplicate rows retained; sessions inferred using 30-minute gaps; 2015-05-03 to 2015-09-18 UTC. |
| Cosmetics | view, cart, remove-from-cart, purchase | User-ID modulo 100 sample: 16,428 users and 216,093 events from five monthly files; rows with missing user or session dropped; their total number was not audited. |

Repeat-user training/test counts are 6,416/1,584 Electronics,
6,387/1,613 Retailrocket, and 4,602/1,133 Cosmetics. Other sampled users are
embedded but excluded from the cluster fit and held-out profiles. The 12,000
sample is not a random draw of all users. Session boundaries also depend on
dataset-specific IDs and/or the 30-minute rule, not observed true session ends.

## Illustrative K=6 profiles

This table deliberately combines similar descriptive patterns across clusters
within a source; it is not a claim of equivalent latent types across sources.
Purchase share counts users with at least one purchase (transaction for
Retailrocket) in the prefix. Input event shares are event-weighted.

| Source / cluster | Held-out users | View / cart / remove / purchase events | Users with purchase | Descriptive pattern |
|---|---:|---|---:|---|
| Electronics c3 | 747 | 96.1 / 2.1 / — / 1.7% | 3.3% | view-dominant |
| Electronics c0, c4 | 183, 168 | 66.8 / 19.8 / — / 13.4%; 62.5 / 20.4 / — / 17.1% | 59.6%, 67.9% | cart and purchase active |
| Retailrocket c0 | 860 | 99.6 / 0.2 / — / 0.1% | 0.5% | view-dominant |
| Retailrocket c3 | 97 | 74.5 / 20.7 / — / 4.8% | 35.1% | cart-active with some purchases |
| Cosmetics c0 | 586 | 92.6 / 5.4 / 1.3 / 0.6% | 0.7% | view-dominant |
| Cosmetics c2 | 74 | 31.0 / 46.8 / 17.2 / 4.9% | 13.5% | cart/remove active, fewer observed buyers |
| Cosmetics c5 | 63 | 25.9 / 40.3 / 15.7 / 18.0% | 100.0% | purchase-active with cart/remove events |

In the Cosmetics K=6 c2 group, 62.2% of the 74 users have a remove event;
in c5, 74.6% of the 63 users have one. These signals do not establish price
sensitivity or explain cart removals. Cosmetics c1 and c4 have median prefix
length 64, suggesting truncation is important for their interpretation.
One Cosmetics group of 16, one Electronics group of 10, and one Retailrocket
group of 10 are small at K=6. Full per-K profiles, transition numerators via
rates/denominators, and cluster sizes are in the JSON outputs.

KMeans seed-42/43 ARI evaluates initialization variation for fixed embeddings
only. It does not measure encoder-seed, sampling, temporal or dataset stability.
No next-event holdout, temporal validation, deployment simulation, statistical
uncertainty or intervention effect is established here. GRU masked reconstruction
is not a causal user policy. Rights to publish or redistribute datasets, derived
embeddings and user-linked assignments are not yet verified.

## Reproduction and artifacts

```
PYTHONPATH='/Users/kimdaewon/Desktop/캡스톤디자인' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python scripts/sweep_three_ecommerce_personas.py
```

Requires the capstone data and `data/models/{source}_masked_embedding_embeddings.parquet`.
Writes `artifacts/three-ecommerce-personas/{source}.json` and user-level
`{source}_memberships.parquet`; the latter contains user IDs and is not for
unreviewed redistribution. Source sample and trained representation may not be
available on other machines. These cross-dataset Electronics embeddings differ
from the earlier separate `electronics_masked_sequence_personas` pilot, so their
K results and cluster labels should not be mixed.
