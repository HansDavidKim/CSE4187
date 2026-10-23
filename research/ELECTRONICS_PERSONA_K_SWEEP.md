# Electronics behavioral persona K sweep

Exploratory K=2–8 sweep using the previously trained seed-42 masked GRU and
standardized embeddings; no K is selected by silhouette. Each KMeans model
uses the 6,396 training users with at least two full-history sessions. The
profiles below use 1,604 held-out users meeting the same eligibility condition,
and are based on the first 64 input tokens, not future behavior. KMeans uses
seed 42 and 10 restarts; a seed-43 fit gives a limited initialization check.
The sampled users were stratified by full-history session count; neither cluster
counts nor shares describe the underlying population. See
`ELECTRONICS_PERSONA_PILOT.md` for source and preprocessing limitations.

| K | Held-out groups: size and rough description | Seed 42/43 ARI |
|---|---|---:|
| 2 | 1,226 view-dominant; 378 cart/purchase-active | 1.000 |
| 3 | 897 view-dominant; 528 mixed; 179 cart/purchase-active | 0.980 |
| 4 | 904 view-dominant; 519 mixed; 175 cart/purchase-active; 6 long-view-heavy with carts | 1.000 |
| 5 | 904 very view-dominant; 281 mostly view; 238 and 175 cart/purchase-active; 6 long-view-heavy | 1.000 |
| 6 | 903 very view-dominant; 281 mostly view; 235 and 174 cart/purchase-active; groups of 6 and 5 | 1.000 |
| 7 | 760 very view-dominant; 277 mostly view; 180 more engaged browsing; 202 and 174 cart/purchase-active; groups of 6 and 5 | 0.999 |
| 8 | 760 very view-dominant; 277 mostly view; 180 more engaged browsing; 202 mixed cart/purchase; 119 purchase-heavy; 55 cart-heavy; groups of 6 and 5 | 1.000 |

At K=8, the purchase-heavy group has 95.0% observed buyers, and the cart-heavy
group has 100% cart users and 7.3% observed buyers within the input prefix.
Their names describe those inputs rather than loyalty or price sensitivity.
The tiny groups have 41 and 5 training users and 6 and 5 held-out users;
interpretation is fragile. High ARI tests only two KMeans seeds on a single fixed
encoder/sample and does not establish robustness across encoders, samples or time.
The same input both defines and describes the groups, so this is not predictive
validation. Inferred session-end tokens, tied timestamps, retained duplicates,
full-history session eligibility and missing promotion/choice data remain limits.

## K=8 example: descriptive user patterns

The following is an exploration of one granular setting, not a claim that K=8 is
optimal. Immediate transitions count only pairs present in the input prefix;
an inferred session-end token counts as a possible next token. A token at the end
of a truncated prefix is excluded from the transition denominator. Percentages are
event-weighted and cannot be interpreted as the chance an individual ultimately
buys. Some cart-transition denominators are small.

| Cluster | Test users | Candidate description | View → next cart (n) | Cart → next purchase (n) | Buyer share |
|---|---:|---|---:|---:|---:|
| 0 | 760 | view-dominant | 0.9% (2,569) | 45.8% (24) | 1.1% |
| 1 | 55 | cart-heavy, few observed purchases | 36.9% (290) | 2.8% (108) | 7.3% |
| 2 | 277 | mostly views, occasional cart | 2.7% (2,176) | 21.3% (61) | 6.5% |
| 4 | 119 | purchase-heavy | 35.4% (625) | 53.4% (234) | 95.0% |
| 5 | 202 | cart/purchase-active | 26.1% (1,167) | 42.2% (313) | 63.9% |
| 7 | 180 | browsing with some cart activity | 4.6% (1,100) | 33.3% (54) | 7.8% |
| 3 | 6 | long-prefix, view-heavy | 4.6% (261) | 30.8% (13) | 66.7% |
| 6 | 5 | purchase-heavy small group | 8.7% (23) | 100.0% (2) | 100.0% |

Clusters 3 and 6 are too small for reliable behavioral conclusions; cluster 3's
median input length hits the 64-token cap. The name “cart-heavy” does not establish
hesitation, price sensitivity or any reason for non-purchase. Cluster IDs are
specific to this K and fit and cannot be compared by ID across K.

Detailed per-K counts, event shares and user-level buyer/cart rates are in
`artifacts/electronics-personas/k_sweep.json`; user assignments for all K are in
`k_sweep_memberships.parquet`. One-session user assignments are extrapolations,
not included in the profiles. All assignments contain user IDs; do not redistribute
without a rights/privacy review. Dataset and model redistribution rights remain
unverified.

Reproduce after the pilot in `ELECTRONICS_PERSONA_PILOT.md`:

```
PYTHONPATH='/Users/kimdaewon/Desktop/캡스톤디자인' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python scripts/sweep_electronics_personas.py
```
