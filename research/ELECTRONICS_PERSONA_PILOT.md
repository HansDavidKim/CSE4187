# Electronics masked-GRU behavioral persona pilot

This is a descriptive exploratory pilot, not uplift, price sensitivity or causal inference.
Source: the existing capstone project's local copy of the Kaggle Electronics Store
events dataset, https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store.
The separate capstone implementation is reused as-is; no source data are copied here.

## Method

Using the capstone preprocessing, missing session identifiers are dropped; session
boundaries are inferred from the source session ID and 30-minute inactivity gaps.
From the first 64 tokens of each user, events are view/cart/purchase and inferred
session end. 12,000 users are sampled in equal strata of one, two, and three or more
sessions, using the full observed history to construct strata. This is not a population
sample and depends on future activity for eligibility; it is not an online experiment.
80/20 user-disjoint splitting precedes model fitting. A bidirectional GRU predicts
randomly masked tokens, a training-fitted standard scaler transforms 48-dimensional
embeddings, and KMeans is fit on training users with at least two observed sessions.
The candidate k is 3–7, chosen by training silhouette subject to a minimum cluster size.
Test silhouettes use only held-out repeat users. The representation is bidirectional
within its input prefix; only information available before a decision may be used
for a prospective simulator. The existing pilot is not a prospective validation.

## Results, seed 42

885,129 raw rows; 165 rows missing session IDs were excluded, 655 exact duplicates
were counted but retained, and 214 source session IDs occur across users. The
sort is stable for tied timestamps but within-timestamp causal ordering is unknown.
Training/test are 9,600/2,400 users; 8,000 sampled users have at least two sessions.
Three clusters selected, test masked-token NLL 0.2683 and repeat-user silhouette
0.4433. Masked NLL is not compared against a baseline in this run; silhouette is
not evidence of cluster stability, generation quality, or genuine customer types.

The following summarizes the first 64 input tokens for 1,604 held-out repeat users;
event shares weight events, while buyer/cart shares weight users. A purchase here
means at least one observed purchase in that input prefix. Cluster numbers are
arbitrary, and sample proportions do not estimate population prevalence.

| Cluster | Users | View/cart/purchase event shares | Buyer share | Cart user share | Descriptive label |
|---|---:|---|---:|---:|---|
| 0 | 528 | 84.4 / 9.4 / 6.2% | 29.7% | 51.3% | browsing with some cart/purchase |
| 1 | 179 | 64.3 / 20.8 / 14.9% | 67.0% | 96.6% | cart/purchase-active |
| 2 | 897 | 97.9 / 1.0 / 1.1% | 2.0% | 2.2% | view-dominant |

Profiles use the same input prefix used to assign clusters, so they describe how
groups differ, not an independent prediction of future behavior. People with one
session are excluded from these profiles. Different seeds, k=2, temporal holdouts,
and stability, prefix-to-future evaluation, uncertainty, and simulation fidelity
remain to be tested. Data collection and purchase observability limits remain unknown.
No observed offer exposure or untreated outcome is established. Dataset license,
redistribution, model artifact and customer-identifier rights need separate review.

## Reproduction

```
PYTHONPATH='/Users/kimdaewon/Desktop/캡스톤디자인' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python scripts/run_electronics_personas.py
```

This requires the separate local capstone project, its dataset and dependencies.
`artifacts/electronics-personas/electronics_personas.*` contains the original
model/assignments and `electronics_embeddings.parquet` the embeddings;
`profiles.json` adds the held-out input-prefix summaries. First-run `personas.*`
files predate a filename collision fix and should not be used. Artifacts contain
user identifiers and should not be published without rights/privacy review.
