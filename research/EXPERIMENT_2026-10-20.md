# AmExpert NTS-NOTEARS pilot — 2026-10-20

## Data and leakage correction

Source: local AmExpert competition CSVs, read from the existing capstone directory. Raw data
rights remain unverified; do not redistribute raw or row-level derivatives.

The prior `amexpert-panel` extended the panel to December 2013 using unassigned campaign metadata
despite transactions ending on 2013-07-03. This incorrectly imputed unobserved weeks as zero. The
old `amexpert-nts-smoke` result is invalid for interpretation. The revised panel retains only full
weeks inside the global transaction range: 2012-01-02 through 2013-06-30 (78 weeks), 1,582
customers and 123,396 customer-weeks. The final partial week excludes 6,377 transaction rows and
346 reported redemptions; 383 redemption reports remain. This is an important sensitivity issue,
not evidence those redemptions did not occur. Coupon start/end weeks remain proxies, and the
absence of an assignment in the training labels is not proven to mean absence of exposure.

The two features `offers_started` and `unique_coupons_started` are equal in 99.9903% of panel
rows, and in 4,210 of 4,222 rows (99.72%) with nonzero offers. Their direction is therefore
effectively unidentifiable. Genuinely missing exposure,
customer entry/exit, unobserved confounding, aggregation and nonstationarity remain concerns.

## Converged pilot

`artifacts/amexpert-nts-observed-100`: 100 customers sampled without replacement (seed 2026),
7,600 lag windows, lag 2, 4 hidden units, lambda1 0.03, lambda2 0.01, threshold 0.15, max 12
outer iterations. Ten iterations reached h=3.8874e-9; the thresholded contemporaneous graph is
acyclic. It contains 15 edges. Same-week purchase counts/spend and coupon bookkeeping variables
dominate; there are no pre-decision parent candidates for transactions or spend and no selected
edges for redemption. No edge is interpreted as a confirmed real-world causal effect.

## Synthetic known-DAG sanity check

`scripts/benchmark_nts_synthetic.py` simulates 50 independent 50-step series with three variables
and three lag-1 edges, including one nonlinear relationship. Seeds 42, 43 and 44 each converged
and recovered 3/3 true edges with 0 false positives (precision/recall/F1 all 1.0). This is a small,
easy sanity check, not an estimate of performance on AmExpert or a broad simulation benchmark.

## Customer-subsample stability

Three independent seeded samples of 100 customers (without replacement within each sample) all
converged and their thresholded instantaneous graphs were acyclic. Seeds 2026, 2027 and 2028
selected 15, 16 and 19 edges respectively; 28 distinct edges appeared in total, of which 6 appeared
in all three runs, 10 in two, and 12 in one. This is a small subsampling diagnostic, not a full
bootstrap confidence interval. The six common edges include two lagged coupon-count edges and
four instantaneous edges. Several are linked to bookkeeping definitions or same-week aggregates,
so stability alone does not establish causal direction or a useful decision-state feature.

Removing `unique_coupons_started` on the same seed-2026 customer sample also converged
(h=3.6915e-9). After removing edges incident to that variable from the original graph, the two
edge sets have 10 common edges out of a union of 20 (Jaccard 0.50). Removing a variable also
changes model dimensionality and random initialization, so this comparison does not isolate the
feature-removal effect from initialization; it is a sensitivity warning rather than a causal test.

## Remaining checks

Held-out temporal prediction, identified interventions, simultaneous-edge robustness to
aggregation, repeated duplicate-feature/initialization sensitivity, missing exposure, richer synthetic DAGs and
IRL/GMM/RL evaluation are not yet established.
Do not use a single fitted graph as ground truth or a completed policy-evaluation simulator.
