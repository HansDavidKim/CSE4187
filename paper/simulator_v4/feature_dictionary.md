# KIT feature contract

The base contract is the ordered 32-column `KIT_FEATURES` registry in `src/simulate/v4/features.py`. The current expanded contract is **KIT32 + seven review/cart features = 39** (`kit39-review-cart-v2`, behavior version `behavior_review_cart_v2`). The user reduced the original PRD eight-event scope to six: view, cart, remove, purchase, coupon usage and review. Search and CS are excluded from new generation and the current feature allowlist. The historical upstream `feature_registry_events_v1.json` describes the previous 45-feature scope and is not included in this checkout; the current [v2 registry](feature_registry_events_v2.json) describes 39. Existing downstream selectors and fitted model inputs are not silently retrained or expanded.

Each row is identified by `customer_id`, `snapshot_id` and exact UTC `as_of_at=t0`. The dataset ID, feature version, currency and minimum currency unit travel with the output; IDs, assignment, outcomes, persona, lifecycle and diagnostics are excluded from predictors. Old eight-event datasets, PDFs and source readers remain historical artifacts and do not establish that search/CS are still generated.

Every t0 must be inside the declared observation range; the end boundary permits features but no known future outcome. All input events satisfy `event_at<t0`. The recent window is `[t0-30d,t0)` and the preceding window is `[t0-60d,t0-30d)`. C30 contains sessions starting in the recent window and ending strictly before t0. Counts and sums are zero for observed absence; undefined means/intervals and incomplete observation are null. Days are elapsed seconds / 86,400. Monetary values remain in the dataset's declared integer minor units; source prices are not renamed KRW.

| Feature | Definition |
|---|---|
| visit_recency_days | Days since the last past session start; null without a session |
| purchase_recency_days | Days since the last confirmed past order; null without an order |
| observed_tenure_days | Days since the first observed session, distinct from signup age |
| session_count_30d | Unique recent session starts, including sessions still open at t0 |
| view_count_30d | Recent view events, including repeated views of a product |
| purchase_count_30d | Unique recent orders, never item-row count |
| net_spend_30d | Recent paid_minor sum after actual coupon discounts |
| mean_net_order_value_30d | Recent paid_minor mean; null without an order |
| mean_session_duration_minutes_30d | Mean completed C30 duration in minutes |
| views_per_session_30d | Views in C30 / C30 session count; null without C30 |
| orders_per_session_30d | Orders linked to C30 / C30 count; an intensity that can exceed 1 |
| mean_purchase_gap_days | Mean successive past-order interval; null with fewer than two orders |
| intervisit_start_gap_mean_days_90d | Mean start-to-next-start interval with the later start in the past 90 days |
| intervisit_start_gap_cv_90d | Sample SD / mean of those intervals; null with fewer than two gaps or zero mean |
| unique_purchase_products_30d | Distinct products in recent order items |
| purchased_qty_30d | Recent order-item quantity sum |
| mean_basket_qty_30d | Purchased quantity / recent order count; null without an order |
| cart_qty_current | Persistent cart quantity replayed from all past cart/remove/cart-checkout records |
| cart_sku_current | Number of products with positive remaining cart quantity |
| cart_gross_value_current | Remaining quantity × fixed catalog price, before coupons |
| last_session_min_viewed_price | Minimum logged display price in the last completed session; null without a priced view |
| last_session_max_viewed_price | Maximum logged display price in the same completed session |
| last_session_start_hour | Last completed session start hour in Asia/Seoul |
| last_session_start_weekday | Same start's local weekday, Monday=0 |
| coupon_uses_30d | Distinct coupons actually used in recent orders |
| coupon_redemption_recency_days | Days since the last past coupon-linked purchase; null without use |
| coupon_discount_amount_30d | Actual applied discounts in recent coupon-linked orders |
| orders_delta_30d | Recent order count minus preceding-window count |
| net_spend_delta_30d | Recent paid sum minus preceding-window paid sum |
| mean_net_order_value_delta_30d | Difference of window order means; null if either window has no order |
| unique_purchase_products_delta_30d | Difference of distinct purchased-product counts |
| purchased_qty_delta_30d | Difference of purchased-quantity sums |

Cart replay uses the shared raw-store function. A cart checkout subtracts its actual item quantities; buy-now leaves the cart intact. Same-time events follow event_id order. Sequence contexts are checked against the shared final cart replay.

The separate DL input contains at most the last 40 past actions, observed inter-event gaps, logged display prices and cart quantity/value immediately before each action. It never contains a next-event gap, a later session ending, or future coupon use. Sequences are right-padded and lengths are used by the LSTM. Context scaling is fitted on training controls only.

PRD compatibility outputs remain outside KIT32: five recent/prior ratios (`visit`, `orders`, `spend`, `basket_qty`, `coupon`), `purchase_cycle_anomaly`, and descriptive `journey_stage`. A zero ratio denominator produces null; no invented denominator or future lifecycle label is supplied. These outputs are not silently appended to the predictor registry.

The seven current `BEHAVIOR_FEATURES`, in registry order, remain separate from KIT32. The first two are the retained review `AUXILIARY_FEATURES` (KIT32 + these two = 34); all seven give 39:

| Feature | Definition |
|---|---|
| `review_count_30d` | All arrived review events in `[t0-30d,t0)`; retains legacy raw4/order-survey counting. Scheduled future reviews are excluded. |
| `mean_review_rating_30d` | Mean rating of those arrived review events; retains legacy rating kinds, null without ratings. |
| `cart_adds_per_nonpurchase_session_30d` | Cart-add action count in completed C30 sessions with no order before t0, divided by the number of those sessions; null for an empty denominator. Quantity does not weight an action. |
| `cart_removes_per_nonpurchase_session_30d` | Remove-action count over the same completed nonpurchase-session denominator; null for an empty denominator. |
| `review_last_order_mean_rating` | Average product rating per reviewed purchase, then the mean across purchases tied at the latest arrived-review clock. Uses all valid past product reviews, not only 30 days. |
| `review_recency_days` | Days since the last valid arrived product review over the observed past; null without one. |
| `review_non5_share_30d` | Within each purchase, fraction of recent valid product reviews rated below 5; then an equal-purchase average. SKU count and purchased quantity do not weight orders. Null without recent valid reviews. |

The three new product-review summaries require valid raw5 `product_star_scenario` observations linked to the same customer's actual purchase/SKU/order line. Duplicate purchase/SKU observations do not multiply a product's weight. Legacy raw4 reviews are not inferred into product reviews: these summaries remain null and `behavior_product_review_observed` records availability. This boolean is a diagnostic, not an extra predictor. Review count/mean retain their earlier compatibility and observation-coverage rules; observed absence is zero for count, while unavailable observation is null.

Reviews do not increment session counts or reset visit recency. Private scheduling scores, latent personas and future events are not predictors. Search/CS columns and related availability diagnostics are absent from the current 39-feature output even when a preserved historical raw file contains those events. Their historical reader/validator support remains intact. Adding or removing these computed outputs does not retrain existing downstream models.

The simulator's shopping IRL state uses a separate 40-dimensional observation representation (core15, run length1, progress bins10, cart bins4, five history values plus five availability flags). This does not alter or enlarge KIT39. Cosmetics does not observe review effects jointly with shopping; no coefficient learned from all39 predictors is claimed. Source reward/P/encoder and heldout strict-prefix evaluation remain train-only. A separate full-source donor posterior bank is used solely for observed-population synthesis, with one persistent private component draw per synthetic customer. It must not supply heldout/live forecasting posteriors or public features.

Native visit timing draws count and offsets jointly from source training windows, preserves same-clock starts, clips partial exposure without retry/time compression, and reports right censoring at the next scheduled visit. The 28-day analytic diagnostic uses the exact same bank's first-offset CDF. These generator changes can change realized histories and therefore feature values; the39 formulas and identifiers remain unchanged. The current source-derived policy-query cap is782 (maximum781 checkout decisions under known-total support plus one END query). Known-total support is not fully observed SKU inventory, and a cap hit is censoring, not a learned completion label. The manifest retains generation_fingerprint and dataset_id includes N and its fingerprint prefix; match raw/model/runtime/input lineage rather than seed alone. These identifiers remain provenance, not predictors. Boundary-truncated sessions are excluded from completed internal `ShoppingHistory`; the existing public C30 feature contract still uses recorded `ended_at<t0` and is not silently redefined.

Missing-value medians and missingness indicators are fitted within each training fold. Saved XGBoost/LightGBM pipelines preserve those transformations for inference. Legacy Parquet without the canonical price/cart/coupon contract is rejected for KIT feature calculation; original legacy files are preserved for historical analysis.
