-- Run from repository root:
-- sqlite3 artifacts/v4/small/raw.sqlite < scripts/query_v4.sql
ATTACH DATABASE 'artifacts/v4/features/features.sqlite' AS derived;

-- Canonical predictor order; IDs, labels and assignments are separate.
SELECT ordinal, feature_name, feature_set_version
FROM derived.feature_registry ORDER BY ordinal;

-- Point-in-time customer snapshots (never use post-assignment fields as predictors).
SELECT customer_id, as_of_at, cart_qty_current, net_spend_30d,
       review_count_30d, review_last_order_mean_rating
FROM derived.features ORDER BY customer_id LIMIT 20;

-- Actual coupon redemption is linked to its purchase and discounted SKU/line.
SELECT e.customer_id, e.event_at, e.related_purchase_event_id,
       e.product_id, e.order_line_no, e.discount_minor, o.paid_minor
FROM events AS e
JOIN orders AS o ON o.purchase_event_id = e.related_purchase_event_id
WHERE e.event_type = 'coupon_usage' ORDER BY e.event_at LIMIT 20;

-- Ratings identify purchased SKU lines, including purchases with several products.
SELECT e.customer_id, e.related_purchase_event_id, e.product_id,
       e.order_line_no, e.rating, e.event_at
FROM events AS e
WHERE e.event_type = 'review' ORDER BY e.event_at LIMIT 20;
