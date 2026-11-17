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

-- Optional Retention analyses. Bind :customer_id and :as_of_at first.
-- Use UTC ISO timestamps with nine fractional digits for the raw cutoff.
-- Completed sessions only; the end of an open session is future information.
SELECT session_id, started_at, ended_at
FROM sessions
WHERE customer_id = :customer_id AND started_at < :as_of_at AND ended_at < :as_of_at
ORDER BY started_at, session_id;

-- Purchase history before the snapshot. Net spend uses paid_minor once per order.
SELECT o.order_id, e.event_at AS purchased_at, e.session_id,
       o.gross_minor, o.discount_minor, o.paid_minor, o.coupon_id
FROM events AS e
JOIN orders AS o ON o.purchase_event_id = e.event_id
WHERE e.customer_id = :customer_id AND e.event_type = 'purchase' AND e.event_at < :as_of_at
ORDER BY e.event_at, e.event_id;
