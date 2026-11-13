"""시뮬레이터와 합의한 컬럼 이름과 기간. 기준 문서: uplift/feature.md."""

RECENT_WEEKS = 4  # 최근 4주 [w0-4, w0)
HISTORY_WEEKS = 12  # 과거 12주 [w0-12, w0)
OUTCOME_WEEKS = 4  # 결과 4주 [w0, w0+4)

# 실험 고객 목록
CUSTOMER_ID = "customer_id"
TREATMENT = "treatment"  # 발송군 1, 미발송군 0
W0 = "w0"  # 기준 주의 시작일(월요일)

# 고객별 주간 합계 (customer_id x reference_week)
REFERENCE_WEEK = "reference_week"
WEEKLY_COLUMNS = ["sessions", "purchases", "net_revenue", "views", "session_minutes"]

# w0 시작 시점 장바구니 상태
CART_COLUMNS = ["cart_qty", "cart_sku", "cart_gross_value"]

LABEL = "y"  # 결과 4주 동안 구매 1회 이상이면 1

FEATURES = [
    # 필수
    "weeks_since_last_purchase",
    "weeks_since_last_session",
    "purchase_count_4w",
    "session_count_4w",
    # 추가 (feature.md 2.1)
    "observed_tenure_weeks",
    "view_count_4w",
    "net_spend_4w",
    "mean_net_order_value_4w",
    "mean_session_minutes_4w",
    "views_per_session_4w",
    "orders_per_session_4w",
    "purchase_weeks_12w",
    "session_weeks_12w",
    "cart_qty_w0",
    "cart_sku_w0",
    "cart_gross_value_w0",
    "orders_delta_4w",
    "net_spend_delta_4w",
]
