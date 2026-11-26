"""시뮬레이터 v5(raw5 · KIT39) 계약: 원시 테이블, 기간, feature 목록. 기준 문서: uplift/feature.md."""

# 기간(일). 실험 창은 28일, feature 창은 30일이다 (설계서 v5 4.2).
ELIGIBILITY_DAYS = 28  # 적격: [t0-28d, t0)에 세션 시작
OUTCOME_DAYS = 28  # 결과: [t0, t0+28d)
RECENT_DAYS = 30  # 최근 창 W30 = [t0-30d, t0), 직전 창 = [t0-60d, t0-30d)
GAP_DAYS = 90  # 방문 간격(F13, F14)을 보는 범위

# 시뮬레이터 실행 디렉터리의 파일
RAW_DATABASE = "raw.sqlite"
RAW_MANIFEST = "raw.manifest.json"
ASSIGNMENT_FILE = "treatments.parquet"

# 원시 업무 테이블 7개와 여기서 읽는 컬럼
RAW_COLUMNS = {
    "customers": ["customer_id", "signed_up_at"],
    "products": ["product_id", "unit_price_minor"],
    "sessions": ["session_id", "customer_id", "started_at", "ended_at"],
    "events": ["event_id", "customer_id", "session_id", "event_at", "event_type", "product_id",
               "quantity", "unit_price_minor", "related_purchase_event_id", "order_line_no",
               "rating", "rating_kind"],
    "orders": ["order_id", "purchase_event_id", "checkout_source", "coupon_id", "discount_minor",
               "paid_minor"],
    "order_items": ["order_id", "line_no", "product_id", "quantity"],
    "coupons": ["coupon_id", "customer_id", "sent_at", "expires_at"],
}
TIME_COLUMNS = {"signed_up_at", "started_at", "ended_at", "event_at", "sent_at", "expires_at"}
PRODUCT_REVIEW_KIND = "product_star_scenario"  # 유효 상품 리뷰의 rating_kind

# 학습 데이터셋의 컬럼
CUSTOMER_ID = "customer_id"
T0 = "t0"  # 배정·발송 시각 (UTC, 현지 월요일 00시)
TREATMENT = "treatment"  # 발송군 1, 미발송군 0
LABEL = "y"  # 결과 28일 동안 주문 1건 이상이면 1

# KIT39: 시뮬레이터 KIT_FEATURES + 확장 7개와 같은 이름·순서
KIT39 = [
    "visit_recency_days",  # F01
    "purchase_recency_days",
    "observed_tenure_days",
    "session_count_30d",
    "view_count_30d",
    "purchase_count_30d",
    "net_spend_30d",
    "mean_net_order_value_30d",
    "mean_session_duration_minutes_30d",  # F09
    "views_per_session_30d",
    "orders_per_session_30d",
    "mean_purchase_gap_days",
    "intervisit_start_gap_mean_days_90d",
    "intervisit_start_gap_cv_90d",
    "unique_purchase_products_30d",
    "purchased_qty_30d",
    "mean_basket_qty_30d",  # F17
    "cart_qty_current",
    "cart_sku_current",
    "cart_gross_value_current",
    "last_session_min_viewed_price",
    "last_session_max_viewed_price",
    "last_session_start_hour",
    "last_session_start_weekday",
    "coupon_uses_30d",  # F25
    "coupon_redemption_recency_days",
    "coupon_discount_amount_30d",
    "orders_delta_30d",
    "net_spend_delta_30d",
    "mean_net_order_value_delta_30d",
    "unique_purchase_products_delta_30d",
    "purchased_qty_delta_30d",
    "review_count_30d",  # F33
    "mean_review_rating_30d",
    "cart_adds_per_nonpurchase_session_30d",
    "cart_removes_per_nonpurchase_session_30d",
    "review_last_order_mean_rating",
    "review_recency_days",
    "review_non5_share_30d",
]

# 과거 쿠폰 이력(F25-F27). 기본 정책은 실험 전 쿠폰을 만들지 않아 전 고객이 같은 값이다.
COUPON_HISTORY_FEATURES = KIT39[24:27]

# 모델 입력. 쿠폰 이력이 있는 시나리오에서는 KIT39 전체를 쓴다.
FEATURES = [name for name in KIT39 if name not in COUPON_HISTORY_FEATURES]
