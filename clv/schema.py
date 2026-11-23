"""CLV 컬럼 계약, 시뮬레이터 컬럼 매핑, 예측 변수 누수 방지 검사 (WBS 4.1·4.3).

구간 계약은 고정이다: 기준시점 ``t0`` 이전 13주가 입력, ``t0``부터 52주가 CLV 정답 구간, ``t0``부터 4주가
조건부 총구매액 정답 구간이다. 예측 변수·정답 이름에 이 길이가 들어가므로 설정값이 아니라 상수로 둔다.
"""
from __future__ import annotations

SEOUL = "Asia/Seoul"
WEEK_SECONDS = 7 * 24 * 3600

INPUT_WEEKS = 13
RECENT_WEEKS = 4
LABEL_WEEKS = 52
# 4주 조건부 총구매액 모델의 정답 구간 길이(쿠폰 유효기간 28일, Uplift의 향후 4주 구매와 같은 구간).
GROSS_WEEKS = 4
# 13주 안에 활동이 없을 때의 경과 주 수 값(구간 바로 바깥인 14).
NO_EVENT_RECENCY = INPUT_WEEKS + 1

# 시뮬레이터 v5 `weekly_panel` 컬럼 이름 -> 합의된 CLV 입력 이름.
SIMULATOR_TO_CLV = {
    "customer_id": "customer_id",
    "reference_week": "week",
    "sessions": "sessions",
    "purchases": "purchases",
    "gross_revenue_minor": "gross_revenue",
    "redemption_discount_cost_minor": "redemption_discount_cost",
    "coupons_sent": "coupons",
    "coupon_actions": "coupon_action",
}

REQUIRED_WEEKLY = ("customer_id", "week", "sessions", "purchases", "gross_revenue",
                   "redemption_discount_cost", "coupons")
OPTIONAL_WEEKLY = ("coupon_action",)
COUNT_COLUMNS = ("sessions", "purchases", "coupons", "coupon_action")
AMOUNT_COLUMNS = ("gross_revenue", "redemption_discount_cost")
# 관측 정보: 관측 주 여부, 부분 관측 주 여부, 그 주의 관측 초. 세 컬럼은 함께 있어야 한다.
COVERAGE_COLUMNS = ("observed_week", "partial_coverage", "coverage_seconds")
PROVENANCE_COLUMNS = ("currency", "amount_unit", "dataset_id")
# 분석·그룹별 평가에만 보존한다. 예측 변수로는 절대 쓰지 않는다.
CONTEXT_COLUMNS = ("treatment", "group", "assigned_at", "as_of_at", "signed_up_at")
# 금액 대조(`net_revenue_minor` == 총매출 - 할인액)에만 읽는 시뮬레이터 컬럼.
CHECK_COLUMNS = {"net_revenue_minor": "source_net_revenue"}
# 실험 배정 열(어댑터가 검사하는 내부 이름). 시뮬레이터 ``treatment``: 0 = Control(쿠폰 미발송군),
# 그 밖의 값 = 쿠폰 발송군. ``group``이 있으면 ``"control"``이 ``treatment == 0``과 일치해야 한다.
# 시뮬레이터 배정 규약이 바뀌면 이 상수와 ``adapter.adapt_weekly``만 고친다.
ASSIGNMENT_COLUMN = "treatment"
CONTROL_CODE = 0
CONTROL_GROUP_LABEL = "control"
# 이 통화 값은 KRW로 간주하지 않는다(단위 환산도 하지 않음).
UNSPECIFIED_CURRENCY = "SOURCE_UNSPECIFIED"

# 변환 후 주문 원장 컬럼과, 원천 컬럼 별칭 -> 표준 이름.
ORDER_COLUMNS = ("order_id", "customer_id", "order_at", "gross_amount", "discount_amount", "paid_amount")
ORDER_ALIASES = {
    "event_at": "order_at", "purchased_at": "order_at",
    "gross_minor": "gross_amount", "discount_minor": "discount_amount", "paid_minor": "paid_amount",
}

REQUIRED_FEATURES = (
    "purchase_count_4w", "purchase_count_13w", "session_count_4w", "session_count_13w",
    "net_spend_4w", "net_spend_13w", "weeks_since_last_purchase",
)
RECOMMENDED_FEATURES = (
    "purchase_active_weeks_13w", "zero_purchase_streak", "purchase_trend", "spend_trend",
    "purchase_per_session_13w", "weeks_since_last_session", "session_active_weeks_13w", "session_trend",
)
AUXILIARY_FEATURES = ("avg_paid_per_purchase_13w", "coupon_sent_13w")
# 모델 입력 순서는 이 튜플 순서로 고정한다(필수 -> 권장 -> 보조).
PREDICTOR_FEATURES = (*REQUIRED_FEATURES, *RECOMMENDED_FEATURES, *AUXILIARY_FEATURES)
# 예측 변수와 함께 계산하지만 상태 표시·정규화·감사용이며 모델 입력은 아니다.
HISTORY_COLUMNS = ("history_weeks_13w", "history_weeks_4w", "history_weeks_prev9w",
                   "feature_window_complete", "coupon_action_13w")

LABEL_COLUMN = "actual_clv_12m"
PREDICTION_COLUMN = "predicted_clv_12m"
# CLV 내부 이름이다. 공용 출력 열 이름은 팀 합의가 더 필요하다.
HIGH_VALUE_COLUMN = "is_high_value_customer"
# 4주 조건부 총구매액: 4주 안 구매 횟수, 구매 고객의 할인 전 총구매액(비구매 고객은 결측), 4주 관측 완전 여부.
GROSS_PURCHASES_COLUMN = "actual_purchases_4w"
GROSS_LABEL_COLUMN = "actual_gross_4w_if_purchase"
GROSS_COMPLETE_COLUMN = "gross_4w_window_complete"
GROSS_PREDICTION_COLUMN = "predicted_gross_4w_if_purchase"
OUTPUT_COLUMNS = ("customer_id", "reference_week", PREDICTION_COLUMN, HIGH_VALUE_COLUMN)

# 누수 방지: 고객 식별자, 절대 시각, 실험 배정, 정답·예측, 기준시점 이후 사실을 드러내는 이름.
_FORBIDDEN_EXACT = {
    "customer_id", "week", "reference_week", "as_of_at", "snapshot_id", "offset_week", "relative_week",
    "treatment", "group", "assigned_at", "assignment_probability", "signed_up_at", "split",
    LABEL_COLUMN, PREDICTION_COLUMN, HIGH_VALUE_COLUMN, GROSS_LABEL_COLUMN, GROSS_PREDICTION_COLUMN,
    GROSS_COMPLETE_COLUMN, "coupons_in_label_window", "policy_eligible",
}
_FORBIDDEN_PREFIXES = ("gap_", "future_", "label", "latent", "posthoc", "factual_", "outcome_",
                       "redeemed", "redemption_", "actual_", "churn", "response_")


def forbidden_predictors(names) -> list[str]:
    """``names`` 중 예측 변수로 쓸 수 없는 이름(식별자·절대 시각·정답·t0 이후 사실)을 돌려준다."""
    return [name for name in names
            if name in _FORBIDDEN_EXACT or any(name.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES)]


def check_predictors(names) -> None:
    """예측 변수 목록에 제외 대상 컬럼이 있으면 ``ValueError``를 낸다(누수 방지)."""
    bad = forbidden_predictors(names)
    if bad:
        raise ValueError(f"excluded columns cannot be CLV predictors: {bad}")


# import 시점에 기본 예측 변수 목록 자체도 검사한다.
check_predictors(PREDICTOR_FEATURES)
