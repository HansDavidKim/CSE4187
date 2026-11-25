- **담당자: 한유승**
- **모델:** Uplift
- **기준:** 시뮬레이터 설계서 v5 (`docs/KIT_feature_simulator_design_v5_daewon.pdf`), raw5 · KIT39

현재 주간 adapter는 동일39열을 공식 Feature Store에서 읽고 manifest의4/8/13주 기간을
사용한다. 아래30/60/90일 산식은 보존한 구형 raw 경로의 설명이며 주간 자료에 재계산하지 않는다.
현재 실행·원장 바인딩·전략 handoff는 [README](readme.md)를 따른다.

## 1. 학습 데이터 기준

- 시뮬레이터의 `features.sqlite`에서 배정 직전 KIT39와 같은 스냅샷의 실험 기록을 읽는다.
    - 저장소 공용 `common/simulator_features.py`가 `feature_registry`의 등록 순서와 버전을 확인하고 SQLite에서 피처를 조회한다.
    - `uplift/common/simulator.py`가 적격 고객의 피처와 실험 배정·정답을 연결한다. 세 모델이 같은 입력을 사용한다.
- **기준 시각 `t0`:** 배정·발송 시각. 현지(Asia/Seoul) 월요일 00시, 저장은 UTC
- **실험 조건 (시뮬레이터 v5):**
    - 대상: `[t0 − 28일, t0)`에 세션을 시작한 적격 고객. 배정 이후 활동이 없는 고객도 포함
    - 배정: 고객별 고정 Treatment/Control 1:1, `assignment_probability = 0.5`
    - Treatment: `t0`에 10% 쿠폰 1회 발송, 28일 유효, 추가 발송 없음. 최고 단가 품목 1개에만 할인
    - Control: 쿠폰 미발송 보장
- **예측할 대상:** 쿠폰 발송에 따른 향후 28일 구매 확률의 변화
    - Y = 1: `[t0, t0 + 28일)`에 주문이 1건 이상
    - Y = 0: 해당 기간에 주문 없음
    - 시뮬레이터 기본 이탈 정의(28일 미방문)와 다른 정의다. 시뮬레이터의 `no_purchase_labels`(`future_no_order_28d`)와 같은 정의
- **데이터 형태:** 적격 고객 1명당 1행. `customer_id, t0, treatment, Feature, y`

## 2. 입력 Feature

시뮬레이터의 KIT39와 같은 이름·순서·산식을 쓴다. 모든 Feature는 `event_at < t0`인 기록만 읽는다.

- 최근 창 `W30 = [t0 − 30일, t0)`, 직전 창 `[t0 − 60일, t0 − 30일)`. 실험 창(28일)과 길이가 다르다
- `C30` = 최근 30일에 시작하고 `t0` 전에 끝난 세션
- 빈 분모·이력 없음은 NULL, 관측된 무행동의 횟수·합계는 0
- 금액은 원천 통화의 minor 단위(원화 아님)

### 2.1 모델 입력 36개

| 번호 | Feature | 계산 | NULL |
| --- | --- | --- | --- |
| F01 | `visit_recency_days` | 마지막 세션 시작부터 경과 일수 | 방문 없음 |
| F02 | `purchase_recency_days` | 마지막 주문부터 경과 일수 | 구매 없음 |
| F03 | `observed_tenure_days` | 최초 세션부터 경과 일수 | 방문 없음 |
| F04 | `session_count_30d` | W30에 시작한 세션 수 | |
| F05 | `view_count_30d` | W30의 view 수 | |
| F06 | `purchase_count_30d` | W30의 주문 수 | |
| F07 | `net_spend_30d` | W30의 `paid_minor` 합 | |
| F08 | `mean_net_order_value_30d` | W30의 `paid_minor` 평균 | 주문 없음 |
| F09 | `mean_session_duration_minutes_30d` | C30의 평균 체류 시간(분) | C30 없음 |
| F10 | `views_per_session_30d` | C30의 view 수 / C30 세션 수 | C30 없음 |
| F11 | `orders_per_session_30d` | C30의 주문 수 / C30 세션 수 | C30 없음 |
| F12 | `mean_purchase_gap_days` | 전체 과거 주문 사이 평균 간격 | 주문 2건 미만 |
| F13 | `intervisit_start_gap_mean_days_90d` | 뒤쪽 시작이 최근 90일인 세션 시작 간격의 평균 | 간격 없음 |
| F14 | `intervisit_start_gap_cv_90d` | 같은 간격의 표본 표준편차 / 평균 | 간격 2개 미만, 평균 0 |
| F15 | `unique_purchase_products_30d` | W30 주문 상품의 종류 수 | |
| F16 | `purchased_qty_30d` | W30 주문 상품 수량 합 | |
| F17 | `mean_basket_qty_30d` | F16 / F06 | 주문 없음 |
| F18 | `cart_qty_current` | `t0` 직전 장바구니 잔존 수량 | |
| F19 | `cart_sku_current` | 잔존 상품 종류 수 | |
| F20 | `cart_gross_value_current` | 잔존 수량 × 상품 단가 합(할인 전) | |
| F21 | `last_session_min_viewed_price` | `t0` 전에 끝난 마지막 세션의 view 표시 단가 최솟값 | 세션·view 없음 |
| F22 | `last_session_max_viewed_price` | 같은 세션의 최댓값 | 세션·view 없음 |
| F23 | `last_session_start_hour` | 같은 세션의 현지 시작 시(0–23) | 세션 없음 |
| F24 | `last_session_start_weekday` | 같은 세션의 현지 요일(월 0 – 일 6) | 세션 없음 |
| F28 | `orders_delta_30d` | 주문 수: 최근 30일 − 직전 30일 | |
| F29 | `net_spend_delta_30d` | 결제액 합: 최근 − 직전 | |
| F30 | `mean_net_order_value_delta_30d` | 결제액 평균: 최근 − 직전 | 한 창이라도 주문 없음 |
| F31 | `unique_purchase_products_delta_30d` | 구매 상품 종류 수: 최근 − 직전 | |
| F32 | `purchased_qty_delta_30d` | 구매 수량 합: 최근 − 직전 | |
| F33 | `review_count_30d` | W30에 도착한 review 수 | |
| F34 | `mean_review_rating_30d` | 같은 review의 별점 평균 | 리뷰 없음 |
| F35 | `cart_adds_per_nonpurchase_session_30d` | C30 중 주문 없는 세션의 cart 수 / 그 세션 수 | 해당 세션 없음 |
| F36 | `cart_removes_per_nonpurchase_session_30d` | 같은 세션의 remove 수 / 그 세션 수 | 해당 세션 없음 |
| F37 | `review_last_order_mean_rating` | 가장 최근에 리뷰가 도착한 구매의 별점 평균 | 유효 리뷰 없음 |
| F38 | `review_recency_days` | 마지막 유효 리뷰 도착부터 경과 일수 | 유효 리뷰 없음 |
| F39 | `review_non5_share_30d` | W30 유효 리뷰의 5점 미만 비율(구매별로 구한 뒤 평균) | 유효 리뷰 없음 |

- 장바구니(F18–F20)는 과거 cart·remove·구매를 `(event_at, event_id)` 순서로 재생한다. `buy_now` 구매는 차감하지 않고, 그 외에는 보유량 한도까지 차감한다
- 유효 리뷰(F37–F39)는 본인의 실제 구매·상품·주문행에 연결된 `product_star_scenario` 별점이다. 구매 × 상품마다 한 번만 센다

### 2.2 계산하지만 입력에서 빼는 3개

| 번호 | Feature | 계산 |
| --- | --- | --- |
| F25 | `coupon_uses_30d` | W30 주문에 사용된 쿠폰 수 |
| F26 | `coupon_redemption_recency_days` | 마지막 쿠폰 사용 구매부터 경과 일수 |
| F27 | `coupon_discount_amount_30d` | W30 쿠폰 사용 주문의 할인액 합 |

- 기본 정책은 실험 전 쿠폰을 만들지 않아 전 고객이 0 / NULL / 0이다. 학습용 스냅샷에 보관하고 세 모델은 나머지 36개(`schema.FEATURES`)를 사용한다

### 2.3 관측 범위에 따른 NULL

창별 관측 범위에 따라 시뮬레이터가 기록한 NULL을 그대로 읽는다.

| 대상 | 관측돼야 하는 구간 |
| --- | --- |
| 전체 | `t0`가 가입 시각 이후. 아니면 39개 모두 NULL |
| 이름이 `_30d`로 끝나는 Feature | `max(t0 − 30일, 가입)` 이후 |
| `_delta_30d` | `max(t0 − 60일, 가입)` 이후 |
| F13, F14 | `max(t0 − 90일, 가입)` 이후 |
| F18–F20 | 가입 시점부터 |

### 2.4 입력 금지

- 배정·실제 발송·`t0` 이후의 사용과 기록
- life-cycle, source donor, 잠재 성분, 사후 반응 타입(`diagnostics`, `evaluation_only`)
- `customer_id`, `t0`, `dataset_id`는 연결용이며 입력이 아니다

## 3. 정답과 시뮬레이터에서 읽는 것

- **정답:** `[t0, t0 + 28일)`에 주문 1건 이상이면 Y = 1. `observation_end_at`이 `t0 + 28일`보다 이르면 NULL로 두고 학습에서 제외
- **uplift:** `쿠폰 발송 시 구매 확률 − 미발송 시 구매 확률`
- **T:** 배정군 그대로(ITT). 쿠폰 사용 여부로 다시 나누지 않는다

| SQLite 테이블 | 읽는 내용 |
| --- | --- |
| `feature_registry` | KIT39 v3 컬럼과 순서 |
| `features` | 배정 직전의 39개 값, 고객·UTC 기준시각·스냅샷·데이터셋 ID |
| `experiment_log` | 동일 스냅샷의 적격성, 배정군·배정 확률, 28일 성과 관측 여부와 실제 미주문 여부 |

원시 로그 입력을 사용하는 기존 실행에서는 `raw.sqlite`, `raw.manifest.json`, `treatments.parquet`를 읽어 같은 데이터셋을 구성한다. 열별 NULL과 관측된 횟수 0은 데이터셋에 각각 보존한다. 기본 히스토그램 부스팅 모델은 각 군의 학습 자료에서 중앙값과 결측 지표를 구성하고, 관측값이 전혀 없는 열도 처리한다.
