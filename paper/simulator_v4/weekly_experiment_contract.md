# 고객별 주간 집계와 고정 배정 실험 로그

`simulate.v4.data.weekly.build_weekly_panel(tables, metadata)`와 `build_experiment_log(tables, metadata)`는 검증된 일곱 원시 업무 테이블과 `labels`·`treatments` sidecar를 읽어 DataFrame을 반환한다. 원본을 수정하거나 새 세션·행동·주문·쿠폰을 만들지 않는다. 파일 저장은 호출자가 수행한다.

현재 export manifest는 `fingerprint.export_schema_version=2`, `fingerprint.event_scope_version=shopping_coupon_review_v1`이다. 검색·CS 집계 컬럼은 제거했으며, 이전 schema1 export는 덮어쓰지 않는다. 새 형식으로 내보낼 때는 새 출력 경로를 지정한다.

## 서로 다른 시간 구간

`reference_week`는 **Asia/Seoul 월요일 00:00**인 timezone-aware 시각이다. 한 주는 `[reference_week, reference_week+7일)`이다. `relative_week=0`은 고객의 실제 배정 시각이 속한 주이고, 같은 고객의 Group은 관측 기간 전체에서 고정된다.

최신 분석의 이탈 기준은 **4주·28일 미방문**, `future_no_session_28d_active30d`다. 의사결정이 월요일 00:00이면 성과 구간은 **w0부터 w3까지**, `[w0 시작,w4 시작)`이다. 직전 방문 적격성은 기존대로 `[as_of_at−30일,as_of_at)`에 세션 시작이 있는지 판정한다. 기존 KIT 기본 lookback과 원천 재방문 모형의 30일 학습 단위를 28일로 바꾸었다고 설명하지 않는다.

성과 구간의 실제 끝은 `labels.outcome_end_at`에서 읽는다. 이 컬럼이 없는 명시적 label 입력만 양의 정수 `metadata.outcome_window_days`를 사용하며 기존 `analysis_outcome_days`도 호환한다. labels sidecar 자체가 없어도 된다는 뜻은 아니다. 기간을 필드명에 30일로 고정하지 않고 `outcome_horizon_days`·`outcome_end_at`을 함께 내보낸다. 아직 성과 기간 전체가 관측되지 않았으면 `outcome_observed=False`이고 관측하지 못한 결과를 0으로 채우지 않는다.

원본 쿠폰 정책의 유효기간은 **발송 후 30일**을 유지한다. 28일 성과 구간으로 바뀌었다는 이유로 원시 `expires_at`을 앞당기지 않는다. 29일째 실제 사용한 쿠폰은 `redeemed=True`지만 `redeemed_in_outcome=False`이며 28일 성과 할인 비용에는 포함되지 않는다. 남은 2일의 사용·주문은 원시 기록과 해당 주간 합계에 그대로 남는다.

## 0과 NULL, 부분 관측

출력 grain은 **고객 × 전체 관측 기간에 걸친 달력 주**다. 각 행의 실제 관측 가능 구간은 주간 경계, 전체 관측 범위, 고객 가입 시각의 교집합이다.

| 상태 | 행동·금액 | 구간 정보 |
|---|---|---|
| 가입 전 또는 관측 범위와 겹침 없음 | `NULL` | `observed_week=False`, `coverage_seconds=0` |
| 관측 구간이 있으며 해당 기록 없음 | `0` | `observed_week=True` |
| 가입일·관측 시작/끝이 주 중간 | 관측된 부분의 합계 | `partial_coverage=True`, 실제 `coverage_seconds`와 경계 제공 |
| 세션 종료 시각이 미상 | 세션 시간은 `NULL` | 해당 시점 이후 가능한 주에 `session_seconds_complete=False` |

이는 가입 이후 선언된 범위에서 연속 관측되는 **현재 시뮬레이터**의 계약이다. 실제 서비스의 수집 중단·추적 누락을 원시 로그의 행 부재만으로 탐지했다고 주장하지 않는다. `no_purchase_this_week`·`no_session_this_week`는 그 행에서 관측된 부분에 대한 표시이며, 부분 주를 완전한 주간 실험 성과로 섞지 않는다.

## 주간 집계 정의

| 필드 | 계산 |
|---|---|
| `sessions` | 그 주에 시작한 고유 세션 수 |
| `purchases` | 그 주에 구매 이벤트가 발생한 고유 Order 수 |
| `converted_sessions` | 그 주에 주문이 하나 이상 있는 고유 세션 수 |
| `views`, `cart_adds`, `cart_removes` | 실제 해당 유형의 이벤트 행 수 |
| `reviews` | 실제 구매 상품별 별점 이벤트 수이며 주문 수가 아님. 방문·주문에 더하지 않음. 검색·CS는 현재 생성/주간 export 범위에서 제외 |
| `mean_review_rating` | 해당 주에 관측된 리뷰 별점 평균; 리뷰가 없으면 NULL |
| `session_seconds` | 세션과 해당 주가 겹치는 실제 시간의 합; 주 경계에서 분할 |
| `gross_revenue_minor` | 주문의 할인 전 금액 합 |
| `redemption_discount_cost_minor` | 실제 주문에 적용된 할인액 합 |
| `net_revenue_minor` | 실제 `orders.paid_minor` 합 |
| `coupons_sent` | 실제 raw Coupon 발송 행 수 |
| `coupon_actions` | 그 주에 기록된 쿠폰 발송 결정 수; 실행 여부와 별개 |
| `marketing_send_cost_minor` | 실제 raw Coupon에 기록된 발송 비용 합 |
| `net_after_marketing_minor` | `net_revenue_minor − marketing_send_cost_minor` |

한 세션에서 주문을 두 번 하면 구매 2건·구매 세션 1개다. 세션이 주 경계를 넘고 양쪽 주에 주문이 있으면 양쪽 `converted_sessions`에 각각 1개가 잡힌다. `sessions`는 시작 주 기준이므로 이 두 열의 비율을 별도 정의 없이 동일 코호트 전환율로 해석하지 않는다.

할인은 이미 실제 결제액에 반영되어 있다. `net_after_marketing_minor`에서 할인액을 다시 빼지 않는다. 원가·배송비·세금·자본 할인율이 없으므로 이 값은 회계 이익이나 할인된 CLV가 아니다. 금액은 정수 minor이고 각 행의 `currency`·`amount_unit`을 따른다. `SOURCE_UNSPECIFIED`를 KRW로 바꾸지 않는다.

raw3 이상 `coupon_usage`는 해당 구매의 사용·할인 대상 감사 이벤트다. `purchases`나 `redemption_discount_cost_minor`에 추가 합산하지 않는다. 최고 단가 한 개의 할인액이 Order에 확정되어 있으므로 기존과 같이 주문 금액을 합산한다. 과거 raw4/5 reader의 8종 허용 집합, 현재 생성의 6종, 학습용 네 쇼핑 행동 시퀀스는 구별한다. 쿠폰 사용은 주문 기준의 쿠폰 피처로 따로 반영한다. raw5의 주문별 리뷰 응답률은 `related_purchase_event_id`를 중복 제거해 계산하며 상품 리뷰 수를 주문 수로 나누지 않는다. 현재 비동기 리뷰만 있는 주는 쇼핑 세션·주문 수가 0일 수 있으며 28일 미방문 판정은 바뀌지 않는다. 보존한 과거 CS도 읽기 호환에서 같은 비방문 원칙을 유지한다.

## Native 방문 시계와 분석 확률

Native timing을 쓴 실행은 실제 source 창의 count와 시작 offsets를 함께 추출한다. Entry는 최초 관측일 UTC 자정, regular 창은 UTC15시(KST 자정)를 기준으로 한다. 부분 관측 창은 원래 schedule과 교집합으로 자르고 동시각 시작을 보존하며, 다음 예정 시작이나 원천 지원에 근거한782회 정책 조회 안전 상한이 세션을 right-censor할 수 있다. 이782는 checkout 결정781회+END판정1회의 예산이며, 완전히 관측된 SKU재고나 학습된 종료시점을 뜻하지 않는다. 이 때문에 기록된 `session_seconds`는 실제 생성·검열된 종료시각으로 합산하고, source template의 예정 시간을 실제 체류시간으로 대체하지 않는다.

28일 analytic 주변확률은 동일 source-frequency-weighted bank의 `p28=p30*F_first(28 | source_profile, active_previous30=True)`를 사용한다. 적격성은 실제 과거 세션에서 판정한다. `F_first`는 실제 sampler와 같은 profile/stratum fallback과 엄격한 `<28d` 경계를 사용하며, 기존 idle sampler의 식을 새 시계에 재사용하지 않는다. 실제 factual label·매출·주간 수치는 계속 원시 로그에서 집계한다. 전환된 확률은 28일 hazard를 새로 학습한 결과가 아니다.

Source donor의 전체 관측 이력으로 재구성한 latent 성분은 합성 전용 private 값이다. 주간 출력·Feature39·과거 시점 정책 입력에는 복사하지 않는다. 이 bank를 사용한 source-population fidelity와 strict-prefix heldout 미래 예측은 서로 다른 평가다. 쌍대 주변확률 평가도 완전한 동일 고객 쇼핑 경로 재생이나 실제 개인별 쿠폰 인과효과를 뜻하지 않는다.

## experiment_log

grain은 고정 실험의 **고객별 배정 1행**이다. 발송 실패·미사용·미방문을 이유로 Treatment나 Control 행을 제거하지 않는다.

- `treatment`, `group`, `assignment_probability`: 기존 고정 배정과 기록된 확률 0.5다. 매주 재추첨하지 않는다.
- `assigned_at`, `assignment_known_at`: 배정이 실제로 알려지는 시각이다. 주간 Group은 과거 주에도 비교용으로 붙인 회고 tag다. `assignment_known_at_week_start`는 엄격한 과거 시점 규칙을 쓰므로 w0 시작과 배정 시각이 같으면 False다. `assignment_known_by_week_end`는 그 주 안에서 알려졌는지 표시한다.
- `coupon_action`, `decision_action`: 발송 결정이다. `actual_sent`는 원시 Coupon 실행 기록 존재 여부로 별도 확인한다. 현재 정책에서 결정 필드가 따로 없으면 고정 배정의 Treatment가 한 번 발송하는 정책 결정을 뜻한다.
- `requested_coupon_id`와 `coupon_id`: 요청 식별자와 실제 발송 쿠폰을 구분한다. 실행되지 않은 요청을 raw Coupon으로 만들어 채우지 않는다.
- `sent_at`, `expires_at`, `send_cost_minor`: 실제 실행 기록이다. 현 정책은 w0 배정 시점 한 번만 발송하고 이후 추가 발송하지 않는다. Control의 t0 이후 새 발송이나 여러 번 발송은 오류로 거절한다.
- `execution_status`, `failure_reason`: 기록된 실행 상태·원인만 보존한다. 결정했지만 발송 기록이 없고 실패 원인도 없으면 `not_sent`와 NULL이며 원인을 추측하지 않는다. 현재 완성 데이터에는 실패 0건이고 `failure_reason`은 NULL이다. 이는 실패 확률을 실데이터로 추정했다는 뜻이 아니다.
- `redemption_order_id`, `redemption_at`, `redemption_discount_cost_minor`: 실제 `orders.coupon_id`와 해당 purchase 이벤트로 찾는다. 발송을 사용으로 간주하지 않는다.
- `redeemed_in_outcome`, `outcome_redemption_discount_cost_minor`: 실제 사용 중 명시된 성과 구간에 속하는 부분만 구분한다.
- `past_coupon_count`, `past_coupon_redemption_count`: t0 전에 실제로 관측한 발송·사용 기록 수다. 현 생성 정책은 t0 이전 쿠폰을 만들지 않아 모두 0이다. 이를 원천 Cosmetics 고객의 과거 쿠폰 이력이 없다는 증거로 해석하지 않는다. 과거 Control의 쿠폰 이력과 실험 후 새 발송도 구분한다.
- `eligible_at_assignment`: 실제 과거 세션에서 적격 여부를 재계산한다. 적격 여부를 미래 방문에서 만들지 않는다.
- `factual_no_session`, `factual_no_order`: 같은 명시적 성과 구간의 미방문과 미구매를 따로 집계한다. 둘은 서로 다른 사건이다. `factual_churn_label`은 주어진 `churn_definition_id`와 함께 보존한다.
- `posthoc_response_type`: factual 단일 실행에서 ST/P/SD/LC를 식별할 수 없어 NULL이다. 숨겨진 `diagnostics.response_type`을 복사하지 않는다. 쌍대 정책 실행의 `paired_outcome_type`과 latent truth는 별도 평가 산출물이다.

모든 행은 `dataset_id`·`seed`·고객/스냅샷 식별자를 보존한다. 새 생성의 dataset ID는 N과 `generation_fingerprint` 일부를 포함하며 설정·모형·runtime 코드·제공 입력의 차이를 구분한다. 같은 seed만으로 다른 크기나 모형의 주간 합계를 같은 실행으로 결합하지 않는다. 과거 불변 출력의 ID와 manifest는 그대로 유지한다. 배정·발송·사용·미래 결과·사후 타입은 t0 이전 예측 입력 X에서 제외한다. 주간 Group을 과거까지 붙였다는 이유로 과거 시점에 배정을 알고 있었다고 취급하지 않는다.

`reference_week`는 집계 bucket의 시작이지 그 합계를 이미 알 수 있는 예측 시점이 아니다. w0 시작에 의사결정하는 X에는 그 전에 끝난 과거 주의 합계만 사용할 수 있다. w0~w3 합계는 성과 자료다. `eligible_at_assignment`도 t0에서 판정한 회고 tag이므로 더 이른 주의 예측 입력으로 가져오지 않는다.

## 실행과 검증

```python
from simulate.v4.data.loader import load_tables
from simulate.v4.data.weekly import build_weekly_panel, build_experiment_log

tables = load_tables("artifacts/v4/run")
metadata = tables["_metadata"]
weekly = build_weekly_panel(tables, metadata)
experiment = build_experiment_log(tables, metadata)
```

```bash
PYTHONPATH=src python -m pytest tests/test_v4_features_sqlite.py tests/test_v4_paired_evaluation.py -q
```

위 명령은 이 체크아웃의 피처·주간 연결과 쌍대 평가 fixture를 검증한다. 아래 상세 항목은 upstream `tests/test_weekly_exports.py`의 원래 검증 범위이며 그 테스트 파일 전체를 이 체크아웃에 복사하지 않았다. 테스트는 작은 결정적 fixture만 사용한다. 주 경계·부분 관측·0/NULL, 주문/구매 세션 분리, 시간 분할, 결제·할인·발송비 합, 고정 배정의 known-at, 실패 기록, 과거 쿠폰, 28일 성과와 30일 유효기간, Control/추가 발송 거절, 입력 불변성을 검증한다. 이 초기 모듈 fixture 검증과 실제 후속 생성·export 검증은 구별한다. 실제 실행의 규모·파일 SHA·완료 여부는 해당 dataset 및 export manifest를 확인한다.
