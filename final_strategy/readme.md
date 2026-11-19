# KIT Final Strategy

## 1. 개요

`final_strategy`는 시뮬레이터 Feature Store와 Churn·Uplift·CLV 예측을 검증하고, 고객 분류 → 리텐션 후보·예상 비용 → 예산 배분 → 예산별 What-if를 수행한다. 모델 학습, 실제 쿠폰 발송, A/B 실험, 실제 ROI 계산은 이 모듈의 실행 범위에 포함하지 않는다.

현재 공개 코드에는 입력 검증, 두 Prediction Adapter, Segmentation, Retention Strategy, Greedy·MILP·Hybrid, 품질 인증, What-if 및 관련 테스트가 포함된다. 실제 학습 번들 세 개를 연결한 추론과 운영 데이터 전체 흐름은 아직 검증되지 않았다.

| 단계 | 공개 진입점 | 주요 반환값 |
| --- | --- | --- |
| 입력 | [inputs.py](inputs.py)의 `load_feature_store()`, `integrate_inputs()` | 검증된 DataFrame, `IntegratedInputs` |
| 입력 검증 | [validation.py](validation.py)의 `validate_feature_frame()`, `validate_prediction_frame()` | 정규화된 복사본; 위반 시 `InputValidationError` |
| 고객 분류 | [segmentation.py](segmentation.py)의 `segment_customers()` | `SegmentationResult` |
| 예측 변환 | [retention_prediction_adapter.py](retention_prediction_adapter.py), [uplift_prediction_adapter.py](uplift_prediction_adapter.py) | 예측 batch·대상 Feature·제외 감사·생산자 메타데이터 |
| 정책·비용 | [retention_strategy.py](retention_strategy.py)의 `build_retention_strategy()` | `RetentionResult` |
| 예산 배분 | [budget.py](budget.py)의 `optimize_budget()` | `BudgetComparison` |
| 품질 인증 | [budget_quality.py](budget_quality.py), [budget_quality_integration.py](budget_quality_integration.py) | 정확한 L/U 및 인증 결과 |
| 예산 시나리오 | [what_if.py](what_if.py)의 `run_what_if()` | `WhatIfResult` |

## 2. 입력 데이터 계약

### Feature Store

현재 시뮬레이터 계약은 KIT39 `kit39-review-cart-v3`, 행동 Feature `behavior_review_cart_v3`이다. Python 패키지 경로는 계속 `simulate.v4`를 사용한다. [시뮬레이터 registry](../simulator/src/simulate/v4/features.py)의 `EXPANDED_FEATURES` 39개와 [storage](../simulator/src/simulate/v4/storage.py)의 `FEATURE_KEYS` 12개를 재사용하며, 여기에서 원시 데이터의 Feature를 다시 계산하지 않는다.

12개 식별·출처 컬럼은 다음과 같다.

```text
customer_id, as_of_at, snapshot_id, reference_week, dataset_id,
currency, amount_unit, raw_database_sha256, source_model_sha256,
feature_set_version, behavior_feature_set_version, expanded_feature_set_version
```

- `feature_set_version`은 기본 Feature 집합의 `kit32-v1`, 전체 39개 버전은 `expanded_feature_set_version`으로 구분한다. Feature 이름과 발행 순서는 registry를 기준으로 한다.
- `customer_id`는 문자열이다. `001`, `NA`를 숫자나 결측으로 바꾸지 않는다. CSV를 별도로 읽는 소비자는 문자열 dtype과 NA 파싱을 명시해야 한다.
- 결합 키는 `(dataset_id, customer_id, as_of_at)`이다. 시간대가 있는 기준 시각을 UTC로 정규화한다. `reference_week`는 Asia/Seoul 기준 해당 주 월요일의 날짜 문자열이다. 같은 주의 다른 기준 시각을 하나의 snapshot으로 합치지 않는다.
- `snapshot_id`는 불투명한 식별자다. 문자열 내부를 해석하거나 새로 구성하지 않고 모델과 Feature의 값을 대조한다. 데이터셋·출처 SHA·통화·단위의 혼합과 중복 키를 거절한다.
- `load_feature_store(path)`는 `feature_store.parquet` 또는 그 디렉터리를 받는다. 함께 발행된 manifest의 버전·컬럼·고객 수·snapshot·출처 정보와 Parquet SHA256을 검사한다. 이 검사는 원시 SQLite를 다시 열어 그 파일의 SHA를 재계산하는 작업과는 다르다.
- `validate_feature_frame(frame)`은 메모리의 컬럼·버전·값·시점·출처 계약을 검사한다. 정상적인 Feature NULL을 유지한다. 파일 manifest 검증을 대신하지 않는다.
- 미래 정답·실험 배정·사후 결과·잠재 반응 유형은 입력 Feature가 아니다. 허용 컬럼만 사용하며 `evaluation_only` 입력을 거절한다. 생산자가 Feature를 실제로 과거 정보만으로 만들었는지까지 소비자 검증만으로 증명하지는 못한다.

### 모델별 예측

`PredictionMetadata(model_version, target_definition_id, horizon_value, horizon_unit, value_semantics)`는 내부 소비 계약이다. `PredictionBatch(frame, metadata)`의 frame에는 `CONTEXT_COLUMNS` 8개, `input_cutoff_at`, `prediction_end_at`, 해당 모델의 값 컬럼이 필요하다.

| 모델 key | 값 컬럼 | 의미·값 검증 | 현재 Adapter/CLV 계약 |
| --- | --- | --- | --- |
| `churn` | `churn_probability` | 유한한 [0, 1], `probability` | Retention Adapter: 향후 28일 무세션 확률 |
| `uplift` | `uplift_score` | 유한한 [-1, 1], `probability_difference` | Uplift Adapter: 향후 28일 구매 발생의 p1 − p0 |
| `clv` | `predicted_clv_12m` | 유한한 금액, 권장 `nonnegative_amount` | 반드시 `52, "weeks"`, 종료는 기준 시각 + 364일 |

`predicted_clv_12m`이라는 기존 컬럼명은 호환성을 위해 유지하지만 실제 기간은 **52주(364일)**다. 달력 12개월이나 단위가 days인 364일 선언을 같은 계약으로 허용하지 않으며 값을 비례 환산하지 않는다. CLV의 `amount` 선언은 입력 단계에서 음수를 표현할 수 있지만, Segmentation은 음수 CLV를 거절한다.

`input_cutoff_at <= as_of_at`이어야 하고, `prediction_end_at`은 명시한 기간과 일치해야 한다. cutoff는 추론 입력의 관측 상한이며 학습 시점을 뜻하지 않는다. 세 모델의 목표와 기간은 각각 보존한다. Uplift와 Churn을 같은 결과 변수로 취급하지 않는다.

`currency`, `amount_unit`을 Feature·예측·비용·예산 사이에 보존한다. 원천 minor 단위의 수치에 단위를 다시 곱하거나 금액을 자동 환산하지 않는다. 예를 들어 `amount_unit="0.01"`인 자료의 비용 100은 그 자료의 금액 단위 100개다. `SOURCE_UNSPECIFIED`를 KRW 등 특정 통화로 해석하지 않는다.

`integrate_inputs(features, predictions)`는 `churn`, `uplift`, `clv`의 `PredictionBatch`를 모두 요구한다. 각 batch는 같은 고객·시각 모집단을 정확히 덮어야 한다. 누락 예측을 0으로 채우거나 고객 교집합을 조용히 취하지 않는다. 반환 `IntegratedInputs`에는 `data`, 모델별 `prediction_metadata`, `prediction_context`가 들어간다.

## 3. Segmentation

`segment_customers(inputs, config=None, eligibility=None)`는 검증된 `IntegratedInputs`를 다시 검사한다. `SegmentationConfig()`의 개발용 기본값은 Churn 0.5, Uplift 0.0, CLV 상위 비율 0.2다. 검증된 사업 정책을 뜻하지 않는다.

| 축 | 분류 |
| --- | --- |
| Churn | 임계값 이상 `high`, 미만 `low` |
| Uplift | 음수 `negative`, 0 이상 임계값 이하 `neutral_low`, 임계값 초과 `positive` |
| CLV | 상위 `high_value`, 나머지 `standard_value` |

각 `(dataset_id, as_of_at)`의 분석 적격 고객은 2×3×2 = 12개 segment에 중복 없이 속한다. CLV 상위 고객 수는 `ceil(N * high_clv_fraction)`이며 CLV 내림차순, 고객 ID 문자열 오름차순으로 결정한다. 작은 모집단에서는 실제 비율이 정확히 20%가 아닐 수 있다.

`priority_score = uplift_score * predicted_clv_12m`이다. 점수 내림차순과 고객 ID 오름차순으로 고객별 순차 순위를 정한다. 음수·0 점수도 분류 결과에 남는다. 서로 다른 예측 기간의 값으로 만든 **전략 우선순위 대리지표**이며 실제 증분 매출·이익·ROI가 아니다.

명시적 `eligibility`를 주면 전체 키에 대한 `eligible`과 `exclusion_reason`이 필요하다. 모델값 누락을 정상적인 제외 사유로 대체하지 않는다. `SegmentationResult`는 `customers`, `excluded`, `summary`, `metadata`를 제공한다. summary는 빈 segment도 포함한 12개 구간의 고객 수·비율, 평균 예측값·점수, 점수 합계를 담는다.

`export_segments(result, csv_path)`는 고객 CSV와 같은 stem의 `_summary.csv`, `_excluded.csv`, `_metadata.json`을 기록하며 기존 파일을 덮어쓰지 않는다. `plot_segments(result, dataset_id, as_of_at)`는 선택한 cohort의 Figure를 반환한다.

## 4. Prediction Adapter

### Retention

`adapt_retention_predictions(predictions, features, *, expected_policy_condition="observed_experiment_mix")`는 Retention 공개 API 형태의 DataFrame을 받는다. 모델 패키지나 학습 라이브러리를 import하지 않는다.

- `target="visit"`, `definition_id="future_no_session_28d_active28d"`, `horizon_days=28`을 검증한다.
- `status="scored"`만 정상 확률로 변환한다. `not_signed_up_at_cutoff`, `no_recent_session`은 원래 NULL 확률과 사유를 `excluded` 감사 DataFrame에 보존한다.
- Feature와 모든 입력 고객, dataset·snapshot·UTC 기준 시각·raw SHA·Feature 버전·통화·단위를 대조한다. snapshot 불일치를 보정하지 않는다.
- `RetentionPredictionResult`는 `features`, `batch`, `excluded`, `producer_metadata`, 입력·예측·제외 고객 수를 제공한다. 전원 제외 시 `batch=None`이므로 입력 통합을 호출하지 않는다. 빈 원본 입력은 거절한다.
- 정책 조건은 생산자 메타데이터에 보존한다. `observed_experiment_mix`는 무쿠폰 조건의 이탈 확률이 아니다. 다른 조건은 호출자의 명시적 기대값과 일치해야 한다.

### Uplift

`adapt_uplift_predictions(model, features, metadata, *, input_cutoff_at, eligibility=None)`는 호출자가 제공한 `model.predict(X)`를 사용한다. 실제 모델 파일을 찾거나 다른 브랜치 코드를 복사하지 않는다.

`UpliftModelMetadata`에는 모델 버전, `feature_names`, 목표 정의, `target`, 기간, Treatment·Control, 정책 조건, 학습 출처 버전, Feature 버전, 통화·단위를 명시한다. `model_artifact_sha256`은 선택 사항이다. 이는 선언 검증이며 학습 자료나 실제 번들의 진위를 증명하지 않는다.

모델에는 KIT39 중 F25~F27 쿠폰 이력 Feature를 제외한 정확한 36개 `UPLIFT_FEATURES`를 정해진 순서로 전달한다. 고객 식별자·Treatment·미래 Y는 전달하지 않는다. 정상 NULL을 보존하고 고유한 index로 고객 대응을 검증한다. 반환 index의 초기화·재정렬·누락·중복과 입력 변조를 거절한다.

반환 컬럼은 순서대로 `p0`, `p1`, `uplift_score`다. 유한한 확률과 `uplift_score = p1 - p0`를 검사하며, 차이 계산에는 코드의 기계정밀도 수준 오차만 허용한다. 음수 Uplift를 자르거나 잘못된 값을 보정하지 않는다. 정의는 `purchase_occurrence`, 28일, `coupon_assignment` 대 `no_coupon_assignment`이다. 쿠폰 사용 확률이나 CLV 증가액이 아니다.

`UpliftPredictionResult`도 batch·대상 Feature·제외 감사·생산자 메타데이터를 반환한다. 제외는 명시적 eligibility로만 결정한다. 예측 누락은 오류다. 두 Adapter의 `producer_metadata`와 `excluded`는 별도로 보존해야 한다. `PredictionMetadata`에 없는 정책·학습 출처가 `integrate_inputs()`로 자동 전달되지는 않는다. 모델별 적격 모집단이 다르면 감사 정보를 유지하면서 사용할 모집단을 명시적으로 합의해야 한다.

## 5. Retention Strategy

`build_retention_strategy(segmentation, eligibility, coupons, redemptions, cost_inputs, cost_metadata, *, dataset_id, as_of_at, currency, amount_unit, history_context, history_policy, policies=None)`는 한 cohort를 평가한다. 원래 segment·점수·순위와 제외 고객을 보존한다.

- `SegmentPolicy`와 `default_segment_policies()`는 12개 segment를 지원한다. 기본 개발 정책은 엄격한 양수 Uplift 후보 중심이며 실제 사업 효과를 확정하지 않는다.
- `HistoryPolicy`의 미사용 쿠폰 차단, 기존 사용 고객 차단, 최소 재발송 간격은 호출자가 명시한다. `HistoryContext`는 dataset·raw SHA, 관측 종료 시각, 출처·버전·인지 시각을 선언한다. 이력이 없다는 것과 완전한 빈 이력이 확인됐다는 것은 다르다.
- eligibility는 `CONTEXT_COLUMNS`와 `business_eligible`, `business_reason`, `history_complete`를 사용한다. 이력이 불충분하면 설정에 따라 unknown/exclude/error로 처리한다.
- 현재 시뮬레이터 기본 적격 기간과 쿠폰 유효기간은 28일이다. 과거 Feature의 30일 집계 창과 구분한다. 이력의 쿠폰 만료·사용·중복·고객 연결을 검사하고 기준 시각 전에 관측 가능한 기록을 사용한다.
- `planned_sent_at`은 `decision_at + send_delay`인 계획 시각이다. 실제 `sent_at`이 아니다. 지연 발송은 적격성 재검증이 필요하다고 표시한다.

비용 입력은 후보별로 다음 식에 필요한 값을 명시한다.

```text
expected_cost = send_cost + redemption_probability * expected_discount_if_redeemed
```

`cost_inputs`에는 `CONTEXT_COLUMNS`, 위 세 값, `source_id`, `source_version`, `source_as_of_at`이 필요하다. 비용은 정확한 Decimal 연산으로 계산하며 임의의 사용률·평균 할인액·0 비용으로 대체하지 않는다. Uplift를 사용 확률로 변환하지 않는다. `CostMetadata(definition_id, version, decimal_places, as_of_at)`에 비용 정의·정밀도·버전을 선언한다.

`RetentionResult`의 `evaluation`과 `cost_estimates`는 감사·계산 결과다. Budget에는 `budget_inputs`, `costs`, `cost_metadata`를 전달한다. 정책 후보는 최종 발송 대상과 다르다.

`audit_observed_coupon_order(items, *, purchase_at, as_of_at, actual_discount_minor, actual_paid_minor)`는 과거 주문의 최고 단가 품목 한 unit에 10% 할인하는 회계를 검사한다. 동률이면 line 번호로 결정하고 minor 단위 반올림 규칙을 적용한다. `paid_minor`는 이미 할인 후 결제액이므로 할인액을 다시 차감하지 않는다. 실제 할인 감사와 미래 예상 비용은 별개다.

## 6. Budget Optimization와 Hybrid

`optimize_budget(segmentation, costs, budget, cost_metadata, *, dataset_id, as_of_at, config=None, algorithm="both", hybrid_config=None)`는 `Budget(amount, currency, amount_unit)` 안에서 이진 선정값을 구한다.

```text
maximize sum(priority_score[i] * selected[i])
subject to sum(expected_cost[i] * selected[i]) <= budget
selected[i] in {0, 1}
```

Budget 비용표의 필수 `COST_COLUMNS`는 `dataset_id`, `customer_id`, `as_of_at`, `snapshot_id`, `raw_database_sha256`, `currency`, `amount_unit`, `expected_cost`다. 단위·출처·snapshot·고객 집합을 검증하며 비용은 외부 명시값이다. 누락·중복·음수·비유한 비용과 음수 예산을 거절한다. 비용 정밀도는 최대 9자리다. Decimal → 정수 tick → 양수 활성 비용의 GCD 축소를 사용하고, 예산 우변은 정수 나눗셈으로 내린다. 이 변환은 이진 실행 가능 집합을 보존한다. 최종 비용은 원래 Decimal 값으로 재합산하고 예산 초과 해를 거절한다. 지원 계수 범위 밖의 입력도 거절한다.

| algorithm | 동작·해석 |
| --- | --- |
| `greedy` | 양수 점수·0 비용은 선정, 비양수 점수는 제외. 점수/비용 내림차순, 점수 내림차순, ID 오름차순. 예산에 안 들어가는 고객은 건너뛴다. 전역 최적성 보장 없음 |
| `milp` | SciPy `milp()` / HiGHS. 이진성·최종 Decimal 비용 검증 후 결과 반환 |
| `both` (기본) | 같은 준비 입력으로 Greedy를 먼저 보존한 뒤 MILP 실행·비교. 실패한 MILP를 Greedy로 위장하지 않음 |
| `hybrid` | Greedy → 정확한 인증 → 필요 시 제한된 Local Improvement → 재인증 → 명시된 선택적 backend → 최종 검증 |

`SolverConfig` 기본값은 `time_limit=10`, `mip_rel_gap=0`, `binary_tolerance=1e-7`, `allow_unproven=False`, `presolve=True`다. `time_limit`은 전체 Python API의 실행시간 상한이 아니다. Presolve 선택만으로 최적해나 속도를 보장하지 않는다.

`BudgetComparison.results`에는 알고리즘별 `AllocationResult(decisions, summary)`가 들어가며 `comparison`, `metadata`를 함께 제공한다. 유효해 여부, 선정 고객 수·점수 합계·예상 총비용·잔여 예산·예산 준수·실행시간·솔버 상태/gap을 확인한다. MILP의 `optimal_numeric`은 수치 솔버 검사 결과이고 `limit_feasible`, `gap_feasible`, `feasible_uncertified`, 해 없음·오류·무효해와 구분한다. 미확정 MILP는 기본적으로 계획 승인되지 않는다. `allow_unproven=True`는 명시적 정책 변경이다.

### 정확한 품질 인증

`prepare_quality_problem()`, `fractional_upper_bound()`, `certify_selection()`은 원래 비용과 실제 이진64 Priority Score를 정확한 Fraction으로 표현한다. 십진수로 반올림한 다른 목적함수로 바꾸지 않는다. L은 실행 가능한 선택의 점수, U는 Fractional Knapsack 완화의 유효 상한이다.

```text
L <= OPT <= U
(U - L) / U <= epsilon  =>  L >= (1 - epsilon) * OPT
```

U가 0인 경우는 코드의 별도 0 오차 처리에 따른다. 이 인증은 해당 입력의 대리 점수 품질 보장이며 실제 사업 성과 보장이나 모든 입력에 대한 시간 보장이 아니다. L=U일 때만 정확한 최적성 증명이 된다. 수치 솔버의 primal/dual bound·MIP gap과 이 정확한 인증을 혼동하지 않는다.

`HybridConfig()` 기본 epsilon은 `"0.0001"`(0.01%), Local Improvement는 pool 32·최대 2 pass·탐색 0.5초, `backend=None`이다. 개선은 제한된 1:1·1:2 교환이며 최적해를 보장하지 않는다. 결과 상태는 `proven_optimal`, `epsilon_certified`, `feasible_uncertified`, `unavailable`로 구분한다. `metadata["hybrid"]`에 인증, 선정 출처, 개선량, 시도·시간과 선택적 솔버 진단을 남긴다.

Hybrid의 `accepted_for_planning`은 인증 성공과 무관하게 **False**다. Greedy 기준선의 승인 필드나 MILP 수치 승인도 실제 쿠폰 발송 권한을 뜻하지 않는다.

선택적 `HybridConfig(backend="highspy")`는 인증 미달일 때만 추가 솔버를 사용한다. 현재 실행기는 Windows Job Object를 이용한다. 부모 프로세스가 검증된 최상 해를 보존하고, 자식의 순서 fingerprint·이진성·정확한 비용·점수를 다시 검사한다. 낮거나 동점인 해, timeout·crash·무효 응답은 기존 해를 대체하지 않는다. 초기해 제출과 실제 수락은 구분해 기록한다. 외부 timeout도 준비·인증·정리 전체를 포함한 API 상한은 아니다.

## 7. What-if와 사후 Budget Quality

`run_what_if(segmentation, costs, base_budget, cost_metadata, *, dataset_id, as_of_at, config=None, algorithm="both", scenarios=None, hybrid_config=None, backend_time_budget=None)`는 같은 입력 모집단·점수·비용을 고정하고 예산만 변경한다. 기본은 `budget_050`, `budget_100`, `budget_200`(50%·100%·200%)이며 각 시나리오는 독립적으로 배분·예산 검증·Hybrid 인증한다. 기존 정책 후보나 점수를 다시 계산하지 않는다.

`WhatIfResult`는 `summary`, `decisions`, `comparisons`, `metadata`, 시나리오별 `results`를 제공한다. 추가 backend 시간 예산은 전체 What-if API의 wall-clock 제한과 다르다. 인증 실패와 솔버 실패, 유효해와 계획 승인 여부를 별도로 해석한다.

`export_what_if(result, output_dir)`는 요약 CSV·고객별 CSV·메타데이터 JSON을 기록하며 기존 출력을 덮어쓰지 않는다. `plot_what_if(result)`는 Figure를 반환한다. 금액·정확한 분수·시각의 JSON 표현과 CSV의 ID 문자열을 읽는 소비자는 출력 스키마를 보존해야 한다.

`certify_budget_result(segmentation, costs, budget, cost_metadata, comparison, *, dataset_id, as_of_at, epsilon="0.0001")`는 기존 결과의 **Greedy** 선택을 원래 입력과 대조해 사후 인증한다. `certify_what_if_result(..., result, ...)`는 이를 시나리오별로 수행한다. Greedy 결과가 없는 MILP-only 또는 Hybrid-only 결과를 임의로 대체 인증하지 않는다. 원래 비교 결과나 승인 상태도 변경하지 않는다. Hybrid 자체의 인증은 해당 결과 메타데이터에서 읽는다.

## 8. 실행 가능한 합성 데이터 예시

저장소 루트에서 아래 Python 블록을 **순서대로 같은 프로세스**에서 실행한다. `PYTHONPATH`에 `simulator/src`가 필요하다. 모델 파일과 테스트 fixture를 import하지 않는다. 모든 예측·학습 출처·비용·완전한 빈 이력은 예제용 선언이다. 실제 Feature Store는 `load_feature_store(path)`로 읽어야 하며, 아래 메모리 자료는 manifest/Parquet 검증 사례가 아니다.

### 입력과 Adapter → Segmentation

```python
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from simulate.v4.features import EXPANDED_FEATURES
from final_strategy.inputs import PredictionBatch, integrate_inputs
from final_strategy.validation import CONTEXT_COLUMNS, PredictionMetadata
from final_strategy.retention_prediction_adapter import adapt_retention_predictions
from final_strategy.uplift_prediction_adapter import (
    UPLIFT_FEATURES, UpliftModelMetadata, adapt_uplift_predictions,
)
from final_strategy.segmentation import segment_customers, export_segments, plot_segments

t0 = pd.Timestamp("2024-03-03T15:00:00Z")
rows = []
for customer in ("001", "NA", "003", "004"):
    row = dict.fromkeys(EXPANDED_FEATURES, 0.0)
    row.update(customer_id=customer, dataset_id="readme-dummy", as_of_at=t0,
               reference_week="2024-03-04", snapshot_id=f"dummy-opaque::{customer}",
               currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
               raw_database_sha256="a" * 64, source_model_sha256="b" * 64,
               feature_set_version="kit32-v1",
               behavior_feature_set_version="behavior_review_cart_v3",
               expanded_feature_set_version="kit39-review-cart-v3")
    row["mean_net_order_value_30d"] = np.nan
    rows.append(row)
features = pd.DataFrame(rows)

predictions = features[["customer_id", "dataset_id", "snapshot_id", "as_of_at",
                        "raw_database_sha256", "currency", "amount_unit",
                        "expanded_feature_set_version"]].copy()
predictions = predictions.assign(
    churn_probability=[0.8, 0.2, 0.7, np.nan],
    status=["scored", "scored", "scored", "no_recent_session"],
    model_version="dummy-retention-v1", target="visit", horizon_days=28,
    definition_id="future_no_session_28d_active28d",
    policy_condition="observed_experiment_mix",
)
retention = adapt_retention_predictions(predictions, features)
assert retention.scored_customer_count + retention.unscored_customer_count == 4
assert retention.batch is not None

class DummyUpliftModel:
    def predict(self, X):
        # Three scored customers, in the exact input order; no trained model.
        p0 = np.array([0.2, 0.5, 0.4])
        p1 = np.array([0.4, 0.45, 0.4])
        return pd.DataFrame({"p0": p0, "p1": p1, "uplift_score": p1 - p0},
                            index=X.index.copy())

declaration = UpliftModelMetadata(
    model_version="dummy-uplift-v1", feature_names=UPLIFT_FEATURES,
    target_definition_id="dummy_purchase_28d_coupon_assignment",
    target="purchase_occurrence", horizon_days=28,
    treatment="coupon_assignment", control="no_coupon_assignment",
    policy_condition="dummy_10pct_coupon_assignment_vs_no_assignment",
    training_source_version="dummy-no-training", feature_set_version="kit39-review-cart-v3",
    currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
)
uplift = adapt_uplift_predictions(DummyUpliftModel(), retention.features, declaration,
                                  input_cutoff_at=t0)
clv = uplift.features[list(CONTEXT_COLUMNS)].copy()
clv["input_cutoff_at"] = t0
clv["prediction_end_at"] = t0 + pd.Timedelta(weeks=52)
clv["predicted_clv_12m"] = [100.0, 200.0, 50.0]
inputs = integrate_inputs(uplift.features, {
    "churn": retention.batch, "uplift": uplift.batch,
    "clv": PredictionBatch(clv, PredictionMetadata(
        "dummy-clv-v1", "dummy_future_paid_52w", 52, "weeks", "nonnegative_amount")),
})
segments = segment_customers(inputs)
audit = {"retention_excluded": retention.excluded, "uplift_excluded": uplift.excluded,
         "retention_producer": retention.producer_metadata,
         "uplift_producer": uplift.producer_metadata}
assert len(segments.customers) == 3 and len(audit["retention_excluded"]) == 1
assert segments.summary.customer_count.sum() == 3
```

### 명시적 후보·비용 → Budget·What-if

```python
from final_strategy.retention_strategy import (
    COUPON_COLUMNS, REDEMPTION_COLUMNS, HistoryContext, HistoryPolicy,
    build_retention_strategy,
)
from final_strategy.budget import Budget, CostMetadata, optimize_budget
from final_strategy.hybrid import HybridConfig
from final_strategy.what_if import run_what_if, export_what_if, plot_what_if
from final_strategy.budget_quality_integration import (
    certify_budget_result, certify_what_if_result,
)

eligible = segments.customers[list(CONTEXT_COLUMNS)].copy()
eligible = eligible.assign(business_eligible=True, business_reason=None, history_complete=True)
# Only this synthetic example declares a complete, empty coupon history.
coupons = pd.DataFrame(columns=COUPON_COLUMNS)
redemptions = pd.DataFrame(columns=REDEMPTION_COLUMNS)
declared_costs = segments.customers.loc[
    segments.customers.uplift_score.gt(0), list(CONTEXT_COLUMNS)
].copy()
declared_costs = declared_costs.assign(
    send_cost="2", redemption_probability="0.25", expected_discount_if_redeemed="10",
    source_id="dummy-cost", source_version="v1", source_as_of_at=t0,
)
cost_metadata = CostMetadata("dummy-send-plus-redemption", "v1", 3, t0)
strategy = build_retention_strategy(
    segments, eligible, coupons, redemptions, declared_costs, cost_metadata,
    dataset_id="readme-dummy", as_of_at=t0, currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
    history_context=HistoryContext("readme-dummy", "a" * 64, t0, "dummy-history", "v1", t0),
    history_policy=HistoryPolicy(False, False, pd.Timedelta(0)),
)
assert strategy.costs.expected_cost.tolist() == [Decimal("4.50")]
budget = Budget(Decimal("10"), "SOURCE_UNSPECIFIED", "0.01")
args = (strategy.budget_inputs, strategy.costs, budget, strategy.cost_metadata)
context = dict(dataset_id="readme-dummy", as_of_at=t0)
comparison = optimize_budget(*args, **context, algorithm="both")
hybrid = optimize_budget(*args, **context, algorithm="hybrid", hybrid_config=HybridConfig())
scenarios = run_what_if(*args, **context, algorithm="both")
certificate = certify_budget_result(*args, comparison, **context)
scenario_certificates = certify_what_if_result(*args, scenarios, **context)
for allocation in [*comparison.results.values(), *hybrid.results.values()]:
    assert allocation.summary["has_valid_solution"]
    assert allocation.summary["total_expected_cost"] <= budget.amount
assert not hybrid.results["hybrid"].summary["accepted_for_planning"]

output = Path(".venv/verification") / ("readme-example-" + uuid4().hex)
export_segments(segments, output / "segments_6plus.csv")
export_what_if(scenarios, output / "what_if")
plot_segments(segments, **context).savefig(output / "segments.png")
plot_what_if(scenarios).savefig(output / "what_if.png")
print(comparison.comparison)
print(scenarios.summary)
print("Synthetic outputs:", output.resolve())
```

이 예시는 scored 모집단만의 통합을 보여준다. 제외 고객의 최종 마케팅 정책을 확정하지 않으며, 감사 정보를 버리지 않는다. 실제 데이터에서는 비용 근거, 모델별 모집단, 정책 조건과 학습·추론 출처를 먼저 확보해야 한다.

## 9. 환경, 테스트 및 검증 범위

[simulator/pyproject.toml](../simulator/pyproject.toml)은 Python >=3.10, NumPy·pandas·SciPy·PyArrow 등을 선언한다. 테스트에 pytest, 시각화에 Matplotlib이 필요하다. 기존 가상환경을 우선 사용한다.

Budget은 `scipy.optimize.milp`를 최상위 import하므로 **SciPy >=1.9**가 필요하다. 시뮬레이터의 선언 하한 `scipy>=1.7.3`만으로는 충분하지 않다. Greedy만 호출해도 Budget 모듈 import에 이 조건이 적용된다. 실제 확인 환경은 Python 3.12.10, SciPy 1.18.1이다.

[requirements-highspy.txt](requirements-highspy.txt)의 `highspy==1.16.0`은 선택적 Windows backend용이다. 기본 Greedy·SciPy MILP·backend 없는 Hybrid에는 설치할 필요가 없다. highspy 자체의 지원 OS와 이 모듈의 Windows 전용 외부 실행기를 구분한다. Linux 등에서 기본 코드 import와 선택적 backend 실행 지원은 별개이며 현재 검증 기록은 Windows 기준이다.

저장소 루트의 PowerShell에서 실행한다. 고유한 pytest 임시 경로와 Matplotlib 설정 경로를 사용하고 로그·JUnit·durations를 남긴다. publisher 테스트를 제외하지 않는다.

```powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$env:PYTHONPATH = (Resolve-Path 'simulator/src').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$run = Join-Path (Resolve-Path '.venv').Path ('verification/final-strategy-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
$env:MPLCONFIGDIR = Join-Path $run 'mplconfig'
$baseTemp = Join-Path $run 'pytest-temp'
$junit = Join-Path $run 'results.xml'
& $python -B -m pytest -c simulator/pyproject.toml final_strategy/tests `
    --basetemp="$baseTemp" -q -ra -p no:cacheprovider `
    --durations=20 --junitxml="$junit" 2>&1 |
    Tee-Object -FilePath (Join-Path $run 'pytest.log')
if ($LASTEXITCODE -ne 0) { throw "pytest failed: $LASTEXITCODE" }
```

테스트 대상에는 입력·Segmentation·두 Adapter·Retention Strategy·Budget·Hybrid·품질 인증·What-if·외부 프로세스 검증이 포함된다.

직전 공개 준비의 독립 스냅샷 검증(2026-11-19, Windows, 9차 Commit 대상 코드)은 **850 collected / 845 passed / 5 skipped / 0 failed / 0 errors**, 143.78초였다. publisher 7개는 통과했다. 5 skip은 기본 환경에 highspy가 없어서 건너뛴 선택적 실제 솔버 테스트이며 highspy 실행 성공 기록이 아니다. 31,016 warnings에는 NumPy Timedelta 변환 관련 DeprecationWarning과 pandas dtype 관련 FutureWarning의 반복이 포함된다. 경고를 숨겨서 해결된 것으로 취급하지 않는다.

위 수치는 기존 실행 기록이며 README를 편집할 때마다 전체 회귀를 재실행했다는 뜻이 아니다. 독립 fixture, 실제 시뮬레이터 publisher, 모의 모델과 솔버 경계 검증을 포함하지만 실제 Retention·Uplift·CLV 학습 모델의 end-to-end 실행은 아니다. 과거 대규모 실험 시간은 호출 경로·설정이 달라 현재 API의 성능 보장으로 기재하지 않는다.

향후 실제 데이터 연동에는 Retention 학습 bundle과 실행 상태, Uplift bundle 및 36개 Feature의 수치 호환성, CLV 출력 Adapter와 52주·출처·회계 계약, 동일한 v5 raw/Feature Store가 필요하다. 세 모델을 연결한 최종 실행, 사업 적격 조건·비용 근거·ROI 정의, 팀 main 통합 후 전체 회귀는 별도로 검증한다.
