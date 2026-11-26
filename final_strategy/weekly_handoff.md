# 모델 예측을 연결하는 쿠폰 전략 출력

이 모듈은 기존 Retention·CLV와 외부 Uplift 예측을 같은 고객·같은 기준일에 맞춰 연결한다.
새 예측 모델이나 생성기의 잠재 반응 유형을 만들지 않는다. 필수 Churn/Uplift/CLV batch 자체가
없으면 입력 오류로 거절한다. batch가 제공한 미관측·산출 불가 값은 0으로 채우지 않고
`availability.csv`에 이유를 남기며 해당 고객에게는 발송하지 않는다.

## 입력과 시간 계약

`load_feature_store(features_dir)`는 기존 체크섬·원장과 공통 `time_spec`의 4/8/13주 정의를 검증한다.
달력월·30/60/90일 자료를 이 주간 계약으로 조용히 해석하지 않는다. `PredictionBatch`는 frame,
`PredictionMetadata`와 선택적인 `producer_metadata`를 받는다. 모델 frame은 대상 고객을
정확히 한 번씩 덮어야 하며 `(dataset_id, customer_id, as_of_at)`과 원장 체크섬·통화·단위를
바꿀 수 없다. 서울 월요일 00:00 기준이며 `input_cutoff_at == as_of_at`이다.

| 역할 | 허용 열 | 정답 의미·예측 종료 |
|---|---|---|
| churn | `churn_prob` 또는 `churn_probability` | 미래 4주 무방문 확률, `t0 + 4주` |
| uplift | `p0/p1` 또는 `purchase_probability_no_coupon/coupon` | 미발송/발송 시 4주 내 구매 확률, `t0 + 4주` |
| clv | `clv` 또는 `predicted_clv_12m` | 미래 52주 할인 후 실결제액, `t0 + 52주` |
| gross_4w | `predicted_gross_4w_if_purchase` | 4주 내 구매 조건부 할인 전 총매출, `t0 + 4주` |

`uplift_score`는 항상 `p1 - p0`이다. Retention의 이탈 확률을 구매 `p0/p1`로 바꾸지 않는다.
52주 CLV를 4주 매출 대신 쓰지 않는다. CLV·Retention이 제공하는 전략 handoff를 그대로
`PredictionBatch(frame, metadata, producer_metadata)`로 감쌀 수 있다. 모델 버전·대상 정의·기간을
각 batch에 선언한다. 관측 104주는 과거 52주·미래 52주, 최소 이력 13주, 적격·쿠폰 유효·Uplift
결과는 4주이며 CLV 입력은 기준일 직전 13주다. 관측 누락을 0매출로 해석하지 않는다.
CLV의 `policy_condition`이 선언되어 있으면 `control_no_coupon`만 허용한다. 오래된 5항목
handoff에 그 선언이 없으면 호환은 유지하되 무발송 조건이 검증되었다고 주장하지 않는다.

## 10개 고객 출력과 6개 업무 세그먼트

`customers.csv`의 열은 다음 10개로 고정한다. 출처 키와 상태는 별도 파일에 보존한다.

`customer_id, churn_prob, p0, p1, uplift_score, clv, segment, priority_score, send, expected_net_value`

main에는 전략 실행 코드가 없고 `docs/request.md`의 6개 분류 예시만 있다. 따라서 이 구현은
아래 완전한 업무 규칙을 명시하며, 경계와 발송 허용 그룹은 `DecisionConfig`로 반드시 입력한다.
데이터에 6개 유형이 모두 나타나도록 강제하지 않는다. Tianchi의 6개 행동 페르소나와도 다르다.

| segment | 규칙 |
|---|---|
| newly_observed | 관측 이력이 `new_history_days`보다 짧음. 실제 가입일·잠재 lifecycle을 추측하지 않음 |
| high_value_persuadable | 신규가 아니며 양의 구매 Uplift·설정한 이탈 위험 이상·고가치 |
| low_value_persuadable | 위 조건의 저가치 |
| high_value_sure_thing | 신규가 아니며 구매 Uplift가 0 근처·미발송 구매 확률 높음·고가치 |
| high_value_no_action | 나머지 고가치 고객 |
| low_value_no_action | 나머지 저가치 고객 |

Uplift 4유형은 양의 효과 Persuadables, 음의 효과 Sleeping Dogs, 0 근처에서 높은/낮은 `p0`를
Sure Things/Lost Causes로 구별한다. 이탈 위험은 별도의 업무 대상 필터로만 사용한다.
Sleeping Dogs는 다른 분류·예산과 관계없이 발송하지 않는다. 이 네 유형은 모델 예측에 대한
업무적 분류이며 개인의 반사실적 반응 유형을 식별했다는 뜻이 아니다.

## 비용·우선순위·예산

통화와 최소 금액 단위는 명시적으로 일치시킨다. `SOURCE_UNSPECIFIED/0.01`을 KRW로
바꾸지 않는다. 할인 기대비용은 발송비와 별개인 **발송 1회당 무조건부 기대 할인액**이다.
설정의 `expected_discount_per_send`로 시나리오 가정을 주거나, 같은 고객·기준일·4주 종료의
`expected_discount_if_sent` 예측표를 준다. 소매 쿠폰은 최고 단가 품목 1개에 10%를 적용하므로,
4주 총매출의 10%를 할인비로 임의 대입하지 않는다.

```
expected_incremental_revenue = (p1 - p0) * predicted_gross_4w_if_purchase
expected_coupon_cost = send_cost + expected_discount_if_sent
expected_net_value = expected_incremental_revenue - expected_coupon_cost
priority_score = max(p1 - p0, 0) * clv
ROI = (선택 고객의 예상 증분 매출 - 예상 쿠폰 비용) / 예상 쿠폰 비용
```

위 매출식은 쿠폰 유무의 구매 조건부 총매출이 같다는 계획 가정이다. 관측된 실제 이익이나
확인된 쿠폰 인과효과가 아니다. `expected_net_value`는 **발송한다면**의 값이며, 합계는 `send=1`
고객만 더한다. 발송 허용 그룹 중 양의 순가치 고객을 우선순위 순서로 예산 내 선택하는 그리디
근사다. 전역 최적해를 주장하지 않는다. 50%·100%·200% 예산에서도 같은 고정 예측을 사용한다.
비용 0의 ROI는 무한대로 만들지 않고 unavailable이다. 현금이 미관측인 Tianchi의 금전 ROI도
unavailable이며 구매일수 proxy를 돈으로 바꾸지 않는다.

## 실제 전략 평가와 그래프

실제 `strategy_group`과 4주 관측 결과를 `strategy_evaluation`에 넣으면 군별 고객 수·전환율·이탈률·
고객당 순매출·비용·ROI·A 대비 차이를 반환한다. 통계 추론에는 아래 선언을 검증해야 한다.

- 무작위 군 배정, 모델·전략 사전 동결 시각, 독립 평가용 cohort
- 모델 개발 고객 `(dataset_id, customer_id)` 목록과 평가 고객의 비중첩
- 배정 시각과 같은 t0, 완전히 관측한 동일 4주 결과
- 사전에 정한 A 기준 군·비교 군 목록·주 지표

평가 metadata에는 `strategy_randomized_assignment=true`로 고객 단위 전략군의 무작위 배정을
별도로 선언하고 `strategy_groups`와 `primary_metric`도 명시한다. 쿠폰 자체가 무작위 배정되었다는
선언만으로 사후 선택한 전략군을 RCT로 취급하지 않는다. 목록이나 주 지표 선언이 빠져도
추론을 제공하지 않는다. 기준 군은 A로 고정한다.

이 조건을 만족하지 않으면 관측 요약만 제공하고 인과 ROI·차이의 신뢰구간·p값을 만들지 않는다.
이항 지표는 큰 표본에서 기존 ITT의 두 비율 Z 검정을 사용하고 작은 기대빈도는 Fisher 정확 검정으로
처리한다. 95% 차이 구간은 Wilson 구간의 Newcombe 결합, 금액 지표는 Welch 구간·검정이다.
미리 선언한 모든 비-A 군의 주 지표 비교를 한 family로 두고 Holm 보정한다. 관측치가 없는 군도
비교 family에서 빼서 보정을 완화하지 않는다. ROI는 A와의 평균 `(gross-discount-send)` 차이를
평균 `(discount+send)` 비용 차이로 나누며 양의 증분 비용이 없으면 계산하지 않는다.
실제 금액 입력은 할인 전 `gross_revenue_minor`, 할인액 `discount_minor`, 발송비
`send_cost_minor`를 별도로 요구한다. main과 동일하게 `paid_revenue_minor`/`paid_minor`/
`net_revenue_minor`를 함께 주면 `gross-discount`, `net_after_marketing_minor`를 주면
`gross-discount-send`와 일치해야 한다. 전략 요약의 `net_revenue_per_customer`는 마케팅비까지
차감한 고객당 순가치다.
실결제액·순매출을 gross로 alias하지 않는다. 예를 들어 A의 gross 80·비용 0과 B의 gross 100·
paid 90·할인 10·발송비 1이면 증분 gross 20, 비용 11, 증분 순가치 9, ROI는 `9/11`이다.

Qini는 독립 평가의 실제 4주 구매·무작위 배정·알려진 propensity의 IPW 누적 구매 증가로 그린다.
모델별 곡선과 `fraction * 전체 IPW 효과`의 Random targeting 기준선을 함께 준다. 이것은 RL의
무작위 발송 정책과 다른 평가 기준선이다. 동점 score block 말미만 곡선에 넣으며 수치 동점은
소수 12자리로 구별한다. 상수 점수 모델의 Qini 면적은 0이며 입력 행 순서로 성능을 만들지 않는다.

`write_outputs`는 아래 6개 그래프를 PNG/PDF 및 CSV 데이터로 함께 저장한다. 필요한 실제 자료가
없으면 수치 0의 가짜 곡선 대신 unavailable 이유를 표시한다.

1. 이탈 위험 분포
2. Uplift 4유형 산점도(x=p0, y=p1-p0)
3. 6개 업무 세그먼트(x=Uplift, y=52주 CLV, 색=이탈 위험)
4. 모델별 Qini·Random targeting 기준선
5. 예산 50%·100%·200% ROI
6. 실제 전략군별 주 지표와 95% 구간

## 실행

### 팀 브랜치의 human profile 연결

`origin/feat/final_strategy`의 `11a1fcd`가 정의한
`strategy_profiles.build_strategy_profiles`와 선언 dataclass·구성 지문을 재사용한다.
`profiles.policies`에는 P0/H0/H1/H2/H3/H4-R/H4-V가 들어가며 P0와 H0는 행동 지문이 같다.
임계값·버전·출처는 호출자가 명시해야 한다. 검증된 효과나 자동 발송 권한을 뜻하지 않는다.

| profile | 엄격한 구매 Uplift 진입 기준 |
|---|---|
| P0, H0 | 0 초과 |
| H1 | 공통 기준 초과 |
| H2 | 고위험은 `risk_high`, 저위험은 `risk_low` 초과 |
| H3 | 고가치는 `high_value`, 나머지는 `standard_value` 초과 |
| H4-R | 고위험/고가치, 고위험/일반, 저위험/고가치, 저위험/일반에 a/b/c/d |
| H4-V | 같은 순서에 a/c/b/d |

팀 선언의 내부 분류는 이탈확률0.5 이상과 CLV 상위 `ceil(20%×고객 수)`를 사용한다.
동점 CLV는 고객 ID로 정렬한다. 완전한 예측이 있는 원래 cohort에서 한 번 계산하며
profile별 후보·예산에 따라 순위를 다시 매기지 않는다. 이12개 내부 분류는 최종6개 업무
세그먼트와 별개다. `build_decisions(..., profile=profile)`은 기존 업무 대상·금액·예산 조건에
해당 profile의 gate를 추가한다. 따라서 효과 비교는 같은 입력·비용·설정으로 해야 한다.

```python
from final_strategy import build_strategy_profiles, build_decisions
profiles = build_strategy_profiles(**declared_profile_thresholds)
profile = next(p for p in profiles.policies if p.policy_id == "H4-R")
result = build_decisions(features, predictions, config, profile=profile)
```

CLI에서는 `--profile-declarations profiles.json --profile H4-R`을 함께 지정한다.
JSON 키는 `version`, `threshold_source`, `common`, `risk_high`, `risk_low`, `high_value`,
`standard_value`, `h4_r`이며 마지막 값은 a/b/c/d 객체다. 선언·지문은 metadata에,
고객별 내부 분류·gate는 `graph_data/human_profile.csv`에 저장한다.

기존 `PredictionBatch(frame, metadata, producer_metadata)`와 DataFrame을 반환하는
`integrate_inputs` 계약은 유지한다. 원격의 score 단독 `IntegratedInputs`·solver/search API는
이 어댑터의 입력이 아니다. 예측 기간·출처가 검증된 구매 p0/p1을 제공해야 하며,
이탈확률이나 uplift score만으로 구매확률을 만들지 않는다. 팀 선언에 필요한 타입만
`profile_contract`에 두며, 새 학습기·solver를 실행하지 않는다.

```python
from final_strategy import PredictionBatch, DecisionConfig, load_feature_store, build_decisions, write_outputs

features = load_feature_store("DATA/features")
predictions = {
    "churn": PredictionBatch(retention.frame, retention.metadata, retention.producer_metadata),
    "clv": PredictionBatch(clv.frame, clv.metadata, clv.producer_metadata),
    "uplift": PredictionBatch(purchase.frame, purchase.metadata, purchase.producer_metadata),
    "gross_4w": PredictionBatch(clv.gross_4w, clv.gross_4w_metadata, clv.producer_metadata),
}
# 경계값·발송비·기대 할인액은 사업에서 사전에 정한 명시적 설정이다.
config = DecisionConfig(**CONFIG)
result = build_decisions(features, predictions, config)
write_outputs(result, "NEW_OUTPUT", qini=held_out_qini, strategy=held_out_strategy)
```

없는 `gross_4w` batch는 dict에서 빼면 된다. 예산 계산은 unavailable이고 다른 예측은 유지된다.
CLI는 기존 모델 파일을 새로 학습하지 않고 파일 handoff를 결합한다.

```sh
PYTHONPATH=simulator/src:. python -m final_strategy --features DATA/features \
  --predictions prediction_batches.json --config decision_config.json --output NEW_OUTPUT
PYTHONPATH=simulator/src:. python -m pytest -q final_strategy/tests
```

`prediction_batches.json`은 역할별 `frame`(CSV/Parquet 경로), `metadata` 5항목,
선택적 `producer_metadata`를 담는다. 최종 10열과 별개로 `customer_identity.parquet`,
`availability.csv`, `segment_summary.csv`, `budget_summary.json`, `budget_what_if.csv`,
`graph_data/`, `graphs/`, 체크섬 `manifest.json`을 저장한다. 기존 출력은 덮어쓰지 않는다.
다중 모델 Qini는 `--qini-outcomes factual.parquet --qini-predictions qini_batches.json`으로
지정하며 JSON 키를 역할 대신 모델명으로 둔다. 전략 결과는 `--strategy-outcomes`로 지정하고
전략 평가의 최초 배정표는 CLI의 `--strategy-assignments`, Python의
`strategy_evaluation(..., assignments=...)`로 전달한다. 고객·데이터셋·기준시각,
snapshot·원장 SHA, 배정 시각·전략군과 원래 배정 확률이 필요하다. 비교에 쓰는 결과는
`outcome_observed=True`이고 4주 관측이 완료되어야 한다. 최초 명부가 없거나 검열된
경우 A 대비 추론을 보류한다. 실현 비용은 실제 발송 상태와 정수 minor 금액으로 검증한다.
두 평가의 무작위 배정·동결·독립 표본 선언은 `--evaluation-metadata` JSON으로 제공한다.

`decision_config.json`의 작은 계약 검증 예시는 아래와 같다. 금액은 위에 선언한 단위이며
이 숫자를 실무 임계값·추정 할인 비용으로 채택한다는 의미는 아니다.

```json
{
  "budget": 60, "send_cost": 10, "expected_discount_per_send": 20,
  "currency": "SOURCE_UNSPECIFIED", "amount_unit": "0.01",
  "high_value_threshold": 100, "churn_threshold": 0.5,
  "uplift_tolerance": 0.05, "baseline_purchase_threshold": 0.5,
  "new_history_days": 91,
  "eligible_segments": ["newly_observed", "high_value_persuadable", "low_value_persuadable"]
}
```
