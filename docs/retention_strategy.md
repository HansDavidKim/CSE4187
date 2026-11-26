# 고객군별 리텐션 전략

동일한 10% 쿠폰을 기준시점에 한 번 요청하고 28일 동안 유효하게 유지하는 조건에서, 어떤 고객을 발송 후보에 포함할지 비교한다. 아래 여섯 전략은 사람이 제안한 가설이다. 임계값과 출처를 호출자가 제공하며, 실제 고객에게 효과가 있는지는 독립 평가 대상이다.

## 고객 분류와 후보 선정

각 고객의 세그먼트 ID는 `위험__uplift__가치`이다. 공통 `SegmentationConfig`는 개발 기준인 이탈 확률 0.5 이상을 `high`, 미만을 `low`로 둔다. Uplift 경계는 0이다. 따라서 `negative`는 음수, `neutral_low`는 0, `positive`는 양수다. 고객 가치는 동일 데이터셋·기준시각의 분석 모집단에서 52주 CLV 상위 20%를 올림한 인원까지 `high_value`, 나머지를 `standard_value`로 분류한다. CLV 동점은 고객 ID 오름차순으로 처리한다. 사업 적격성과 예산으로 모집단이 좁아져도 가치 등급을 다시 매기지 않는다.

아래 각 행은 Uplift 등급 3개에 대응하므로 모두 12개 세그먼트를 포함한다. 각 칸은 해당 위험·가치 조합의 `min_uplift`이다. 모든 행에서 **구매 Uplift가 임계값을 엄격히 초과**해야 한다. 따라서 음수와 0인 8개 조합은 모두 미발송이고, 양수인 4개 조합은 표에 따라 후보가 된다. 작은 양수도 적용 기준을 넘으면 후보가 될 수 있다. 양수 분류 경계를 변경하면 `neutral_low`의 의미가 달라지므로 이 비교에서는 다른 분류 설정을 거절한다.

| 위험 | 가치 | H0 / P0 | H1 | H2 | H3 | H4-R | H4-V |
|---|---|---:|---|---|---|---|---|
| high | high_value | 0 | t | r_high | v_high | a | a |
| high | standard_value | 0 | t | r_high | v_standard | b | c |
| low | high_value | 0 | t | r_low | v_high | c | b |
| low | standard_value | 0 | t | r_low | v_standard | d | d |

- **H0**는 적격 양의 Uplift 전체를 검토하는 기준선이다. 기존 P0의 12행 규칙과 설정 지문을 그대로 사용한다.
- **H1**은 `0 < t ≤ 1`인 공통 기준으로 작은 구매 증분 효과를 제외하는 가설이다.
- **H2**는 `0 ≤ r_high < r_low ≤ 1`로 고위험군의 진입 기준을 낮춘다. 같은 위험군에서는 CLV 등급에 같은 기준을 쓴다. 목표인 재방문은 실제 28일 무세션율로 평가해야 한다.
- **H3**는 `0 ≤ v_high < v_standard ≤ 1`로 고가치군의 진입 기준을 낮춘다. 예산 점수에도 연속형 CLV가 들어가므로 이 추가 우대가 도움이 되는지를 검증한다.
- **H4-R / H4-V**는 공통으로 `0 ≤ a ≤ b < c ≤ d ≤ 1`인 네 수준을 입력받는다. H4-R은 고위험·일반가치에 낮은 중간값 b를, H4-V는 저위험·고가치에 b를 준다. 양 끝 그룹과 임계값 수준이 같으므로 중간 두 집단의 우대를 직접 비교할 수 있다.

## 선언과 실행

`final_strategy.strategy_profiles.build_strategy_profiles()`의 반환형은 기존 `CandidateSet`이다. 아래 변수는 호출자가 사전에 정한 임계값과 그 근거다. 수치 기본값은 없다.

```python
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.policy_search import run_policy_search

profiles = build_strategy_profiles(
    version=policy_version, threshold_source=threshold_source,
    common=t, risk_high=r_high, risk_low=r_low,
    high_value=v_high, standard_value=v_standard,
    h4_r={"a": a, "b": b, "c": c, "d": d},
)
plans = run_policy_search(candidates=profiles.policies, **shared_inputs)
```

`shared_inputs`에는 같은 고객 명부와 snapshot, 모델 입력, 모집단 적격성, 사업 적격성, 쿠폰 이력과 관측 범위, 공통 비용표, 예산·통화·단위, 솔버 설정을 준다. 이력 정책에는 비교 계약의 28일 유효기간을 지정한다. 후보 선언은 기준시점 즉시 발송 계획을 사용한다. 실제 쿠폰 조건과 단일 요청은 실험 로그에서 별도로 검증한다. `profiles.to_dict()`는 공통 분류 설정 및 지문별 각 선언의 ID·버전·가설·근거·우대 방향·임계값·출처를 제공한다.

`eligible_for_coupon`은 사업·이력 조건의 결과다. 규칙까지 통과한 `budget_candidate`에 대해 기존 Budget 엔진이 `selected`를 계산한다. 정책별 예산 예약은 없다. 유효해 여부와 `accepted_for_planning`을 보존하고 발송 실행은 별도 단계로 둔다. 고객별 예상 비용은 명시된 `send_cost + redemption_probability × expected_discount_if_redeemed`다. 사용 확률은 별도로 공급한다. 구매 증분 확률에서 쿠폰 사용 확률을 유도할 수 없다.

목적함수는 `maximize Σ(uplift_score × predicted_clv_12m × action)`이다. 기존 이름의 CLV 열은 52주 할인 후 결제액 예측이다. 이탈 확률은 위험군 분류에 쓰이며 목적함수 가중치에는 들어가지 않는다. H1~H4의 후보는 H0 후보의 부분집합이어서 비용·점수·예산이 같으면 가능한 선택 조합도 H0의 부분집합이다. 따라서 **정확 최적화한 대리 점수의 최댓값은 H0를 초과할 수 없다.** Greedy나 미확정 솔버 해의 실측 점수에는 이 순서를 요구할 수 없다. 이 점수로 실현 이익이나 ROI를 계산할 수 없다. 기존 혼합 학습 CLV 번들의 정책 조건과 4주 조건부 모델 미학습 상태는 그대로다.

`profiles.report`와 `profiles.to_dict()`는 조회할 때마다 독립된 사본을 반환한다. 사본의 임계값이나 근거를 편집해도 이후 평가에 사용하는 선언은 유지된다. 다른 기준을 평가하려면 변경된 입력으로 프로필을 다시 생성한다.

## 검증과 정책 선택

1. **임계값 근거를 확보한다.** 사람이 정한 값, discovery 분위수, 선택적인 Phase C XGBoost 잔차 분할값의 출처를 기록한다. 변경한 각 위험·가치군에서 임계값 양쪽의 Treatment/Control 고객 수와 구매 사건 수를 discovery와 confirmation에서 확인한다. 기존 P1~P4 생성기는 한 번에 한 대상 그룹의 기준을 바꾼다. H2~H4의 여러 그룹 간 관계는 위 선언에서 고정한다.
2. **고정 로그에서 후보를 비교한다.** 소수 후보에는 기존 `search_fixed_log(method="exhaustive")`를 사용한다. 설정 지문이 같은 H0와 P0는 P0의 별칭으로 보존된다. 한 코호트에서 선정 고객·발송 시각이 같아도 설정이 다른 정책은 다른 코호트의 후보로 유지한다. `evaluate_policy_value()`와 `compare_policy_values()`로 H1·H2·H3·H4 각각과 H0, H2와 H3, H4-R과 H4-V의 차이를 동일 로그의 전체 ITT 고객 기준으로 계산한다. 비교 방향은 왼쪽에서 오른쪽을 뺀 값이다. 유효하지 않은 계획, 미승인 계획, 불완전 관측의 보류 사유를 지표와 함께 확인한다.
3. **평가 지표를 사전에 합의한다.** 제안 주지표는 28일 무세션율, 주요 보조지표는 28일 구매율이다. 고객당 할인 후 결제액, 발송비·쿠폰 할인액·총 마케팅 비용도 비교한다. 예상 예산 준수를 필수 제약으로 두며 실제 지출은 원장의 금액으로 평가한다. 허용 가능한 성과 저하폭, 통계적 채택 기준, 증분 가치·ROI의 회계 정의는 최종 분석 전 팀 합의가 필요하다.
4. **최종 실험 모집단을 분리한다.** 학습·임계값 발견·확인·후보 선택 고객과 별도의 고객으로, 정책을 동결한 뒤 무작위 전략 배정과 실제 실행, 28일 관측을 검증한다. 고정 로그 IPS는 탐색용 추정이다. 다수 후보의 독립 관측 비용이 클 때 Bandit을 고려할 수 있지만, 현재 독립 관측 경로는 `producer_verification_unavailable`로 보류된다. 기존 Bandit 보상은 구매율이며 정책 평가 결과의 `dispatch_authorized`는 false다. 구매율만으로 정책을 자동 채택하지 않는다. 이 문서의 지표 제안은 최종 A/B 실험 명세와 별도로 관리한다.
