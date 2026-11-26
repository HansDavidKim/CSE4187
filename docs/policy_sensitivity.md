# 전략 민감도와 선정 안정성

`final_strategy.policy_sensitivity.run_policy_sensitivity()`는 동일한 최초 고객 명부에서 기준 계획과 지정한 변경 계획을 비교한다. 각 조건은 기준 입력에서 독립적으로 시작한다. 고객별 예측과 메타데이터, 적격성, 이력, 평가 로그와 관측 기간은 한 번만 공급한다. 아래 예시는 검증된 입력을 준비한 뒤 사용하는 호출 형태다.

```python
from final_strategy.policy_sensitivity import SensitivityScenario, run_policy_sensitivity

# shared는 run_policy_search의 공통 입력이다. 비용표와 CostMetadata는 estimate에서 읽는다.
shared = {k: v for k, v in search_inputs.items() if k not in {"cost_inputs", "cost_metadata"}}
result = run_policy_sensitivity(
    search_inputs=shared,
    cost_estimate=estimate,             # estimate_coupon_costs() 결과
    profile_settings=profile_settings,  # build_strategy_profiles()의 인수
    scenarios=[
        SensitivityScenario("half", "budget", "budget_050", "예산 절반 가정"),
        SensitivityScenario("reference", "budget", "budget_100", "기준 예산 재표시"),
        SensitivityScenario("double", "budget", "budget_200", "예산 두 배 가정"),
        SensitivityScenario("fee", "send_cost", "1.2", "발송비 20% 상승 가정"),
        SensitivityScenario("cut", "common", proposed_cut, "사전 선언한 H1 후보"),
        SensitivityScenario("solver", "algorithm", "hybrid", "동일 예산에서 해 비교"),
    ],
    max_scenarios=7,  # 기준 포함
    pareto_directions={"purchase_rate_28d": "max"},
    experiment_log=log, evidence=evidence,
)
report = result.to_dict()
change = report["differences"]["half"]["H0"]
```

| 변경 필드 | 입력값과 실행 규칙 |
| --- | --- |
| `budget` | `budget_050/100/200`. 하나라도 요청하면 기존 `build_budget_scenarios()`가 세 금액의 정확한 tick 표현을 검사한다. 100%는 원래 기준 결과를 참조하며 `reused_from`에 기준 ID를 기록한다. |
| `send_cost`, `redemption_probability`, `expected_discount_if_redeemed` | 입력값은 원본의 해당 성분에 적용하는 비음수 배율이다. 보고서의 before=1과 after는 배율을 뜻한다. 고객별 원래 값은 `original_cost_inputs`에 보존한다. 변경 후 확률 상한과 예상 비용의 tick 정밀도를 검사한다. |
| `monetary_components` | 발송비와 사용 조건부 할인액에 같은 배율을 적용한다. 사용 확률은 고정한다. |
| `common`, `risk_high/low`, `high_value/standard_value` | 필드 하나의 절대 임계값을 바꾸고 H 프로필 전체를 재구성한다. 위험군·가치군 우대 관계는 기존 검증을 따른다. |
| `h4_r` | a,b,c,d 전체를 전달하는 복합 변경이다. 기존 생성기가 같은 값으로 H4-V의 중간 두 임계값을 교환한다. |
| `algorithm` | `greedy`, `milp`, `hybrid`. 공통 SolverConfig를 사용한다. 적용된 설정과 품질 근거는 시나리오의 hybrid_config와 comparison.plans에 있다. |

시나리오 ID, 예산 종류, 알고리즘, 비용 배율, 임계값과 전략별 우대 관계는 비용 추정 상태를 확인하기 전에 검사한다. 선언이 유효하고 비용 추정이 보류됐을 때만 결과가 withheld이며 비교 호출은 0회다. 고객별 배율 적용 후 사용 확률과 예상 비용의 tick 정밀도는 완료된 비용표에서 검사한다. 비용 시나리오는 과거 자료로 만든 추정치에 적용하는 사전 가정이다. 평가 기간의 관측값은 IPS 경로에서만 사용한다. 공식 프로필의 이탈 분류 기준과 CLV 상위 20%를 유지한다.

민감도 API는 독립 Decimal 문맥에서 예산 생성부터 정책 비교·IPS·차이 집계까지 실행한다. 유효 계획의 금액은 최대 `10^18`, 최소 tick은 `10^-9`이므로 잔여 예산에는 최대 28자리면 충분하다. 비용 성분의 곱셈은 별도의 고정밀 검사, 차이와 비율은 Fraction 연산을 사용한다. 유효 계획마다 선정 총비용과 잔여 예산의 합을 정확히 대조한다. 정상 종료와 예외 모두 호출자의 Decimal 설정을 보존한다.

## 결과 읽기

각 시나리오의 `comparison`은 기존 `compare_policies(pairs=[])`의 전체 보고서다. audit에는 공통 입력 지문, 비용 정의, 모델 및 평가 로그 근거가 있고 plans에는 선언별 정책 설정·결정 지문과 대표 정책 ID가 있다. 시나리오별 시간은 비교와 차이 집계에 걸린 벽시계 시간이다. 전체 시간은 기준 입력 준비도 포함한다. 정책별 솔버 단독 시간은 제공되지 않는다. 예시의 비교 호출은 기준을 포함해 6회이며, 비교 안에서는 고유 설정별로 실행한다.

`differences[scenario_id][policy_id]`에서 모든 변화량의 방향은 변경값 − 기준값이다. 상대 변화율은 기준값으로 나누며 기준값 0이나 미확정 값은 null이다. 선정·후보 Jaccard는 두 집합의 교집합 수 / 합집합 수이고, 유효한 두 빈 집합은 1이다. added는 새로 들어온 수, removed는 빠진 수다. 무효인 선정 결과와 비교하면 선정 차이는 null이다. 세그먼트 선정률의 분모는 해당 세그먼트 전체 고객 수다. 정책 ID·버전을 대응시킨 뒤 고객 키별 출처와 실제 분류를 대조한다.

예상 지출은 발송비 + 사용 확률 × 사용 조건부 할인액의 선정 고객 합계다. IPS의 마케팅 비용은 같은 로그의 실현 비용에 대한 고객당 점추정치다. 행동이 같으면 계획 비용이 달라져도 동일한 평가 조건에서 이 IPS 값은 같다. IPS 6개 지표의 ESS, 추정 가능 여부, 보류 사유와 시나리오 내부 Pareto 상태를 함께 확인한다. Decimal은 문자열, 정확한 비율과 차이는 분자·분모로 내보낸다.

동결 선언은 `baseline_fixations`로 기준 실행에만 전달한다. 100% 참조는 동일 실행의 원래 증거 상태를 보여준다. 변경 계획의 고정 로그 평가는 탐색 단계다. 미승인 계획의 IPS에는 기존 명시적 탐색 옵션이 필요하다. Hybrid 승인 상태와 검열 보류 사유를 보존한다. 고정 로그 차이로 독립 실험 효과, 통계적 유의성, 실제 최적 임계값을 확정할 수 없다.

## 시뮬레이터 인계

`policy_search.run_policy_search()`에서 구성한 고객별 결정은 `comparison.decisions`에서 대표 정책별로 읽는다. `plans.representative_id`로 별칭을 연결한다. 최초 명부의 dataset_id, customer_id, as_of_at, snapshot_id, raw_database_sha256, selected, planned_sent_at, expected_cost와 통화·단위를 사용한다. 무효 계획의 NULL 결정을 유지한다. 정책 ID·버전·설정 지문·결정 지문·예산·계획 승인 상태는 plans에 있다. 발송 요청의 10% 최고단가 한 단위 할인과 28일 유효기간 계약은 `policy_value.INTERVENTION_SPEC`에서 확인한다. 실제 sent_at, 발송 성공·실패, 과금과 사용 결과는 생산자가 기록해야 한다. 민감도 결과의 dispatch_authorized는 항상 false다.
