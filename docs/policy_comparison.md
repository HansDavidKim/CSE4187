# H0 기준 정책 비교

`final_strategy.policy_comparison.compare_policies()`는 한 코호트의 고객 선정 계획, 같은 실험 로그의 IPS 점추정치, 정책 간 차이를 하나의 보고서로 반환한다. 로그가 없어도 계획과 세그먼트 표를 볼 수 있다.

## 호출

아래 예시는 검증된 입력과 비용 추정 결과가 준비된 시점에서 시작한다. `shared`에는 Phase A의 `run_policy_search()`에 전달하는 고객·예측 문맥, 적격성, 쿠폰 및 사용 이력, 비용 메타데이터, 기준시각, 예산, 알고리즘과 솔버 설정을 넣는다.

```python
from final_strategy.policy_comparison import ScopedPolicyFixation, compare_policies, evaluation_fingerprint
from final_strategy.strategy_profiles import build_strategy_profiles

profiles = build_strategy_profiles(**profile_settings)
shared = dict(search_inputs, cost_inputs=cost_estimate.cost_inputs,
              cost_metadata=cost_estimate.cost_metadata)
# H1 버전·설정·시각은 보관된 사전 동결 기록에서 읽는다.
# 평가 로그 발행 후에는 같은 실험의 명부·원장을 확인하고 그 자료의 지문을 연결한다.
fixation_by_policy_id = {"H1": ScopedPolicyFixation(
    policy_id="H1", policy_version=frozen_version,
    configuration_fingerprint=frozen_configuration_fingerprint,
    evaluation_fingerprint=evaluation_fingerprint(parquet_path, experiment_evidence),
    fixation=frozen_declaration,
)}
result = compare_policies(
    profiles, search_inputs=shared, scenario_id="baseline-budget",
    experiment_log=parquet_path, evidence=experiment_evidence,
    fixations=fixation_by_policy_id,
    pareto_directions={"no_session_rate_28d": "min",
                       "marketing_cost_minor_per_customer": "min"},
)
plans = result.table("plans").set_index("policy_id")
segments = result.table("segments")
h0_segments = segments[segments.representative_id == plans.loc["H0", "representative_id"]]
report = result.to_dict()  # json.dumps(report, allow_nan=False)
```

비용 추정기를 사용할 때는 `estimate_coupon_costs()`의 완료된 결과에서 비용표와 메타데이터를 함께 전달한다. 사업 적격성·이력 적격성의 3상태 인원 통계는 현재 실행 결과에서 복원할 수 없어 `eligibility_counts_status="not_provided"`로 표시한다.

## 실행 범위와 근거

Phase D의 등록 로직으로 설정 지문별 대표와 별칭을 구하고 Phase A를 호출한다. H0와 P0는 대표 실행을 공유한다. H2/H4-R 또는 H3/H4-V도 설정이 일치하면 별칭이 된다. 서로 다른 설정이 한 코호트에서 같은 계획을 만들어도 두 설정은 별개로 보존한다. 정책 선언과 설계 근거는 `registration`, 정책별 대표 ID와 두 지문은 `plans`에서 조회한다.

Phase B는 같은 호출에서 생성한 원래 `PolicyRun`과 같은 로그로 평가한다. 동결 선언은 별칭 ID별로 제공한다. 선언이 없는 별칭에 다른 별칭의 근거를 전파하지 않는다. 같은 대표 실행과 같은 동결 선언의 평가만 재사용한다. 고객별 기여분을 대조하는 쌍 비교는 기존 API 내부에서 양쪽을 재평가한다. `audit`의 호출 횟수와 소요시간에서 이 비용을 확인할 수 있다.

공통 입력 지문은 제공한 실행 인자 전체를 대상으로 하며 DataFrame의 고객 행 순서에 독립적이다. 예측값·메타데이터, 이력, 적격성, 예산, 고객별 비용까지 포함한다. 정책 설정 지문, 실제 선정·시점 지문, 로그의 출처와 동결 증거는 별도로 기록한다. 입력 표현이나 선택 인자의 명시 여부가 다르면 지문이 달라질 수 있다. 원본 금액 단위 검사는 호출자가 전달한 값에서 시작한다. 이미 상류에서 반올림한 금액 단위의 과거 정밀도는 복원할 수 없다. 생산자의 진위, 모집단 독립성 및 사전 정책 동결에는 별도 증거가 필요하다.

보고서 v2의 예산·비용·사용 확률은 실행 엔진의 십진수 해석을 거친 뒤 유효 숫자와 자리 위치로 기록한다. 예를 들어 NumPy float32와 Python float가 서로 다른 예산으로 실행되면 지문도 달라진다. 소수 끝의 0과 입력 행 순서는 금액의 의미를 바꾸지 않는다. 예측값과 솔버 설정에는 각 입력의 표현을 보존한다. v1 지문을 v2 결과의 비교 키로 사용하지 않는다.

`evaluation_fingerprint(experiment_log, evidence)`는 Phase B의 실험·데이터셋·원장 선언과 평가 로그를 결합한다. Parquet는 선언된 체크섬, DataFrame은 명부 및 관측값을 포함한 내용 지문을 사용한다. 생산자가 같은 동결 기록과 평가 로그를 어떻게 연결했는지는 별도로 입증해야 한다. 결과가 생기기 전에는 최종 로그 체크섬을 알 수 없으므로 사전 동결 시각과 사후 평가 자료의 연결 시점을 구분해 보관한다. 스코프가 일치하면 Phase B의 호출자 선언 수준을 유지한다. 기존 `PolicyFixation`만 제공하면 시각·사후 정보 사용 여부를 검증한 뒤 `scope_unavailable`, `unverified`로 기록하고 탐색 등급으로 평가한다. 명시한 ID·버전·설정·평가 자료가 다르면 호출을 거절한다.

사람 설계 프로필의 보고서는 기존 생성기로 재구성해 정책별 12개 규칙, 버전, 설정 지문, 별칭별 임계값과 대조한다. `registration.design_evidence_status`는 일치하면 `human_profile_rules_matched`이다. 일반 후보와 Phase C의 모델 분석·출처는 `producer_metadata_only`로 전달한다. 설명 문구와 출처가 실제 사업 근거인지는 이 검사로 판정할 수 없다.

`pareto_directions`는 고유한 IPS 지표를 키로 하고 `min` 또는 `max`를 값으로 갖는 비어 있지 않은 매핑을 받는다. 중복을 담을 수 있는 쌍 목록은 거절한다. 호출자가 이미 딕셔너리 생성 과정에서 덮어쓴 키는 복구할 수 없으므로 원천 설정을 읽을 때도 중복을 확인해야 한다.

## 표 읽기

| 표 | 단위와 해석 |
|---|---|
| `plans` | 정책 선언당 한 행. 전체 요청 계획률은 선정 수 / 최초 모집단, 후보 선정률은 선정 수 / 후보 수, 예산 사용률은 계획 예상 비용 / 예산이다. 분모 0은 NULL이다. 분석 모집단은 상류 제외 이후의 인원이다. |
| `decisions` | 대표 실행별 전체 고객. 요청당 예상 비용, 예산 후보, 선정 여부, 계획 시점, 비용 출처를 확인한다. |
| `segments` | 대표 실행별 기존 12개 세그먼트와 상류 제외가 있을 때의 `unclassified`. 분류 고객 및 후보를 각각 분모로 한 선정률, 선정 비용, 목적값을 표시한다. Budget에서 누락되는 비후보의 분류는 같은 전체 입력에 같은 Segmentation 설정을 적용해 키로 연결한다. 세그먼트별 재순위화나 예산 배분은 수행하지 않는다. 설정이 다른 실행의 동일한 세그먼트 이름은 서로 다른 고객을 포함할 수 있다. |
| `ips` | 정책과 지표당 한 행. 28일 구매율·무세션율, 고객당 할인 후 결제액·발송비·쿠폰 할인액·총 마케팅 비용의 기존 Phase B 추정치와 근거. 분모는 배정된 전체 ITT 고객 수다. 행동 일치 수, ESS, 검열 수, 지표별 관측 수를 함께 읽는다. 세그먼트별 IPS는 제공하지 않는다. |
| `comparisons` | 좌측 − 우측. 기본은 H1·H2·H3·H4-R·H4-V 각각 − H0, H2 − H3, H4-R − H4-V다. `pairs=[("H0", "P0")]`처럼 직접 지정할 수 있다. 계획 차이와 고객 집합 차이, 대응 IPS 차이를 구분한다. 비율 차이 0.01은 1 percentage point다. 상대 변화율은 차이 / 우측 추정치이며 우측이 0이면 NULL이다. |
| `pareto` | 같은 호출에서 명시한 IPS 지표·방향과 같은 근거 등급 안에서만 비교한다. 모든 축에서 열등하지 않고 하나 이상 엄격히 우수할 때 지배한다. 동점과 지표 간 상충은 함께 비지배가 될 수 있다. |

무효 계획의 선정 수·비용·목적값은 NULL이다. 정상적인 전원 미선정은 0이다. Hybrid의 유효성, 품질 인증, 계획 승인 여부는 원래 Budget 판정을 따른다. 비교 결과의 `dispatch_authorized`는 항상 false다.

## 관측, 비용 및 보류

로그 미제공은 `not_evaluated`, 검증 실패는 `contract_violation`과 원인으로 남는다. 검열된 경우 지표마다 `estimable`과 `hold_reason`을 확인한다. 발송비에 충분한 근거가 있어도 구매나 할인액은 보류될 수 있다. IPS 구매율은 유한 표본에서 1을 넘을 수 있다. 보고서는 원래 값을 보존한다. 금액은 Decimal, IPS와 대응 차이는 유리수로 다루며 JSON에는 각각 십진 문자열과 분자·분모로 기록한다.

Parquet는 Phase B의 읽기 함수로 파일 접근·체크섬·디코딩을 먼저 확인한다. 정상 파일은 원래 경로로 평가하므로 Phase B가 바이트를 다시 검사한다. 파일을 찾지 못하거나 디코딩에 실패하면 계획을 보존하고 사유를 기록한다. 계산 중 발생한 예기치 않은 예외는 호출자에게 전파한다.

요청당 예상 비용에서 선정 고객의 비용을 더한 값이 계획 총비용이다. IPS 마케팅 비용 단위는 **평가 고객당 원천 minor**다. 실제 외부 집행액 및 실제 예산 준수 여부는 별도 원장으로 확인해야 한다. 할인 후 결제액은 그대로 사용한다. 실패한 요청도 과금하는 비용표는 계획에 사용할 수 있지만, 미발송에 양수 발송비가 기록된 로그는 현 Phase B 계약을 위반한다. 계획 결과는 유지하고 IPS에 오류 사유를 남긴다.

미승인 계획 평가, 체크섬을 확인할 수 없는 DataFrame 평가, 탐색 결과의 Pareto 포함은 각각 `allow_unapproved`, `allow_unverified_log`, `include_exploratory`로 명시한다. 지정 지표 보류 또는 탐색 제외 시 Pareto 상태는 `held`가 된다. 포함한 경우에도 로그·동결·승인 근거가 다른 정책끼리는 지배 관계를 부여하지 않는다.

## 전략 선정에 필요한 추가 근거

예산 배율, 비용, 임계값 또는 솔버를 바꿀 때는 새 시나리오 ID로 다시 호출한다. 지문과 실행 조건을 대조한 뒤 민감도를 해석한다. What-if 출력을 IPS 입력으로 변환하지 않는다. 실제 평가는 독립 고객의 배정 근거, 기준시각 이전의 정책 동결, 원장 체크섬, 실패 과금 내역 및 28일 관측 완료 여부가 필요하다. 28일 무세션율의 주지표 채택, 허용 성과 저하폭, 최소 ESS 및 경제성 기준은 팀 협의 사항이다. Pareto 점추정만으로 정책 우열의 통계적 유의성, 실제 효과, ROI 또는 최종 A/B 결과를 확정할 수 없다.
