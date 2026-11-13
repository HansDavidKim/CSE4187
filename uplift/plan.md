# Uplift T/X/H-learner 학습 Plan

목표: 10% 쿠폰 1회 발송이 4주 유지(구매 1회 이상)에 주는 효과 `τ(x) = P(Y=1|X,T=1) − P(Y=1|X,T=0)`를 고객별로 추정한다. 기준 문서는 `usecase.md`, `feature.md`, `docs/simulator_ERD.md`.

## 1. 시뮬레이터와 합의

확정된 것:

- 받을 형태: 고객별 주간 합계(`customer_id × reference_week`). 비활동 주는 0, 가입·관측 전은 NULL. 모든 기간은 주 단위(과거 12주, 최근 4주, 결과 4주)이며 일·시간 단위 기록은 쓰지 않는다.
- 실험 조건: 적격 고객(`w0` 이전 4주 방문자)을 고객별 고정 배정(확률 0.5). Treatment는 `w0`에 10% 쿠폰 1회(4주 유효), Control은 미발송, 추가 발송 없음.
- 평가용으로 ST/P/SD/LC 잠재 진단값과 `dataset_id`·seed를 받는다.

남은 것 (`feature.md` 4절):

- 4주 미구매 정의의 `definition_id`. 시뮬레이터 기본은 4주 미방문이다.
- 쌍대 실행 결과를 같은 정의로 받을 수 있는지.
- `views`, `session_minutes`, `w0` 장바구니 상태의 내보내기 형식.
- 규모: small(5,000명)로 개발, full(각 군 10,000명 이상)로 최종 학습.

## 2. 학습 데이터셋 생성 (UC-01)

- 고객 1명당 1행: `customer_id, w0, X, T, Y`.
- X는 `w0` 이전 12주의 주간 합계만 사용. `feature.md`의 필수 4개 + 추가 14개. 기본 시나리오에는 과거 쿠폰 이력이 없어 쿠폰 Feature는 뺀다.
- Y는 `[w0, w0 + 4)` 주의 `purchases` 합이 1 이상인지 여부. 이 구간이 다 관측된 고객만 포함.
- T는 배정군 기준(ITT). 쿠폰 사용 여부로 재분류하지 않는다.
- 검증: 미래 주를 추가해도 X가 변하지 않는지, T/C 간 X 분포가 균형인지(SMD) 확인.

## 3. 전처리와 분할

- train / valid / test를 고객 단위로 분할하고 T 비율을 유지. 기준 주가 여러 개면 시간 순으로 분할.
- null은 그대로 두고 결측을 처리하는 트리 모델(LightGBM 등)을 base learner로 사용. 대체값·스케일러가 필요하면 train에서만 적합.

## 4. 모델 학습 (UC-02)

| 모델 | 방법 |
| --- | --- |
| T-learner | `μ1`(T=1), `μ0`(T=0)를 따로 학습. `τ = μ1(x) − μ0(x)` |
| X-learner | T-learner의 `μ0, μ1`로 대조군·처치군의 가상 효과 `D0 = μ1(x) − Y`, `D1 = Y − μ0(x)`를 만들고 각각 회귀한 뒤 propensity `e = 0.5`로 가중 평균 |
| H-learner | 직접 추정(X/DR 방식)과 간접 추정(T 방식)을 섞는 hybrid 방식. 표준 라이브러리에 없어 직접 구현 |

- T·X는 `causalml` 또는 `econml` 사용, 설치가 안 되면 scikit-learn/LightGBM으로 직접 구현.
- UC-04가 `p1`, `p0`를 모두 요구하므로 세 모델 모두 `μ0, μ1`도 함께 저장.
- 하이퍼파라미터는 valid의 Qini 계수로 선택.

## 5. 평가 (UC-03)

- test에서 Qini curve / Uplift curve, Qini 계수(AUUC), 상위 decile별 실제 uplift.
- 기준선: 무작위 타기팅, 전체 평균 효과(ATE).
- 쌍대 실행 결과를 같은 정의로 받으면 고객별 실제 uplift와 직접 비교(순위 상관, 세그먼트별 평균 오차).
- 시뮬레이터의 잠재 진단값(ST/P/SD/LC)은 입력에 넣지 않고, 예측 세그먼트와의 일치도 점검에만 사용.

## 6. 예측과 내보내기 (UC-04–06)

- 선택한 모델로 `p1, p0, uplift_score = p1 − p0` 산출.
- `uplift_score`와 `p0` 기준으로 4개 세그먼트 분류, 임계값은 valid에서 결정.
- CSV: `customer_id, w0, p1, p0, uplift_score, segment` + `model_version`, `definition_id`.

## 확인 필요

- H-learner의 정확한 정의(참고 논문). 위 설명은 추정이므로 구현 전에 확정해야 한다.
- 현재 `.venv`는 Python 3.14라 `causalml`/`econml` 설치 가능 여부 미확인.
