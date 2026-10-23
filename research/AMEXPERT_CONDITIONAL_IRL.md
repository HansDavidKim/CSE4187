# AmExpert 개입 조건부 구매 trajectory IRL 파일럿

## 모델 정의

사용자 행동은 `transactions > 0`으로 구성한 구매/비구매이며, `coupon_discount`는 행동 정의에 쓰지 않는다. 별도 입력 `z_t`는 같은 주의 기록된 활성 쿠폰 오퍼가 1개 이상인지 나타낸다. `z_t=0`은 **기록 없음**이지 실제 미제공이라고 확인된 것이 아니다. 기존 패널의 쿠폰 캠페인 유효기간을 이용한 근사치이며, 실제 발송·구매 순서는 확인되지 않는다.

사용자 상태는 직전 주 구매 여부 `s_t`, 구매 행동 `a_t`와 오퍼 기록 `z_t`를 구분한다. 구매 reward는 `a_t * (w0 + w1*s_t + w2*z_t + w3*s_t*z_t)`이며, 비구매 reward는 0이다. 학습 데이터의 오퍼 기록 전이를 `P(z_{t+1}|z_t)`로 추정하고, `s_{t+1}=a_t`인 유한 horizon의 maximum causal entropy soft Bellman recursion으로 관측 구매 trajectory likelihood를 최대화한다. 계획에서 미래 오퍼는 주어진 동일 분포 전이로 처리하고 사용자 구매에 반응하지 않는다는 단순화가 있다. 이는 오퍼가 실제 무작위라는 뜻이 아니다.

0~51주의 구매·오퍼 기록으로 masked GRU를 학습하고 임베딩을 GMM으로 군집화한다. 각 그룹의 선형 reward 계수를 먼저 추정하고, 각 사용자 계수는 자기 그룹의 계수로 shrinkage하여 추정한다. 사용자 계수들의 그룹별 평균·공분산과 그룹 비율로 Gaussian mixture를 구성하여 새 계수를 샘플링할 수 있다. 다항 선택의 특수한 경우인 binary logit 정책이며 별도 아이템 선택 집합은 없다.

기존 NTS-NOTEARS 결과에서 `transactions`의 **lag > 0 부모는 없다**. 동시점 `unique_items -> transactions`는 같은 주의 결과일 수 있으므로 결정 전 feature로 사용하지 않는다. 따라서 IRL feature인 직전 구매와 오퍼는 명시적 설계 가정이며 NTS에서 발견된 parent-set이라고 주장할 수 없다. 추가 feature가 발견되더라도 현 IRL state/reward에는 자동 주입되지 않으며, 이 파일럿은 DAG 기반 IRL이라는 주장을 뒷받침하지 않는다.

## 시간 분할과 평가

IRL 학습 에피소드는 주 1~12, 13~24, 25~36, 37~48이며 오퍼 전이는 0~48주로 추정한다. GRU 그룹은 0~51주를 사용한다. 52~64주는 검증이 아니라 **보정 적합**에 사용한다: 전체 사용자 공통 구매 intercept와 오퍼 기록 계수 offset을 해당 구간의 구매 log loss로 학습한다. 이 검증 수치는 보정에 대한 in-sample 결과다. 65~77주는 보정에 사용하지 않은 테스트다. 동일 사용자에 대한 미래 예측이며 새 사용자 generalization을 뜻하지 않는다.

시드 2026, 4그룹, 20 epoch에서 테스트:

| 측정 | 결과 |
|---|---:|
| 관측 구매율 | 68.06% |
| teacher-forced 사용자 예측 평균 | 69.30% |
| 관측 오퍼 스케줄 replay 사용자별 rollout 평균 (100회) | 69.29% |
| 사용자별 teacher-forced log loss | 0.5924 |
| 학습 구매율 상수 기준 log loss | 0.6526 |
| 오퍼 미기록: 관측 / 예측 구매율 | 60.88% / 62.68% |
| 오퍼 기록: 관측 / 예측 구매율 | 74.97% / 75.66% |

100회 rollout 표준편차는 Monte Carlo 변동만 나타내며 추정의 신뢰구간이 아니다. 관측 테스트 오퍼 스케줄을 replay하므로 개입 정책의 결과를 검증한 실험이 아니다. 합성 계수 rollout도 관측 테스트의 초기 구매 상태·오퍼 스케줄을 사용한다. 평균 구매율과 정답 이력 조건부 예측이 맞는다고 장기 trajectory 분포가 검증된 것은 아니다.

학습 구간 오퍼 기록 비율은 5.85%, 보정 구간 47.56%, 테스트 50.97%로 분포 차이가 크다. 보정 전 테스트 예측의 log loss는 0.6567, rollout 구매율은 50.49%였다. 보정이 시기별 차이를 흡수할 수 있으나, 이 관측 차이의 원인을 쿠폰 효과라고 해석할 수는 없다. 시간·사용자 교란, 캠페인 중첩, 측정과 selection을 별도로 다뤄야 한다.

이전 그룹·비선형 실험은 구매를 쿠폰 할인 사용 여부에 따라 세 범주로 나눴고, 현시점 오퍼를 외생 개입으로 명시하지 않았다. 기존 결과를 이 파일럿의 causal uplift나 개입 정책 비교에 사용하지 않는다. 이전 파일을 삭제하지는 않았다.

## 재실행 및 안전한 사용

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_conditional_trajectory_irl.py
PYTHONPATH=src pytest -q
```

`artifacts/amexpert-conditional-trajectory-irl/summary.json`은 요약이고 `local_model.npz`에는 사용자 ID·계수와 파생 시퀀스가 들어 있다. 데이터 권리가 확인되기 전 외부 재배포하지 않는다. NTS 새 학습, 실험 처치로서 인과 식별, HeartSteps·Complete Journey 적용, 여러 시드·새 사용자 검증, RL/GRU4Rec 비교는 아직 수행하지 않았다.
