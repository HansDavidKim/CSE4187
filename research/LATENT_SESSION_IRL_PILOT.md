# 잠재 사용자 특성·MaxEnt 세션 IRL·재방문 간격: 파일럿

## 결과와 범위

Electronics와 Cosmetics 각각에서 첫 두 관측 세션만으로 사용자 행동 요약의 GMM posterior를 만들고, 다음 세션의 짧은 행동 시퀀스와 세션 종료를 유한 horizon MaxEnt IRL로 학습했다. 세션 종료 후에는 두 세션 요약과 직전 세션 행동으로 다음 세션까지의 이산 시간 위험을 계산하는 실행 가능한 사용자 모델을 추가했다.

**이 모형은 아직 전체 사용자 시뮬레이터나 쿠폰 인과 모형이 아니다.** 다음 세션이 30일 내 기록됐고 길이가 1~3개 이벤트인 경우로 IRL 학습·평가를 제한한다. 3번째 이벤트 이후 종료는 강제된다. Cosmetics의 보류 짧은 세션 38개에서 1이벤트 세션 실제 비중 84.2%, 모형 예상 비중 70.9%로 보정 부족도 확인됐다. 세션 종료는 30분 timeout으로 추정한 것이고 실제 로그아웃이 아니다.

## 1. 데이터, 시간 및 잠재 특성

사용자별 관측 이벤트를 30분 공백 기준으로 세션화한다. Electronics 이벤트는 사용자 해시 10% 표본, Cosmetics는 ID modulo 100 표본을 사용한다. 원본 세션 ID를 세션 경계로 사용하지 않으므로 기존 K=6 페르소나 분석과 세션이 다르다. 행위는 Electronics의 view/cart/purchase, Cosmetics의 view/cart/purchase/remove다. 이벤트 순서는 타임스탬프 동률일 때 입력 순서를 따른다.

두 번째 세션 종료를 기준 시점으로 한다. 첫 두 세션의 행동 수·총 이벤트 수와 첫 세션 후 공백을 이용하며, 이후 세션이나 재방문 결과는 입력하지 않는다. 사용자 해시로 학습/테스트 사용자를 분리하고, 학습 기준 시점은 시간 cutoff 이전, 테스트 기준 시점은 cutoff 이후로 제한한다. 양쪽 모두 30일 후속 관측 가능 조건을 둔다. 이미 두 번 관측된 사람만 포함하므로 신규 사용자 전체를 대표하지 않는다. 시간 cutoff는 Electronics 2021-01-12 UTC, Cosmetics 2020-01-15 UTC다.

학습 사용자만으로 표준화기와 대각 공분산 GMM 3개 성분을 추정한다. 사용자 잠재 상태 `z`는 첫 두 세션 요약을 입력했을 때 첫 두 성분의 posterior 확률이며, 나머지 한 성분은 기준 성분이다. 기존 GRU K=6 페르소나와 동일하지 않고, 세션 내내 고정된 행동 요약 잠재 표현이지 개인별 IRL reward나 canonical type이 아니다. 3개 성분은 파일럿의 설정값이며 최적성·견고성을 검증하지 않았다.

## 2. 유한 horizon MaxEnt IRL과 다항 로짓 선택

허용된 세션 경로는 1~3개의 기록 행동 이후 `END`이며, 후보 경로를 전부 열거한다. `END`는 timeout으로 추정한 세션 종료다. 세 번째 이벤트에서는 종료가 강제되며 4번째 이상의 행동을 허용하지 않는다. 따라서 길이가 4 이상인 실제 세션은 제거한다. view를 기준 reward 0으로 두고 나머지 행동과 END에 대해:

\[
r_\theta(s,a,z)=\theta_a^\top[1,\ t/3,\ \mathbf 1(a_{t-1}=\mathrm{purchase}),\ z_1,z_2].
\]

`t`는 현재 세션 내 이전 행동 개수이고, 모든 state 변수는 선택 이전의 것이다. END에도 같은 특징 정의를 적용한다. 초기에는 이전 구매 지표가 0이다. L1 계수 0.1, 온도 1을 사용한다. 학습 목적은 관측 경로 feature와 선택 가능한 전체 경로의 reward 합의 log-partition을 비교하는 MaxEnt trajectory log likelihood다. 동역학은 행동 이력의 결정적 갱신이며 세션 당 horizon이 제한된다. 일반적인 확률적 환경의 완전한 maximum causal entropy IRL을 검증한 것은 아니다. [Ziebart 등의 MaxEnt IRL](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.pdf), [maximum causal entropy](https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.html)를 참고했다.

경로 확률은 `P(τ|z) ∝ exp(Σ rθ)`이다. 실제 각 단계 선택은 현재 prefix와 일치하는 경로의 확률 질량을 합해 조건화한다. 이는 가능한 미래의 soft value를 포함한 `softmax(Q)` 선택으로, 일반적으로 즉시 `softmax(r)`과 다르다. 구현은 완전 경로를 샘플링하는 방식이지만 동일한 조건부 다항 선택분포다. 시퀀스에 원래 없던 장바구니·상품 노출 제약은 모델링하지 않았다.

이 reward는 가정한 모델 내 행동을 재현하는 파라미터이지 사람의 참된 효용, 복지, 인과효과 또는 개별 사용자별 reward가 아니다. view 기준 및 온도 고정은 reward의 모든 식별 문제를 해결하지 못한다. 긴 세션을 제외한 조건부 샘플에만 맞춘다.

## 3. 세션 종료 후 상태와 방문 간격

세션 단위로 이전 공백과 이번 세션의 view/cart/purchase/remove 수, 이벤트 수, 고정 GMM posterior를 사용한다. 행동 수는 20개에서 clipping 후 20으로 나누고, 이전 공백은 720시간에서 clipping 후 720으로 나눈다. 종료 직후 해당 세션 요약을 갱신하고, 다음 방문의 위험을 예측한다. 사용자의 별도 만족도 값이나 장기적 선호 변화는 추정하지 않았다.

조건부 위험 `h_j=P(T∈[b_j,b_{j+1}) | T≥b_j,x,z)`에 bin 지시자와 상태를 넣어 로지스틱 회귀를 학습한다. 시간 구간은 0, 1, 6, 24, 72, 168, 336, 720시간이다. 각 구간의 전체 후속 기간이 행정 cutoff 안에 있을 때만 그 구간의 학습 행을 사용한다. 모형은 미관측을 영구 이탈로 처리하지 않는다. 생존 기반 예측 접근의 일반적 배경은 [discrete-time survival 모델 논문](https://bmcmedresmethodol.biomedcentral.com/counter/pdf/10.1186/s12874-022-01679-6)을 참고했다.

생성 시 각 구간의 위험을 순차적으로 샘플링하고, 선택된 구간 안에서는 **균일한 시각을 가정**한다. 모든 구간에서 방문이 없으면 `None`을 반환해 `>30일 또는 관측 밖`으로 표시한다. 30일을 넘는 정확한 시각이나 영구 이탈을 생성하지 않는다. 짧은 IRL 세션 출력이 위험모형의 전체 세션 학습 분포와 다를 수 있다.

## 4. 사용자·시간 분리 보류 평가

| 항목 | Electronics | Cosmetics |
|---|---:|---:|
| 30일 추적 가능한 학습 / 테스트 기준 사용자 | 1,741 / 116 | 1,833 / 96 |
| IRL 학습 / 테스트 짧은 다음 세션 | 497 / 34 | 573 / 38 |
| 다음 세션이 30일 내 관측된 전체 학습 / 테스트 사용자 | 558 / 41 | 899 / 54 |
| IRL 보류 NLL / 결정(END 포함) | 0.540 | 0.397 |
| 동일 IRL에서 latent 제외 NLL / 결정 | 0.571 | 0.401 |
| 30일 재관측률 실제 / 예측 | 35.3% / 33.3% | 56.2% / 48.1% |
| 30일 Brier: 상태+latent / 시간 bin 기준선 | 0.215 / 0.230 | 0.232 / 0.251 |

NLL의 비교 기준은 같은 후보 경로·같은 특징 구조에서 잠재 특징만 제거한 모형이다. Brier 기준선은 시간 bin만을 사용한다. 모두 낮을수록 좋지만 한 번의 분할, 작은 보류 IRL 표본과 선택된 짧은 세션에서 얻은 수치다. latent 자체의 통계적 유의성, 전체 사용자 일반화, 평균 효과 또는 인과 개선을 보장하지 않는다. Cosmetics는 보류 30일 재관측률을 8.1%p 낮게 예측했다.

Electronics 길이 1/2/3 실제 비중은 73.5/8.8/17.6%, 모형은 70.3/19.4/10.2%다. Cosmetics 실제는 84.2/10.5/5.3%, 모형은 70.9/20.5/8.6%다. 세션 길이의 불일치는 현재 모델을 채택·고정하기에 충분하지 않다는 근거다. 세션 내 상태를 자유롭게 굴리는 장기 rollout이나 여러 seed의 분포 적합도는 평가하지 않았다.

## 5. 쿠폰 canonical type과 인과 범위

각 데이터의 로그에는 쿠폰의 발송·노출·무작위 배정이 없으므로 latent posterior에서 sure thing, persuadable, sleeping dog, lost cause의 개인별 유형이나 집단 분포를 추정하지 않았다. 개인의 잠재결과 두 개는 동시에 관측되지 않으며 무작위 실험의 주변 확률만으로도 결합 분포가 일반적으로 미식별이다. 참고: [Tian–Pearl의 잠재결과 결합 확률 경계](https://escholarship.org/uc/item/44s6x4c7).

`coupon_outcomes(p0, p1, u)`는 오직 **가상 민감도 시나리오**다. 동일 `u`를 두 처치 결과에 쓰는 coupling을 가정한다. `p1 ≥ p0`이면 단조성을 강제하고 sleeping dog가 불가능하다. `p1 < p0`이면 반대로 persuasion이 없다. 이는 네 가지 유형이 함께 존재하는 일반 모형이 아니며, 결합 분포를 식별한 결과도 아니다. 이 함수는 쿠폰의 시간에 따른 효과나 행동·간격에 연결되어 있지 않다. 관측자료에서 `p0,p1`을 추정했다고 주장하지 않는다.

사용자의 요청에서 언급한 인과추론 관점은 처치/결과 분리, 시간 순서, 사용자·시간 분리와 잠재결과 식별 한계로 반영했다. 이것들이 교란을 제거하는 인과추정 기법이나 무작위 개입 검증을 대신하지는 않는다. 데이터의 한계로 실제 쿠폰 효과, 인과적 이질성, 관측 사용자별 canonical type을 추정하지 않았다.

## 6. 산출물·재현

- 모델: `src/ecommerce_sim/latent_session_model.py`, 학습: `scripts/run_latent_session_pilot.py`.
- 결과: `artifacts/latent-session-pilot/{electronics,cosmetics}.json` 및 로컬 직렬화 모델 `.joblib`. 직렬화 파일은 로컬에서만 사용하고 신뢰할 수 없는 pickle/joblib를 열지 않는다. 원본·모델의 재배포 권리는 별도 확인해야 한다.
- `LatentSessionUser(model, prefix, rng)`는 두 세션 요약에서 posterior를 만들고 `sample_session()`, `finish_session(events)`로 세션 및 30일 내 간격을 샘플링한다. prefix 컬럼과 스케일은 직렬화 모델의 `prefix_cols` 및 코드의 `session_return.sessions` 정의를 따른다. 시간대·표본 등의 입력 정의를 맞추어야 한다.
- 실행: `PYTHONPATH='.:src:/Users/kimdaewon/Desktop/캡스톤디자인' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python scripts/run_latent_session_pilot.py electronics cosmetics`.

다음 필요한 검증은 완전한 세션 horizon/검열 모델, 액션 가용성, latent 차원·초기화·표본 민감도, 사용자·시간 분리 반복 평가 및 긴 시퀀스 rollout이다. 개입 효과를 위해서는 별도 쿠폰 발송·수령·반응 로그와 명시적인 식별 가정 및 overlap 점검이 필요하다.
