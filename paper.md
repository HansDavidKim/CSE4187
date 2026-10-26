# 교차 도메인 행동표현과 합성 Uplift를 결합한 전자상거래 쿠폰 정책 시뮬레이터

> KSC 3쪽 원고의 읽기용 초안이다. 제출용 원본은 `paper/paper.tex`, 렌더 결과는
> `output/pdf/paper.pdf`이다.

## 요약

실서비스에서 쿠폰 정책을 장기간 탐색하면 직접 비용과 사용자 피로가 발생하지만,
공개 전자상거래 로그 하나만으로는 무처치 행동 시퀀스와 개인별 쿠폰 반응을 함께
학습하기 어렵다. 본 연구는 서로 다른 공개 로그의 역할을 분리한 순차 정책
시뮬레이터를 제안한다. Cosmetics 행동 로그에서 masked bidirectional-GRU 표현과
행동 페르소나를 학습하고, 사용자별 maximum-entropy 효용계수, lifecycle 및 이산
세션간격을 결합한다. Electronics와 Tianchi는 공통 과거행동 특징으로 Cosmetics
페르소나에 정렬하며 각 도메인의 이질성은 대각 Gaussian mixture로 보존한다.

Tianchi의 관측 수령 차이를 persona별 합성 uplift 보정목표로 변환한다.
사전행동 대리군 내 매칭으로 얻은 구매율은 네 canonical response type의 합성
보정 모멘트로만 사용한다. 48차원 사용자 latent에서 canonical type 확률을 생성하는
행렬은 L1--L2 정규화로 학습하고, 쿠폰효과는 반복 수령과 시간 감쇠를 거쳐 행동 및
복귀효용에 결합한다. 세 도메인별로 1,000명·13주 온라인 학습 후 독립
5,000명·52주를 평가하며, 발송비용과 첫 구매 할인비용의 두 경제조건에서 UCB, TS,
LinUCB, LinTS, PPO, discrete SAC를 비교한다.

## 1. 문제와 증거 경계

쿠폰은 현재 구매뿐 아니라 다음 방문과 이탈확률을 바꿀 수 있다. 실제 사용자에게
여러 정책을 반복 적용하는 방법은 비싸고, 오프라인 로그에는 선택하지 않은 행동의
결과가 없다. 행동 로그와 쿠폰 로그가 서로 다른 서비스에서 수집됐다면 사용자
식별자와 행동공간도 일치하지 않는다.

이를 세 층으로 분리한다.

1. 관측 시퀀스에서 무쿠폰 행동과 세션 복귀를 학습한다.
2. 공통 과거행동 특징으로 도메인별 집단을 정렬한다.
3. Tianchi의 수령 차이를 공개된 합성 uplift 규칙으로 주입한다.

따라서 쿠폰항을 0으로 두면 무쿠폰 행동 생성기로 정확히 환원된다. Canonical type과
정책 순위는 동일한 합성 사용자와 비용조건에서 비교한다.

## 2. 데이터 역할

| 데이터 | 관측 행동 | 역할 |
|---|---|---|
| REES46 Cosmetics | view, cart, remove, purchase, session | 행동표현, 페르소나, 사용자별 효용, 세션간격 |
| REES46 Electronics | view, cart, purchase, session | 공통 행동패턴의 외부 기술적 확인과 도메인 보정 |
| Tianchi O2O Coupon | click, purchase, coupon receipt | 사전행동 대리군, 합성 canonical 비율, 쿠폰 보정 모멘트 |

원본 데이터는 저장소에 재배포하지 않는다. Electronics의 `remove`, Tianchi의
`cart/remove`는 0으로 대치하지 않고 해당 도메인의 미관측 행동으로 비활성화한다.
쿠폰 수령은 정책 개입으로, click과 purchase는 사용자 행동으로 분리한다.

Cosmetics에서 `remove`를 삭제한 Cosmetics-drop과 Electronics는 동일한
`view/cart/purchase` 어휘로 하나의 masked-GRU와 K-means를 공동학습해 K=4
기준공간을 구성한다. 두 도메인은 같은 J0--J3 군집에 일대일로 놓인다. Tianchi는
K=6으로 군집화한 뒤 coupon receipt를 제외한 13개 탐색·구매 특징으로 이 기준공간에
many-to-one 정렬한다. 여섯 Tianchi 군집은 J0--J2의 세 기준군으로 합쳐진다. 이는
쿠폰이 관측되는 도메인에서 행동집단이 더 세분화된 구조를 보인다.

## 3. 방법

### 3.1 사용자 표현과 행동 페르소나

행동 사이에 SESSION 토큰을 삽입하고 유효 위치의 15%를 마스킹해 원래 토큰을
복원한다. 16차원 토큰 임베딩과 방향별 hidden size 24의 양방향 GRU에서 48차원
사용자 표현을 얻는다. 저장된 표현에 K=4 K-means를 적용한다. 얕은 조회군은 고정
NEW와 CHURNING lifecycle의 혼합상태로 해석하며,
성숙 home persona는 P0, P2, P3으로 둔다.

| 군집 | 관측 행동 | 시뮬레이터 내 역할 |
|---|---|---|
| P0 | 구매·장바구니 전이가 모두 큰 구매 혼합형 | 구매지향 성숙 home persona |
| P2 | 긴 시퀀스에서 cart·remove가 반복되는 장기 수정형 | 숙고·수정형 성숙 home persona |
| P3 | cart 비중이 높고 구매가 낮은 장바구니 집중형 | 저전환 탐색형 성숙 home persona |
| C1 | 짧은 view 중심 시퀀스 | NEW·CHURNING·목적종료가 혼재하여 lifecycle로 분리 |

### 3.2 Proxy group 배정

구매비중, 총 행동량, 활동일, 일별 행동량, 구매일 비중, 중앙간격, 최초·최종
구매위치, 탐색↔구매 전이와 지속성 등 처치 이전 13개 공통 특징을 사용한다. Pooled
robust scaling 후 Electronics와 Tianchi 사용자를 Cosmetics K=4 centroid에 최근접
배정한다. Electronics에서 독립적으로 재현되지 않은 C1은 제외하고 C0, C2, C3을
P0, P2, P3의 대리군으로 사용한다.

Tianchi 각 대리군 안에서는 과거 구매여부와 과거 쿠폰 수령횟수를 exact match하고,
propensity-logit 0.2 표준편차 caliper의 1:1 비복원 매칭을 적용한다. 기준일 이후
행동은 배정과 매칭에 사용하지 않는다. P-LT는 기준일 이전에 활동일 10일 이상,
구매일 3일 이상, 관측기간 30일 이상을 만족한 사용자 내부에서 별도로 매칭한다.

### 3.3 Canonical type 비율

매칭된 비수령·수령 구매율을 각각 $p_{c,0},p_{c,1}$이라 한다. 두 잠재결과의
주변확률을 보존하고 추가 상관을 가정하지 않는 최대엔트로피 독립결합은 다음과 같다.

$$
\pi_c=\left(p_{c,0}p_{c,1},(1-p_{c,0})p_{c,1},
p_{c,0}(1-p_{c,1}),(1-p_{c,0})(1-p_{c,1})\right).
$$

순서는 sure thing, persuadable, sleeping dog, lost cause다.

| 그룹 | Sure thing | Persuadable | Sleeping dog | Lost cause |
|---|---:|---:|---:|---:|
| P0 | 2.82% | 19.50% | 9.82% | 67.85% |
| P2 | 4.69% | 21.98% | 12.89% | 60.44% |
| P3 | 2.72% | 19.54% | 9.50% | 68.25% |
| P-LT | 10.49% | 23.86% | 20.05% | 45.60% |

Sure thing은 무쿠폰과 쿠폰에서 모두 구매하고, persuadable은 쿠폰을 받을 때만
구매하며, sleeping dog은 무쿠폰에서만 구매하고, lost cause는 두 조건 모두
구매하지 않는 유형이다.

### 3.4 Feature engineering과 MaxEnt 효용

행동 직전의 12개 기본특징은 절편 1개, $\log(1+\text{세션 위치})$·
$\log(1+\text{세션 경과분})$·$\log(1+\text{직전 이벤트 간격분})$ 3개,
$\log(1+\text{누적 행동횟수})$ 4개(view·cart·remove·purchase), 직전 행동
one-hot 4개(view·cart·remove·purchase)다($1+3+4+4=12$). 모두 매 의사결정에서 과거 로그만으로
관측되어 결측과 희소 교차항이 없으므로 개인별 계수 추정에 사용한다.

$$
r_u(s,a)=\theta_{u,a}^{\top}\phi(s),\qquad
\pi_u(a\mid s)=\frac{\exp r_u(s,a)}{\sum_b\exp r_u(s,b)}.
$$

세션 내부는 one-step contextual MaxEnt IRL로 정의한다. 이 경우 entropy-regularized
최적 정책이 위 다항 logit user choice model과 같으므로, IRL은 reward 계수를
복원하고 choice model은 이를 확률적 행동으로 변환한다. Temperature는 1로 고정하고
각 특징에서 행동계수 합을 0으로 제약한다. Persona별 maximum-likelihood 계수를
사전중심으로 학습한 뒤 각 centroid 최근접 사용자 20명의 계수를 MAP으로 추정한다. 계수에는
대각 GMM을 적합하며, 적합도 향상과 모수 증가를 함께 벌점화하는 Bayesian
information criterion(BIC)이 최소인 성분 수를 선택한다(선택값 3). 혼합비는 성숙
페르소나 모집단 비중으로 사후층화한다. 2차 78특징 모형은 pooled ablation에서
NLL을 0.648에서 0.600으로 낮췄지만 개인별 표본 안정성을 위해 실제 사용자 GMM은
12개 기본특징을 사용한다.

### 3.5 세션 간격과 lifecycle

세션 종료 후 간격은 `0–1h`, `1–6h`, `6–24h`, `1–3d`, `3–7d`, `7–14d`,
`14–30d`, `NO_RETURN_30D`의 여덟 토큰으로 샘플링한다. Engagement, friction,
satiation, inactivity momentum, gap acceleration, habit strength는 과거 세션만으로
재귀 갱신한다. 첫 간격은 학습평균 51.33시간으로 초기화하되 결측 지시자를 유지한다.
NEW→home persona↔CHURNING 전이와 구매 후 30일 비복귀 종료를 별도로 모델링한다.

### 3.6 Latent-conditioned uplift와 순차 보정

표준화한 48차원 latent와 절편에서 네 canonical logit을 만드는 49×4 행렬 $W$를
$L_1=0.001$, $L_2=0.01$로 학습한다.

$$
c_u\sim\operatorname{Categorical}
\left(\operatorname{softmax}([1;z_u]^\top W)\right).
$$

P-LT에는 별도 logit offset을 학습한다. 무쿠폰 행동모형은 동결하고 $W$와
페르소나별 쿠폰 강도만 보정하므로 이 단계는 continual learning보다 sequential
calibration에 가깝다. Persuadable과 sleeping dog에는 서로 반대 부호의 구매·복귀
offset을 준다. 반복 수령 노출은 제곱합의 제곱근으로 누적하며 각 수령효과는 6주
안에 단조 감소한다.

### 3.7 도메인 GMM

세 도메인의 공통 13특징에 대각 GMM을 적합하고 BIC와 최소 성분크기 조건으로
Cosmetics 4개, Electronics 5개, Tianchi 6개 성분을 선택한다. 먼저 도메인 GMM
성분을 뽑고, 성분별 P0/P2/P3 조성에서 home persona를 뽑은 뒤 해당 persona의
효용계수를 샘플링한다. 서로 다른 시간단위의 중앙간격 비율은 제곱근 수축 후
[0.5, 4]로 제한한다.

## 4. 정책 벤치마크

도메인별 1,000명·13주에서 온라인 학습하고, 독립 seed의 5,000명·52주에서 정책을
고정해 평가한다. 의사결정은 주 1회이며 동일 사용자 seed를 사용한다. Agent가 보는
19개 상태에는 persona, lifecycle, 최근 7/30일 행동, 마지막 세션·구매 이후 시간,
쿠폰노출과 발송가능 여부가 포함된다. Canonical type과 미래결과는 숨긴다.

경제조건은 두 가지다.

1. **발송비용(S):** 판매 1회 1,000, 실제 쿠폰 발송 1회당 100을 차감한다.
2. **첫 구매 할인(D):** 발송비는 없고 쿠폰을 42일 FIFO로 보관한다. 각 쿠폰이
   처음 적용되는 후속 구매에만 10% 할인비용을 적용하며 이후 구매는 정상가격이다.

최종 수치와 그림은 `artifacts/agent-benchmark/summary.json`,
`img/analysis/agent_policy_comparison.png` 및
`img/analysis/agent_coupon_cost_per_purchase.png`에서 생성한다. 표에는 사용자당
순이익의 평균±사용자 간 표준편차와 평균 쿠폰 수를 함께 보고한다.

## 5. 한계

- Canonical 결합, 6주 감쇠와 반복 수령함수는 공개된 합성 보정 규칙이다.
- 세 도메인의 사용자 ID는 연결되지 않으며 공통 특징에 대한 집단 수준 전이만 한다.
- 정책 비교는 단일 전역 seed와 고정 경제가정에 조건부다.
- 후속 검증은 비용·감쇠 민감도와 다중 seed 비교로 확장한다.
