# Uplift T/X/H-learner 학습 Plan

목표: 10% 쿠폰 1회 발송이 28일 유지(주문 1건 이상)에 주는 효과 `τ(x) = P(Y=1|X,T=1) − P(Y=1|X,T=0)`를 고객별로 추정한다. 기준 문서는 `usecase.md`, `feature.md`, `docs/simulator_ERD.md`, 시뮬레이터 설계서 v5.

## 1. 시뮬레이터 v5에서 달라진 점과 대응

| v5 변경 | Uplift 쪽 대응 |
| --- | --- |
| 이벤트 6종(coupon_usage, review 추가), Event 17컬럼, `mixed` 결제 | ERD 재작성. 장바구니 재생과 리뷰 Feature에 반영 |
| Feature가 KIT39로 확정(F33–F39 리뷰·cart 추가, F21–F24는 전체 이력의 마지막 완료 세션) | `feature.md`를 KIT39 이름·산식으로 재작성. 입력은 쿠폰 이력 3개를 뺀 36개 |
| 실험의 적격·쿠폰 유효·성과 창이 모두 28일, Feature 창은 30일 | 주 단위(4주·12주) 집계를 버리고 일 단위 창으로 계산 |
| 실행 결과가 `raw.sqlite` + 사이드카 파일로 저장 | 주간 합계 대신 원시 로그를 직접 읽음 |
| 쌍대 rollout은 `evaluation_only/`에 분리 | 학습 입력 단계에서는 읽지 않음 |

입력 단계는 세 모델이 함께 쓴다.

```
시뮬레이터 실행 디렉터리
  └ common/raw.py       load_raw()       원시 테이블 7개 + 배정 → RawData
      └ common/features.py  build_dataset()  KIT39 + T + y (고객 1명당 1행)
          └ Tlearner / Xlearner / Hlearner   fit(X, t, y) → predict(X)
```

## 2. 시뮬레이터 담당자에게 확인할 점

1. **파일 계약.** v5 코드에서도 실행 디렉터리에 `raw.sqlite`, `raw.manifest.json`(`metadata.observation_start_at`·`observation_end_at`, `database_sha256`), `treatments.parquet`(`customer_id`, `as_of_at`, `treatment`, `assignment_probability`)가 같은 이름·컬럼으로 나오는지. `raw.py`가 이 셋만 읽는다.
2. **KIT39 v3 대조용 실행 결과.** 같은 실행의 `raw.sqlite`와 `features.sqlite`를 받아 우리 계산과 39개 전부를 대조하려 한다. 현재 저장소의 v2 코드와는 일치를 확인했지만, v3에서 바뀐 두 가지(F21–F24의 마지막 완료 세션 범위, 관측 누락 시 NULL 규칙)는 설계서 문장만 보고 구현했다(`feature.md` 2.3).
3. **`mixed` 결제의 저장 값.** `orders.checkout_source`에 `mixed`가 실제로 기록되는지. 현재 저장소의 SQLite 스키마는 `cart | buy_now`만 허용한다.
4. **28일 미구매 기준의 실제 uplift.** 쌍대 rollout은 조건부 재방문식(방문 기준) 평가라고 되어 있다. 주문 기준(`future_no_order_28d`)의 고객별 발송/미발송 확률을 받을 수 없으면, 실제 uplift와의 직접 비교는 못 하고 Qini로만 평가한다.
5. **`t0`를 가로지르는 세션.** 설계서도 별도 검증이 필요하다고 적었다. 그런 세션은 C30(종료 `< t0`)에서 빠지고 F04(시작 기준)에는 들어간다고 이해했다. 맞는지.
6. **학습용 실행.** 최종 학습에 쓸 실행(각 군 10,000명, 28일 규칙)을 디렉터리째 받을 수 있는지, 아니면 `SIM_ASSETS`(모형 파일)를 받아 직접 생성해야 하는지. 생성기는 기본 v10으로 가정한다.
7. **기준 시점 수.** 실행 하나에 `t0`가 하나다. 시간 순 검증이 필요하면 `t0`가 다른 실행이 추가로 필요하다.

## 3. 할 일

### 3.1 학습 데이터셋 (UC-01)

- `load_raw` → `build_dataset`으로 `customer_id, t0, treatment, KIT39, y` 생성.
- 대상은 적격 고객(`[t0 − 28일, t0)` 방문). y는 `[t0, t0 + 28일)` 주문 여부이고, 이 구간이 다 관측된 고객만 학습.
- T는 배정군 기준(ITT).
- 검증: `t0` 이후 기록을 지워도 X가 같은지, 시뮬레이터 `features.sqlite`·`no_purchase_labels`와 일치하는지, T/C 간 X 분포가 균형인지(SMD).

### 3.2 전처리와 분할

- train / valid / test를 고객 단위로 분할하고 T 비율을 유지.
- NULL은 그대로 두고 결측을 처리하는 트리 모델(scikit-learn `HistGradientBoosting`)을 base learner로 사용.

### 3.3 모델 (UC-02)

| 모델 | 방법 |
| --- | --- |
| T-learner | `μ1`(T=1), `μ0`(T=0)를 따로 학습. `τ = μ1(x) − μ0(x)` |
| X-learner | `μ0, μ1`로 가상 효과 `D0 = μ1(x) − Y`, `D1 = Y − μ0(x)`를 만들어 각각 회귀한 뒤 `e = 0.5`로 가중 평균 |
| H-learner | `f0, f1`을 `(1−λ)·결과 손실 + λ·(f1 − f0 − 가상 효과)²`로 함께 학습. λ는 검증 데이터로 선택 (`H_learner.pdf`) |

- 모델 코드는 X의 내용에 의존하지 않는다. v5 대응으로 바뀐 것은 입력 단계와 Feature 목록이다.
- UC-04가 `p1`, `p0`를 모두 요구하므로 세 모델 모두 `μ0, μ1`도 함께 저장.
- 하이퍼파라미터는 valid의 Qini 계수로 선택.

### 3.4 평가 (UC-03)

- test에서 Qini curve, Qini 계수, 상위 decile별 실제 uplift. 기준선은 무작위 타기팅과 ATE.
- 실제 uplift와의 직접 비교는 2절 4번 답에 따라 결정.
- 잠재 진단값은 입력에 넣지 않고 예측 세그먼트와의 일치도 점검에만 사용.

### 3.5 예측과 내보내기 (UC-04–06)

- `p1, p0, uplift_score = p1 − p0` 산출, `uplift_score`와 `p0`로 4개 세그먼트 분류(임계값은 valid에서 결정).
- CSV: `customer_id, t0, p1, p0, uplift_score, segment` + `model_version`, `dataset_id`.

## 현재 상태

- 구현됨: 원시 로그 읽기(`common/raw.py`), KIT39·정답·데이터셋(`common/features.py`), 세 모델, Qini 평가, 학습 스크립트(`train.py`), 테스트 12개 통과.
- 확인됨: 임시 원시 로그 300명에서 KIT39 39개가 현재 저장소의 시뮬레이터 Feature 코드(v2) 결과와 모두 일치.
- 시뮬레이터 실행 결과가 로컬에 없어 `common/synthetic.py`의 임시 원시 로그로만 실행했다. 성능 수치는 의미가 없다.
- 남은 일: 실제 실행 결과 연결과 2절 대조, 하이퍼파라미터 선택, 세그먼트 분류와 CSV 내보내기(UC-05, UC-06), `config/uplift.yaml`.
- 위험: v5의 1,000명 검증 실행에서 쿠폰 사용이 500건 중 13건이었다. 주문 기준 Y의 양성 비율과 효과 크기가 작을 수 있어 full 규모가 필요하다.
