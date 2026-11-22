# CLV (Customer Lifetime Value)

시뮬레이터 v5 출력의 **과거 13주**로 **미래 52주 실결제금액(CLV)** 을 예측한다.
비교 모델은 과거 평균 Baseline, BG/NBD + Gamma-Gamma, XGBoost, 두 파트 XGBoost 4종이고,
최종 결과는 `results/clv_predictions.csv`와 Python API(`clv.api`)로 전략·예산 모듈에 넘긴다.
PyTorch는 쓰지 않는다.

> **다른 모듈 담당자는 [api.md](api.md)부터 본다.** 학습된 번들(`artifacts/`), 전략 모듈 입력 batch를 만드는
> `predict_for_strategy`, 버전·입력 데이터 정보, 학습 고객 중복 규칙이 정리돼 있다.
> 예측 기간은 52주(364일)이고, `predicted_clv_12m`은 호환을 위해 유지하는 열 이름이다(달력 12개월 아님).

## 구성 (WBS)

| WBS | 내용 | 파일 |
|---|---|---|
| 4.1 | 시뮬레이터 Adapter, Parquet/CSV 로딩, 스키마·타입·중복·결측·음수·금액식·통화·관측 검증, 원장↔주간 대조 | `adapter.py`, `validation.py`, `schema.py` |
| 4.2 | 65주(13+52) 구간, 적격 기준시점 탐색, `actual_clv_12m`, 관측 부족 처리, 고객 단위 분할, 누수 방지, 원장 대조 | `dataset.py` |
| 4.3 | 4주·13주 RFM·행동 Feature 17개 | `features.py` |
| 4.4 | 평균 Baseline, BG/NBD + Gamma-Gamma(주문 원장 기반, scipy 최우추정) | `baseline.py`, `probabilistic.py` |
| 4.5 | XGBoost 회귀, 두 파트 XGBoost(구매 확률 × 구매 시 결제액), Validation 그리드 선택, seed 고정, 저장·재로딩 | `xgboost_model.py` |
| 4.6 | MAE·RMSE·편향·분포·Top-N·10분위·상위 20% 일치율 | `evaluation.py` |
| 4.7 | 학습/평가/추론 분리, CSV 출력, 외부 API, CLI, 모델 카드·체크섬, 전략 모듈 입력 batch | `pipeline.py`, `api.py`, `cli.py`, `bundle.py`, `inference.py` |

설정은 `config.py`(`CLVConfig`), JSON 출력은 `io.py`, 테스트는 `tests/`.

## 설치

```bash
pip install -r clv/requirements.txt      # numpy, pandas 2.x, pyarrow, scipy, xgboost, PyYAML, pytest
```

pandas 3.x에서는 시뮬레이터 `fit_shopping`이 copy-on-write로 실패하므로 공용 환경은 pandas 2.x를 쓴다.
xgboost가 없으면 두 XGBoost 모델만 `skipped`로 기록하고 나머지는 계속 실행한다.

## 실행 (저장소 루트)

입력은 시뮬레이터 산출물 두 가지다.

* `--weekly`: `ecommerce-sim v4 features` 출력 디렉터리(또는 `weekly_panel.parquet`, CSV)
* `--ledger`: `ecommerce-sim v4 generate` 출력의 `raw.sqlite`(또는 데이터셋 디렉터리, 주문 CSV/Parquet)

```bash
python -m clv validate --weekly artifacts/features --ledger artifacts/small/raw.sqlite
python -m clv cutoffs  --weekly artifacts/features                      # 적격 65주 구간 확인
python -m clv train    --weekly artifacts/features --ledger artifacts/small/raw.sqlite --output results/clv
python -m clv evaluate --weekly artifacts/features --ledger artifacts/small/raw.sqlite --model-dir results/clv
python -m clv predict  --weekly artifacts/features --ledger artifacts/small/raw.sqlite --model-dir results/clv \
                       --output results/clv_predictions.csv [--reference-week 2025-01-06] [--model xgboost]
python -m clv run      --weekly ... --ledger ...                        # train + evaluate + predict
python -m clv info     --model-dir results/clv                          # 모델 버전·학습 데이터·코드 버전
python -m clv strategy --features <평가용 features 디렉터리> --ledger <평가용 dataset> --model-dir results/clv \
                       --output results/clv_strategy_predictions.parquet # 전략 모듈 입력 batch(api.md 참고)
```

`train --cutoff`로 학습 기준시점을, `--config`로 JSON/YAML 설정(`clv:` 섹션 허용)을 지정한다.

```python
from clv.api import load_clv_predictions, predict_clv, train_clv

clv = load_clv_predictions("results/clv_predictions.csv")   # 전략·예산 모듈용 DataFrame
clv = predict_clv("artifacts/features", ledger="artifacts/small/raw.sqlite", model_dir="results/clv",
                  reference_week="2025-01-06")              # 정답 없는 시점도 예측
```

## 입력 계약과 Adapter

| CLV 이름 | 시뮬레이터 `weekly_panel` |
|---|---|
| `customer_id` | `customer_id` |
| `week` | `reference_week` (서울 월요일 00:00, 주간 행의 시작) |
| `sessions`, `purchases` | 동일 |
| `gross_revenue` | `gross_revenue_minor` |
| `redemption_discount_cost` | `redemption_discount_cost_minor` |
| `coupons` | `coupons_sent` |
| `coupon_action` (보조) | `coupon_actions` (주간 발송 결정 건수, 고객별 Boolean 아님) |

* `paid_amount = gross_revenue - redemption_discount_cost`. 시뮬레이터 `net_revenue_minor`와 주마다 일치하는지 확인한다.
* `marketing_send_cost_minor`(쿠폰 발송비)는 차감하지 않고 읽지도 않는다.
* 금액은 원천 최소 단위 그대로다. `currency`·`amount_unit`을 그대로 전달하며, `SOURCE_UNSPECIFIED`는 KRW로 바꾸지 않고 경고만 남긴다. 통화가 없으면 오류다.
* `observed_week`가 False인 주(가입 전·관측 밖)와 행이 없는 주는 **미관측**이다. 0으로 채우지 않는다. `partial_coverage` 주는 관측은 됐지만 완전 관측이 아니다.
* CLV 스키마 CSV에 관측 컬럼이 없으면 값이 있는 행은 완전 관측, 없는 행은 미관측으로 보고 그 가정을 기록한다. 시간대 없는 날짜는 서울 시각으로 읽는다.
* BG/NBD 입력은 `raw.sqlite`의 `orders`와 구매 `events.event_at`을 조인한 실제 주문이다(manifest 체크섬 확인). 주간 집계에서 주문을 복원하지 않는다.

## 기간 정의 (WBS 4.2)

* 기준시점 `t0`는 서울 월요일 00:00이다. 출력 `reference_week`는 **예측 기준시점 t0**이며 시뮬레이터 주간 행의 `reference_week`(그 주의 시작)와 다른 개념이다.
* Feature: `[t0 - 13주, t0)`, 정답: `[t0, t0 + 52주)`의 실결제금액 합 `actual_clv_12m`.
* 정답은 52주가 모두 완전 관측일 때만 만든다. 기본 학습 대상은 정답이 있는 고객 전체다(`require_complete_history=False`). 13주 입력이 모두 완전 관측인 고객만 학습하려면 `--config clv/configs/complete_history_only.json`을 쓴다.
* 시뮬레이터는 월 단위로 생성하므로 `cutoffs`로 실제 데이터에서 13 + 52주가 들어가는 `t0`를 찾는다. 기본 학습 시점은 적격 고객이 가장 많은 `t0`(동률이면 최신)다. 처음·마지막 주는 부분 관측이어서 최소 67개 달력 주가 필요하다. 24개월 생성(`months>=24`)이면 실험 시점 `as_of_at`도 적격이다.
* 시뮬레이터 `future_revenue_12m`(달력 12개월, Control만 제공)은 쓰지 않는다.
* 정답이 있는 모든 고객을 고객 단위로 Train/Validation/Test(기본 0.6/0.2/0.2, seed 42)로 나눈다. 한 고객은 한 분할에만 속한다. 분할이 `require_complete_history`와 무관하므로 두 설정을 같은 Test 고객으로 비교할 수 있다. 학습·모델 선택은 그중 학습 대상 고객만 쓴다.
* 정답은 원장 합계와 고객별로 대조하고, 다르면 오류를 낸다.

## Feature (WBS 4.3)

필수 `purchase_count_4w/13w`, `session_count_4w/13w`, `net_spend_4w/13w`, `weeks_since_last_purchase`;
권장 `purchase_active_weeks_13w`, `zero_purchase_streak`, `purchase_trend`, `spend_trend`, `purchase_per_session_13w`,
`weeks_since_last_session`, `session_active_weeks_13w`, `session_trend`; 보조 `avg_paid_per_purchase_13w`, `coupon_sent_13w`.
순서는 `schema.PREDICTOR_FEATURES`, 자료형은 float64다.

* 관측된 주만 합산한다. 관측됐지만 활동이 없으면 0이다.
* `weeks_since_last_*`: 직전 주 활동이면 1, 13주 안에 활동이 없으면 14.
* `zero_purchase_streak`: 최근부터 연속된 관측 무구매 주 수(구매 주나 미관측 주에서 멈춤).
* 추세 = 최근 4주 주당 평균 - 이전 9주 주당 평균. 평균의 분모는 관측 주 수이고, 한쪽이 전혀 관측되지 않았으면 NaN이다.
* 분모가 0인 비율(세션 없음, 구매 없음)은 0이 아니라 NaN이다. XGBoost의 결측 분기가 처리한다.
* 제외: 고객 ID, 절대 주차, 정답, `gap_*`, 잠재 상태, 실험 배정·사후 결과, t0 이후 기록(`schema.check_predictors`가 막는다).
* KIT39와 별개로 CLV가 직접 계산한다.

## 모델 (WBS 4.4\~4.5)

* **Baseline**: `(net_spend_13w / 13) * 52`. 부분 이력 고객(추론만 해당)은 13 대신 관측 주 수로 나눈다.
* **BG/NBD + Gamma-Gamma**: 거래 단위는 서울 날짜별 주문 묶음(Lifetimes 관례, `transaction_period="order"` 선택 가능), 시간 단위는 주, 52주 비할인 합계다. 보정 기간은 기본적으로 다른 모델과 같은 13주(`calibration="feature_window"`)이고 `full_history`도 선택할 수 있다. Train 고객으로만 적합한다.
  * 적용 조건(`registry.json → model_details.bgnbd_gg`의 `checks`·`rejections`·`warnings`, `probabilistic.applicability`)
    * 필수(위반 시 `rejected`로 기록하고 저장·평가·최종 선택에서 제외): 두 최적화 수렴, 모수가 유한한 양수, `q>1`(Gamma-Gamma 평균 유한), 수치 오류 고객 비율 ≤ 1%.
    * 경고만: `a+b > 1000`(이탈 확률 Beta가 한 점 `a/(a+b)`로 무너져 이탈 이질성 미식별), |corr(frequency, monetary)| > 0.3(독립 가정 위반), 반복 구매 고객 < 30명.
    * 기록만: `a>1`(`a≤1`이면 생애 거래 수 평균이 발산하지만 52주 조건부 기대값은 유한하며 Monte Carlo로 확인).
    * 최소 적합 인원(구매 고객 10명, 반복 구매 고객 5명)이 부족하면 `failed`로 기록하고 평가 결과를 만들지 않는다.
  * 벌점 민감도(`reports/bgnbd_sensitivity.csv`): `bgnbd_sensitivity_penalizers`(기본 0, 0.001, 0.01, 0.1)마다 Train 고객으로 다시 적합해 모수, `a/(a+b)`, `a+b`, Validation MAE·RMSE, 선택 모델과의 예측 순위상관을 남긴다. 보고용이며 선택 모델은 바꾸지 않는다.
  * 보정 기간 구매가 없는 고객은 BG/NBD가 정의되지 않는다. `no_purchase_fallback="zero"`(기본) 또는 `train_mean`(같은 처지 Train 고객의 평균 정답)을 쓰고 `status=no_purchase_in_calibration`으로 표시한다.
  * 반복 구매 0회 고객은 Gamma-Gamma 모집단 평균 `p·v/(q−1)`을 받는다.
  * 13주 보정 기간이면 T가 13주 이하로 잘린다(좌측 절단). 생성 과정의 `a`, `b`는 개별 식별이 약하고 `a/(a+b)`는 잘 식별된다(테스트로 확인).
  * Lifetimes 0.11.3과 모수·기대 구매수가 2% 이내로 일치하는지 테스트한다(설치된 경우).
* **XGBoost**: `objective`(squarederror/tweedie)·`max_depth`·`min_child_weight` 그리드를 Validation RMSE로 고르고(조기 종료), 고른 트리 수로 Train에서 다시 학습한다. seed와 `n_jobs=1`을 고정해 같은 입력이면 같은 예측이 나온다. Feature 이름·순서·자료형은 메타데이터와 대조한다. 음수 예측은 0으로 자르고 그 건수를 기록한다.
* **두 파트 XGBoost**(`xgboost_two_part`): CLV = P(52주 안에 구매) × E[52주 결제액 | 구매].
  * 1파트: 학습 고객 전체로 구매 여부(`actual_clv_12m > 0`)를 분류한다(`binary:logistic`, Validation 로그 손실로 선택).
  * 2파트: 구매한 학습 고객만으로 결제액을 회귀한다(`squarederror`/`gamma`, 구매한 Validation 고객의 RMSE로 선택).
  * 예측 변수는 XGBoost와 같은 17개다. 학습 구매 고객이 20명 미만이면 `failed`로 기록한다.
  * 예측 CSV에 `purchase_probability_52w`, `expected_amount_if_purchase` 보조 열을 붙이고, 1파트의 AUC·로그 손실·Brier 점수를 `evaluation.json → two_part_classifier`에 남긴다.
* 최종 모델(`final_model="auto"`)은 4개 모델 중 Validation RMSE가 가장 낮은 모델이다. Test는 보고에만 쓴다.

## 평가 (WBS 4.6)

모든 모델을 같은 Test 고객, 같은 52주 정답으로 평가한다(`reports/evaluation.json`).
MAE, RMSE, 편향, 총액, R², Spearman, 예측·실제 분포 분위수, 10분위별 실제 대 예측(`deciles_test.csv`),
모델별 Top-20(`top20_test.csv`), 상위 20% 일치율(같은 k: |예측 ∩ 실제| / k)과 포착 가치 비율을 낸다.
Treatment/Control별 지표와 Baseline 대비 MAE·RMSE 비율(`vs_baseline`, 1 미만이면 Baseline보다 나음)도 함께 낸다.
같은 분할 안에서 13주 이력 완전/미완전 고객을 따로 평가한 결과는 `evaluation.json → by_history`와 `registry.json → test_metrics_by_history`에 있다. 실행하지 못했거나 제외된 모델은 평가에서 빠지고 사유만 남는다.

## 출력 (WBS 4.7)

`results/clv_predictions.csv` (앞 4개 열이 계약 열)

| 열 | 의미 |
|---|---|
| `customer_id` | 고객 ID |
| `reference_week` | 예측 기준시점 t0 (ISO, +09:00) |
| `predicted_clv_12m` | 향후 52주 예상 실결제금액(원천 최소 단위) |
| `is_high_value_customer` | 예측 상위 20% 여부. **CLV 내부 이름이며 팀 합의 필요** |
| `clv_rank`, `model_name`, `model_version`, `horizon_weeks`, `prediction_end_at`, `history_complete`, `history_weeks_13w`, `currency`, `amount_unit`, `dataset_id` | 보조 |

상위 20%는 `ceil(0.2·N)`명이고, 동점은 `customer_id` 오름차순으로 정한다. 예측에는 정답이 필요 없다.
13주 이력이 부족한 고객도 예측하되 `history_complete=False`로 표시한다. 기본 설정은 이런 고객도 학습에 포함한다.
기본 예측 시점은 패널의 실험 `as_of_at`이 하나일 때 그 값, 아니면 마지막 완전 관측 주의 다음 월요일이다.

학습 산출물(`results/clv/`): `model_card.json`(모델 버전·코드·라이브러리 버전·학습 데이터 출처·파일 체크섬),
`registry.json`(기준시점·계약·설정·모델 상태·최종 모델·지표·경고),
`splits.csv`(분할, `history_complete`, `training_population`), `models/`(baseline.json, bgnbd_gg.json, xgboost/, xgboost_two_part/),
`reports/`(validation.json, eligible_cutoffs.csv, evaluation.json, bgnbd_sensitivity.csv, deciles_test.csv,
top20_test.csv, labelled_predictions.csv).

## 테스트

```bash
python -m pytest clv/tests -q
```

작은 합성 Fixture(시뮬레이터 스키마)와 시뮬레이터 저장 코드 통합 테스트(`test_simulator_integration.py`)가 있다.
통합 테스트는 시뮬레이터 동봉 설정(`simulator/src/simulate/v4/config/population.yaml`, `source-shopping-v10`,
외부 자산 불필요)으로 60명·24개월을 생성해 엔진 → `raw.sqlite` → `features` → CLV 전 과정을 실행한다.
표본이 작아 출력 형식·금액·관측 계약만 검증하며, 모델 성능의 근거가 아니다.

실데이터 생성·학습 명령(5,000명·24개월, 학습 seed 42와 평가 seed 43)과 체크섬은 [api.md 7장](api.md#7-입력-데이터와-재현)에 있다.

## 협의·주의 사항

* 출력 열 이름 `is_high_value_customer`와 `reference_week`의 의미(예측 기준시점)는 전략·대시보드 담당과 합의해야 한다.
* 실험 시점 이후 52주 정답 구간에는 Treatment 고객의 쿠폰 효과가 들어간다. 학습에는 모두 포함하고, `treatment_in_label_window` 표시와 그룹별 지표를 남긴다. Uplift × CLV 결합 시 이중 계상 여부를 협의해야 한다.
* 통화가 `SOURCE_UNSPECIFIED`이면 금액은 원천 최소 단위(`amount_unit`)다. KRW 환산은 하지 않는다.
* `simulator.yaml`(원천 고객별 프로필, `data/models` 필요) 데이터에서는 아직 실행하지 않았다. 동봉 `population.yaml` 데이터와 분포·dataset_id가 다를 수 있다.
* 실데이터 결과(`population.yaml` 24개월, 5,000명, seed 42, 기준시점 2025-01-06). 시뮬레이터 `main` 84372a7로 다시 생성해도
  raw SQLite 체크섬이 같아 아래 수치가 그대로이며, 학습에 쓰지 않은 seed 43 데이터셋 결과는 [api.md 8장](api.md#8-현재-번들-성능)에 있다. 13주 이력이 완전한 고객은 1,601명(32%)이고,
  Test 고객 1,000명(완전 329, 미완전 671)은 두 설정에서 같다. MAE / RMSE / 상위 20% 일치율:

  | 학습 대상 | Baseline | BG/NBD+GG | XGBoost | 두 파트 XGBoost | 최종 모델 |
  |---|---|---|---|---|---|
  | 정답 고객 전체(기본) | 7,136 / 26,617 / 0.46 | 3,080 / 9,160 / 0.46 | 3,512 / 8,935 / 0.47 | 3,529 / 8,945 / 0.425 | XGBoost |
  | 이력 완전 고객만(Test 329명) | 10,118 / 23,244 / 0.39 | 6,286 / 13,484 / 0.33 | 6,498 / 12,977 / 0.26 | 6,286 / 13,035 / 0.32 | 두 파트 |

  같은 이력 구간끼리 비교하면 전체 학습 쪽 XGBoost MAE가 완전 6,498→6,039, 미완전 3,946→2,274로 낮았고 Validation에서도 같은 방향이었다.
  그래서 기본값을 `require_complete_history=False`로 바꿨다. 두 파트 XGBoost는 구매 여부 AUC 0.75(Test)지만 구매 고객
  518명으로 학습한 금액 파트가 트리 9개에서 멈춰 예측 폭이 좁다. 전체 학습에서는 기존 XGBoost와 비슷했다.
* BG/NBD의 `a`, `b`는 표본에 따라 크게 달라진다(완전 이력 학습의 이전 분할 약 19만·96만, 현재 분할 2.3·10.7, 전체 학습 22.6·102.1).
  `a/(a+b)`는 0.17\~0.18로 안정적이고, 전체 학습에서 벌점 0\~0.1의 Validation 예측 순위상관은 0.999 이상, RMSE는 7,588\~8,258이었다.
