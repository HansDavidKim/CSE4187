# CLV 전달 문서: 모델 파일·추론 API·입력 데이터·버전

전략·예산 최적화(`final_strategy`)와 대시보드가 CLV 예측을 같은 평가 고객에 대해 직접 만들 수 있도록 정리한 문서다.
설치·학습 방법과 모델 설명은 [readme.md](readme.md)에 있다.

## 0. 요약

| 요청 항목 | CLV가 제공하는 것 | 위치 |
| --- | --- | --- |
| 학습된 모델 파일 | 체크섬이 기록된 번들(모델 4종, registry, 고객 분할, 모델 카드) | [`artifacts/population-24m-n5000-seed42/`](artifacts/population-24m-n5000-seed42/) |
| 추론 API | `clv.api.predict_for_strategy`, `clv.api.load_model`, CLI `python -m clv strategy` | [api.py](api.py), [inference.py](inference.py) |
| 입력 데이터 | 시뮬레이터 `features/`(Feature Store + 주간 패널)와 `dataset/raw.sqlite`, 재생성 명령과 체크섬 | [7장](#7-입력-데이터와-재현) |
| 버전 정보 | `model_card.json`, `PredictionMetadata` 5항목, `producer_metadata` | [6장](#6-번들-구조와-버전-정보) |

- 예측 대상은 기준 시각부터 **52주(364일)** 동안의 고객별 결제액 합이다. 열 이름 `predicted_clv_12m`은 전략 모듈
  계약과의 호환 때문에 유지하는 이름일 뿐이며, 달력 12개월이 아니다.
- 최종 전략 평가는 CLV 학습에 쓰지 않은 고객으로 해야 한다. 전달한 번들은 seed 42 데이터셋으로 학습했으므로,
  평가용 Feature Store는 **다른 seed의 데이터셋**(예: seed 43)이 기본이다([3장](#3-평가-고객과-학습-고객)).

## 1. 예측 대상 계약

| 항목 | 값 |
| --- | --- |
| 기준 시각 `t0` | Feature Store의 `as_of_at`. 서울 월요일 00:00이어야 한다(시뮬레이터 실험 시각이 그렇다) |
| 입력 구간 | `[t0 - 13주, t0)`의 주간 집계. `t0` 이후 행은 쓰지 않는다 |
| 예측 구간 | `[t0, t0 + 52주)`. `prediction_end_at = as_of_at + 52주` |
| 예측값 | 고객별 결제액 합 = `gross_revenue - redemption_discount_cost`. 쿠폰 발송비는 빼지 않는다 |
| 값의 성질 | 0 이상의 유한한 금액(`nonnegative_amount`). 모든 모델이 음수 예측을 0으로 자른다 |
| 금액 단위 | 원천 최소 단위 그대로. 현재 데이터는 `currency=SOURCE_UNSPECIFIED`, `amount_unit=0.01`이며 KRW로 환산하지 않는다 |
| 정책 조건 | `observed_experiment_mix`. 학습 정답 구간에 학습용 실험의 쿠폰 발송(Treatment 50%)이 섞여 있다 |
| 대상 정의 ID | `clv_paid_amount_52w_v1`. 정의가 바뀌면 번호를 올린다 |

정책 조건 때문에 이 CLV는 "쿠폰을 보내지 않았을 때의 CLV"가 아니다. `Uplift × CLV` 우선순위 점수에서 쿠폰 효과를
두 번 세지 않는지는 팀 합의가 필요하다([9장](#9-제한과-합의가-필요한-점)).

## 2. 빠른 시작

### 전략 모듈 입력 batch

저장소 루트에서 실행한다. `simulator/src`와 `final_strategy`를 import할 수 있어야 한다.

```python
from clv.api import HANDOFF_BUNDLE, predict_for_strategy   # HANDOFF_BUNDLE = clv/artifacts/population-24m-n5000-seed42
from final_strategy.inputs import PredictionBatch, load_feature_store
from final_strategy.validation import PredictionMetadata

DATA = "results/simulator/main-84372a7/population-24m-n5000-seed43"   # 학습에 쓰지 않은 데이터셋

clv = predict_for_strategy(
    f"{DATA}/features",                   # feature_store.parquet과 weekly_panel.parquet이 있는 디렉터리
    ledger=f"{DATA}/dataset",             # raw.sqlite: Feature Store의 raw_database_sha256과 대조
    model_dir=HANDOFF_BUNDLE,             # 생략하면 results/clv(직접 학습한 번들)를 찾는다
)
batch = PredictionBatch(clv.frame, PredictionMetadata(**clv.metadata))
features = load_feature_store(f"{DATA}/features")
# integrate_inputs(features, {"churn": ..., "uplift": ..., "clv": batch})
```

- `clv.frame`은 Feature Store의 모든 `(dataset_id, customer_id, as_of_at)`을 같은 순서로 정확히 한 번씩 덮는다.
  고객을 빼거나 0으로 채우지 않는다.
- `clv.metadata`는 `PredictionMetadata`와 필드 이름이 같은 dict다. CLV는 `final_strategy`를 import하지 않는다.
- `ledger`를 생략해도 최종 모델(XGBoost)은 예측할 수 있다. 다만 원장과의 체크섬 대조는 빠지고,
  Feature Store manifest와의 대조만 남는다.

### 명령행과 파일 전달

```bash
python -m clv strategy \
  --features results/simulator/main-84372a7/population-24m-n5000-seed43/features \
  --ledger   results/simulator/main-84372a7/population-24m-n5000-seed43/dataset \
  --model-dir clv/artifacts/population-24m-n5000-seed42 \
  --output   results/clv_strategy_predictions.parquet
```

출력은 세 파일이다.

- `clv_strategy_predictions.parquet`: frame
- `clv_strategy_predictions_metadata.json`: 메타데이터와 체크섬
- `clv_strategy_predictions_audit.csv`: 감사용 표

다시 읽을 때는 `clv.api.load_strategy_predictions(path)`를 쓴다. 체크섬을 대조한 뒤 같은 `StrategyPredictions`를 돌려준다.

### 대시보드·분석용 고객별 예측

```python
from clv.api import HANDOFF_BUNDLE, load_model
model = load_model(HANDOFF_BUNDLE)
table = model.predict(f"{DATA}/features", reference_week="2025-01-06")   # 이력이 있는 모든 고객
top = model.predict(f"{DATA}/features", reference_week="2025-01-06", customers=["U0000001", "U0000002"])
```

`customers`를 주면 그 고객만 예측한다. 상위 20% 표시와 순위는 반환되는 고객들 안에서 다시 정한다.

## 3. 평가 고객과 학습 고객

요구사항([docs/request.md](../docs/request.md))상 최종 전략 평가는 모델 학습·튜닝에 쓰지 않은 고객으로 해야 한다.

전달한 번들은 seed 42 데이터셋의 실험 고객 5,000명 전원을 기준 시각 2025-01-06에서 고객 단위로 나눠 학습했다.

| 분할 | 인원 | 역할 |
| --- | --- | --- |
| train | 3,000 | 적합 |
| validation | 1,000 | 모델 선택 |
| test | 1,000 | 보고 |

`predict_for_strategy`는 고객마다 `clv_training_role`을 계산한다.

| `clv_training_role` | 의미 | 기본 동작 |
| --- | --- | --- |
| `train` | CLV 적합에 쓴 고객 | **오류** |
| `validation` | CLV 모델 선택(하이퍼파라미터·최종 모델)에 쓴 고객 | **오류** |
| `test` | 학습 데이터셋의 보고용 고객. 적합·선택에는 쓰지 않음 | 허용 |
| `unused` | 이 모델이 본 적 없는 고객(다른 데이터셋이면 전원) | 허용 |

- 같은 데이터셋(seed 42)의 Feature Store를 그대로 넣으면 4,000명이 `train`·`validation`이므로 오류가 난다.
- 권장 1: 전략 평가용으로 다른 seed의 데이터셋을 생성해 쓴다. 고객 ID(`U0000001` 등)는 데이터셋마다 다시 매겨지지만
  `dataset_id`가 다르므로 학습 고객과 섞이지 않는다.
- 권장 2: seed 42를 꼭 써야 하면 Test 고객 1,000명만 고른다. 고른 표는 manifest 행 수와 달라 `load_feature_store`를
  다시 통과하지 못하므로, 전략 모듈도 같은 고객으로 Feature Store를 걸러 써야 한다. 다른 모델(Churn·Uplift)의 학습
  고객과 겹치지 않는지는 각 모델이 따로 확인해야 한다.

  ```python
  import pandas as pd
  from clv.api import HANDOFF_BUNDLE, load_model, predict_for_strategy
  FEATURES = "results/simulator/main-84372a7/population-24m-n5000-seed42/features"
  store = pd.read_parquet(f"{FEATURES}/feature_store.parquet")
  roles = load_model(HANDOFF_BUNDLE).training_roles(store["dataset_id"].iloc[0], store["customer_id"])
  test_only = store.loc[roles.eq("test").to_numpy()]
  clv = predict_for_strategy(test_only, weekly=FEATURES, model_dir=HANDOFF_BUNDLE)   # DataFrame이면 weekly 필수
  ```
- `allow_training_customers=True`(CLI `--allow-training-customers`)는 개발 중 배선 확인용이다. 결과를 만들되
  `producer_metadata["warnings"]`에 겹친 인원을 남긴다.
- 한계: 학습 고객 판정은 `dataset_id`가 같은지로 한다. seed 42로 설정만 바꿔 다시 만든 데이터셋은 `dataset_id`가 달라
  모두 `unused`가 되지만 학습 고객을 다시 만든 것일 수 있다. 이 경우 입력 seed가 학습 seed와 같다는 경고를 남기므로,
  평가용 데이터셋은 seed부터 다르게 만든다.

## 4. API

모두 `clv.api`에서 import한다.

| 함수·클래스 | 용도 | 반환 |
| --- | --- | --- |
| `predict_for_strategy(features, weekly=None, ledger=None, model_dir="results/clv", model=None, output=None, allow_training_customers=False)` | 전략 모듈 입력 batch | `StrategyPredictions` |
| `load_strategy_predictions(path)` | 저장한 batch를 체크섬 대조 후 읽기 | `StrategyPredictions` |
| `load_model(model_dir="results/clv", model=None)` | 번들을 체크섬 대조 후 불러오기 | `CLVModel` |
| `model_info(model_dir, model=None)` | 모델·버전·학습 데이터·Test 지표 요약 | dict |
| `predict_clv(weekly, ledger=None, model_dir=..., reference_week=None, model=None, output=None, customers=None)` | 고객별 예측(CSV 선택) | DataFrame |
| `load_clv_predictions(path)` | `results/clv_predictions.csv`를 자료형 맞춰 읽기 | DataFrame |
| `train_clv`, `evaluate_clv`, `run_clv` | 학습·평가·일괄 실행 | dict |

인자 설명:

- `features`: `feature_store.parquet`, 그 디렉터리, 또는 키 8열(`STRATEGY_KEY_COLUMNS`)이 있는 DataFrame.
- `weekly`: 생략하면 `features` 디렉터리의 `weekly_panel.parquet`을 쓴다. `features`가 DataFrame이면 필수다.
- `model`: 생략하면 registry의 최종 모델(현재 `xgboost`)이다. 다른 이름은 `baseline`, `bgnbd_gg`, `xgboost_two_part`이다.
  `bgnbd_gg`만 예측에 `ledger`가 필요하다.

`CLVModel`의 주요 속성·메서드는 다음과 같다.

- 속성: `name`, `model_version`, `metadata`, `needs_ledger`, `training_reference_week`, `card`(모델 카드), `registry`
- 메서드: `predict(...)`, `predict_for_strategy(...)`, `info()`, `training_roles(dataset_id, customer_ids)`

### 거절하는 입력

모두 `ValueError`(번들 파일 없음은 `FileNotFoundError`)이며, 조용히 보정하지 않는다.

| 상황 | 이유 |
| --- | --- |
| 모델 카드가 없거나 번들 파일이 학습 후 바뀜 | 검증되지 않은 모델 사용 방지 |
| 학습 때와 `currency`·`amount_unit`이 다름 | 금액 단위가 섞이지 않게 함 |
| Feature Store와 주간 패널·원장의 `dataset_id` 또는 `raw_database_sha256`이 다름 | 다른 데이터셋을 섞지 않음 |
| `weekly_panel.parquet`·`feature_store.parquet`이 각 manifest 체크섬과 다름 | 출처 보장 |
| `as_of_at`이 서울 월요일 00:00이 아님, 시간대 없음 | 주 단위 입력과 기준 시각이 어긋남 |
| 키 중복, 한 batch에 여러 데이터셋·통화·단위 | 전략 모듈 계약과 같음 |
| 기준 시각 직전 13주에 관측 주가 하나도 없는 고객 | 이력을 지어내지 않음 |
| `train`·`validation` 고객 포함(기본) | 3장 |

## 5. 출력 형식

### 전략 batch frame (`StrategyPredictions.frame`)

열 순서까지 고정이며, 이 밖의 열은 없다.

| 열 | 자료형 | 내용 |
| --- | --- | --- |
| `dataset_id` | str | Feature Store 값 그대로 |
| `customer_id` | str | Feature Store 값 그대로 |
| `as_of_at` | `datetime64[ns, UTC]` | 기준 시각 |
| `reference_week` | str | Feature Store 값 그대로(서울 월요일 날짜) |
| `snapshot_id` | str | Feature Store 값 그대로 |
| `currency`, `amount_unit` | str | Feature Store 값 그대로 |
| `raw_database_sha256` | str | Feature Store 값 그대로 |
| `input_cutoff_at` | `datetime64[ns, UTC]` | 입력 관측 상한(배타적). `as_of_at`과 같다 |
| `prediction_end_at` | `datetime64[ns, UTC]` | `as_of_at + 52주` |
| `predicted_clv_12m` | float64 | 52주 예상 결제액(원천 최소 단위, 0 이상) |

### `metadata`

| 키 | 값(현재 번들) |
| --- | --- |
| `model_version` | `clv-0.2.0+xgboost.<12자>`. 번들마다 다르며 `model_card.json`에 있다 |
| `target_definition_id` | `clv_paid_amount_52w_v1` |
| `horizon_value`, `horizon_unit` | `52`, `"weeks"` |
| `value_semantics` | `"nonnegative_amount"` |

### `audit`

키 3열(`dataset_id`, `customer_id`, `as_of_at`)에 다음 열이 붙는다. 두 파트 모델이면 구매 확률과 구매 시 기대 결제액도 있다.

- `history_complete`: 13주를 모두 관측했는지
- `history_weeks_13w`: 관측한 주 수(부분 관측 주는 비율)
- `clv_training_role`

13주 이력이 일부만 있는 고객도 예측한다. 이런 고객은 seed 43에서 69%다.

### `producer_metadata`

| 키 | 내용 |
| --- | --- |
| `model_name`, `model_version`, `bundle` | 사용한 모델과 번들 위치 |
| `code`, `inference_code` | 학습 당시와 이번 추론 때의 `clv_version`, `git_commit`, `clv_dirty`. 버전이 다르면 경고 |
| `training_environment`, `inference_environment` | 학습 때와 추론 때의 라이브러리 버전. XGBoost 버전이 다르면 경고 |
| `target`, `policy_condition` | 1장 계약 |
| `training` | 학습 기준시점, 학습 `dataset_id`·`raw_database_sha256`, 학습 정답 구간 끝 |
| `inputs` | 예측 입력 `dataset_id`, `raw_database_sha256`, 대조에 성공한 출처, 주간 파일 경로·체크섬, 원장 경로 |
| `snapshots` | 기준 시각별 `input_cutoff_at`, `prediction_end_at`, 고객 수 |
| `customers`, `history_incomplete_customers` | 인원 |
| `training_overlap` | 같은 데이터셋 여부, 역할별 인원, 허용 여부 |
| `warnings`, `created_at` | 경고와 생성 시각 |

`raw_database_sha256_verified_by`에는 파일 해시까지 확인한 출처만 들어간다. 값은 `weekly_manifest`(features manifest)와
`ledger_manifest`(raw.sqlite manifest)다. `verify_checksum=False`(CLI `--skip-checksum`)이면 이 목록은 비고,
manifest 값이 일치한 출처만 `raw_database_sha256_matched`에 남는다. 이때 `checksums_skipped=true`와 경고도 함께 남는다.

### 고객별 예측 CSV (`results/clv_predictions.csv`)

앞 4열은 계약 열이다.

- `customer_id`
- `reference_week`(서울 시각)
- `predicted_clv_12m`
- `is_high_value_customer`(상위 20%, `ceil(0.2 × N)`명, 동점은 ID 오름차순)

뒤에 보조 열이 붙는다.

- `clv_rank`, `model_name`, `model_version`
- `horizon_weeks`(52), `prediction_end_at`
- `history_complete`, `history_weeks_13w`
- `currency`, `amount_unit`, `dataset_id`
- 두 파트 모델이면 `purchase_probability_52w`, `expected_amount_if_purchase`

## 6. 번들 구조와 버전 정보

```text
artifacts/population-24m-n5000-seed42/
  model_card.json      버전·코드·라이브러리·학습 데이터 출처·파일 체크섬
  registry.json        기준시점, 계약, 설정, 모델 상태·진단, 최종 모델, 지표
  splits.csv           customer_id -> train/validation/test (학습 고객 역할 판정에 사용)
  models/              baseline.json, bgnbd_gg.json, xgboost/, xgboost_two_part/
  reports/evaluation.json  Validation·Test 지표(이력 완전/미완전 구간 포함)
```

`model_card.json`의 주요 항목은 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| `model_versions` | 모델별 버전. `clv-<CLV 버전>+<모델>.<registry·splits·모델 파일 체크섬 요약>`이므로 파일이 하나라도 바뀌면 달라진다 |
| `final_model` | Validation RMSE가 가장 낮은 모델 |
| `target`, `policy_condition` | 1장 계약 |
| `code` | 학습 당시 CLV 버전, git 커밋, `clv/` 미커밋 변경 여부 |
| `environment` | Python·numpy·pandas·scipy·xgboost·pyarrow 버전 |
| `training_data` | 학습 기준시점, `dataset_id`, `raw_database_sha256`, 주간 파일 체크섬, 시뮬레이터 manifest 일부(seed, months, 버전, 설정·코드 체크섬, 실험 시각, 최소 이력 일수 등), 분할 인원 |
| `files_sha256` | registry, splits, 모델 파일의 SHA256. `load_model`이 불러올 때마다 대조한다 |

`artifacts/.gitattributes`가 줄바꿈 변환을 끈다. 그래서 Windows에서 받아도 체크섬이 유지된다.
`python -m clv info --model-dir <번들>`로 요약을 볼 수 있다.

## 7. 입력 데이터와 재현

입력 데이터(raw.sqlite 약 120MB, features 약 240MB)는 저장소에 올리지 않는다. 같은 명령으로 다시 만들고 체크섬으로 확인한다.
루트 `.gitignore`에는 아직 `results/`가 없다(팀 공용 파일이라 CLV 작업에서 바꾸지 않았다). `git add .`로 올리지 않도록 주의하거나
각자 `.git/info/exclude`에 `/results/`를 추가한다.
이 문서의 수치는 `main` 84372a7의 시뮬레이터 코드로 만들었다(`population.yaml`, 외부 자산 불필요).

```bash
cd simulator
for SEED in 42 43; do
  OUT=../results/simulator/main-84372a7/population-24m-n5000-seed$SEED
  PYTHONPATH=src python -m cli v4 generate --config src/simulate/v4/config/population.yaml --mode full \
    --n-customers 5000 --months 24 --seed $SEED --output $OUT/dataset          # 약 3분
  PYTHONPATH=src python -m cli v4 validate --data $OUT/dataset/raw.sqlite
  PYTHONPATH=src python -m cli v4 features --data $OUT/dataset --output $OUT/features   # 약 1분
done
```

| seed | 용도 | `raw_database_sha256` | `dataset_id` 끝부분 |
| --- | --- | --- | --- |
| 42 | CLV 학습(번들) | `8c35f7639425fb8cfdffee21a608b1624db8cbd15ea222e0123a37bc2c202579` | `...-seed42-n5000-36a6b3788c9bd6a2-28d` |
| 43 | 전략 평가 예시(학습 미사용) | `c8a322428d84c181fcbf23be294265c5bd92b1cce7b86dd4f72de749206d1710` | `...-seed43-n5000-52b60a4299c7a349-28d` |

생성 데이터의 `raw.manifest.json`에 있는 `database_sha256`이 위 값과 같아야 한다. 이 기계에서는 seed 42의 raw
SQLite 체크섬이 시뮬레이터 커밋 8b5c03c와 84372a7에서 같았다. 다른 환경에서 달라지면 번들은 그대로 쓰고,
그 데이터셋의 체크섬이 Feature Store·원장 사이에서만 일치하면 된다.

공통 조건은 다음과 같다.

- 실험 기준 시각: `as_of_at = 2025-01-05T15:00:00Z`(서울 2025-01-06 월요일 00:00)
- 관측 기간: 2024-01-01 ~ 2026-01-05T15:00Z(52주 정답 구간 전체 포함)
- 고객 수: 데이터셋마다 5,000명. 실험 적격: 배정 직전 28일 내 세션, `min_customer_history_days=0`

번들을 다시 학습하려면 다음 명령을 쓴다(약 10초).

```bash
python -m clv train --weekly results/simulator/main-84372a7/population-24m-n5000-seed42/features \
  --ledger results/simulator/main-84372a7/population-24m-n5000-seed42/dataset --output results/clv
```

다시 학습하면 `model_version`이 바뀐다. 전략 결과에는 사용한 `model_version`을 함께 기록한다.

## 8. 현재 번들 성능

모두 실제로 실행한 결과다. 금액은 원천 최소 단위이며, 실제 52주 결제액은 고객 82%가 0원이다.

**seed 42 Test 1,000명**(학습 데이터셋의 보고용 고객, 평균 실제값 2,557)

| 모델 | MAE | RMSE | R² | Spearman | 상위 20% 일치율 | 상위 20% 포착 가치 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 7,136 | 26,617 | -7.52 | 0.32 | 0.46 | 0.48 |
| BG/NBD + GG | 3,080 | 9,160 | -0.01 | 0.32 | 0.46 | 0.48 |
| **XGBoost (최종)** | 3,512 | 8,935 | 0.04 | 0.36 | 0.47 | 0.49 |
| 두 파트 XGBoost | 3,529 | 8,945 | 0.04 | 0.34 | 0.43 | 0.44 |

**seed 43 전체 5,000명**(학습에 쓰지 않은 데이터셋, 평균 실제값 2,079)

| 모델 | MAE | RMSE | R² | Spearman | 상위 20% 일치율 | 상위 20% 포착 가치 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 11,422 | 74,865 | -86.5 | 0.33 | 0.49 | 0.49 |
| BG/NBD + GG | 2,558 | 8,047 | -0.01 | 0.33 | 0.49 | 0.49 |
| **XGBoost (최종)** | 3,060 | 7,748 | 0.06 | 0.34 | 0.43 | 0.55 |
| 두 파트 XGBoost | 3,086 | 7,760 | 0.06 | 0.34 | 0.41 | 0.51 |

해석:

- 최종 모델(Validation RMSE 기준)인 XGBoost는 학습에 쓰지 않은 데이터셋에서도 RMSE가 가장 낮다. 상위 20% 고객이
  가진 실제 가치의 55%를 포착한다.
- 고객 수로 센 상위 20% 일치율과 MAE는 BG/NBD가 더 좋다. 우선순위를 "누가 상위인가"로 쓰면 BG/NBD도 후보다
  (`model="bgnbd_gg"`, 원장 필요).
- R²는 0.06 이하다. 13주 이력만으로 52주 금액을 맞추는 정확도는 낮으므로, 점수는 순위·규모 신호로 해석한다.
- seed 43 이력 완전 고객 1,546명은 XGBoost MAE 4,880 / RMSE 9,596이다. 미완전 고객 3,454명은 2,246 / 6,759다.

## 9. 제한과 합의가 필요한 점

- 한 batch는 한 데이터셋이다. 기준 시각은 서울 월요일 00:00만 지원한다(현재 시뮬레이터 실험 시각과 같다).
- 학습 기준시점(2025-01-06)과 다른 시각에도 예측할 수 있다. 하지만 학습 분포와 달라질 수 있어 성능은 위 표와 다를 수 있다.
- 정책 조건이 `observed_experiment_mix`다. Uplift 점수와 곱할 때 쿠폰 효과 중복 반영 여부는 팀 합의가 필요하다.
- `is_high_value_customer`는 CLV 내부 이름이다. 전략 모듈의 Segmentation은 상위 비율을 스스로 다시 계산한다.
- `currency=SOURCE_UNSPECIFIED`는 KRW가 아니다. 예산(`--budget 50000000`)과 비교하려면 단위 합의가 먼저다.
