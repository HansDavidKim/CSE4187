# Retention

시뮬레이터의 고객 행동으로 **향후 28일 미방문 확률**을 예측한다. 학습·배치 추론·독립 실행 평가·분포 모니터링을 제공하고, 전략 파트가 Uplift·CLV와 같은 고객·시점으로 결과를 연결할 수 있게 한다.

**2026-11-18 인계 상태** — 기본 파이프라인과 통합 실행은 검증했다. 전체 규모 테스트 AUC는 **0.7587**로 요구 기준 **0.78에 미달**한다. 마지막에 추가한 장기 이력 피처 후보는 기본 설정에서 꺼져 있으며, 추가 이후 전체 회귀 검증은 남아 있다.

## 1. 예측 문제와 데이터 계약

| 항목 | 현재 기준 |
| --- | --- |
| 적격 고객 | 기준시각 `t0` 직전 28일에 방문했고 `t0` 전에 가입한 고객 |
| 정답 | `[t0, t0+28일)`에 새 세션 시작이 없으면 이탈 1, 있으면 0 |
| 입력 경계 | `t0` 이전 기록. 진행 중 세션의 미래 종료 결과는 완료 세션 통계에서 제외 |
| 학습 입력 | 동일 실행의 공식 피처 디렉터리와 `raw.sqlite`·`raw.manifest.json` |
| 피처 기간 | 기본104주 자료는 최근4주·변화8주·간격13주(28/56/91일). 명시적 legacy 월간 자료는30/60/90일. 예측 기간은28일 |
| 실험 조건 | 자동1:1 배정·전체 미발송·고객 지정/인원·비율 고정 배정을 지원. 발송군은 기준일 쿠폰1회,28일 유효, 결과 구간 추가 발송 없음 |
| 라벨 관측 | 결과 28일을 모두 관측한 고객으로 학습·성능 평가 |
| 시간·금액 | 시간대가 있는 기준시각을 UTC로 정규화. 금액은 원천 통화·최소 단위 유지 |

학습은 시뮬레이터 공개 API `load_retention_inputs()`의 KIT39와 성숙한 라벨을 사용한다. 데이터셋 ID·원장 체크섬·피처 버전·고객·snapshot·기준시각·통화·단위를 대조하고, 실제 미래 세션과 쿠폰 발송으로 라벨·정책을 확인한다.

raw schema 5는 전체 SQLite 체크섬을 확인한 뒤 필요한 행·열을 조회한다. 학습은 추가 피처와 라벨 확인에 필요한 구간을 읽고, 추론은 공식 피처 재구성에 필요한 과거 이력을 읽는다. 이전 raw schema는 공식 전체 로더를 사용한다.

## 2. 빠른 실행

모든 명령은 **프로젝트 루트**에서 실행한다. 프로젝트 `.venv`의 Python을 명시한다. 의존성 범위는 [requirements.txt](requirements.txt), 현재 Windows/Python 3.12 환경의 설치 기록은 [requirements-lock.txt](requirements-lock.txt)에 있다.

다음 명령은 500명·104주(과거52주·미래52주) 데이터를 두 seed로 생성하고, 첫 실행에서 학습한 모델을 다음 실행에 고정 적용한다. 기준시점은 실행마다13주씩 이동한다. 생성은 기존 시뮬레이터의 `target` 모드·목표 이탈률20% 설정을 사용한다. 실제 이탈률은 실행마다 측정한다.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m retention experiment --config retention/config.yaml --output retention/artifacts/demo --customers 500 --weeks 104 --seeds 42 43
```

주요 결과는 `demo/model/`과 `demo/seed_43/evaluation/`에 생성된다. 기존 실험을 재실행하면 설정·원천이 일치하는 완성 데이터를 재사용한다. 설정을 바꾼 실험에는 새 출력 경로를 사용한다.

기본 `--coupon-mode auto`는 기존1:1 무작위 배정을 유지한다. 전체 미발송은 `--coupon-mode none`, 별도 무작위25% 발송은 `--coupon-mode assign --coupon-ratio .25 --coupon-assignment-seed 7`이다. 비율 대신 `--coupon-count 100` 또는 `--coupon-customer-ids id1 id2`를 지정할 수 있다. 명시적 `--months`는 이전 월간 자료 재현용이다.

원장만으로 배치 추론할 때는 실제 데이터셋의 기준시각을 지정한다.

```powershell
$retentionAsOf = (Get-Content -Raw -Encoding UTF8 retention/artifacts/demo/seed_42/dataset/raw.manifest.json | ConvertFrom-Json).metadata.as_of_at
.\.venv\Scripts\python.exe -m retention predict --bundle retention/artifacts/demo/model/bundle.joblib --input retention/artifacts/demo/seed_42/dataset --as-of $retentionAsOf --output retention/artifacts/demo/prediction
```

이미 생성된 데이터로 학습하려면 [config.yaml](config.yaml)의 `input_dir`, `feature_dir`, `output_dir`을 지정한다. 경로는 프로젝트 루트 기준이다.

```powershell
.\.venv\Scripts\python.exe -m retention validate --input retention/input/dataset --features retention/input/features --output retention/artifacts/validation
.\.venv\Scripts\python.exe -m retention train --config retention/config.yaml
```

## 3. 설정과 학습 방식

| 설정 | 기본값과 의미 |
| --- | --- |
| `policy_condition` | `observed_experiment_mix`는 발송·미발송군을 함께 학습한 관측 정책의 위험. `no_new_coupon`은 실제 미발송군으로 별도 학습 |
| `calibration` | `auto`. 보정용 고객에서 sigmoid를 학습하고, 별도 선택 고객의 Log Loss·Brier가 모두 개선될 때 적용. `none`·`sigmoid`도 지원 |
| `extra_features` | `true`. KIT39에 구매 주기 이상·여정 단계·시퀀스 요약·상대 변화율 10개를 추가해 49개 후보 생성 |
| `history_features` | 미설정 시 `false`. 관측된 장기 방문·구매·행동 비율 9개를 추가하는 실험 후보 |
| `max_length` | 최근 128개 timestamp 행동 그룹. 같은 시각의 행동은 개수로 묶음 |
| `trials / iterations / epochs` | 3 / 120 / 12. 초기 튜닝·학습 예산 |
| `psi_threshold / ks_alpha` | 0.2 / 0.01. 분포 변화 알림 기준 |
| `max_metric_drop` | 검증 AUC·Precision·Recall 대비 0.05 초과 하락 시 알림 |

비교 모델은 상수 확률·방문 최근성 로지스틱, LightGBM·CatBoost·XGBoost, PyTorch LSTM·GRU, 선택한 ML·DL의 1:1 평균 앙상블이다.

고객을 60/20/20으로 분리한다. 60% 안에서 트리의 5-fold 튜닝과 DL의 내부 Early Stopping을 수행한다. 기본 `auto` 보정은 선택용 20%를 보정용 10%·최종 선택용 10%로 나눈다. 모델은 선택 고객의 ROC-AUC, 동률이면 Log Loss로 고르고, 임계값은 같은 선택 고객의 F1로 정한다. 최종 테스트 20%는 결과 보고에 사용한다.

검증 예측을 생성한 학습 모델을 최종 저장한다. 학습 분할의 0.5%·99.5% 분위수로 수치를 클리핑하고, 중앙값 대체·결측 지시자·시퀀스 정규화도 학습 자료에서 적합한다. 미관측, 분모 0, 간격 표본 부족은 결측으로 표현한다.

피처 사전, 학습 표본의 결측·상수 통계, KIT39·전체 후보·리뷰 제외·최근 방문일의 개발 비교를 저장한다. 장기 이력을 켜면 장기 이력 제외 비교도 추가한다. 최종 사용 피처 수는 학습 자료의 전체 결측 여부에 따라 달라진다.

## 4. 독립 평가와 모니터링

```powershell
.\.venv\Scripts\python.exe -m retention evaluate --bundle retention/artifacts/demo/model/bundle.joblib --input retention/artifacts/demo/seed_43/dataset --features retention/artifacts/demo/seed_43/features --output retention/artifacts/demo/manual_evaluation
```

기본 평가 경로는 다른 데이터셋·생성 seed를 요구한다. 같은 학습 실행의 재현에는 `--allow-training-data`를 명시한다. `--features`를 생략하면 원장에서 피처와 정답을 계산하며, `--as-of`로 시각을 지정할 수 있다. 결과 기간이 덜 관측되면 확률·분포와 라벨 대기 상태를 기록한다.

평가에는 모델별 AUC·bootstrap 구간·PR-AUC·Log Loss·Brier, 고정 임계값의 Precision·Recall·F1·혼동행렬·대상 인원, 정책·이력 길이별 결과를 포함한다. ML AUC 0.78 달성 여부는 검증에서 선택한 트리로 판정한다.

모니터링은 학습 자료에서 고정한 분포와 새 실행을 PSI·KS로 비교한다. 결측률도 PSI에 포함하고 KS는 Benjamini–Hochberg로 다중 검정을 보정한다. 같은 모델 버전의 시점별 성능을 기록한다. seed 변경은 표본 변동을, 시작일 변경은 다른 기준시점의 결과를 확인하는 데 사용한다. 분포 변화 탐지는 통제된 변화 시나리오로 별도 확인한다.

실제 배정군과 미래 방문 기록으로 A/B 이탈률 차이·p-value·95% 구간을 계산한다. 기본 표본 설계는 이탈률 20%, 절대 감소 5%p, 유의수준 0.05, 목표 검정력 0.8이다.

## 5. Uplift·CLV 연결과 Python API

`--integrate`를 추가하면 별도로 설치한 Uplift T/X/H와 CLV 공개 API로 모델을 학습하고 새 실행의 결과를 연결한다. 기본104주 자료가 CLV의 미래52주 평가 기간을 포함한다. Uplift는 기록된 실제 배정확률을 사용하며, 양쪽 군의 관측이 없는 미발송·전원발송이나 고객ID 지정 배정에는 인과 효과 학습을 허용하지 않는다. 이 경우 Retention·CLV는 각 공개 API로 실행한다.

```powershell
.\.venv\Scripts\python.exe -m retention experiment --config retention/config.yaml --output retention/artifacts/integration_demo --customers 500 --weeks 104 --seeds 101 102 103 --integrate
```

| 파트 | 전달하는 값 |
| --- | --- |
| Retention | 정책 조건이 명시된 28일 미방문 확률 |
| Uplift | 28일 구매확률의 발송 효과. `p0`는 미발송 구매확률 |
| CLV | 향후 52주 할인 후 실결제금액. 출력명은 `predicted_clv_12m` |

`dataset_id`·`customer_id`·실제 기준시각·통화·금액 단위를 맞춰 결합한다. Retention의 `as_of_at`, Uplift의 `t0`, CLV 출력의 `reference_week`를 같은 순간으로 정규화한다. 시뮬레이터 주간 행의 `reference_week`는 그 주의 시작이다.
Uplift는 공식 `load_feature_dataset(feature_dir, ledger=raw_dir)`로 같은 Feature Store의 공개39개 피처·4주 구매 정답·실제 배정확률을 읽고 원장 배정 설계와 체크섬을 대조한다. 주간 자료를 과거30/90일 raw 피처로 다시 계산하지 않는다.
저장 모델은 실제 학습 창·피처 계약을 기록하고, 캐시 재사용과 예측은 공식 `predict_for_strategy` 검증을 거친다. 출처 없는 모델을 주간 모델로 해석하지 않는다. `--integrate`는104주 주간 자료만 지원하며, 명시적 legacy 월간 자료와 SQL 모델은 각 모델의 호환 API에서 별도로 사용한다.
동결 모델의 예측은 전원 발송 자료에도 가능하지만, Control-only CLV의 성숙한 평가 고객이 없으면 CLV 평가는 unavailable로 남기며 예측을 버리지 않는다.

```python
from pathlib import Path
import pandas as pd
from retention.api import predict_retention, evaluate_retention

root = Path("retention/artifacts/demo")
bundle = root / "model" / "bundle.joblib"
as_of = pd.read_parquet(root / "model" / "samples.parquet").as_of_at.iloc[0]

predictions = predict_retention(bundle, root / "seed_42" / "dataset", as_of)
report = evaluate_retention(
    bundle, root / "seed_43" / "dataset", root / "api_evaluation",
    feature_dir=root / "seed_43" / "features",
)
```

`predict_retention()`은 파일 저장 없이 DataFrame을 반환한다. 학습 API `train_retention(config)`도 제공하며, 설정 읽기는 `retention.pipeline.read_config()`를 사용한다.

전략 결합에는 `retention.api.predict_for_strategy(bundle, dataset, as_of)`를 쓴다.
반환값의 `frame`은 같은 고객·기준시각·snapshot·원장 체크섬에 `churn_prob`,
`input_cutoff_at`, `prediction_end_at`을 붙이고, `metadata`는4주 이탈확률의 대상·기간·모델 버전을 선언한다.
`producer_metadata`에 관측 혼합/무발송 정책 조건과 제외 수를 기록하며 부적격 고객의 확률은 결측으로 유지한다.
기존 `predict_retention()`의 `churn_probability` 출력 이름은 유지한다.

주간 KIT39 버전과 legacy 버전은 혼용하지 않는다. `_30d`·`_90d` 열 이름은 호환용으로 유지하되,
주간 자료의 추가 피처·시퀀스·장기 이력도28/91일을 사용하고 생성한 피처 사전에 실제 기간을 기록한다.

예측에는 식별 키, `churn_probability`, `predicted_churn`, `status`, 타깃·기간·정책 조건·임계값·모델 버전이 포함된다. 부적격 고객은 확률을 비워 두고 제외 상태를 반환한다.

## 6. 결과 파일과 코드 위치

모든 출력은 `retention/` 아래에 저장한다. `artifacts/`·`input/`은 Git 제외 대상이므로 새 checkout에서는 데이터를 준비하거나 생성한다.

| 파일 | 내용 |
| --- | --- |
| `bundle.joblib`, `run_status.json` | 모델·전처리·보정기·모니터링 기준과 완료 상태. 새 학습의 완료 표시는 `complete` |
| `config.json`, `input_validation.json`, `run_manifest.json` | 설정·원천·체크섬·버전·환경·시간·메모리 |
| `samples.parquet`, `split_membership.parquet`, `splits.json` | 라벨·고객 키·튜닝/보정/선택/테스트 분할 |
| `feature_dictionary.md`, `feature_quality.csv`, `feature_ablation.csv` | 피처 산식·결측 규칙·학습 분포·개발 비교 |
| `metrics.json`, `requirements_report.json`, `comparison.md` | 모델 성능·수치 기준 달성 여부·선택 근거 |
| `selection.json`, `cross_validation.csv`, `best_params.json` | 보정·임계값·튜닝 결과 |
| `lstm.pt`, `gru.pt`, 학습·Early Stopping 로그 | DL 모델과 epoch 선택 기록 |
| `shap_*.csv/json/png` | 설명 모델·고객·시각·기준값·출력 척도와 트리 기여 |
| `predictions.parquet`, `runtime_profile.json` | 배치 예측과 입력·피처·추론 처리 비용 |
| `evaluation.json`, `monitoring_report.json`, `ab_test_report.json/md` | 독립 평가·분포/성능 알림·실제 A/B 분석 |
| `combined_predictions.parquet`, `monitoring_history.csv` | 파트별 점수 결합과 반복 실행의 성능 이력 |

학습은 작업 디렉터리에서 완성한 뒤 최종 경로에 공개한다. 실패한 작업에는 오류 상태를 남긴다. 기존 완성 모델의 경로로 새 학습을 실행하면 새 경로를 요구한다. SHAP은 선택한 트리의 보정 전 출력을 설명하며, 실제 우승 모델과 설명 모델을 각각 기록한다.

| 코드 | 역할 |
| --- | --- |
| [simulator_input.py](simulator_input.py) | 공식 입력·원장 검증, SQL 조회, 추가 피처·시퀀스 |
| [dataset.py](dataset.py) | 고객 분할·내부 5-fold, 기존 시간순 분할 |
| [models.py](models.py), [sequence.py](sequence.py) | 트리·기준선·전처리·확률 보정, LSTM·GRU |
| [feature_dictionary.py](feature_dictionary.py) | 피처별 산식·결측·비즈니스 의미 |
| [pipeline.py](pipeline.py), [evaluate.py](evaluate.py) | 학습·저장·추론·지표·SHAP |
| [api.py](api.py), [monitoring.py](monitoring.py), [experiments.py](experiments.py) | 공개 API·분포 모니터링·반복 생성·파트 연결 |
| [data.py](data.py), [features.py](features.py), [fixture.py](fixture.py) | 기존 7종 CSV/Parquet 계약과 fixture 경로 |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | 합의와 구현 계획 |

## 7. 확인된 결과와 남은 작업

실측 결과는 `retention/artifacts/implementation_verification/`에 있다.

| 검증 | 마지막 확인 결과 |
| --- | --- |
| 테스트·환경 | 테스트 36개 통과, `pip check` 통과 |
| 전체 데이터 | 20,000명·24개월·seed 42, 이벤트 7,429,128건, 이탈률 19.04% |
| 전체 모델 | 검증에서 CatBoost 선택. 테스트 4,000명 AUC 0.7587 |
| 독립 평가 | seed 43의 2,000명 AUC 0.7448. 별도 6개월·500명 실행 AUC 0.7473 |
| 원장 재추론 | 20,000명 추론 완료. 보류 테스트 4,000명의 식별 키·예측값 일치 |
| 3개 파트 연결 | 500명·24개월의 두 독립 실행에서 Retention·Uplift·CLV 결과 연결 완료 |
| 처리 비용 | 한 번의 전체 실행에서 생성·저장 약 29분, 공식 피처 생성 약 10분, 학습 약 3분, 원장 추론 약 5분 |
| 메모리 | 공식 피처 생성 등을 포함한 동일 프로세스의 최대 RSS 약 9.4 GiB |

장기 이력 9개 후보를 더한 개발 비교에서는 기존 피처 대비 개선이 없었다. CatBoost의 선택 구간 AUC는 0.7852에서 0.7827이었다. `history_features`는 현재 기본 설정에서 꺼져 있다. 테스트 36개 통과 기록은 이 후보 코드 추가 전이며, 이후 전체 회귀 테스트는 다시 실행해야 한다.

남은 작업은 ML 테스트 AUC 0.78 달성을 위한 개발 검증, 변경 후 회귀 테스트, 추가 튜닝을 수행할 경우 새 seed의 최종 독립 평가다. 최종 전략의 대상 선정·ROI는 전략 파트의 실행 코드가 연결되면 검증한다. 현재 통합 검증은 세 모델의 예측·정답·키 연결 범위까지 확인했다.

`MODEL_COMPARISON.md`에는 초기 7종 fixture 결과가 남아 있다. 현재 공식 시뮬레이터의 성능은 위 표와 실행별 `metrics.json`·`evaluation.json`을 기준으로 확인한다.

## 8. 테스트와 작업 재개

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$retentionTestTemp = Join-Path $env:TEMP ("retention-tests-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -m pytest retention/tests -q -p no:cacheprovider --basetemp $retentionTestTemp
```

작업을 재개할 때는 현재 설정, 위 검증 상태, 마지막 실행의 `run_status.json`을 먼저 확인한다. 변경 범위는 Retention 내부이며 외부 파트의 공개 계약을 사용한다. 외부 변경이 필요한 장애는 원인과 필요한 변경을 사용자에게 먼저 설명한다. 기간 기준은 합의된 28일을 적용한다.

기존 fixture는 `python -m retention fixture --output retention/input/fixture --customers 120`으로 생성할 수 있다. 생성된 설정은 과거 30일 타깃·시간순 평가를 점검하는 별도 경로다. 현재 기본 설정과 제출 검증은 위의 28일 시뮬레이터 경로를 사용한다.
