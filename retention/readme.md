# Retention 구현 및 실행

확정된 7종 입력을 받아 기준시각별 피처·정답을 만들고 **LightGBM, CatBoost, XGBoost, LSTM, GRU**를 비교한다. 상수 확률·방문 최근성 로지스틱 기준선과 ML+DL 고정 평균 앙상블도 포함한다. 모든 프로젝트 출력은 `retention/`에 저장한다.

## 실행 환경

프로젝트 루트의 PowerShell에서 실행한다. activate 여부와 무관하게 프로젝트 가상환경의 Python을 명시한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r retention/requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest retention/tests -q -o cache_dir=retention/.pytest_cache
```

`requirements.txt`는 직접 의존성 범위, `requirements-lock.txt`는 이 Windows/Python 3.12 환경에서 설치·검증한 정확한 버전이다. 다른 OS/Python의 바이너리 호환성까지 보장하는 범용 lock은 아니다. 현재 PyTorch는 CPU 빌드이며 CUDA를 요구하지 않는다.

## 입력 파일

한 디렉터리에 다음 이름의 파일을 둔다. 각 테이블은 CSV 또는 Parquet 중 하나만 제공한다. ID는 문자열로 보존하며, CSV의 ID 앞자리 0도 유지한다.

| 파일명 | 필드 |
| --- | --- |
| `customers.parquet` 또는 `.csv` | `customer_id`, `signed_up_at` |
| `sessions.parquet` 또는 `.csv` | `session_id`, `customer_id`, `started_at`, `ended_at` |
| `events.parquet` 또는 `.csv` | `event_id`, `customer_id`, `session_id`, `event_at`, `event_type` |
| `orders.parquet` 또는 `.csv` | `order_id`, `customer_id`, `session_id`, `purchased_at` |
| `items.parquet` 또는 `.csv` | `order_id`, `product_id`, `unit_price`, `quantity` |
| `coupons.parquet` 또는 `.csv` | `coupon_id`, `customer_id`, `policy`; 선택 `sent_at`, `expires_at` |
| `redemptions.parquet` 또는 `.csv` | `coupon_id`, `order_id` |

빈 테이블도 해당 스키마를 가진 파일로 제공한다. 내부 persona·canonical type·미래 gap 등 추가 컬럼은 모델 입력으로 읽지 않는다. 쿠폰 사용 시각은 연결 주문의 `purchased_at`이며, 주문금액은 상품 단가×수량을 주문별로 먼저 합산한다. 환불·취소를 임의로 생성하지 않는다.

`sent_at`·`expires_at`은 컬럼 자체를 생략할 수 있다. 제공한다면 컬럼 내 시각은 완전해야 하며 일부 행만 미확인인 경우 검증에서 안내한다. 해당 시각이 없으면 과거 발송·유효 쿠폰 피처는 계산하지 않는다. `policy`가 단순 문자열인 경우 할인율로 임의 변환하지 않는다.

세션은 고객+세션 ID 조합으로 연결한다. 결측 세션 ID는 공통 가짜 세션으로 묶지 않으며, 결측 종료 시각으로 미래의 완료 세션 길이를 만들지 않는다. 동일 timestamp 안의 이벤트는 행동별 개수/존재를 묶어 시퀀스로 사용하므로 파일 행 순서와 무관하다.

## End-to-end smoke test

아래 명령은 외부 데이터 대신 **7종 입력 계약에 맞는 테스트 전용 합성 fixture**를 만든다. 실제 고객 생성기나 쿠폰 인과효과를 재현한 데이터가 아니다. 새 경로를 사용해야 하며 기존 데이터는 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe -m retention fixture --output retention/artifacts/demo/input --customers 120
.\.venv\Scripts\python.exe -m retention validate --input retention/artifacts/demo/input --output retention/artifacts/demo/validation
.\.venv\Scripts\python.exe -m retention train --config retention/artifacts/demo/input/smoke_config.yaml
.\.venv\Scripts\python.exe -m retention predict --bundle retention/artifacts/demo/smoke_run/bundle.joblib --input retention/artifacts/demo/input --as-of 2024-10-01 --output retention/artifacts/demo/inference
```

fixture는 2024-10-01 이전 기록만 포함하므로 마지막 추론은 미래 정답 없이 실행된다. 재학습 시에는 `train --output retention/artifacts/새실행명`으로 새 출력 경로를 지정할 수 있다.

## 실제 데이터로 학습

1. 받은 7종 파일을 `retention/input/` 또는 읽기 가능한 입력 경로에 둔다. 원본은 수정하지 않는다.
2. `config.yaml`에서 입력/출력 경로, 기준시각 범위, `test_start`, 실제 `observation_end`와 `observation_assumption`을 설정한다. 기본 예시의 `REPLACE` 문구는 실제 확인 내용으로 바꿔야 실행된다.
3. 입력 검증 후 학습하고 `comparison.md`, `metrics.json`, `input_validation.json`을 확인한다.

```powershell
.\.venv\Scripts\python.exe -m retention validate --input retention/input --output retention/artifacts/real_validation
.\.venv\Scripts\python.exe -m retention train --config retention/config.yaml
.\.venv\Scripts\python.exe -m retention predict --bundle retention/artifacts/real_run_001/bundle.joblib --input retention/input --as-of 2026-01-01 --output retention/artifacts/real_inference
```

이탈 기본값은 향후 30일 새 세션 시작 없음이다. `target: purchase`로 바꾸면 주문 발생 기준을 사용한다. 가입 후 최소 30일·최근 30일 방문 고객을 초기 대상으로 한다. 제외 고객은 추론에서 사유와 함께 반환하고 확률을 임의로 채우지 않는다.

시각은 UTC로 통일한다. 입력은 기준시각 미만, 정답은 기준시각 이상·기간 끝 미만이다. 실제 수집 누락 여부와 종료가 불명확하면 정답의 의미도 보장되지 않으므로 확인한 가정을 설정에 남긴다. 파일의 마지막 이벤트를 자동으로 수집 종료라고 가정하지 않는다.

충분한 기간·클래스 표본이 없어 5개 시간 fold 또는 독립 Early Stopping 분할을 구성할 수 없으면 명시적으로 실패한다. 임의 랜덤 분할로 대체하지 않는다. 데이터 길이가 짧으면 기간·표본 설정을 재검토해야 한다.

## 학습 및 비교 방식

- 51개 내부 피처: RFM, 변화율 5종, 세션 품질, 구매 주기, 시간대, 여정, 시각 그룹 기반 시퀀스 요약, 쿠폰 이력. 정의는 실행별 `feature_dictionary.md`에 저장한다.
- ML 3종은 같은 purged rolling-origin 5-fold와 제한된 Optuna 탐색으로 비교한다. 클래스 가중치는 해당 학습 표본에서만 계산한다.
- LSTM·GRU는 PyTorch의 검증된 표준 레이어를 사용한다. 두 모델은 같은 입력·은닉 크기·최대 epoch 예산을 사용하지만 파라미터 수까지 같지는 않다. 시각 그룹의 행동 개수는 학습 가능한 투영층으로 임베딩하고 packed sequence로 padding을 제외한다.
- 각 DL의 epoch는 공통 외부 검증보다 앞선 별도 내부 시간 구간에서 Early Stopping으로 선택한다. 이후 고정 epoch로 다시 학습한다.
- 공통 개발 검증에서 선택한 트리와 RNN의 확률을 1:1 평균해 앙상블 후보를 만든다. 최종 모델과 임계값은 개발 검증에서 정하고 마지막 테스트로 바꾸지 않는다.
- AUC, PR-AUC, Log Loss, Brier, 보정 그래프와 고객 단위 bootstrap AUC 구간을 보고한다. 불균형 가중치가 확률 보정에 불리할 수 있어 AUC만 보고 운영 적합성을 판단하지 않는다.
- SHAP은 검증에서 선택한 트리의 전역 상위 10개·대표 고객 기여값을 저장한다. 이는 이탈의 인과 원인이 아니다.

## 주요 출력

| 파일 | 내용 |
| --- | --- |
| `comparison.md`, `metrics.json` | 모델 비교·선택 근거·한계 |
| `cross_validation.csv`, `best_params.json` | ML 3종의 탐색·5-fold 결과 |
| `splits.json`, `samples.parquet` | 시간 분할과 성숙한 정답·고객 키 |
| `feature_dictionary.md`, `features.parquet`, `sequences.npz` | 피처 정의·파일 기반 피처 저장·시퀀스 |
| `selection.json` | 개발 검증 선택 모델·앙상블 구성·임계값 |
| `lstm.pt`, `gru.pt`, `*_early_stopping.json`, `*_training_log.json` | DL 파라미터·epoch 선택·학습 로그 |
| `bundle.joblib` | 입력 변환과 전체 모델을 포함한 배치 추론 묶음 |
| `test_predictions.parquet`, `validation_predictions.parquet` | 같은 표본의 모델별 확률 |
| `shap_summary.png`, `shap_importance.csv`, `shap_customer.csv` | SHAP 설명 |
| `probability_and_pr.png` | 확률 보정·Precision/Recall 곡선 |
| `run_manifest.json`, `config.json`, `input_validation.json` | 실행 환경·원본 식별·입력 검증·가정 |

`joblib` 모델은 이 프로젝트에서 생성한 신뢰할 수 있는 파일만 로드한다. 외부에서 받은 임의 pickle/joblib을 로드하지 않는다.

## 범위와 현재 한계

- 현재 결과는 확정 입력 스키마의 smoke test다. 테스트용 합성 데이터의 높은 AUC를 실데이터 AUC≥0.78 달성으로 주장하지 않는다.
- 일반 Retention 확률을 제공하며 추가 쿠폰 미발송 조건의 인과적 위험을 보장하지 않는다. T/X/H 학습은 Uplift 담당자의 역할이다.
- 과제의 공용 `docs/`·`results/` 대신 동등 자료를 `retention/artifacts/` 안에 생성한다. 다른 디렉터리를 수정하지 않는 작업 범위를 따른다.
- DuckDB로 계약 컬럼을 읽되 현재 피처 생성은 메모리에 적재된 테이블을 사용한다. 대규모 실데이터의 메모리·처리량은 아직 검증하지 않았다. 로더를 고객 파티션으로 확장할 필요가 있는지는 실제 규모에서 측정한다.
- 직접 작성한 부분은 데이터 처리·모델 연결·학습/평가 흐름이다. 트리 알고리즘과 LSTM/GRU 셀을 재구현하거나 검증되지 않은 사전학습 가중치를 가져오지 않았다.

참조: 프로젝트의 `RETENTION_REQUIRED_FEATURES.md`, `docs/request.md`의 피처·ML·DL 요구사항, `논문.pdf` §3.3·§4. 논문 내부 합성 uplift 규칙을 예측 입력으로 사용하지 않는다.

구현 API 확인 자료: [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html), [packed sequence](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html), [LightGBM의 클래스 가중치와 확률 보정 주의](https://lightgbm.readthedocs.io/en/stable/_modules/lightgbm/sklearn.html).
