# Final Strategy: 입력 통합과 고객 Segmentation

## 1. 모듈 개요

KIT 프로젝트의 WBS 5.1 입력 로딩·정합성 검증과 WBS 5.2 고객 Segmentation을 제공한다.
시뮬레이터가 발행한 Feature Store와 생산자가 정의한 Churn·Uplift·CLV 예측값을
검증·결합한 뒤, 고객 분류·우선순위·요약 통계·CSV/JSON 출력·기본 시각화를 수행한다.

```text
KIT39 Feature Store + 세 모델의 내부 PredictionBatch
  → integrate_inputs()
  → IntegratedInputs
  → segment_customers()
  → SegmentationResult → export_segments() / plot_segments()
```

공개 구현은 [inputs.py](inputs.py), [validation.py](validation.py),
[segmentation.py](segmentation.py)와 두 테스트 파일이다.
원시 데이터 생성·KIT39 재계산·모델 학습은 수행하지 않는다.
**실제 Churn·Uplift·CLV 출력 연결은 이 공개 범위에서 완료되지 않았다.**
아래 예측 형식은 내부 소비자 계약이며, 외부 모델의 CSV 경로·함수·컬럼 계약을 확정한 것이 아니다.

## 2. 입력 데이터 계약

### Feature Store

현재 v5 시뮬레이터의 Python 패키지 경로는 `simulate.v4`다.
`simulate.v4.storage.FEATURE_KEYS`의 식별·출처 12개와
`simulate.v4.features.EXPANDED_FEATURES`의 KIT39 Feature 이름·순서를 재사용한다.
총 51컬럼을 허용하며 누락·중복·추가 컬럼은 거절한다.
정규화된 반환 순서는 12개 키·출처 컬럼 다음 39개 Feature다.

12개 키·출처 컬럼:

```text
customer_id, as_of_at, snapshot_id, reference_week, dataset_id,
currency, amount_unit, raw_database_sha256, source_model_sha256,
feature_set_version, behavior_feature_set_version, expanded_feature_set_version
```

| 버전 필드 | 현재 요구값 |
| --- | --- |
| DataFrame의 `feature_set_version` | `kit32-v1` |
| DataFrame의 `behavior_feature_set_version` | `behavior_review_cart_v3` |
| DataFrame의 `expanded_feature_set_version` | `kit39-review-cart-v3` |
| manifest의 `feature_set_version` | `kit39-review-cart-v3` |

manifest의 `predictors`와 `keys_and_provenance`는 registry의 이름·순서와 일치해야 한다.
시뮬레이터 상수를 import하므로 이후 버전 변경 시에도 호환성 검증이 필요하다.

- `customer_id`와 다른 식별자는 비어 있지 않은 문자열이다. `001`을 숫자로 변환하지 않는다.
- 결합·중복 기준은 `(dataset_id, customer_id, as_of_at)`이다. 같은 고객의 서로 다른 시각은 구분한다.
- 한 입력은 한 dataset·raw SHA·통화·금액 단위·생성 모형 SHA에 속해야 한다.
- `as_of_at`은 timezone-aware 값만 받아 UTC로 정규화한다.
- `reference_week`는 서울 기준 월요일 날짜 또는 월요일 00시 timestamp이며,
  `as_of_at`이 속한 주차와 대조해 `YYYY-MM-DD`로 정규화한다.
  `as_of_at` 자체를 월요일로 강제하지 않으며, 같은 주의 다른 시각을 합치지 않는다.
- `snapshot_id`는 불투명 문자열이다. 내부 구조를 해석하거나 다시 만들지 않고 모델별 값과 대조한다.
- 두 SHA 필드는 소문자 64자리 digest여야 한다. 누락된 출처를 임의 생성하지 않는다.
- 통화 변환은 하지 않는다. `amount_unit`은 양의 유한 Decimal로 검증하고 동등한 표기만 정규화한다.
  예를 들어 `0.0100`은 `0.01`이 되며, `SOURCE_UNSPECIFIED`를 KRW로 추정하지 않는다.
- 정상 Feature NULL은 유지한다. 예측값의 결측·무한대·복소수·bool·시각형 값은 거절한다.

파일 로더는 `feature_store.parquet`와 인접 `manifest.json`을 사용한다.
storage schema 1, 버전, 허용 컬럼, 서울 주차/UTC 저장 선언, Parquet SHA256, 행 수,
snapshot 시각 목록, dataset·raw SHA·생성 모형 SHA를 검증한다.
manifest의 다른 파일이나 원시 SQLite를 다시 읽거나 원장 자체의 SHA를 재계산하지는 않는다.
`validate_feature_frame()`만 호출하면 파일 manifest·checksum 검증은 수행되지 않는다.

### 모델별 내부 예측 계약

`predictions`는 정확히 `churn`, `uplift`, `clv` 세 키의 `PredictionBatch` 매핑이다.
각 batch의 frame은 아래 8개 문맥 컬럼, 두 기간 컬럼, 해당 모델 예측 컬럼만 포함한다.

```text
dataset_id, customer_id, as_of_at, reference_week, snapshot_id,
currency, amount_unit, raw_database_sha256,
input_cutoff_at, prediction_end_at, <해당 모델 예측 컬럼>
```

| 모델 | 예측 컬럼 | `value_semantics`와 검증 |
| --- | --- | --- |
| Churn | `churn_probability` | `probability`, 유한한 0~1 |
| Uplift | `uplift_score` | `probability_difference`, 유한한 -1~1, 음수 허용 |
| CLV | `predicted_clv_12m` | `amount` 또는 `nonnegative_amount`; 유한한 금액, 후자는 음수 거절 |

CLV 기간은 반드시 `horizon_value=52`, `horizon_unit="weeks"`이며 정확히 364일이다.
`predicted_clv_12m`은 호환성을 위해 유지하는 기존 컬럼명이며 달력 12개월을 뜻하지 않는다.
종료 시각은 `as_of_at + pd.Timedelta(weeks=52)`와 일치해야 한다.
`12 calendar_months`, `364 days`, 51주·53주 선언은 거절하며 기간이나 예측값을 자동 환산하지 않는다.
빈 입력을 포함한 Segmentation에서도 같은 CLV 기간 계약을 검증한다.
Segmentation에서는 미래 결제금액 CLV를 사용하므로 `amount` 선언이어도 음수를 거절한다.
입력 단계와 분류 단계의 검증 범위를 구분한다.

`input_cutoff_at`은 과거 입력의 배타적 종료 시각이며 `as_of_at` 이하여야 한다.
`prediction_end_at`은 `[as_of_at, prediction_end_at)`의 종료로, UTC 기준 `as_of_at`에
선언한 일·주·달력 개월을 더한 값과 일치해야 한다. 두 시각 모두 timezone-aware여야 한다.
생산자가 선언한 기간·결과 정의를 보존하며, 방문 이탈과 구매 유지 등을 같은 결과로 취급하지 않는다.

각 모델의 고객·시각 키 집합은 Feature Store와 정확히 같아야 한다.
누락·추가 고객과 중복은 오류이며, snapshot·주차·출처·통화·단위도 행별로 대조한다.
고객을 조용히 제외하거나 예측 NULL을 0으로 채우지 않는다.
미래 정답·실험 컬럼 등 미등록 컬럼과 `evaluation_only` 경로/속성은 거절한다.
단, 생산자의 잘못된 cutoff 선언이나 정상 컬럼 안에 숨은 미래 정보까지 증명·검출하지는 못한다.

## 3. 주요 입력 API

| 클래스 / 생성자 | 역할과 필드 |
| --- | --- |
| `PredictionMetadata(model_version, target_definition_id, horizon_value, horizon_unit, value_semantics)` | 모델 버전·예측 정의·기간·수치 의미. 버전과 정의는 비어 있지 않은 문자열, 기간은 양의 정수, 단위는 `days`/`weeks`/`calendar_months` |
| `PredictionBatch(frame, metadata)` | 정규화 대상 DataFrame과 해당 `PredictionMetadata` |
| `IntegratedInputs(data, prediction_metadata, prediction_context)` | Feature 51개 컬럼과 예측 3개 컬럼, 모델별 메타데이터, 모델별 키와 cutoff·결과 종료 시각 |

생성자 자체가 전체 입력을 검증하지는 않는다. 아래 검증·통합 함수를 사용한다.
클래스는 frozen dataclass이지만 내부 DataFrame은 변경 가능하므로 Segmentation에서도 재검증한다.

| 함수 시그니처 | 주요 입력·반환값·오류 |
| --- | --- |
| `load_feature_store(path: str \| Path) -> pd.DataFrame` | 기존 파일 또는 디렉터리 → 검증된 Feature 복사본. 파일명·manifest·checksum·출처 불일치 거절 |
| `validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame` | 메모리 Feature → 복사·정규화된 51컬럼. 빈 입력·스키마·키·버전·시각·출처·수치 오류 거절 |
| `validate_prediction_frame(frame: pd.DataFrame, metadata: PredictionMetadata, model: str) -> pd.DataFrame` | 내부 batch frame → 검증·정규화된 복사본. 모델명·기간·cutoff·예측 범위 오류 거절 |
| `integrate_inputs(features: pd.DataFrame, predictions: Mapping[str, PredictionBatch]) -> IntegratedInputs` | 세 batch를 고객·정확한 시각으로 결합. Feature 행 순서를 유지하며 고객 집합·문맥 불일치 거절 |

`PredictionMetadata`, 검증 함수, `InputValidationError`는 `validation.py`에서,
나머지 입력 클래스·로더·통합 함수는 `inputs.py`에서 import한다.
계약 오류는 `InputValidationError(ValueError)`로 보고한다.
일부 파일 I/O 오류와 Parquet 엔진 오류는 원래 예외가 전달될 수 있다.
입력 DataFrame은 직접 수정하지 않는다.

## 4. Segmentation

분류는 `(dataset_id, as_of_at)`별 분석 모집단 안에서 수행하며 같은 주의 다른 시각을 분리한다.
세 축의 조합으로 2 × 3 × 2 = **12개 세그먼트**를 만든다.

| 축 | 실제 값과 경계 |
| --- | --- |
| Churn | `high`: 확률 ≥ 임계값, `low`: 임계값 미만 |
| Uplift | `negative`: 0 미만, `neutral_low`: 0 이상 임계값 이하, `positive`: 임계값 초과 |
| CLV | `high_value`: 상위 고객, `standard_value`: 나머지 |

`segment_id`는 `high__positive__high_value`처럼 세 축을 조합한다.
`segment_name`은 같은 축의 표시용 이름이다. 빈 세그먼트도 통계에 남긴다.

개발용 기본값은 Churn 임계값 0.5, Uplift 임계값 0.0, CLV 상위 비율 0.2다.
따라서 기본 Uplift 중립은 정확히 0이다. **확정된 사업 정책이나 검증된 최적 임계값이 아니다.**

CLV 상위 인원은 분석 고객 N에 대해 `ceil(N × high_clv_fraction)`이다.
CLV 내림차순·문자열 `customer_id` 오름차순으로 정하며, 동점 때문에 인원을 늘리지 않는다.
작은 모집단의 실제 상위 비율은 20%보다 클 수 있다.

```text
priority_score = uplift_score × predicted_clv_12m
```

Priority Score는 **전략용 대리지표이며 실제 증분 매출·이익·ROI가 아니다.**
음수 Uplift의 음수 점수를 유지하고, Uplift 또는 CLV가 0이면 점수도 0이다.
점수 내림차순·고객 ID 오름차순으로 정렬해 기준시각별 `priority_rank`를 1부터 연속 부여한다.
서로 다른 모델의 결과 기간은 원래 선언대로 메타데이터에 남긴다.

`eligibility=None`이면 검증된 전체 입력 고객을 분석한다. 쿠폰 발송 적격을 뜻하지 않는다.
제외하려면 모든 입력 키를 한 번씩 포함하는 아래 DataFrame을 명시적으로 제공한다.

```text
dataset_id, customer_id, as_of_at, eligible, exclusion_reason
```

`eligible`은 bool이다. False 행에는 비어 있지 않은 사유, True 행에는 NULL 사유가 필요하다.
제외 고객은 `excluded`에 원래 사유와 함께 보존하며 상위 CLV·순위·통계 분모에서 제외한다.
예측 결측은 eligibility로 우회하지 못한다. 입력 재검증이 먼저 수행된다.

요약 통계는 고객 수·비율, Churn/Uplift/CLV/점수 평균, 점수 합계, 통화·단위를 포함한다.
고객 수 합계는 분석 모집단과 일치해야 한다. 빈 세그먼트는 평균 NULL, 점수 합계 0이다.
전원 제외 시 해당 모집단의 통계 12행은 유지한다.
스키마·선언·context를 유지한 빈 `IntegratedInputs` 슬라이스는 빈 결과가 되지만,
일반 입력 통합 함수는 빈 입력을 받지 않는다.

## 5. 공개 Segmentation API

모두 `final_strategy.segmentation`에서 import한다.

```python
SegmentationConfig(
    churn_threshold=0.5,
    uplift_threshold=0.0,
    high_clv_fraction=0.2,
    segmentation_version="segmentation-12-v1",
)
```

세 수치 설정은 유한한 0~1 실수여야 하며 bool은 거절한다. 버전은 비어 있지 않은 문자열이다.

`SegmentationResult(customers, excluded, summary, metadata)`의 필드:

| 필드 | 내용 |
| --- | --- |
| `customers` | 분석 고객의 Feature·출처·예측값, 분류 축·ID·이름, `is_high_clv`, 점수·순위 |
| `excluded` | 제외된 입력 고객과 원래 `exclusion_reason` |
| `summary` | 모집단별 12개 세그먼트 통계 |
| `metadata` | 설정·분류 버전, 모델별 정의·기간·버전, 입력 snapshot·예측 context, 분석/제외/상위 고객 수와 분류 규칙 |

| 함수 시그니처 | 동작 |
| --- | --- |
| `segment_customers(inputs: IntegratedInputs, config: SegmentationConfig \| None = None, eligibility: pd.DataFrame \| None = None) -> SegmentationResult` | 기존 통합 검증을 재사용하고 분석·제외·통계·메타데이터 반환. 잘못된 context·음수 CLV·eligibility 불일치 거절 |
| `export_segments(result: SegmentationResult, csv_path: str \| Path) -> dict[str, Path]` | `.csv` 경로에 고객 CSV와 `_summary.csv`, `_excluded.csv`, `_metadata.json` 저장. 반환 키는 `customers`, `summary`, `excluded`, `metadata` |
| `plot_segments(result: SegmentationResult, dataset_id: str, as_of_at)` | 선택한 단일 모집단의 matplotlib `Figure` 반환. 고객 수·비율·평균 CLV·평균 점수 4개 그래프. 없는 모집단은 오류 |

`export_segments()`는 지정한 네 파일 중 기존 파일이 있으면 `FileExistsError`로 거절한다.
고객 CSV에는 문맥 8컬럼·예측 3컬럼·분류 및 점수 파생 컬럼만 내보내며 Feature 39개 전체는 넣지 않는다.
CSV 재로딩 시 문자열 ID와 단위를 보존하고 timezone-aware 시각으로 읽어야 한다.
`NA` 같은 ID가 기본 결측 토큰으로 변하지 않도록 문자열/결측 파싱 정책도 명시해야 한다.
`plot_segments()`는 파일을 저장하지 않는다. 저장하려면 반환 Figure의 `savefig()`를 호출한다.
빈 세그먼트의 평균을 0으로 그리지 않는다. matplotlib은 시각화 호출 시 필요하다.

## 6. 사용 예시

아래는 **검증용 더미 예측**을 만드는 예시다. 실제 모델 추론이나 고객별 운영 예측이 아니다.
`features`는 같은 시점·출처로 발행된 Store를 `load_feature_store()`로 읽은 DataFrame이어야 한다.
임의 원장·정답 테이블을 전달하지 않는다. 더미 값과 정의를 운영에 사용하면 안 된다.

```python
import pandas as pd

from final_strategy.inputs import PredictionBatch, integrate_inputs, load_feature_store
from final_strategy.validation import CONTEXT_COLUMNS, PredictionMetadata
from final_strategy.segmentation import SegmentationConfig, segment_customers


def demo_segment(features):
    declarations = {
        "churn": ("churn_probability", 0.25, PredictionMetadata(
            "dummy-churn", "dummy_no_session_28d", 28, "days", "probability")),
        "uplift": ("uplift_score", 0.1, PredictionMetadata(
            "dummy-uplift", "dummy_purchase_difference_4w", 4, "weeks", "probability_difference")),
        "clv": ("predicted_clv_12m", 100.0, PredictionMetadata(
            "dummy-clv", "dummy_paid_amount_52w", 52, "weeks", "nonnegative_amount")),
    }
    predictions = {}
    for model, (column, value, metadata) in declarations.items():
        frame = features[list(CONTEXT_COLUMNS)].copy()
        frame["input_cutoff_at"] = frame.as_of_at  # 더미 계약의 cutoff 선언
        offset = (pd.Timedelta(weeks=52) if model == "clv"
                  else pd.Timedelta(days=28))
        frame["prediction_end_at"] = frame.as_of_at.map(lambda point: point + offset)
        frame[column] = value
        predictions[model] = PredictionBatch(frame, metadata)
    inputs = integrate_inputs(features, predictions)
    return segment_customers(inputs, SegmentationConfig())


# 기존 Feature Store 경로를 지정해 실행한다.
# features = load_feature_store("<existing_features_directory>")
# result = demo_segment(features)
# print(result.customers[["customer_id", "segment_id", "priority_score", "priority_rank"]])
```

이 예시의 cutoff와 종료 시각은 더미 계약을 재현한 것이다. 실제 연결에서는 모델 생산자가
제공한 정의·기간·cutoff·출처를 검증해야 하며 Feature에서 값을 복사하는 것만으로 모델 출처가 증명되지 않는다.
출력이 필요하면 `export_segments(result, "<new_output>/segments_6plus.csv")`를 호출한다.
시각화는 `plot_segments(result, dataset_id, as_of_at)`으로 모집단을 명시해 선택한다.

## 7. 테스트

- [test_inputs.py](tests/test_inputs.py): 102개 사례. 키·출처·시간대·기간·수치·NULL·미래 컬럼·입력 비변경성,
  실제 publisher의 Parquet·manifest·checksum 소비를 검증한다.
- [test_segmentation.py](tests/test_segmentation.py): 73개 사례. 12개 조합·경계·동점·상위 비율·점수·제외 사유,
  다중 시각 분리·통계·CSV/JSON 재로딩·시각화·비변경성을 검증한다.

두 파일 합계는 175개다. CLV 52주·364일 종료 시각, 윤년·월말, 이전 기간 선언 거절과 빈 Segmentation을 검사한다. 실제 모델 번들·외부 예측 파일은 사용하지 않는다.
publisher 관련 7개는 raw 로딩과 Feature 계산을 fixture로 대체하고 실제 v5 publisher를 실행한다.
SQLite/Parquet 저장과 manifest·로더 호환성 검사이며 실제 대규모 시뮬레이션 데이터 검증은 아니다.

환경은 저장소의 `simulator/pyproject.toml`을 참고한다. Python 3.10 이상,
NumPy·pandas·PyYAML·pyarrow·pytest 및 시각화용 matplotlib이 필요하다.
기존 호환 가상환경에서 **저장소 루트 기준**으로 실행한다.
`python -m pytest`가 저장소 루트를, 설정 파일이 `simulator/src`를 import 경로에 추가한다.

Windows 기본 pytest 임시 경로 접근 문제를 피하도록 매 실행마다 새로운 GUID 기반
`--basetemp`를 사용한다. pytest가 해당 경로를 정리할 수 있으므로 사용자 데이터 경로를 지정하지 않는다.

```powershell
$env:PYTHONPATH = (Resolve-Path 'simulator/src').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path (Resolve-Path '.venv').Path 'verification/mplconfig-segmentation'

$baseTemp = Join-Path (Resolve-Path '.venv').Path (
    'verification/pytest-first-commit-' + [guid]::NewGuid().ToString('N')
)

.venv/Scripts/python.exe -B -m pytest `
    -c simulator/pyproject.toml `
    final_strategy/tests/test_inputs.py `
    final_strategy/tests/test_segmentation.py `
    --basetemp="$baseTemp" `
    -q -ra -p no:cacheprovider
```

publisher 제외 조건을 넣지 않는다. 테스트용 임시 산출물은 Git 제외 `.venv/verification/` 아래에 둔다.

CLV 52주 변경 전인 2026-11-19, Python 3.12.10의 기존 Windows 가상환경에서 두 파일을 실행한 기록은
**165 passed, 0 failed, 0 skipped (21.17초)**였다. publisher 관련 7개도 포함한다.
NumPy timedelta 관련 경고 2,398개를 숨기지 않았으며, 향후 라이브러리 호환성 검토가 필요하다.
CLV 52주 변경 후 Phase A의 기존 실행 기록은 **175 passed, 0 failed, 0 skipped (36.17초)**다.
publisher 7개를 포함했고 경고 3,438개를 숨기지 않았다. 이번 스테이징 준비에서 재실행한 결과는 아니다.
이 검증은 실제 모델 추론 성공을 의미하지 않는다.

## 8. 현재 제약

- 실제 Churn·Uplift·CLV 출력 연결은 미완료이며 내부 batch 계약과 생산자 계약의 대조가 필요하다.
- 현재 입력·Segmentation은 CLV 52주(364일)를 요구하며 달력 12개월과 동일시하지 않는다.
  기존 컬럼명 `predicted_clv_12m`은 유지한다. 기간 계약 변경과 실제 CLV 출력 Adapter 연동은 별개이며,
  실제 모델의 출처·snapshot·금액 단위·예측 정의 연결은 추가 검증이 필요하다.
- 개발용 Segmentation 임계값과 분석 eligibility는 확정 사업 정책이나 실제 발송 승인 기준이 아니다.
- Priority Score는 결과 기간이 다른 예측의 곱인 대리지표다. 실제 증분 매출·이익·ROI 산출은 이 공개 범위 밖이다.
- 시뮬레이터 v5와의 독립 fixture 검증은 실제 운영 데이터·모델 추론·전체 생산자 출처 검증과 구분한다.
- 허용된 정상 NULL을 보존하지만, 고객별 예측값 누락은 거절한다. 미확정 모집단 정책으로 고객을 자동 제거하지 않는다.
