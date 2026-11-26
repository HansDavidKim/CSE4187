# 28일 구매 Uplift

적격 고객에게 10% 쿠폰을 발송하도록 배정했을 때의 구매 확률 변화량을 T/X/H-learner로 추정한다. 세 모델은 동일한 고객 분할과 배정 전 피처 36개를 사용한다. 설계와 개별 산식은 [구현 계획](plan.md)과 [피처 사전](feature.md)에 있다.

## 실행

저장소 루트에서 실행한다. 최신 주간 계약은 과거/미래 각각52주, 특징 창4/8/13주,
결과4주다. 이름에 `_30d`/`_90d`가 남은 열도 주간 manifest가 실제 기간을 결정한다.
시뮬레이터의 공식 Feature Store를 읽으며 39개 특징을 주간 집계로 다시 만들지 않는다.

```bash
python -m pip install -e "./simulator[dev]" -r uplift/requirements.txt
PYTHONPATH=simulator/src:. python -m uplift.train \
  --data simulator/artifacts/features --ledger simulator/artifacts/small \
  --out uplift/artifacts/run1
```

아직 시뮬레이터 출력이 없다면 시뮬레이터 의존성이 설치된 환경에서 다음 명령으로 작은 실험을 만들 수 있다. 생성 경로는 새 디렉터리 또는 빈 디렉터리여야 한다.

```bash
PYTHONPATH=simulator/src:. python -m cli v4 generate \
  --config simulator/src/simulate/v4/config/population.yaml \
  --n-customers 200 --weeks 104 --seed 42 --output simulator/artifacts/small
PYTHONPATH=simulator/src:. python -m cli v4 features \
  --data simulator/artifacts/small --output simulator/artifacts/features
```

분리한 features 경로로 학습할 때는 `--ledger`에 원본 데이터 디렉터리 또는 `raw.sqlite`를 준다.
해시가 같은 원장과 그 manifest의 무작위 배정 설계를 확인한다. `auto`와 무작위 `assign`의
단일 propensity(예: .25)는 지원하고 `none`, 전원 처치, 명시 고객 지정은 양군의 무작위
비교를 식별하지 못하므로 학습을 거절한다. 예측은 이들 모드도 지원한다.

구형 달력 자료의 raw 입력/SQLite API는 보존한다. 그 재계산 창은30/60/90일이며,
주간 데이터에는 적용하지 않는다. 입력 생략 시 합성 예제는 구현 점검용이며 새 모형 성능의
증거가 아니다. 학습·평가 분할 시드는 `--seed`로 지정한다(기본0).

## 최종 전략에 구매확률 연결

학습은 기존 T/X/H 구현을 그대로 쓴다. 모델에는 실제 특징 순서와 기간 계약을 저장한다.
기간 정보가 없는 구형 모델이나30/60/90일 학습 모델을 주간 특징에 적용하면 거절한다.
새 시드로 만든 별도 결정용 Feature Store에서 한 모델을 선택하여 추론한다.

```bash
PYTHONPATH=simulator/src:. python -m uplift.predict \
  --model uplift/artifacts/run1/tlearner.pkl \
  --features simulator/artifacts/decision_features --output uplift/artifacts/handoff1
```

추론은 특징만 읽으며 처치 배정과 미래 라벨은 읽지 않는다. 학습/선택 고객이 포함된
동일 데이터에 대한 재사용은 기본 거절하고 `--allow-training-customers`는 개발 점검에만 쓴다.
출력 `prediction_batches.json`의 `uplift` 항목을 Churn/CLV handoff와 결합하면 기존
`python -m final_strategy`의 `--predictions`로 읽을 수 있다. 다른 필수 모델을 만들거나
이탈 확률을 구매 확률로 변환하지 않는다. Python API는 다음과 같다.

```python
from uplift.api import predict_for_strategy
from final_strategy import PredictionBatch
result = predict_for_strategy("uplift/artifacts/run1/tlearner.pkl", "DATA/features")
batch = PredictionBatch(result.frame, result.metadata, result.producer_metadata)
```

batch는 동일 `(dataset_id, customer_id, as_of_at)`/snapshot/원장 해시,
`input_cutoff_at=t0`, `prediction_end_at=t0+4주`, 구매 `p0/p1`과 차이값을 보존한다.
Tianchi 일별 자료는 `purchased_28d`를 읽고 미관측 금액/장바구니를0으로 만들지 않는다.
Retention의 `--integrate`도 `load_feature_dataset(..., ledger=raw_dir)`와 동일 API를 쓴다.
세그먼트는 예측에 대한 기술적 규칙이며 개인의 반사실적 유형이나 실제 인과효과를 식별하지 않는다.

## 통합 출처와 의존성

T/X/H learner, raw/SQLite 계약, 원 평가 산식과 기존 테스트는 `origin/feat/uplift`
커밋 `d451cf8`에서 선별 도입했다. `origin/yuseung`은 이 실행 코드보다 오래된 문서 브랜치다.
기존 코드 수식은 유지하고 주간 adapter, 저장 모델의 기간 선언, 전략 handoff와 연결 검사를 추가했다.
공용 `common/simulator_features.py`도 같은 브랜치의 유지 API를 재사용한다.
연구 PDF/그림·모델 파일은 가져오지 않는다. Uplift 자체는 NumPy/pandas/PyArrow/scikit-learn>=1.4를
사용하고, 시뮬레이터와 Retention/CLV 전체 실행에는 각 모듈의 requirements도 설치해야 한다.

## 출력

| 파일 | 내용 |
| --- | --- |
| `uplift_predictions.csv` | 적격 고객 × 모델당 한 행. `dataset_id, customer_id, t0, split, p0, p1, uplift_score, model, model_version, segment` |
| `evaluation.csv` | 모델별 평가 고객 수·관측 ATE·전체 순위 Qini(`qini`)·10분위 Qini(`qini_decile`)·구매 ROC AUC·평균 예측 효과 |
| `qini_curve.csv` | 평가 고객의 타기팅 비율별 누적 증분 구매와 무작위 기준선 |
| `qini_decile_curve.csv` | 배정군별 상위 0%부터 100%까지 10% 간격의 누적 구매율 차이와 무작위 기준선 |
| `dataset.parquet` | 학습 입력의 원본 스냅샷, KIT39, 처치와 구매 정답 |
| `{t,x,h}learner.pkl` | 재사용할 학습 모델 |

`p0`와 `p1`은 각각 미발송·발송 시의 28일 구매 확률이고, `uplift_score = p1 − p0`다. 예를 들어 0.05는 구매 확률 5%p 증가에 해당한다. X와 H는 추정 효과를 미발송 확률에 더해 발송 확률을 구성하고 [0, 1] 범위로 제한한다. `split=test` 행으로 평가하고, 정답 관측이 끝나지 않은 고객은 `split=unobserved`로 예측만 제공한다. 후속 전략에서 고객별 점수를 사용할 때는 `model`로 사용할 모델 하나를 선택한다.

구매 ROC AUC는 각 평가 고객의 실제 배정군에 해당하는 확률(`p0` 또는 `p1`)이 구매 여부를 얼마나 잘 구분하는지 측정한다. 평가 라벨에 한 클래스만 있으면 공란으로 기록한다. Uplift 순위 성능은 Qini로 평가한다.

`qini_decile`은 배정군별로 Uplift 점수를 정렬하고 같은 상위 비율까지의 누적 구매 수를 각 군의 전체 고객 수로 나눈 뒤 차이를 구한다. 10분위 점들을 직선으로 연결해 무작위 기준선과의 면적을 적분한다. 분위 경계가 동점 구간을 가르면 그 구간의 평균 구매율을 적용한다. 논문 표 7의 표시 단위와 맞추려면 CSV 값을 1,000배 한다. [Gubela 외 §5–6.1](https://www.researchgate.net/publication/331791032_Conversion_Uplift_in_E-Commerce_A_Systematic_Benchmark_of_Modeling_Strategies)

```bash
PYTHONPATH=simulator/src:. python -m pytest -q uplift/tests
```
