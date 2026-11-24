# Uplift 모델 API 인계

10% 쿠폰 발송 배정이 고객의 향후 28일 내 구매 확률을 얼마나 바꾸는지 추정한다. 점수 `0.05`는 구매 확률 5%p 증가를, 음수는 구매 확률 감소를 뜻한다.

## 1. 학습 및 CSV 생성

저장소 루트에서 실행한다. 학습 데이터에는 고객당 한 번의 1:1 무작위 배정(`treatment=1` 발송군, `0` 미발송군)과 배정 후 28일 구매 결과가 필요하다.

```bash
python -m pip install -r uplift/requirements.txt
python -m uplift.train --data simulator/artifacts/features \
  --out uplift/artifacts/handoff --seed 0
```

`--data`에는 `experiment_log`가 있는 `features.sqlite` 또는 해당 파일의 디렉터리를 지정한다. YAML의 `source_simulation.min_customer_history_days`를 바꿨다면 데이터를 새로 생성한다. 학습 로더는 SQLite에 저장된 적격 조건을 확인하므로 `0`, `84`, `90`일 설정 모두 같은 명령으로 학습할 수 있다. [시뮬레이터 실행 안내](../simulator/README.md)

| 출력 파일 | 사용 방법 |
| --- | --- |
| `uplift_predictions.csv` | `model`에서 확정한 모델 하나(`T-learner`, `X-learner`, `H-learner`)를 선택하고, `dataset_id`, `customer_id`, `t0`(배정 기준 시각, UTC)로 같은 실행의 다른 모델 결과와 연결한다. 고객·시각·모델당 한 행이며 `segment`에 추정 세그먼트가 담긴다. |
| `evaluation.csv` | 모델별 `qini`, `qini_decile`, `roc_auc`, `n_test` 등을 확인한다. Qini는 타기팅 순위를, ROC AUC는 실제 배정군에서의 구매 확률 예측을 평가한다. |
| `tlearner.pkl`, `xlearner.pkl`, `hlearner.pkl` | 선택한 모델을 저장해 새 고객에 적용한다. |

## 2. 새 고객 점수 계산

아래 함수의 `batch`는 같은 배정 시각에 평가할 적격 고객의 사전 피처 DataFrame이다. `features.sqlite`에서는 `common.simulator_features`의 `open_feature_store`와 `read_kit_features(connection, FEATURES)`로 읽을 수 있다. 배정 전 데이터로 만든 피처 36개를 `FEATURES` 순서대로 전달한다. [피처 사전](../uplift/feature.md)

```python
from uplift.common.schema import FEATURES
from uplift.Xlearner import XLearner


def score_customers(batch, model_path="uplift/artifacts/handoff/xlearner.pkl"):
    batch = batch.reset_index(drop=True)
    model = XLearner.load(model_path)
    predictions = model.predict(batch[FEATURES])
    keys = batch[["dataset_id", "customer_id", "as_of_at", "snapshot_id"]]
    return keys.rename(columns={"as_of_at": "t0"}).join(predictions)
```

예시는 X-learner를 사용한다. 최종 확정 모델에 따라 `TLearner` 또는 `HLearner`와 해당 파일로 바꾼다. 세 모델의 `predict(X)`는 같은 행 순서와 인덱스로 다음 값을 반환한다.

| 컬럼 | 의미 |
| --- | --- |
| `p0` | 미발송 배정 시 28일 내 구매 확률 |
| `p1` | 발송 배정 시 28일 내 구매 확률 |
| `uplift_score` | `p1 - p0`. Uplift 기준 타기팅은 값이 큰 고객부터 선정한다. |

## 3. 최종 전략 A/B 테스트

- 모델 자체의 비교 지표는 `evaluation.csv`에 있다. 대응하는 예측 행은 `split=test`이고, `split=unobserved`는 28일 결과 관측이 완료되지 않은 고객이다.
- 최종 전략 비교에는 학습·튜닝에 쓰지 않은 별도 고객을 사용한다. 모델과 발송 정책을 고정한 후 사전 피처로 점수를 계산하고, 전략군에 무작위 배정해 실제 관측 결과를 비교한다. [프로젝트 요구사항](request_specification.md)
