# 28일 구매 Uplift

적격 고객에게 10% 쿠폰을 발송하도록 배정했을 때의 구매 확률 변화량을 T/X/H-learner로 추정한다. 세 모델은 동일한 고객 분할과 배정 전 피처 36개를 사용한다. 설계와 개별 산식은 [구현 계획](plan.md)과 [피처 사전](feature.md)에 있다.

## 실행

저장소 루트에서 실행한다. 시뮬레이터의 `v4 features`로 만든 디렉터리 또는 그 안의 `features.sqlite` 파일을 지정한다.

```bash
python -m pip install -r uplift/requirements.txt
python -m uplift.train --data simulator/artifacts/features --out uplift/artifacts
```

아직 시뮬레이터 출력이 없다면 시뮬레이터 의존성이 설치된 환경에서 다음 명령으로 작은 실험을 만들 수 있다. 생성 경로는 새 디렉터리 또는 빈 디렉터리여야 한다.

```bash
PYTHONPATH=simulator/src python -m simulate.v4.cli v4 generate \
  --config simulator/src/simulate/v4/config/population.yaml \
  --n-customers 200 --months 6 --seed 42 --output simulator/artifacts/small
PYTHONPATH=simulator/src python -m simulate.v4.cli v4 features \
  --data simulator/artifacts/small --output simulator/artifacts/features
```

기존 raw 실행 디렉터리도 `--data`에 지정할 수 있다. 입력 없이 실행하면 합성 예제를 만든다. 학습·평가 분할 시드는 `--seed`로 지정한다(기본 0).

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
python -m pytest -q uplift/tests
```
