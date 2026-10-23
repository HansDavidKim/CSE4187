# Backend-TS v2: feature와 실험 명세

GPT-5.6 Sol high에게 feature 설계를 의뢰한 결과를 바탕으로 구현했다. 주간 episode는 서로 다른 시점의 이력과 label을 별도 배열로 보관한다. 자료 제공자의 [생성 코드](https://github.com/nannullna/ts4uplift/blob/main/dataset/utils.py)와 [전처리 노트북](https://github.com/nannullna/ts4uplift/blob/main/preprocess/preprocess_time_series.ipynb)의 기준 시각을 이용하며, 기준 전 이벤트만 사용한다.

## Feature schema

| 역할 | 필드 | 정의 |
|---|---|---|
| 사전 이력 | `history[N,168,5]` | 기준 전 7일, 1시간 bin의 `LOGIN`, `GET`, `POST`, `PUT`, `DELETE` 기록 수 `log1p` |
| grid mask | `history_grid_mask[N,168,5]` | 고정 grid 존재 여부. 전부 true이며 서비스의 실제 관측 완전성이나 사용자 활동 여부를 뜻하지 않음 |
| 개입 | `treatment[N]` | episode 단위 실제 push 시점(1) 또는 일주일 전 pseudo-control(0). 시간별 값이 아님 |
| 결과 | `outcome_y3[N]` | 기준 이후 `[0,3시간)` 로그인 여부 |
| 라벨 mask | `endpoint_mask[N,2]` | 공개된 T/Y 라벨의 존재 여부. 원자료에서 전부 true |
| metadata | `episode_labels.csv` | user, campaign, index, history 시작, 결과 window 끝, pair/episode ID, split 등. 모델 node가 아님 |

0 로그는 기록이 없었다는 뜻일 뿐 서비스 전체 활동 부재나 로그 수집 완전성을 증명하지 않는다. 기준 후 raw login 원자료는 archive에 없어 Y3를 독립 검증할 수 없다. 공식 코드에서 T/Y의 관측 방식과 7일 전 pseudo-control을 확인했다. 제공된 episode는 사전 7일 내 로그인 사용자를 대상으로 선택되므로 전체 사용자 집단이 아니다. 실제 push 열람·기기 및 숨은 교란은 관측되지 않는다.

## User-disjoint 분리

6월 테스트 사용자 전체를 4–5월 train/validation에서 제외한다. 773 episode가 제외되고 train 2,024 / validation 540 / test 821 episode다. 각 분할 고객 수는 각각 1,506 / 387 / 632명이다. 세 집합 간 고객 교집합은 0이다. 동일 사용자의 paired episodes는 같은 집합에 둔다. 6월의 사용자별 중복 episode는 유지했으므로 원 논문의 test pair 처리와 동일한 실험은 아니다. 원래의 warm-user temporal test와 다른 모집단이다.

기존 패널의 동일 고객·캠페인 1,011쌍에서 control Y3가 treated 이력의 첫 3시간에 그대로 포함됐다. 이것은 cross-split leakage 위험의 실제 예시이며, 같은 고객의 다른 시점 결과들이 counterfactual 쌍이라는 뜻은 아니다. 고객 분할은 분할 간 누출을 막지만 train 내 통계적 의존성을 제거하지 않으므로 재표본추출은 고객 단위로 한다.

## NTS와 prediction

현재 NTS는 시간별 이력 노드만 입력한다. T/Y를 시간별 0으로 채우거나 반복하지 않는다. T/Y에 대한 endpoint-aware masked causal model은 아직 구현하지 않았으므로 처치 및 결과를 포함한 DAG를 얻은 것이 아니다. 이전 시간 bin → 이후 시간 bin edge는 후보 패턴이고 인과적으로 식별된 edge가 아니다. 1시간 내 동시점 방향은 알 수 없어 `--forbid-instantaneous`를 사용했다. 이로 인한 acyclicity 만족은 사전 제약의 결과지 인과 정확도 증거가 아니다.

1-lag, 임계값 0.3, 고객 150명 복원추출을 5회만 시행한 탐색에서 POST(t−1)→GET(t)와 POST(t−1)→POST(t)는 5/5회, GET(t−1)→GET(t)와 GET(t−1)→POST(t)는 4/5회, POST(t−1)→LOGIN(t)는 3/5회였다. 표본 고객 수가 전체 훈련 고객보다 작고 실행마다 초기화 seed도 함께 바뀌었다. 5회는 안정성 판정에 부족하며 표본·초기화·threshold 효과가 섞여 있다. edge norm은 비음수 강도로 방향의 부호나 인과효과 크기가 아니다. [요약 결과](../artifacts/backend-ts-bootstrap/summary.json).

비교용으로 prehistory 합계와 T로 Y3를 예측하는 logistic baseline을 만들었다. 학습에서만 scaler를 학습하며 0/1 T를 유지한다. 훈련과 test 사용자가 다른 시간 test에서 AUC는 약 0.878, log loss는 약 0.280이다. 이는 예측 진단일 뿐 uplift나 인과 효과가 아니다. 원본 시점별 차이, pseudo-control 및 선택 편향이 남는다.

훈련 데이터에 대해서만 고객 묶음 5-fold out-of-fold propensity 진단을 실시했다. 선택된 이력 요약에 따른 propensity AUC는 약 0.585, `[0.05,0.95]` 밖 비율은 약 0.05%다. 이것은 미관측 교란이 없다는 증거나 충분한 causal positivity를 증명하지 않는다. IPW ESS는 episode 단위 가중치에 대한 통계이고 고객 상관을 반영한 ESS는 아니다.

## 실행

```bash
PYTHONPATH=src python scripts/prepare_backend_ts.py --archive data/raw/backend-ts/backend-ts_v0.1.zip
PYTHONPATH=src python scripts/audit_backend_episode_splits.py
PYTHONPATH=src python scripts/prepare_backend_ts_v2.py
PYTHONPATH=src python scripts/run_backend_outcome_baseline.py
PYTHONPATH=src python scripts/audit_backend_overlap.py
PYTHONPATH=src pytest -q
```

`artifacts/backend-ts-features-v2/`의 데이터는 개인 단위의 로컬 파생 자료다. 논문에 공유하기 전 데이터 이용조건과 [공식 저장소 안내](https://github.com/nannullna/ts4uplift#disclaimer)를 검토한다.
