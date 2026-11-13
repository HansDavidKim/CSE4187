업리프트 모델 워크스페이스
시뮬레이터에 이미 어느정도 있어서 안쓸수도

## 구조

| 경로 | 내용 |
| --- | --- |
| `Tlearner/`, `Xlearner/`, `Hlearner/` | 세 uplift 모델. 모두 `fit(X, t, y)` 후 `predict(X)`로 `p0, p1, uplift_score` 반환 |
| `common/schema.py` | 시뮬레이터와 합의한 컬럼 이름, 기간(4주·12주), feature 목록 |
| `common/features.py` | 고객별 주간 합계 → 고객 1명당 1행 학습 데이터셋 |
| `common/evaluate.py` | Qini curve, Qini 계수, 분위별 실제 uplift |
| `common/synthetic.py` | 시뮬레이터가 나오기 전까지 쓰는 임시 데이터 (점검용) |
| `train.py` | 세 모델 학습·비교·저장 |

## 실행

저장소 루트에서 실행한다.

```bash
pip install -r uplift/requirements.txt
python -m uplift.train                     # 임시 데이터로 점검
python -m uplift.train --dataset data.csv  # build_dataset 결과로 학습
python -m pytest uplift/tests
```
