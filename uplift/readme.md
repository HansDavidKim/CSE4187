업리프트 모델 워크스페이스
시뮬레이터에 이미 어느정도 있어서 안쓸수도

## 구조

| 경로 | 내용 |
| --- | --- |
| `Tlearner/`, `Xlearner/`, `Hlearner/` | 세 uplift 모델. 모두 `fit(X, t, y)` 후 `predict(X)`로 `p0, p1, uplift_score` 반환 |
| `common/schema.py` | 시뮬레이터 v5 계약: 원시 테이블 컬럼, 기간(28일·30일), KIT39 목록 |
| `common/raw.py` | 시뮬레이터 실행 디렉터리(`raw.sqlite`, 배정)를 읽어 `RawData`로 반환. 세 모델 공용 |
| `common/features.py` | 원시 로그 → KIT39 + T + y, 고객 1명당 1행 학습 데이터셋. 세 모델 공용 |
| `common/evaluate.py` | Qini curve, Qini 계수, 분위별 실제 uplift |
| `common/synthetic.py` | 시뮬레이터 실행 결과가 없을 때 쓰는 임시 원시 로그 (점검용) |
| `train.py` | 세 모델 학습·비교·저장 |

## 실행

저장소 루트에서 실행한다.

```bash
pip install -r uplift/requirements.txt
python -m uplift.train                                    # 임시 원시 로그로 점검
python -m uplift.train --data simulator/artifacts/v5_raw  # 시뮬레이터 실행 디렉터리로 학습
python -m pytest uplift/tests
```
