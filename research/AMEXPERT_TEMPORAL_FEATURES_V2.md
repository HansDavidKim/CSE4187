# AmExpert 주간 feature 명세 v2

고객 × 완전한 월요일~일요일 주 단위의 탐색·감사용 패널이다. 인과 DAG, NTS-NOTEARS 입력 또는 처치 효과 데이터가 아니다. v1 산출물은 보존하되 고객별 오퍼 시점에 관한 미래정보가 섞일 수 있어 분석에 사용하지 않는다.

| 역할 | 변수 | 정의 |
|---|---|---|
| 식별자 및 시간 | `customer_id`, `week_start`, `week_end_exclusive` | 고객 ID, 주 시작일과 다음 주 시작일 |
| 과거 상태 후보 | `prev_purchase`, `prev_transaction_rows`, `prev_spend`, `prev_quantity`, `prev_coupon_discount`, `prev_other_discount`, `prev_unique_items` | 해당 고객 직전 주 관측 결과. 기록이 없으면 미확인 null |
| 이력 유무 | `history_available` | 패널에 직전 주 행이 존재하면 true. 실제 관측 완전성을 뜻하지 않음 |
| 개입 | 없음 | 개인별 발송·열람·배정 시점이 확인되지 않아 어떤 주에도 고객별 개입을 배정하지 않음 |
| 행동 | `purchase` | 거래 기록이 있으면 1, 없으면 미확인 null. 비구매 0을 확인할 근거는 없음 |
| 구매 후 결과 | `transaction_rows`, `quantity`, `spend`, `other_discount`, `coupon_discount`, `unique_items` | 거래 행 수, 수량·가격 합, 할인 부호 반전 합, 고유 품목 수. 거래 기록이 없는 주는 null |

## 중요한 제한

- `train.csv` 고객별 쿠폰 기록과 캠페인 날짜는 무결성 및 중복 집계에만 사용한다. 고객별 배정 시각을 알 수 없으므로 해당 기록을 고객-주 행에 결합하지 않는다.
- `purchase` 결측은 관측된 비구매와 미관측을 구분할 근거가 없다는 뜻이다. 따라서 이 패널로 비구매 확률, 행동 선택 확률, uplift 또는 개입 효과를 바로 식별할 수 없다.
- 고객 목록은 전체 기간에 거래가 한 번 이상 존재한 고객을 기준으로 한다. 연구 목적의 사후 선택이며 매주 관측 대상이었는지를 증명하지 않는다. 거래 없는 고객과 고객별 가입·이탈 정보도 알 수 없다.
- `purchase`는 거래 행 존재 여부와 결정적으로 연결된다. 직전 주 변수도 원래 결과 열의 lag 복제다. 이들을 NTS-NOTEARS에 전부 독립 노드로 투입하지 않는다. 노드·시차 선택, 동시점 순서 제약, 결측 처리 및 데이터 적합성을 별도로 설계해야 한다.
- 캠페인·거래 시각, 숨은 교란, 선택 편향은 해결되지 않았다. `redemption_status`도 정확한 발생 시점을 몰라 제외한다. feature 역할은 인과 방향의 증거가 아니다.
- 개별 고객 단위 데이터의 권리가 확인되기 전까지 원본 또는 파생 패널을 외부에 배포하지 않는다.

## 실행

```bash
PYTHONPATH=src python scripts/prepare_amexpert_temporal_features.py --data '/Users/kimdaewon/Desktop/캡스톤디자인/data/raw/kaggle/amexpert'
PYTHONPATH=src pytest -q
```

결과는 `artifacts/amexpert-temporal-features-v2/`에 저장된다. v1 결과는 자동 수정하거나 삭제하지 않는다.
