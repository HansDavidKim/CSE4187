# AmExpert 시간·역할 분리 feature 명세

> **주의: 이 문서는 v1의 보존 기록이며 사용 중지.** 개인별 오퍼 기록 시점 누출과 비구매 0 가정 문제를 수정한 [v2 명세](AMEXPERT_TEMPORAL_FEATURES_V2.md)를 사용한다.

이 산출물은 NTS-NOTEARS나 IRL 학습 결과가 아니라 이들에 사용하기 위한 **후보 feature 패널**이다. 기존 산출물을 덮어쓰지 않는다. 행 단위는 `customer_id × 완전히 관측된 월요일~일요일 주`이며 `week_end_exclusive`는 다음 월요일이다.

## 변수와 역할

| 역할 | 변수 | 정의와 조건 |
|---|---|---|
| 과거 상태 후보 | `prev_purchase`, `prev_transaction_rows`, `prev_spend`, `prev_quantity`, `prev_coupon_discount`, `prev_other_discount`, `prev_unique_items` | 동일 고객의 직전 주 관측 결과. 현재 주 결과를 당겨 쓰지 않음 |
| 이력 가용성 | `history_available` | 첫 주는 false이고 위의 과거 변수는 null이다. 첫 주 이력을 0으로 만들어내지 않는다 |
| 개입 기록 대리 | `preweek_active_offer_record_count` | `train.csv`에 기록된 고객·캠페인·쿠폰 중 캠페인 시작일이 주 시작일보다 앞서고 종료일이 주 시작일 이상인 조합 수. 중복 키는 1개로 센다 |
| 개입 기록 상태 | `preweek_offer_record_status` | 기록이 있으면 `prior_start_active_record`, 없으면 `no_record_exposure_unknown`. 양쪽 모두 실제 전달 시각을 확인했다는 의미는 아니다 |
| 해당 주 시작 기록 | `within_week_campaign_start_record_count` | 해당 주 시작한 캠페인의 기록 수. **주 시작 시점에는 미래일 수 있으므로 결정 전 feature가 아니며**, 주간 처치와 구매의 선후를 단정하는 용도로 쓰지 않는다 |
| 사용자 행동 | `purchase` | 해당 주 거래 행이 있으면 1, 없으면 0. 관측기간 전체에서 거래파일이 완전하다는 가정 아래의 관측 구매·비구매 대리 |
| 구매 후 관측 결과 | `transaction_rows`, `quantity`, `spend`, `other_discount`, `coupon_discount`, `unique_items` | 해당 주 거래의 행 수·수량/가격 합·할인 부호 반전 합·서로 다른 품목 수. 같은 주 구매 결정을 예측하는 사전 상태로 사용하지 않는다 |

역할별 분리는 인과성이 확인되었다는 뜻이 아니다. 특히 거래와 가격·할인·품목 수의 같은 주 관계는 거래 생성 방식상 구조적 관련이 있으므로, 이를 구매의 선행 원인으로 읽지 않는다. `redemption_status`는 정확한 redemption 시점이 확인되지 않아 제외했다. 할인 결과를 다음 주 이력에 쓰는 것과 같은 주 결정 전 원인으로 쓰는 것은 다르다.

## 중요한 데이터 가정

- 쿠폰 기록만으로 실제 고객 전달 여부와 전달 시점을 검증할 수 없다. 캠페인이 주 시작 전 시작된 기록만 남겨도 **기록의 실제 도착이 주 시작 전이었다고 보장하지 않는다**. 외생 처치라고 단정하지 않고, 민감도 분석 및 추가 근거가 필요하다.
- `train.csv` 기록만 읽으며 `test.csv`에서 온 오퍼는 포함하지 않는다. 미기록 상태에는 이 문제를 포함한 불확실성이 남는다. 기록 수 0은 관측된 기록의 개수일 뿐, 처치 0이 아니다. 원자료 `redemption_status`는 feature에 결합하지 않는다.
- 전체 거래 관측 시작·종료 사이의 모든 가구·주는 관측됐다고 가정하고 거래 미기록을 구매 0으로 표시한다. 이 가정은 파일 자체로 완전히 검증되지 않았으며 누락 거래 또는 이탈이 있다면 결과가 편향된다.
- 전체 고객에 동일 관측기간을 적용하므로 고객별 가입·관측 시작·이탈 시점은 보장되지 않는다. 첫 완전 주의 직전 이력은 미확인 null이다. 외곽 불완전 주의 거래는 보고서에 별도로 센다.
- campaign start/end는 포용적인 날짜로 처리한다. 주 중 시작된 캠페인과 주 중 종료되는 캠페인에 대한 연속 노출이나 구매 시점의 적용 여부를 보장하지 않는다. 실제 offer를 스케줄대로 replay하거나 무작위 처리로 취급해서는 안 된다.
- 단일 컬럼/lag/type에 기반한 변수 분리는 temporal leakage를 줄이기 위한 입력 명세일 뿐이며, 숨은 교란·선택 편향·인과 방향을 해결하지 않는다. NTS에서는 이에 맞는 그래프 제약과 결측 처리를 별도로 구현하고 검증해야 한다. 현재 NTS 학습기는 null을 안전하게 처리하는 인터페이스가 없으므로 이 패널을 무턱대고 넣지 않는다.

## 실행

```bash
PYTHONPATH=src python scripts/prepare_amexpert_temporal_features.py --data '/Users/kimdaewon/Desktop/캡스톤디자인/data/raw/kaggle/amexpert'
PYTHONPATH=src pytest -q
```

`artifacts/amexpert-temporal-features/feature_report.json`에 집계와 역할 정의, `local_customer_week_features.csv.gz`에 로컬 고객별 패널을 저장한다. 데이터 권리가 확인되기 전에는 개별 고객 단위 원본 또는 파생 파일을 외부에 배포하지 않는다.
