# eCommerce-Sim

관측 로그를 학습해 고객 행동을 생성하는 이커머스 시뮬레이터다.
조회·장바구니 추가·제거·구매·쿠폰 사용·구매 상품별 별점을 생성하고,
세션 간 장바구니와 주문 금액을 유지한다. 검색·CS·리뷰 본문은 생성하지 않는다.

## 코드 구조

```text
src/
  cli.py                   # 공통 실행 진입점
  simulate/v4/
    configuration.py       # 설정과 자료 경로
    simulator/             # 행동·방문·생애주기·거래 원장
    irl/                   # 원자료 기반 모형 학습
    data/                  # 데이터 로딩·검증·주간/실험 집계
    features.py            # 시점 이전 이력의 39개 피처
    storage.py             # 피처 SQLite 저장
    validation.py          # 원장·정답·원자료 정합성 대조
    quality.py             # 생성 분포 비교
    config/                # 실행에 필요한 번들 YAML/JSON
  analysis/                # 기존 연구 분석
  models/                  # 기존 정책·학습 코드
  preprocessing/           # 원자료 전처리
  utils/                   # 공통 경로·이벤트 도구
scripts/                   # 기존 실행 진입점과 SQL 예시
tests/                     # 회귀 검사
```

Git에는 실행 코드·테스트와 합성 프로필 생성에 필요한 집계 설정을 둔다.
원자료, 원천 고객별 학습 파일, 생성 데이터와 로컬 연구 산출물은 Git에서 제외한다.

## 실행

Python 3.10 이상에서 저장소의 `simulator/` 디렉터리 기준으로 실행한다.

```bash
cd simulator
python -m pip install -r requirements.txt
ecommerce-sim v4 generate --config src/simulate/v4/config/population.yaml --mode small \
  --n-customers 500 --months 6 --seed 42 --output artifacts/small
ecommerce-sim v4 validate --data artifacts/small/raw.sqlite
ecommerce-sim v4 features --data artifacts/small --output artifacts/features
```

설치 전에는 `PYTHONPATH=src python -m cli v4 ...`를 사용할 수 있다.
저장소 루트에서 설치할 때는 `python -m pip install -e "./simulator[dev]"`를 사용한다.
두 설치 방식 모두 `pyproject.toml`의 pandas 2.3.3 고정을 적용한다.
[population.yaml](src/simulate/v4/config/population.yaml)은 같은 디렉터리의 집계
설정을 읽어 고객 프로필, 상품 카탈로그, 구매 바구니와 방문 일정을 생성한다.
학습된 쇼핑·재방문·리뷰 모형과 방문·가격·바구니·리뷰 지연의 집계 분포를 사용한다.
원천 고객별 관측 프로필을 쓰는 실행과 생성 분포 및 데이터셋 ID가 달라질 수 있다.

합성 프로필 설정은 이산 쇼핑 정책과 GMM 재방문 모형을 사용한다. 각 고객의 쇼핑
보상에 따른 재방문 오프셋을 GMM에서 한 번 뽑아 전체 이력에 적용한다. GMM의
학습 범위에서 벗어난 쇼핑 보상에는 이산 모형의 재방문 오프셋을 적용한다.
구매 바구니의 상품은 같은 단가의 선호 SKU로만 바꿔, 가격 전이 모형이 선택한
주문의 금액과 수량을 보존한다.

동일 단가 상품 교체 규칙 적용 전, 별도 시드 222·223으로 각 2,000명·6개월을
생성해 고객별 프로필 기반 v10과 비교했다.
방문 수·주문 수·28일 미방문율·행동 비중·행동 전이의 오차를 같은 가중치로 비교했을 때
평균 오차는 이산 모형 0.1136, 전체 GMM 0.0810, 결합 모형 0.0642였다.

학습 자료는 [REES46](https://rees46.com/en/datasets)의
[Cosmetics Shop 행동 기록](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop),
[Tianchi CCF2016 O2O](https://tianchi.aliyun.com/competition/entrance/231593/information),
[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)다.

기존 학습 파일로 집계 모델을 다시 만들 때는 다음 명령을 사용한다.

```bash
PYTHONPATH=src python -m simulate.v4.irl.export_population \
  --models /path/to/assets/data/models --reward-gmm /path/to/fitted-gmm.json \
  --output artifacts/population/data/models
```

이 파일들을 사용하려면 `generate --config src/simulate/v4/config/population.yaml`에
`--assets-root artifacts/population/data/models`를 추가한다.

### 원천 고객별 프로필 기반 실행

[simulator.yaml](src/simulate/v4/config/simulator.yaml)에는 원천 고객별 프로필을
사용하는 실행 설정이 있다. `--assets-root`에 `data/models`를 포함한 디렉터리를
지정한다. 실제 원자료를 준비하고 `generate --fit-missing`으로 누락 모형을
학습할 수 있다. 기존 출력·모형은 덮어쓰지 않는다.

v5 기본 설정은 이산 성분 정책(`source-shopping-v10`)을 사용한다. 고객마다 원천
행동에서 학습한 대표 성향 하나를 배정하고, 해당 성향의 보상 계수와 고객의 방문 이력,
장바구니 상태를 사용해 행동을 샘플링한다.

Gaussian 경로(`source-shopping-v11`)를 재현하려면 기본 YAML과 같은 디렉터리에
설정 복사본을 만들고 `source_simulation.gaussian_reward_model_path`를
`data/models/shopping_gaussian_rewards.json`으로 지정해 `generate --config`에 전달한다.
Gaussian은 고객별 쇼핑 보상 offset 4개와 재방문 logit offset 1개를 한 번 뽑는다.
학습은 `python -m simulate.v4.irl.fit_gaussian_rewards --help`에서 확인할 수 있다.

`source-shopping-v12`는 고객별 쇼핑과 재방문 기록의 우도에서 혼합 비중·평균·전체
공분산을 근사 EM으로 학습한 GMM이다. `python -m simulate.v4.irl.fit_joint_reward_gmm --help`에
재학습 인자가 있다. 설정 복사본의 `source_simulation.reward_gmm_model_path`에 학습한
`data/models/shopping_reward_gmm.json`을 지정하고 `gaussian_reward_model_path`는 `null`로 둔다.
이 복사본을 `generate --config`에 전달한다. `generate --fit-missing`은 모집단 사전확률을 쓰는
GMM을 적합한다. 합성용 donor 성분 확률까지 저장하려면 학습기에 `--with-donor-posteriors`를
전달한다. 2단계 MAP-GMM 비교는 `simulate.v4.irl.fit_reward_gmm`에서 재현할 수 있다.

v12는 고정된 전이를 미리 묶고 Newton 갱신으로 soft Bellman 정책을 계산한다.
Bellman 잔차가 기준을 충족하면 정책을 사용하고, 계산이 수렴하지 않으면 기존 반복법으로
계산한다. 이전 30일 적격성·쿠폰 조건으로 Apple M3에서 진행한 20,000명·24개월·seed42
비교의 원시 생성과 저장 CPU 시간은
원래 v12 1,866.67초에서 1,132.63초로 줄었다. SQLite와 Parquet 14개의 체크섬은
같았다. 같은 조건의 v10은 807.97초였다. Cosmetics 5개월 분포 검증에서 GMM
보완안은 독립 고객 표본을 바꿨을 때 행동 비중, 행동 전이, 세션 길이 오차의 허용
기준을 모두 충족하지 못했다. 현재 28일 조건의 5개월 전체 분포에서는 v12가 두
보류 표본의 세 지표를 개선했다. 배정 이후 28일 대조군의 행동 비중 오차는 두
표본 모두에서 v10이 낮았다. GMM 설정을 선택하면 최적화한 v12로 실행한다.

업무 기록은 `raw.sqlite`의 고객·상품·세션·이벤트·주문·주문상품·쿠폰 7개 관계에
저장한다. `features.sqlite`는 공개 이력의 39개 피처와 배정·정답·주간/실험 테이블을
분리한다. 피처 이름과 순서는 [features.py](src/simulate/v4/features.py)의
`EXPANDED_FEATURES`와 파생 DB의 `feature_registry`를 따른다.
조회 예시는 [query_v4.sql](scripts/query_v4.sql)에 있다.

기본 실험의 적격성은 배정 직전 28일 방문 여부이며, 쿠폰은 배정 때 한 번 발송해
28일 동안 유효하다. 미래 28일 미방문은 기본 이탈 정답이고, 28일 미구매 정답은
실험 출력에서 별도로 제공한다. KIT의 `_30d` 피처와 원천 방문 모형의 30일 학습 창은
별도로 사용한다. 과거 30일 적격성·쿠폰 산출물은 해당 manifest 기준으로 읽는다.
쿠폰은 고객별 1:1 고정 배정하며 실제 발송과 사용을 구분한다. 별점은
실제 구매한 상품별 최대 한 번이다. 금액은 원자료의 통화·단위를 유지한다.
잠재 성향과 미래 정답은 예측 피처에 포함하지 않는다.

### 쿠폰 배정 전 최소 고객 이력

두 YAML의 `source_simulation.min_customer_history_days`로 쿠폰 배정 전 최소 이력
일수를 지정한다. 기본값 `0`은 제한 없음이며, `30`이면 첫 쇼핑 행동 기록부터 배정
시각까지 30일 이상 지난 고객만 선정한다. 가입일이나 행동한 날짜 수가 아닌 경과
기간이며, 정확히 설정 일수에 도달한 고객도 포함한다. 기존 `eligibility_window_days`의
최근 방문 조건과 함께 처치·대조 1:1 배정 전에 적용한다.

값은 비음수 정수여야 하며 사전 관측 기간을 넘을 수 없다. 외부 입력 고객이 조건을
충족하지 않으면 오류로 알린다. 적용값은 manifest와 실험 로그에 기록하고, 검증·내보내기에도
같은 기준을 적용한다. 값을 높이면 신규 고객이 제외되고 적격 고객 모집 시간이 늘어날 수 있다.
기존 데이터에는 소급 적용하지 않으므로 설정 변경 후 새 출력 경로로 생성한다.

### 쿠폰 효과 보정

`population.yaml`과 `simulator.yaml`은 발송에 따른 방문 효과, 10% 할인에 따른 추가
구매 효과, 발생한 구매에서의 쿠폰 사용을 각각 모의한다. 구매는 장바구니와 쇼핑 상태에
따른 행동 확률로 생성된다. 쿠폰 사용은 이 중 유효기간과 사용 가능 시점을 충족하는 구매에
기록한다. 아래 계수는 합성 프로필에서 보정했고 원천 고객별 프로필에도 같은 값을 적용한다.

- [Hillstrom 이메일 실험](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html?m=0)의
  발송군과 미발송군에서 14일 방문율 차이 +6.09%p와 구매율 차이 +0.50%p를 구했다.
  발송은 첫 방문 후보가 14일 안에 있는 경우의 방문 확률에만 직접 영향을 준다.
  발송으로 유도된 방문에서의 구매 확률을 보정해 발송군 전체의 구매율 차이를 맞췄다.
- [Gubela 외 실험 §4.1·표 2](https://doi.org/10.1007/s10796-022-10283-4)의
  할인 없는 뉴스레터 대비 10% 할인 뉴스레터의 12일 구매율 차이 +0.32%p를
  할인에 따른 추가 효과의 기준으로 썼다. 실제 실험의 할인은 장바구니에 적용됐다.
- [Gopalakrishnan·Park 실험 표 2](https://thearf-org-unified-admin.s3.amazonaws.com/MSI/2020/06/MSI_Report_19-106-1.pdf)의
  7일 구매자 중 사용 비중 16.4%(0.56% ÷ 3.42%)에 맞춰 Tianchi 사용 오즈를
  조정했다. Tianchi는 구매 이력별 상대 오즈와 사용 지연 분포를 제공한다. 비중의 기준이
  된 실험은 20달러 이상 구매에 7달러를 할인했다.

시드 310–313에서 각 2,000명·6개월을 생성했다. 같은 고객과 난수로 효과 없음,
발송 효과만, 발송과 할인 효과를 모두 켠 경우를 비교해 `coupon_effect_calibration`의
계수를 적합했다. 동일 단가 교체 규칙을 적용한 후 별도 시드 324–331에서 각
2,000명·6개월을 생성해 다시 평가했다. 14일 발송 방문 효과는 +5.89%p,
14일 발송 구매 효과는 +0.575%p, 12일 할인 추가 구매 효과는 +0.475%p였다.
같은 표본에서 7일 구매자 129명 중 26명(20.2%)이 쿠폰을 사용했다.
이 기준들을 다른 상품과 발송 조건에 옮겨 적용했으므로 28일 효과와 고객별 효과는
시뮬레이션 추정치다.

평가 전용 반응 결과는 업무 기록과 다른 경로로 내보낸다.

```bash
ecommerce-sim v4 export-experiment --data artifacts/small \
  --output artifacts/experiment --evaluation-output artifacts/evaluation_only/small
```

쌍대 평가는 조건부 재방문 확률을 비교한다. Uplift의 28일 구매 학습 정답은
`no_purchase_labels.parquet`의 실제 주문에서 계산한 값을 사용한다.
기존 `run`·Starbucks·`campaign` 환경은 별도 연구 경로로 보존한다.

### Retention 입력

v5의 공통 입력 `kit39-review-cart-v3`는 v4 사전의 39개 컬럼과 순서를
따른다. F21–24는 기준시점 전에 끝난 마지막 세션을 전체 관측 이력에서 찾는다.
30일 집계, 두 30일 창의 차이, 90일 방문 간격은 각자 필요한 구간이 관측된
경우만 계산한다. 가입 이후 장바구니 이력이 일부 빠지면 현재 장바구니는
결측으로 둔다. 미방문 정답은 배정 직전 28일 적격 고객의 배정 후 28일을
관측해서 계산한다. `kit39-review-cart-v2` 산출물도 이 적격성과 정답 정의를
기록한 경우 Retention 로더로 읽을 수 있다. 30일 적격성을 사용한 과거 자료는
현재 설정으로 다시 생성해야 한다.

```python
from simulate.v4.data.loader import load_retention_inputs

snapshots, X, y = load_retention_inputs("artifacts/features")
```

`X`는 등록 순서대로 KIT39만 담는다. `y=1`은 28일 미방문이다. 같은
고객·기준시각·snapshot의 피처와 라벨을 연결하고 성숙한 결과만 반환한다.
`snapshots.policy_id`는 관측된 실험 정책이다. 단일 실험 snapshot은
입력 점검에 사용할 수 있고 시간순 학습에는 서로 다른 기준시점의 성숙한
샘플이 필요하다.

추가 분석은 [기준 시점 이전 세션·주문 조회](scripts/query_v4.sql)에서 시작한다.
raw DB의 UTC 시각은 `2025-01-05T15:00:00.000000000+00:00` 형식으로 바인딩한다.
조회 전용 장바구니 전환율은 최근 30일 완료 세션 중 담기가 있었던 세션을 분모로
하고, 첫 담기 뒤 같은 세션에서 주문한 비율이다. 관측 창이 불완전하거나 분모가
0이면 NULL이다. 장바구니 이후 구매라는 구성개념은
[Esmeli와 Gokce (2025)](https://www.mdpi.com/0718-1876/20/1/28)를 참고했다.
30일 집계는 프로젝트의 산식이며 모델 피처 채택에는 별도 검증이 필요하다.

### Uplift 입력

`features` 명령이 생성한 디렉터리에서 실험 고객, 주간 합계, 배정 시 장바구니를 읽는다.

```python
from simulate.v4.data.loader import load_uplift_inputs

customers, weekly, cart = load_uplift_inputs("artifacts/features")
```

`w0`는 서울 월요일 00시의 배정 시점이다. 주간 `session_minutes`는 초를 분으로
환산하고, `net_revenue`와 장바구니 금액은 원천 최소 단위로 유지한다. 각 입력의
`attrs`에는 dataset ID, seed, 통화와 금액 단위가 있다.
Retention과 Uplift 로더는 피처 디렉터리의 manifest 체크섬으로 각 입력 파일을
검증한다.
`weekly_panel.session_seconds`는 관측을 마친 뒤의 주간 합계다. 배정 시점에
열려 있던 세션이 있다면 종료 확인 여부가 과거 주간값에 영향을 줄 수 있다.
검증한 v5 1,000명 자료에서 배정 시점을 가로지르는 세션은 0건이었다.

## 검증

```bash
PYTHONPATH=src python -m pytest -q
```

기존 정책의 로컬 학습 파일이 필요한 테스트는 파일이 없을 때 사유를 표시하고
건너뛴다. 원자료 대조와 생성 품질 비교는 각각 `simulate.v4.validation`,
`simulate.v4.quality`의 `--help`에서 실행 인자를 확인할 수 있다.
