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

Git에는 실행 코드·테스트·설정·이 안내만 관리한다. 원자료·학습 가중치·데이터베이스·
생성 결과·문서·노트북·가상환경·캐시는 로컬에서 사용하고 Git에서 제외한다.

## 실행

Python 3.10 이상에서 저장소의 `simulator/` 디렉터리 기준으로 실행한다.

```bash
cd simulator
python -m pip install -e ".[dev]"
ecommerce-sim v4 generate --assets-root /path/to/assets --mode small \
  --n-customers 500 --months 6 --seed 42 --output artifacts/small
ecommerce-sim v4 validate --data artifacts/small/raw.sqlite
ecommerce-sim v4 features --data artifacts/small --output artifacts/features
```

설치 전에는 `PYTHONPATH=src python -m cli v4 ...`를 사용할 수 있다.
저장소 루트에서 설치할 때는 `python -m pip install -e "./simulator[dev]"`를 사용한다.
기본 설정은 [simulator.yaml](src/simulate/v4/config/simulator.yaml)이다.
`--assets-root`는 `data/models`와 원자료를 가진 디렉터리다. 기존 자료가 다른 위치에
있으면 해당 경로를 명시한다. 생략하면 현재 디렉터리를
사용한다. 가중치가 없으면 필요한 경로를 표시하며, 실제 원자료를 준비한 뒤
`generate --fit-missing`으로 누락 모형을 학습한다. 기존 출력·모형은 덮어쓰지 않는다.

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
계산한다. Apple M3에서 20,000명·24개월·seed42의 원시 생성과 저장에 든 CPU 시간은
원래 v12 1,866.67초에서 1,132.63초로 줄었다. SQLite와 Parquet 14개의 체크섬은
같았다. 같은 조건의 v10은 807.97초였다. Cosmetics 5개월 분포 검증에서 GMM
보완안은 독립 고객 표본을 바꿨을 때 행동 비중, 행동 전이, 세션 길이 오차의 허용
기준을 모두 충족하지 못했다. GMM 설정을 선택하면 최적화한 v12로 실행한다.

업무 기록은 `raw.sqlite`의 고객·상품·세션·이벤트·주문·주문상품·쿠폰 7개 관계에
저장한다. `features.sqlite`는 공개 이력의 39개 피처와 배정·정답·주간/실험 테이블을
분리한다. 피처 이름과 순서는 [features.py](src/simulate/v4/features.py)의
`EXPANDED_FEATURES`와 파생 DB의 `feature_registry`를 따른다.
조회 예시는 [query_v4.sql](scripts/query_v4.sql)에 있다.

기본 이탈 정답은 미래 28일 미방문이며, 과거 피처·적격성 창과 쿠폰 유효기간은
30일이다. 쿠폰은 고객별 1:1 고정 배정하며 실제 발송과 사용을 구분한다. 별점은
실제 구매한 상품별 최대 한 번이다. 금액은 원자료의 통화·단위를 유지한다.
잠재 성향과 미래 정답은 예측 피처에 포함하지 않는다.

평가 전용 반응 결과는 업무 기록과 다른 경로로 내보낸다.

```bash
ecommerce-sim v4 export-experiment --data artifacts/small \
  --output artifacts/experiment --evaluation-output artifacts/evaluation_only/small
```

이 평가는 조건부 재방문 확률의 비교이며 실제 개인 인과효과의 식별이 아니다.
기존 `run`·Starbucks·`campaign` 환경은 별도 연구 경로로 보존한다.

## 검증

```bash
PYTHONPATH=src python -m pytest -q
```

기존 정책의 로컬 학습 파일이 필요한 테스트는 파일이 없을 때 사유를 표시하고
건너뛴다. 원자료 대조와 생성 품질 비교는 각각 `simulate.v4.validation`,
`simulate.v4.quality`의 `--help`에서 실행 인자를 확인할 수 있다.
