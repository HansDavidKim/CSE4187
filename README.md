# eCommerce-Sim

Electronics·Cosmetics의 행동 이질성과 세션 간격을 재현하고, Tianchi의 쿠폰 반응을
참고하여 개입 시나리오를 실험하기 위한 사용자 시뮬레이터 연구 프로젝트다.
최종 목표는 페르소나별 MaxEnt IRL, 세션 종료 및 확률적 재방문, 쿠폰 개입을
결합하고 오프라인 합성 데이터와 온라인 에이전트 평가 환경을 제공하는 것이다.

현재 파이프라인은 proxy 배정, canonical-type 합성비율, 사용자별 MaxEnt 계수,
세션 lifecycle·이산 간격, latent-conditioned uplift, 도메인 GMM 및 여섯 정책의
장기 평가를 한 환경으로 연결한다. Tianchi 관측차이는 합성환경의 보정 모멘트로
사용하며, 페르소나별 쿠폰 반응과 canonical type은 명시된 시뮬레이션 가정으로
생성한다. 세부 근거와 한계는 [analysis.md](analysis.md)를 기준으로 읽는다.

최종 KSC 원고는 [paper.docx](paper.docx), 검증된 3페이지 PDF는
[output/pdf/paper-docx.pdf](output/pdf/paper-docx.pdf)에 있다.

## 구조

```text
src/
  preprocessing/  # 원자료 검증, 시퀀스 및 이력 특징
  analysis/       # 행동·세션·비활동 분석, 그래프 특징
  models/         # MaxEnt IRL, 선택모형, 그래프·임베딩 학습
  simulate/       # 세션·간격 샘플러 및 합성 패널 생성
  utils/          # 공통 경로, 이벤트 로더 및 마스킹
  cli.py          # 기존 CLI
scripts/          # 현재 연구에 사용하는 실행 스크립트
tests/            # 원자료가 필요 없는 회귀 테스트
analysis.md       # 연구 결과 및 실험별 상세 기록
img/              # 분석 그림
data/raw/         # 로컬 원자료, Git 제외
artifacts/        # 생성 데이터·모델·통계, Git 제외
paper/            # 논문 원고
output/pdf/       # 기존 논문 PDF
```

코드는 역할별 패키지에서 import한다. 예:
`from simulate.latent_session_model import LatentSessionUser`.
기존 `ecommerce_sim.*` import와 이를 참조하는 pickle은 자동 호환되지 않는다.
옛 pickle은 이전 코드에서 열어 내보내거나 이전 버전에서 재생성해야 한다.

## 설치와 검증

저장소 루트에서 Python 3.10 이상을 사용한다. 의존성의 기준은 `pyproject.toml`이다.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m cli --help
```

분석 그림과 Parquet 입출력까지 사용하려면 분석 의존성도 설치한다.

```bash
python -m pip install -e ".[dev,analysis]"
```

## 현재 연구 경로

| 작업 | 실행 진입점 | 결과·한계 |
|---|---|---|
| masked-GRU 임베딩 학습 | `python -m analysis.persona.sequence --source electronics` | 행동·세션 경계 기반이며 데이터 로더는 외부 프로젝트가 필요 |
| Cosmetics persona MaxEnt IRL | `python -m analysis.irl.persona --help` | 선형·2차 persona reward, 30일 세션-gap hazard와 사용자별 MAP 계수; 쿠폰 효과는 포함하지 않음 |
| Electronics·Cosmetics 행동 군집 | `python -m analysis.persona.cluster compare electronics cosmetics` | [페르소나 분석](analysis.md#2-행동-시퀀스-기반-페르소나); 저장된 임베딩과 외부 데이터 로더 필요 |
| Cosmetics 고유 행동 제거 정렬 | `python -m analysis.persona.cluster electronics-drop` | remove_from_cart 제거 후 GRU 재학습 산출물이 필요하며 로깅 스키마 ablation이지 반사실 행동 추정이 아님 |
| 공통 로그 공동 군집 | `python -m analysis.persona.sequence --source retail_combined --cluster-count 4 --max-users 24000 --output artifacts/retail-combined-personas/retail_combined_clusters_k4.json` | Cosmetics의 remove_from_cart를 제거하고 Electronics와 하나의 GRU·KMeans로 공동 학습 |
| 공동 군집 대응 요약 | `python -m analysis.persona.cluster retail-combined` | 공동 K=4의 도메인별 군집 분포와 공통 행동 조성 유사도 산출 |
| Tianchi 최초 수령 전후 정렬 | `python -m analysis.uplift.observed first-receipt-alignment` | 최초 관측 쿠폰 수령 전 행동을 Cosmetics/Electronics K=4에 매칭하고 28일 전후 변화를 기술적으로 비교 |
| Tianchi 쿠폰 수령 이후 행동 | `python -m analysis.uplift.observed reactivation`, `python -m analysis.uplift.observed persistence` | 관측 비교이며 인과 uplift가 아님 |
| Canonical type 조건부 범위 | `python -m analysis.uplift.canonical` | [가정 기반 시나리오](analysis.md#archive-canonical-type-bounds-pilot); 실제 유형 분포 추정이 아님 |
| 페르소나별 세션 간격 | `python -m analysis.persona.behavior delays electronics cosmetics` | 사후 기술 분석이며 예측용 분포로 바로 사용할 수 없음 |
| Cosmetics·Electronics lifecycle 교차검증 | `python -m analysis.persona.behavior lifecycle-cross-domain` | 공통 30일 미복귀 head, 만족 이탈 후보, 공통 프로파일 및 희소 장기 구매자 `P_LT` 산출; 두 원본 경로가 필요 |
| 잠재 세션 IRL·재방문 | `scripts/run_latent_session_pilot.py` | [파일럿](analysis.md#archive-latent-session-irl-pilot); 짧은 세션 및 30일 내 간격 |
| 분석 요약 그림 | `scripts/plot_analysis_summary.py` | 기존 JSON 산출물이 있어야 실행 가능 |
| 도메인 GMM 구성 | `PYTHONPATH=src python -c "from simulate.domain import build_domain_adapters; build_domain_adapters()"` | Cosmetics/Electronics/Tianchi의 공통 13특징에 대각 GMM을 적합하고 Cosmetics 행동 backbone에 정렬 |
| 주간 쿠폰 정책 비교 | `PYTHONPATH=src python scripts/run_agent_benchmark.py` | 도메인별 1,000명·13주 학습 후 독립 5,000명·52주 평가; 발송비·첫 구매 할인비를 각각 비교 |

최종 합성 패널은 `artifacts/synthetic-benchmark-datasets/`에 도메인별 Parquet과
메타데이터 JSON으로 저장된다. 정책 결과는 `artifacts/agent-benchmark/summary.json`,
그림은 `img/analysis/agent_policy_comparison.png`와
`img/analysis/agent_coupon_cost_per_purchase.png`에 저장된다.

인자가 있는 스크립트는 [실행 스크립트 목록](scripts/README.md)을 확인한다.
예를 들어 세션 파일럿은 다음처럼 실행한다.

```bash
python -m scripts.run_latent_session_pilot electronics cosmetics
```

## 데이터와 외부 코드

이 체크아웃에는 원자료, 학습 체크포인트, 실험 JSON이 포함되어 있지 않다.
데이터셋 사이 사용자를 연결하거나 서로 다른 군집 번호를 같은 유형으로 취급하지 않는다.
논문 사용·재배포 권리는 [데이터 정책](analysis.md#archive-data-policy)을 별도로 확인한다.

이전 capstone 프로젝트의 자료를 읽는 스크립트는 그 디렉터리의
`data/raw/kaggle/`와 `data/models/` 구조를 사용한다. 기본 루트는 현재 저장소이며,
별도 위치를 쓸 때는 환경변수 `ECOMMERCE_CAPSTONE_ROOT`를 지정한다.
`src.irl`을 사용하는 이전 실험과 일부 데이터 로더는 해당 외부 프로젝트 코드도 필요하다.

PowerShell 예:

```powershell
$env:ECOMMERCE_CAPSTONE_ROOT = 'C:\path\to\capstone'
$env:PYTHONPATH = "$env:ECOMMERCE_CAPSTONE_ROOT;."
python -m analysis.persona.cluster compare electronics cosmetics
```

Tianchi 스크립트는 기존 `data/raw/tianchi-o2o/` 경로를 사용한다.
자료 위치를 정리하기 위해 원자료를 자동으로 옮기거나 덮어쓰지 않는다.

## 이전·보조 실험

Starbucks, AmExpert, Backend-TS 및 인과 발견의 이전 실험 결과와 한계는
[분석 기록](analysis.md#archive-legacy-pipelines)에 보존한다. 관련 과거 실행
스크립트는 정리했으므로 기록의 명령이 현재 체크아웃에서 동작하지 않을 수 있다.
이 결과를 현재 페르소나별 쿠폰 인과효과의 검증 근거로 사용하지 않는다.
