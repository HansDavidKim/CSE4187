# eCommerce-Sim

Electronics·Cosmetics의 행동 이질성과 세션 간격을 재현하고, Tianchi의 쿠폰 반응을
참고하여 개입 시나리오를 실험하기 위한 사용자 시뮬레이터 연구 프로젝트다.
최종 목표는 페르소나별 MaxEnt IRL, 세션 종료 및 확률적 재방문, 쿠폰 개입을
결합하고 오프라인 합성 데이터와 온라인 에이전트 평가 환경을 제공하는 것이다.

현재는 분석·모델별 파일럿 단계다. 일별 빈도 제한 쿠폰 정책 환경과 여섯 정책
baseline은 실행 가능하지만, **페르소나별 인과적 쿠폰 효과와 canonical type
분포가 식별된 모델은 아니다.**
16개 목표별 현재 상태와 주요 결과는 [analysis.md](analysis.md)를 기준으로 읽는다.

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
| Tianchi 쿠폰 수령 이후 행동 | `python -m analysis.uplift.observed reactivation`, `python -m analysis.uplift.observed persistence` | 관측 비교이며 인과 uplift가 아님 |
| Canonical type 조건부 범위 | `python -m analysis.uplift.canonical` | [가정 기반 시나리오](analysis.md#archive-canonical-type-bounds-pilot); 실제 유형 분포 추정이 아님 |
| 페르소나별 세션 간격 | `python -m analysis.persona.behavior delays electronics cosmetics` | 사후 기술 분석이며 예측용 분포로 바로 사용할 수 없음 |
| 잠재 세션 IRL·재방문 | `scripts/run_latent_session_pilot.py` | [파일럿](analysis.md#archive-latent-session-irl-pilot); 짧은 세션 및 30일 내 간격 |
| 분석 요약 그림 | `scripts/plot_analysis_summary.py` | 기존 JSON 산출물이 있어야 실행 가능 |
| 주간 쿠폰 정책 비교 | `PYTHONPATH=src python -m scripts.run_agent_benchmark` | 1,000명·13주 학습 후 5,000명·52주 평가; 실제 인과 정책 평가는 아님 |

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
