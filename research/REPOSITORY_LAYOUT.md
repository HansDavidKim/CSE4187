# 저장소 구조와 기능 보존 기준

이 저장소의 목표는 `AGENTS.md`의 16개 연구 항목이다. 이번 정리는 기존 분석과
모델의 계산을 유지하면서 코드의 역할, 실행 방법, 실험의 한계를 드러내는 작업이다.
새로운 쿠폰 효과 추정기나 RL 에이전트를 구현하는 작업은 포함하지 않는다.

## 코드의 역할

| 위치 | 책임 |
|---|---|
| `src/ecommerce_sim/preprocessing/` | 원자료 검증, 시퀀스·시간·이력 특징 구성 |
| `src/ecommerce_sim/analysis/` | 캠페인 감사, 인과 그래프 특징, 행동 변화·비활동·세션 간격 분석 |
| `src/ecommerce_sim/models/` | 선택모형, MaxEnt IRL 변형, 그래프 학습, 임베딩 |
| `src/ecommerce_sim/simulate/` | 사용자 상태, 세션·간격 샘플링, 합성 패널 생성 |
| `src/ecommerce_sim/cli.py` | 기존 `ecommerce-sim` 및 `python -m ecommerce_sim.cli` 명령 |
| `src/ecommerce_sim/paths.py` | 저장소 및 외부 capstone 자료의 위치 |
| `scripts/` | 기존 파일명으로 실행하는 연구·전처리 진입점; [목록](../scripts/README.md) 참고 |
| `tests/` | 합성 입력 기반 회귀 테스트; 원자료 없이 실행 가능 |
| `analysis.md` | 현재 연구 목표별 결과와 식별·검증 한계 |
| `research/` | 실험별 상세 기록; 과거 결과와 최종 설계를 구분 |
| `research/figures/` | 보고서에서 참조하는 그림 |
| `data/raw/` | Git에 포함하지 않는 원자료 |
| `artifacts/` | Git에 포함하지 않는 생성 패널·모델·실험 통계 |
| `paper/`, `output/pdf/` | 논문 원고와 기존 PDF |

이미 사용하는 `ecommerce_sim` 패키지와 `pyproject.toml`을 유지한다. 아직 없는
에이전트 구현이나 데이터 산출물을 위해 빈 패키지·폴더를 만들지는 않는다.

## 기능 보존

- 모델의 보상식, 최적화 설정, 난수 seed, 샘플링 및 데이터 검증은 유지한다.
- 기존 `ecommerce_sim.<module>` import는 새 역할별 모듈과 같은 객체를 가리킨다.
  이전 모듈 경로로 저장된 Python 객체도 같은 클래스를 찾을 수 있게 한다.
- `scripts/*.py`의 파일명과 인자, CLI 하위 명령, 산출물 이름은 유지한다.
- 이전 개발자의 절대경로 대신 저장소를 기본 위치로 사용한다. 기존 별도 capstone
  자료를 사용하려면 `ECOMMERCE_CAPSTONE_ROOT`를 해당 디렉터리로 설정한다.
  외부 `src.irl`·`src.analysis` 코드가 필요한 실행에는 그 프로젝트도
  `PYTHONPATH`에 넣어야 한다. 외부 코드를 다른 구현으로 대체하지 않는다.
- 순수 집계 함수는 외부 프로젝트 없이 import할 수 있도록 외부 코드 import를
  실제 분석 실행 위치로 옮긴다.

실험 스크립트마다 대상 모집단·세션 정의·학습 조건이 달라, 이름이 비슷하다는
이유만으로 삭제하지 않는다. 기능 보존을 확인하는 테스트도 유지한다.

## 검증 범위

정리 전후 동일 환경의 pytest 결과, 기존 import와 새 import의 객체 동일성,
CLI 실행, 합성 패널 생성 및 설치 패키지 구성을 확인한다. 원자료·외부 capstone
코드·학습된 체크포인트가 없는 환경에서는 전체 실험 재실행과 수치 재현을
완료했다고 주장할 수 없다. 최종 검증 결과는 이 문서에 덧붙인다.

### 2026-10-23 검증 결과

- 동일 가상환경의 원본 복사본: 68 passed, 1 failed, 6 collection errors.
  수집 오류는 누락된 외부 capstone import였고, 실패는 gCastle import가
  전역 torch dtype을 바꾸어 생긴 LightGCN 테스트 간 간섭이었다.
- 정리 후 전체 테스트: **94 passed**. 기존 80개 테스트를 그대로 유지하고,
  모듈 경로·이전 pickle 클래스 참조·외부 자료 경로·import 부작용에 관한
  14개 검증을 추가했다. gCastle import 전후 dtype을 복원하여 클래스 접근과
  기존 진단 기능을 유지했다.
- 26개 모듈을 역할별로 이동했다. 원본의 221개 함수·클래스 정의를 AST로
  비교해 import 위치·경로를 제외한 계산 구조가 같음을 확인했다.
- 기존 Python 실행 파일 58개의 경로와 기존 테스트 파일의 내용을 유지했다.
  기존 직접 실행을 보존하고 조건부 IRL 스크립트의 `-m` 실행도 확인했다.
- `python -m ecommerce_sim.cli --help`와 wheel 빌드가 성공했다.
  wheel 내부에 모든 패키지 모듈이 포함됐고, 소스 디렉터리와 분리한
  Python 프로세스에서도 wheel import와 이전 API 경로가 동작했다.
  Ruff의 F401/F811/F841 검사에서 소스·실행 스크립트·신규 테스트가 통과했다.
- pytest에 남은 4개 경고는 PyTorch sparse 검사 및 pandas API의 기존 경고다.
  원자료, 외부 capstone 코드 및 체크포인트가 없어 전체 학습·실험 수치 재현과
  R 추출기는 실행하지 않았다.
