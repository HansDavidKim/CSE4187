### 규칙
- 각자의 기능은 feat/기능명 브랜치에서 개발
    - 고장은 fix/기능명
- conflict 없을 시 메인에 병합알아서. conflict 나면 카톡방에 보고
- 웬만하면 자신의 디렉토리에서 작업. 
- docs, common은 필요하다 싶으면 추가 가능.

### 시뮬레이터

시뮬레이터 코드·설정·테스트는 [simulator/](simulator/README.md)에서 관리한다.
기존 `simulator_etc/`의 실행 코드는 이 디렉터리로 통합했다.

```bash
cd simulator
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

### 주 단위 모델 연결

기본 생성은 104주(기준 시점 이전 52주 + 이후 52주)다. 최소 고객 이력은
13주이며, 최근 방문·쿠폰 유효기간·Uplift 결과는 4주다. 최근 특징은
4/8/13주로 집계하고 CLV는 과거 13주에서 미래 52주를 예측한다.
`--months`를 명시한 기존 월간 생성은 해당 manifest의 기간을 유지한다.

- [시뮬레이터](simulator/README.md): `auto`는 기존 1:1 무작위 배정,
  `none`은 미발송, `assign`은 고객 ID·인원·비율 지정이다. 데이터 생성 모드는
  t0에 한 번 발송하며 주간 RL 정책과 구별한다.
- [Uplift](uplift/readme.md): 4주 구매확률 `p0`, `p1`과 `p1-p0`를 제공한다.
- [Retention](retention/readme.md): 같은 고객·스냅샷의 4주 미방문 확률을 제공한다.
- [CLV](clv/readme.md): 미발송 학습 모집단과 52주 예측 범위를 확인한다.
- [최종 전략](final_strategy/readme.md): 모델별 예측 시점·결과 기간·금액 단위를
  검증한 뒤 고객 10필드, 6세그먼트 요약, 예산별 분석과 그래프를 생성한다.

확률 예측은 실제 모델 출력을 연결한다. 금액 예측이나 무작위 실험의 관측 결과가
없으면 예상 이득·ROI·전략 평가를 계산할 수 없는 사유를 기록한다.
원자료와 원천별 학습 모형, 생성 산출물은 로컬에 보관한다.

저장소 루트에서 별도 가상환경에 모듈 의존성을 설치하고 전체 계약 검사를 실행한다.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e "./simulator[dev]" -r clv/requirements.txt \
  -r retention/requirements.txt -r uplift/requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=simulator:simulator/src:simulator/tests:. \
  python -m pytest -q --import-mode=importlib \
  simulator/tests clv/tests retention/tests uplift/tests final_strategy/tests
```

macOS에서는 트리 모델과 신경망을 함께 검증한 PyTorch 2.8.0을 사용한다.
Retention requirements와 lock 파일에 이 범위를 반영했고, 학습기는 호출자의
1스레드 제한을 유지한다. 원천별 로컬 학습 파일이 없는 기존 검사는 사유를 표시한다.
