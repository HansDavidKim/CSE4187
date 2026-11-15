# 설정 위치

설정 공통 로더는 [`simulate.v4.configuration.load_config`](../src/simulate/v4/configuration.py)다. 실행 가능한 시뮬레이터 기본 설정은 [`src/simulate/v4/config/simulator.yaml`](../src/simulate/v4/config/simulator.yaml) 한 곳이다. `python -m cli v4 generate --config <파일> --assets-root <자료 루트>`로 설정 파일과 로컬 자료 루트를 지정할 수 있다. 원자료·학습 모형 경로는 자료 루트에서, 번들 응답 JSON은 번들 설정 디렉터리에서 해석한다.

main의 `simulator.yaml`, `features.yaml`, `items.yaml`, `system.yaml`, `retention_ml.yaml`, `retention_dl.yaml`, `uplift.yaml`, `clv.yaml`은 모두 0바이트였다. 실행 가능한 설정으로 오해하지 않도록 복사하지 않았다. 후속 모델 설정은 해당 모델이 구현될 때 추가한다.

| 현재 계약 | 기준 위치 |
|---|---|
| 고객 수·기간·seed, 28일 label, 원천 모형 경로 | 번들 `simulator.yaml`의 `simulation`, `churn`, `source_simulation` |
| KIT39 이름·순서·버전 | [`features.py`](../src/simulate/v4/features.py)의 `EXPANDED_FEATURES`, [`feature_registry_events_v2.json`](../paper/simulator_v4/feature_registry_events_v2.json) |
| raw SQLite 관계·버전·제약 | [`raw_store.py`](../src/simulate/v4/simulator/raw_store.py)의 `RAW_COLUMNS`, `_schema`, `_validate` |
| 개별 데이터의 통화·금액 단위·관측 범위·SHA | `raw.manifest.json` 및 생성 `manifest.json`; 실제 생성 데이터의 메타데이터 |

코드의 피처 registry나 SQLite schema를 비어 있는 YAML로 덮어쓰지 않는다. [main 통합 기록](../docs/main_integration.md)에 원본 revision과 선택 근거가 있다.
