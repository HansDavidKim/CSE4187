# main 기능·문서 통합 기록

비교 기준은 `origin/main` **`f5f8d6f0c06e863eed03e14762be2315ecc2100e`**(`Merge pull request #2 from HansDavidKim/yuseung`)와 통합 전 daewon **`3e551b1`**이다. [해당 main 트리](https://github.com/HansDavidKim/CSE4187/tree/f5f8d6f0c06e863eed03e14762be2315ecc2100e)를 고정해 확인했으며, 현재 브랜치에는 `src/simulate/v4/`를 추가한 최신 시뮬레이터 구조를 유지한다.

## 실제 구현과 계획

main 최상위의 새 팀 디렉터리에는 Python 구현이 없다. `simulator_etc/`에 이동된 기존 파일은 경로를 제거하고 daewon과 비교하면 **`Additional.md` 한 개만 새 문서**이고 나머지는 동일한 Git blob이다. 특히 `simulator_etc/src/` 28개·`tests/` 48개·`scripts/` 14개는 기존 daewon 파일과 각각 동일하다. 같은 구현을 두 위치에 복사하지 않았다.

| main 내용 | 확인 결과 | 현재 반영 |
|---|---|---|
| 기존 시뮬레이터·분석·정책 코드 | `simulator_etc/`로 이동한 동일 코드 | 기존 `src/`를 유지하고 V4 구현 사용 |
| 팀별 디렉터리·공통 설정 | readme 또는 빈 YAML뿐 | 문서 경계만 보존; 실행 코드로 간주하지 않음 |
| 전체 ML/DL·Uplift·CLV·예산·대시보드 요구사항 | 목표·실행 예시·담당자 계획 | 요구사항으로 보존; 완료 기능으로 표기하지 않음 |
| Uplift 주간 입력·개입 계약 | feature·plan·usecase 문서 | 현재 export 이름·유효기간·평가 범위에 맞춰 연결 |
| V3 네 행동 ERD | 정수 ID·네 행동·KRW 기본 가정 | 실제 V4 TEXT ID·여섯 생성 이벤트·비동기 리뷰·원천 통화로 갱신 |
| `Additional.md` | 후속 모델별 입력·정답 표 | `docs/model_input_contracts.md`로 보존 |

현재 코드 경계는 `simulate.v4.configuration.load_config`와 `simulate.v4.data.loader.load_tables`를 공유하며 기존 CLI·피처 import 별칭도 유지한다. 현재 시뮬레이터에는 여섯 생성 이벤트, KIT39, 일곱 관계의 불변 SQLite, 과거 시점 피처, 개입·결과 sidecar, 주간 집계, 별도 미구매 label과 평가 전용 쌍대 출력이 있다. 20,000명 설정이나 과거 감사 결과가 있다는 이유로 새 실행·전체 팀 모델·12개월 미래 CLV 검증이 완료됐다고 주장하지 않는다. 최신 실행 규모·모형 SHA·품질은 해당 데이터의 manifest와 독립 감사 결과로 확인한다.

## 충돌 시 적용할 계약

| 주제 | 적용 기준 |
|---|---|
| 이벤트 | 현재 생성은 `view/cart/remove_from_cart/purchase/coupon_usage/review`. 검색·CS는 새 생성·KIT39에서 제외하고 과거 reader 호환만 유지 |
| 피처 | KIT32 + 리뷰·장바구니 7개 = KIT39. main의 빈 `config/features.yaml`은 계약이 아님 |
| Persona | 현재 원천 기반 latent IRL과 private 개인 성분을 유지. 과거 P0·P2·P3 고정 유형 문서만으로 재분류하거나 공개 입력에 latent 값을 추가하지 않음 |
| 시간 | 원천 학습·생성·직전 방문 적격성·쿠폰 유효기간은 30일. 기본 결과는 `future_no_session_28d_active30d`의 28일. 4주 결과를 위해 쿠폰을 28일로 줄이지 않음 |
| Uplift 입력 | main의 최근 4주·12주 변수는 후속 학습 계획. 현재 KIT39의 30일 창과 별도로 정의하고 과거 완료 주만 사용 |
| Uplift 정답 | 방문 label과 `future_no_order_28d_active30d`의 미구매 label을 구분. latent ST/P/SD/LC는 평가 파일에만 포함; 실제 배정·발송·사용을 혼동하지 않음 |
| 쌍대 평가 | 조건부 재방문 모형의 주변 결과를 평가. 전체 쇼핑·개별 미구매 반사실의 재생이나 관측 자료의 개인별 인과효과 식별로 해석하지 않음 |
| 코호트·CLV | 현재 관측 가능한 범위와 검열 상태만 해석. main의 M12 곡선·모든 후속 모델·12개월 CLV 요구를 충족하려고 미래를 채우거나 이탈 전 징후를 강제하지 않음 |
| 통화·회계 | 원본 `currency`·`amount_unit` 유지. `SOURCE_UNSPECIFIED`를 KRW로 바꾸지 않음. 쿠폰은 최고 단가 한 개 할인, 발송비는 별도 |
| 설정 | [`src/simulate/v4/config/simulator.yaml`](../src/simulate/v4/config/simulator.yaml)이 실행 기본값. 빈 main YAML은 추가하지 않음 |

## 가져온 텍스트와 위치

아래 원본 경로는 모두 위에 고정한 main revision을 기준으로 한다. 원본 내용은 Git에서 확인할 수 있으며 원자료·모형·DB·이미지·PDF·PPTX는 가져오지 않았다.

| 현재 경로 | main 원본 | 반영 내용 |
|---|---|---|
| `docs/request.md`, `docs/request_specification.md` | 동일 경로 | 원문 내용 보존·빈 줄 공백 정리, 계획 문서·V4 우선순위 안내 추가 |
| `docs/simulator_ERD.md` | 동일 경로 | 현재 schema 5·생성 범위로 갱신 |
| `docs/readme.md` | 동일 경로 | 실제 문서의 탐색 링크로 보완 |
| `docs/model_input_contracts.md` | `simulator_etc/Additional.md` | 입력 요구사항 표와 계획 상태 보존 |
| `uplift/feature.md`, `uplift/plan.md`, `uplift/usecase.md` | 동일 경로 | 후속 계획 보존; 30일 적격성·유효기간 및 실제 주간 열 이름 수정 |
| `uplift/readme.md` | 동일 경로 | 구현된 export·정답·평가 범위를 명시 |
| `config/readme.md` | 동일 경로 | 실제 설정·registry·SQLite 계약 위치를 명시 |
| `data/readme.md` | 동일 경로 | 기존 데이터 디렉터리에 안내 파일만 추가 |
| `clv/readme.md`, `common/readme.md`, `dashboard/readme.md`, `features/readme.md`, `final_strategy/readme.md`, `retention/readme.md` | 동일 경로 | 팀 경계 보존, 실제 구현 위치 또는 미구현 상태 표시 |
| `docs/feature_dictionary.md` | 새 연결 문서 | 기존 V4 정의·registry로 안내; 사전 복제 없음 |

main의 여덟 빈 설정 파일(`clv`, `features`, `items`, `retention_dl`, `retention_ml`, `simulator`, `system`, `uplift`)과 `simulator_etc/` 전체 사본은 제외했다. 현재 root README·`.gitignore`·로컬 자료는 덮어쓰지 않았다.

## 브랜치 협업 규칙

main README의 규칙을 보존한다: 기능은 `feat/<기능명>`, 수정은 `fix/<기능명>`에서 개발하고 담당 영역에서 작업한다. `docs`·`common`은 공유한다. 충돌이 없는 변경은 협의한 절차로 main에 병합하고, 충돌은 팀에 보고한다. 이 통합의 배포 대상은 사용자가 지정한 `daewon`이며 main을 자동으로 갱신하지 않는다.

## 검증 범위

main과 기존 daewon 파일의 Git blob 비교, 빈 설정 확인, 가져온 텍스트·현재 Python schema/registry/주간 export의 대조와 문서 링크 확인을 수행했다. 이 문서 통합은 생성기 알고리즘을 바꾸거나 후속 모델을 학습하지 않는다. 시뮬레이터 코드의 별도 테스트·원천 적합·충실도 감사는 [기존 검증 기록](../paper/simulator_v4/daewon_final_independent_audit.json)과 이번 변경의 실행 결과를 함께 확인한다.
