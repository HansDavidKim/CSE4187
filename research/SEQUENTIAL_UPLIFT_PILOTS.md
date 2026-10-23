# 시계열 uplift 후보: 전처리 및 NTS 탐색 실험

## Backend-TS

공식 저장소의 archive와 [공식 전처리 노트북](https://github.com/nannullna/ts4uplift/blob/main/preprocess/preprocess_time_series.ipynb)에 표시된 캠페인 기준 시각을 사용한다. 처리군은 실제 방송 기준, 대조군은 정확히 7일 이전의 pseudo-control이다. 이력은 기준 시각 직전 168시간의 로그인 및 HTTP method별 `log1p` 이벤트 수이며, 처치 및 3시간 로그인 결과는 별도 episode 테이블이다. 해당 시간에 관측된 로그가 0개라는 의미이지 시스템 전체 활동이 반드시 없었다는 보장은 아니다. 4–5월은 사용자 단위 해시로 train/validation을 나누고, 6월은 별도 시간 테스트다. 따라서 6월 테스트 사용자는 과거 train 사용자와 겹칠 수 있다. 생성 스크립트 `scripts/prepare_backend_ts.py`.

기존 다중 시퀀스 NTS 구현은 각 시점의 모든 노드를 완전 관측한 상태로 다루므로 시퀀스 전체의 미관측 T/Y를 0으로 대치해 학습하지 않았다. 대신 50개 훈련 episode의 이력에만 1-lag NTS를 시험했다. 이것은 **개입 그래프가 아니다**. 10회 반복 후 h=2.79e-8로 h_tol=1e-8에 미도달, 임계처리한 동시점 그래프는 DAG다. 작은 표본, 수렴 미달, 안정성 평가 부재로 선정 근거가 아니다. T/Y를 포함한 endpoint-aware masked loss 및 시간 제약은 미구현이다.

## KuaiRand

[공식 페이지](https://kuairand.com/)에 따르면 Pure는 일부 video만 포함한 불완전 시퀀스이므로 예비 동작 시험에만 쓴다. 1K는 완전한 사용자 시퀀스를 위한 버전이다. `is_rand`는 알림이 아니라 random video replacement 정책 여부이며 사용자 반응 `is_click`, `is_like`, `long_view`와 구분한다. 노출 단위 sequence는 동일 사용자 30분 이내 연속 관측 세션이며 동일 타임스탬프 중복은 모호해 제거한다. 사용자 해시 구분과 날짜 분리 모두 적용한다. 이 방식은 원 보고서에서 제안한 session 단위의 초기 실험이나, Pure의 누락 때문에 실제 인접 노출이 보장되지 않는다. 각 세션의 user ID를 결과에 저장해 추후 user-level bootstrap이 가능하다.

NTS는 동시점 feedback→policy를 금지하지만 다른 인과 제약, user-level bootstrap, hyperparameter validation, held-out forecast와 반사실/uplift 성능 평가는 아직 없다. 동시점 feedback들 간 순서도 확정되지 않았다. 학습된 edge는 검증된 인과효과가 아니다. 스크립트 `scripts/prepare_kuairand.py`.

Pure의 training session에서 random 비율은 약 88.9%로 전체 서비스의 정책 비율을 대표하지 않는다. 1,500개 세션·6,501개 window의 1-lag 시험은 h=4.42e-9로 수렴했으며 0.3 threshold에서 `is_rand(t-1) → is_rand(t)`, `is_rand(t) → long_view(t)`, `long_view(t) → is_click(t)` 세 edge가 남았다. `long_view`와 `is_click`은 일부 UI에서 시청시간을 공유해 계산상 중복될 수 있으므로 인과관계로 해석하지 않는다. 이진 변수를 0/1 그대로 두고 `is_click`을 제외해 다시 돌리면 threshold 0.3에서 edge는 0개다. 이는 scaling과 변수 선택을 동시에 바꾼 민감도 시험으로 어느 한 변경의 효과를 분리하지 못한다. 표본은 user 단위가 아닌 세션 단위 추출이고 제거된 동시시각 기록으로도 관측 간 누락이 생길 수 있다. 이 결과로 최종 그래프 또는 데이터셋을 선정하지 않는다.

1K 예비실험은 [제3자 미러](https://huggingface.co/datasets/numberbeat6/kuairand)의 두 로그를 사용했다. 표준 로그 SHA-256은 미러가 제공하는 값 `548daf771e54e2b73086cc8e7c6f56787d44421f5026c2100421810e47ae9dd4`와 일치하지만, 공식 archive와의 대조는 아직 못 했다. 4/22–5/08 표준 로그 6,657,061건 중 98.66%가 동일 고객·동일 타임스탬프 중복 그룹에 속해, 해당 행을 모두 제외하는 현 처리의 대표성에 심각한 문제가 있다. 1,500개 남은 train 세션의 예비 NTS(0/1 유지, click 제외, 1 lag, 0.3 threshold)는 edge 0개다. timestamp 동률을 안전하게 처리하는 단위의 재설계 및 표본 보존율 검증 전까지 1K 결과는 채택 판단에 사용할 수 없다.

## 다른 후보 및 선정 상태

ASSISTments 논문의 구 다운로드 주소는 404였고 현행 접근·약관 확인이 필요하다. PhysioNet과 Freddie Mac은 자격/이용조건 확인 없이 접근할 수 없다. 로컬 Complete Journey의 campaign 테이블은 고객별 실제 접촉 시각을 입증하지 않으므로 이를 개입 시각으로 가정해 학습하지 않는다. 아직 후보 간 동일한 실험이나 검증을 마치지 않아 채택 대상은 결정하지 않았다.

Backend-TS 공식 저장소는 CC BY-NC-SA 4.0과 별개로 사용 결과 공유 시 연락을 요청한다. 논문 공개 전 [공식 안내](https://github.com/nannullna/ts4uplift#disclaimer)를 확인한다. 각 원자료 및 파생 개인별 데이터는 공개 저장소에 포함하지 않는다.
