# Uplift 연결 계약

이 디렉터리의 [feature](feature.md)·[plan](plan.md)·[usecase](usecase.md)는 main에서 가져온 후속 모델 설계다. T/X/H-learner 학습·Qini 평가·4분면 예측 파이프라인은 아직 구현되지 않았다. 현재 범위는 시뮬레이터의 합성 데이터와 평가 자료 제공이다.

현재 시뮬레이터는 고객별 확률 0.5로 한 번 배정한다. 적격성은 `t0` 이전 **30일** 세션 시작, Treatment는 `t0`에 **10% 쿠폰 1회·30일 유효**, Control은 실험 후 미발송이다. 할인은 주문 내 최고 단가 상품 **한 개**에 적용한다. 생성·원천 재방문 단위는 30일이고, 현재 결과 label은 별도로 **28일**이다. 월요일 기준 주간 집계와 이 세 시간 구간을 혼동하지 않는다.

| 제공 자료 | 실제 필드·제약 |
|---|---|
| `weekly_panel.parquet` | `customer_id × reference_week`; Asia/Seoul 월요일 시작. `sessions`, `purchases`, `views`, `session_seconds`, `net_revenue_minor` 사용. 분 단위가 필요하면 초를 60으로 나눈다. |
| `experiment_log.parquet` | 배정 1행/고객. 배정·발송 결정·실제 발송·사용을 분리하고 비활동·미사용 고객도 유지한다. |
| `no_purchase_labels.parquet` | `future_no_order_28d_active30d`; 28일 동안 주문이 없으면 `no_purchase_churn=1`. 구매 유지 Y는 관측 완료 행에 한해 `1-no_purchase_churn`. |
| 방문 label | `future_no_session_28d_active30d`; 미구매와 다른 사건이다. |
| KIT39 장바구니 상태 | `cart_qty_current`, `cart_sku_current`, `cart_gross_value_current`. `t0`의 피처 파일에서 별도로 가져와 고객·시점을 맞춘다. 주간 export에 `cart_qty_w0` 같은 별칭은 자동 생성되지 않는다. |
| 평가 전용 | 별도 `evaluation_only/`의 `paired_rollouts.parquet`, `latent_truth.parquet`. 조건부 재방문 방정식의 쌍대 평가이며 전체 쇼핑 경로나 미구매 정답의 개별 반사실을 재생하지 않는다. |

비활동이 관측된 주는 0, 가입·관측 전은 NULL이다. 주 중간 가입·관측 경계는 부분 관측으로 표시되므로 완전한 4주와 구별한다. 미래 결과·배정·처치 후 쿠폰·persona·latent 반응 타입을 과거 피처에 넣지 않는다. 공개 factual 로그의 `posthoc_response_type`은 NULL이고 ST/P/SD/LC 잠재값은 평가 파일에서만 받는다.

현재 구현은 `simulate.v4.data.weekly.build_weekly_panel`, `build_experiment_log`, `simulate.v4.data.experiment_exports.export_tables`다. 실행은 저장소 루트의 `python -m cli v4 export-experiment --help`를 따른다. 상세 시간·회계 정의는 [주간 계약](../paper/simulator_v4/weekly_experiment_contract.md), 버전 충돌과 원본 출처는 [main 통합 기록](../docs/main_integration.md)에 있다.
