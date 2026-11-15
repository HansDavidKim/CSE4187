# 시뮬레이터 피처 사전

현재 예측 입력은 **KIT32 + 리뷰·장바구니 7개 = KIT39**, 버전 `kit39-review-cart-v2`다. 전체 정의는 [피처 사전 원본](../paper/simulator_v4/feature_dictionary.md), 기계 판독 registry는 [feature_registry_events_v2.json](../paper/simulator_v4/feature_registry_events_v2.json), 계산 구현은 [`features.py`](../src/simulate/v4/features.py)와 [`behavior_features.py`](../src/simulate/v4/behavior_features.py)에 있다.

각 입력은 `event_at < as_of_at`인 과거만 사용한다. 금액은 원본 통화의 정수 minor 단위를 유지하며 IDs·배정·정답·persona·latent 진단은 예측 입력에 포함하지 않는다. 검색·CS 피처는 현재 39개 계약에서 제외한다. 코드의 피처 순서를 기준으로 사용하며 후속 모델 학습이나 입력 확장을 자동으로 수행하지 않는다.
