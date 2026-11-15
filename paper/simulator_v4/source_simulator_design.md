# 원천 프로필 기반 사용자 시뮬레이터

> upstream 설계 문서를 현재 import·테스트·설정 경로에 맞췄다. `upstream audit/...` 표시는 이 체크아웃에 복사하지 않은 역사적 참조다. 현재 저장소의 실행·검증 범위는 [프로젝트 README](../../README.md)와 [통합 기록](../../docs/main_integration.md)을 따른다.

기본 진입점은 `simulate.v4.simulator.EcommerceSimulator`다. Cosmetics 관측 사용자 프로필과 가격·장바구니 조건부 선택 모형을 읽어 세션·행동·주문을 생성하고 SQLite 원시 업무 기록을 저장한다. 이전 6개 고정 persona와 구매율을 사용한 생성기는 upstream `legacy_engine.py`의 역사적 비교 경로이며 이 체크아웃에는 포함하지 않았다.

현재 생성 범위는 **조회·담기·제거·구매·쿠폰 사용·상품별 별점 리뷰 6종**이다. PRD가 처음 요구한 8종에서 사용자가 검색·CS 제외를 지시한 범위 변경이다. 새 기본 생성과 bootstrap은 검색·CS 모형을 학습·로드·호출하지 않는다. 리뷰를 사용하는 새 기록은 raw schema 5의 7개 업무 테이블·17개 Event 컬럼을 유지한다. 쇼핑 IRL·재방문·기존 행동 시퀀스는 원래 네 쇼핑 행동을 사용하며, 쿠폰 사용은 구매 품목 중 최고 단가 한 개 할인에 붙는 회계 기록이다. 쿠폰 이벤트 계약(upstream `audit/simulator/coupon_usage_events.md`)을 따른다.

upstream의 기존 8종 데이터·PDF·`feature_registry_events_v1.json`·검색/CS 원천 학습 코드는 역사적 기록이다. 이 체크아웃은 현재 6종 경로와 과거 raw reader 호환을 포함하며 해당 자료 전체를 복사하지 않았다. 이전 8종 원천 설명(upstream `audit/simulator/eight_event_models.md`)과 `full_product_reviews_v7`/`small_product_reviews_v7`는 과거 계약·실행의 자료이며 현재 6종 결과가 아니다. 현재 확장 관측 피처는 KIT32에 리뷰·cart 7개를 더한 39개이며 [피처 사전](feature_dictionary.md)을 따른다.

리뷰는 Olist 주문 후 28일 내 관측 설문 기록의 발생·지연과 Amazon All Beauty 과거 별점 기반 다항 모형을 분리해 사용한다. **주문 응답 여부와 지연은 주문당 한 번만** 뽑고, 응답이 발생하면 그 주문의 서로 다른 구매 상품마다 별점 한 개를 생성한다. A를 3개·B를 1개 산 주문은 A와 B 각각 최대 한 개의 리뷰를 갖는다. 같은 상품을 다른 주문에서 다시 구매하면 그 새 구매에 대한 리뷰는 별도로 허용한다. 상품 수만큼 Olist 응답 확률을 독립 적용해 주문 응답률을 높이지 않는다.

raw5 리뷰는 `related_purchase_event_id`로 실제 구매를, `product_id/order_line_no`로 그 구매의 실제 품목을 가리킨다. 별점은 1–5, `rating_kind=product_star_scenario`이며 세션·수량·가격·offer·쿠폰·할인 필드는 NULL이다. 하나의 주문에 여러 상품 리뷰가 있어도 구매·금액·장바구니는 추가로 갱신하지 않는다. `review_diagnostics`도 구매×서로 다른 상품 단위이며, 주문 응답률은 구매 식별자로 중복을 제거해 계산한다.

Amazon 별점 문맥은 실제 도착 시각보다 먼저 도착한 상품 리뷰만 사용한다. 같은 시각에 도착한 A/B 별점은 서로의 값이나 순서를 입력으로 쓰지 않는다. Olist 예약 문맥의 과거 리뷰 수는 응답한 주문 수이므로, 도착한 이력을 구매 식별자로 중복 제거해 상품 수에 따라 부풀지 않게 한다. 현재 별점 모형은 고객의 과거 별점에 조건화하며 SKU별 평균·품질 효과를 학습한 모형은 아니다.

Olist에는 상품별 응답 분모가 없으므로 주문 응답을 구매 상품 전체로 전개하는 규칙은 명시적 시나리오다. 상품별 선택적 작성률·한 주문 내 별점의 결합분포를 실측했다고 설명하지 않는다. 상품 FK는 실제 생성 주문의 품목 연결이며 Amazon ASIN과 Cosmetics 상품을 실측 매칭한 것이 아니다. 과거 raw4의 구매당 최대 한 개·상품 NULL 리뷰는 그대로 읽고, 새 SKU 리뷰로 추정 변환하지 않는다. 리뷰 텍스트는 생성하지 않는다.

리뷰 모형은 쇼핑 RNG와 분리된 seed를 사용한다. 리뷰는 새 세션·주문·장바구니 변화를 만들지 않으며, 비동기 도착이 28일 미방문 life-cycle과 label을 되돌리지 않는다. 검색·CS의 역사적 NULL/FK 계약은 과거 raw4/5를 읽을 때만 유지한다.

## 현재 실행 경로

저장소 루트에서 다음 CLI를 사용한다. `--assets-root`에는 필요한 원자료·학습 모형이 있는 로컬 루트를 지정하고 출력은 각각 새 디렉터리로 지정한다. 모형을 새로 적합할 때만 실제 원자료를 준비하고 `generate --fit-missing`을 명시한다. `simulate.v4.data.make_dataset`은 모형 준비 함수 모듈이며 단독 데이터 생성 CLI가 아니다.

```bash
PYTHONPATH=src python -m cli v4 generate --assets-root /path/to/assets --output artifacts/v4/run --mode small
PYTHONPATH=src python -m cli v4 validate --data artifacts/v4/run
PYTHONPATH=src python -m cli v4 features --data artifacts/v4/run --output artifacts/v4/features
PYTHONPATH=src python -m cli v4 export-experiment --data artifacts/v4/run --output artifacts/v4/exports --evaluation-output artifacts/v4/evaluation_only/run
```

원천 분포와의 대조는 `simulate.v4.validation`, 작은 동일 창 품질 비교는 `simulate.v4.quality`가 담당한다. 이 작업도 실제 원천 자료·그 SHA와 일치하는 prepared cache 및 학습 모형이 필요하며 후속 모델을 학습하지 않는다.

```bash
PYTHONPATH=src python -m simulate.v4.validation --data-dir artifacts/v4/run --source-cache /path/to/prepared_source.parquet --output artifacts/v4/validation.json
PYTHONPATH=src python -m simulate.v4.quality --assets-root /path/to/assets --source-cache /path/to/prepared_source.parquet --customers 200 --seeds 17 29 --output artifacts/v4/quality.json
```

설정 로딩은 `simulate.v4.configuration.load_config`, 생성 자료 로딩은 `simulate.v4.data.loader.load_tables`가 공통 진입점이다. 아래 원천 적합 설명은 시뮬레이터 내부 모형의 계보이며 이탈 ML/DL·Uplift·CLV를 학습하는 명령이 아니다. `src.main --mode ...` 같은 main 요구사항의 전체 시스템 명령은 후속 구현 목표이고 현재 CLI가 아니다.

## 관측 자료가 연결되는 곳

기초 학습은 `PYTHONPATH=src python -m simulate.v4.irl.fit_shopping --checkout-episodes`로 `data/models/shopping_profiles_checkout.json`을 만든다. 기존 MNL 비교 경로는 그 위에 `fit_donor_prior --joint-policy`, `fit_checkout_transition`을 적합한 `shopping_profiles_aligned_v6.json`이다. 6종 고도화 경로 `src/simulate/v4/config/simulator.yaml`은 재학습한 joint latent IRL, 합성 전용 donor 성향 posterior, source-native 방문 timing bank와 재학습한 기존 리뷰 head를 연결한다. 이 native 구성을 현재 기본 설정에 반영했으며 실제 사용 모형과 판정 근거는 실행 manifest·평가 결과로 확인한다. 누락 artifact의 준비 순서는 `simulate.v4.data.make_dataset.ensure_source_models`가 담당하며 검색·CS는 제외한다. 같은 사용자 표본의 전체 관측 이력을 유지하며 원본 사용자·상품 ID를 문자열로 보존한다. 원본 가격은 Decimal로 0.01 원천 단위 minor로 변환하고 상품별 관측 양수 최빈가격을 고정 카탈로그에 사용한다. 통화가 검증되지 않았으므로 `SOURCE_UNSPECIFIED`이며 KRW 환산으로 설명하지 않는다. 0원·음수 가격의 원본 행동은 보존하되 양수 카탈로그 적합에서 분리한다.

프로필에는 관측 행동 횟수, 상품·카테고리 선호, 구매 반복, 가격대, 세션 수·길이, 행동 간격, 세션 종료 이후 idle gap, 관측 구매 세션 간격 등이 들어간다. 사용자별 표본이 없으면 같은 행동군 또는 전체 자료의 관측 분포를 사용한다. 원본에 없는 주문 ID·실제 수량·가입 시각·초기 cart·쿠폰 노출을 실측값으로 만들어 채우지 않는다. 구매 session 간격과 product-event 개수는 실제 주문 주기·수량과 구별한다.

기존 비교 actor는 즉시 contextual MaxEnt/MNL이다. 재학습 IRL은 과거 행동, 제시가격·참조가격, cart, 구매·반복 이력, 세션 진행과 완료된 과거 세션을 train-only 40차원 관측 표현으로 만들고 soft Bellman 정책의 causal likelihood를 학습한다. 내부 상태 표현40과 후속 예측 KIT39는 별개다. 원본의 마지막 관측 조회 상품은 실제 offer를 대신하는 proxy다. 학습·보류예측은 사용자·시간을 분리하고 전처리는 학습 범위에서만 적합한다. 선택 상관을 인과적 가격탄력성이나 실제 심리 선호 식별로 해석하지 않는다.

원본의 같은 고객·세션·동일 시각에 연속된 purchase 상품 행은 하나의 checkout episode로 묶는다. 표본의 14,194개 상품 구매 행은 1,695개 묶음이 된다. 실제 주문 ID를 복원했다고 주장하지 않는다. 중간 purchase 행의 선택 label만 제외하며 모든 상품 unit에 대한 재고·구매 이력·가격 참조값은 갱신한다. `action_counts/session_lengths`는 원래 unit 집계, `decision_action_counts/session_choice_lengths/choice_gap_seconds`는 묶음 결정 집계다.

기본 설정의 `return_model.json`은 과거 방문 최근성, 최근 두 30일 방문량, 구매·cart·제거·상품 탐색 이력으로 다음 30일 재방문을 예측하는 별도 원천 적합 모형이다. 생성 중에도 해당 window 이전의 실제 생성 이력으로 같은 context를 만든다. 학습 범위로 clip하는 runtime 규칙과 유지율 절편 보정은 원천의 순수 성능과 구별한다. 모델을 명시적으로 끄면 donor의 기술적 방문확률 baseline을 사용한다. 두 모델의 경로와 SHA를 metadata에 남긴다.

## 생성 순서와 장바구니

사용자 프로필을 샘플링한 뒤 해당 사용자의 관측 상품 빈도로 제시상품을 정한다. 가격과 행동 직전 cart·구매 이력을 먼저 만든 뒤 행동별 선택확률을 계산한다. 개인의 관측 행동 횟수는 공통 모형에 개인차를 더한다. 따라서 상품가격은 결제액뿐 아니라 행동 선택 입력에도 들어간다. 입력 카탈로그를 제공할 때도 같은 원천 통화·minor 단위여야 한다.

기존 MNL 경로는 donor의 episode 길이와 행동 빈도 보정을 사용한다. 현재 native IRL은 고객당 한 잠재 성분의 학습 정책과 END를 사용하며 donor 행동 배율이나 donor 길이로 종료를 덮어쓰지 않는다. 학습 END 대신 source 길이를 뽑아 비종료 정책을 적용한 별도 `source_length` ablation은 채택하지 않았다. 현재 기본 상한은 782회 정책 조회다. current known-total 지원 source 세션의 최대 checkout 결정781회에 마지막 END 판정1회를 더한 값이며, source 상한 근거(upstream `audit/simulator/source_supported_tail_cap_v4.json`)에 독립 재계산을 보존했다. known-total은 관측 총량이 행동을 지원한다는 뜻이며 특정SKU의 과거 재고가 완전히 관측됐다는 뜻은 아니다. 상한과 다음 예정 세션·관측 종료 경계는 안전·검열 조건이고, 782에서 반드시 자연 종료하도록 학습한 것은 아니다. 이전64 상한 실험은 역사적 결과로 유지한다. `session_diagnostics.termination_reason/truncated_session`으로 `causal_policy_end`, `max_events`, `scheduled_next_start`, `window_end` 등을 구별한다. cart 잔량 진단 시각은 세션 종료 시각이다.

cart는 사용자·상품별로 세션 사이에 유지한다. 담기 한 번에 1개, 제거는 보유 상품에서 1개다. 기본 구매는 개인 관측 checkout 묶음을 관측 빈도만큼 소진한 뒤 같은 행동군·전체 source 후보를 사용한다. 후보 수량을 맞추고 이전 주문의 unit 평균 가격에 대한 학습 전이로 표본을 뽑는다. 개인 자료가 없는 cold profile은 source 빈도를 사용한다. 해당 품목의 보유 잔량만 차감하고 나머지 cart는 보존한다. 전량 보유한 묶음은 `cart`, 일부만 보유하면 `mixed`, 보유가 없으면 `buy_now`다. source에 없는 초기 담기를 만들지 않는다. 하나의 구매 Event에 하나의 다품목 Order를 연결하며, 수량은 source 상품 행 수의 proxy다. raw v2는 부분·혼합·다품목 직접 구매를 지원하고 v1의 기존 계약도 읽고 검증한다.

구매 후 선택 context와 재방문 context의 상품 횟수는 Order header 수가 아닌 OrderItem unit 수다. 직전 category도 마지막 view가 아니라 실제 직전 상호작용 상품을 쓴다. 묶음 뒤에는 마지막 source 상품·동일 시각 unit 간격을 유지한다. 모든 과거 prefix의 수량·가격·쿠폰·회계를 저장 전 검증하며, 독립 표본 journal로 재대조한다.

## Source-native 방문 시계와 개인차 재구성

`visit_timing_v4.json`은 train hash 고객과 `window_end < 2020-01-29`인 완성 source 창만 사용한다. Entry는 최초 관측일의 UTC 자정을 anchor로 하여 실제 첫 방문의 time-of-day를 보존한다. 실제 가입 후 지연을 실측한 것이 아니다. Regular 창은 UTC15시(KST 자정)의 30일 창이며, profile×직전30일 활동 지원이 없으면 같은 과거활동 stratum pool, 그다음 regular pool로 돌아간다. Entry에는 별도 profile→entry pool을 쓴다. count와 모든 시작 offsets는 같은 관측 창에서 함께 뽑고 중복 template는 관측 빈도로 가중한다.

짧은 실제 노출은 full30 schedule과 교집합으로 잘라낸다. 시간을 압축·당기거나 재추출해 count를 유지하지 않는다. 동시각 시작을 보존하고 1초 jitter를 넣지 않는다. 다음 예정 시작은 앞 세션의 시간 상한이며 그 뒤 행동은 right-censor한다. 학습 END를 대신 만들어 붙이지 않고, 경계·안전 상한으로 잘린 세션은 완료 `ShoppingHistory`에서 제외한다. 세션 종료=다음 시작은 허용하지만 양수 시간 중첩은 허용하지 않는다. 이 변경은 기존 독립 count/idle 추출의 예정 방문 손실을 바로잡으며 방문 발생 확률을 임의 배율로 높이지 않는다.

파라미터 학습과 보류 next-action/활동 예측은 Jan29 train 경계와 strict-prefix posterior를 유지한다. 별도 `shopping_joint_latent_source_traits.json`은 **고정된** reward/P/encoder/prior/활동계수에서 전체 source 관측기간의 지원 이력으로 donor별 성분 posterior만 재구성한 합성 전용 산출물이다. 성분은 합성 고객당 한 번 뽑고 유지하며 private diagnostics에만 둔다. 전체 source bank는 holdout 예측에 사용할 수 없다. 이는 source donor의 관측 개인차를 조건으로 한 population reconstruction이며 새 실제 고객의 미래 예측 성능을 입증하지 않는다.

Timing bank의 train-only 표본과 합성 trait bank의 전체 관측 source 범위는 서로 다르다. 활동 head의 원래 UTC00 학습 snapshot과 runtime UTC15 query도 구별한다. 활동 발생·방문 시각·쇼핑을 모두 동일한 공동 생성 likelihood로 새로 적합했다는 주장은 하지 않는다.

개발 비교는 MNL/IRL actor×기존/native 시계의 2×2 ablation이다. 동일 native 시계의 MNL 대비 효과와 원래 전체 시스템 대비 변화를 별도로 제시한다. 원래 시스템에 비해 모든 지표가 개선됐다고 포괄해 쓰지 않는다. 후보를 고정한 뒤 확인 seed를 사용하며 확인 결과에 맞춰 다시 튜닝하지 않는다. 실제 값은 비교 프로토콜(upstream `audit/simulator/latent_irl_protocol.md`)과 실행별 benchmark를 따른다.

## 진입기·행동 persona·이탈기

`lifecycle`는 원시 세션 이력에서 별도로 계산한다. 최초 관측은 `ENTRY`, 첫 재방문부터 `HOME`, 마지막 세션 시작 이후 실제 28일이 지나면 `EXIT`다. 이탈 뒤 재방문하면 `HOME`으로 돌아간다. 다음 30일에 방문할 예정인지 미리 보고 과거 상태를 정하지 않는다. 정확히 28일 시각의 귀환은 EXIT timer 후 HOME 순서로 기록하며, 관측 종료 이후의 timer는 생성하지 않는다.

ENTRY는 실제 신규 가입, EXIT는 영구 계정 종료의 정답이 아니다. 원천에서 28일 이상 공백 뒤 귀환한 고객도 관측된다. `home_persona`는 source 전체 이력의 행동군을 가진 latent donor 태그이며 미래를 배제한 persona 분류 정답이 아니다. life-cycle, 세션 종료와 미래 churn label을 서로 복사하지 않는다. 반응 타입 ST/P/SD/LC도 실제 개인 인과효과로 식별되지 않는다.

## 실험·시간·결과 계약

의사결정 시점 t0는 Asia/Seoul 월요일 00:00이고 저장 시각은 UTC다. 기본 적격군은 `t0−30일,t0)`에 세션이 있는 고객이다. 이력만 사용해 적격군을 구성한 뒤 1:1 무작위 배정하고 Treatment에 10% 쿠폰을 정확히 한 번 보낸다. 유효기간은 30일이며 추가 발송하지 않는다. 쿠폰은 한 번만 사용할 수 있고 할인액은 정수 minor로 half-up 반올림한다. 실제 발송과 주문 사용을 구분한다.

기본 label은 `[t0,t0+28일)`에 새 세션이 없으면 churn=1이다. 미래 세션과 주문 로그에서 label·확정 순매출을 직접 계산한다. 목표 비율을 맞추기 위해 생성 후 label을 덮어쓰지 않으며, 작은 표본의 비율을 강제로 고정하지 않는다.

활동 발생 여부의 30일은 **생성 기준**이며 이탈 관측 4주와 구별한다. `src/simulate/v4/simulator/outcomes.py`가 원시 business record·배정·쿠폰을 수정하지 않고 `[t0,t0+28일)`의 세션과 결제를 재집계한다. `future_revenue_28d`와 기본 매출 alias는 28일이고 원래 `future_revenue_30d`·12개월 target은 유지한다. native timing 경로의 28일 확률은 실제 sampler와 같은 source profile·과거활동 stratum bank에서 계산한 `p28=p30*P(first_offset<28d)`다. 28일 경계는 엄격한 미만이며 label은 실제 세션으로 따로 계산한다. 기존 idle timing 경로는 원래의 analytic 식을 호환용으로 유지한다. 어느 쪽도 새로 학습한 28일 hazard라는 뜻은 아니다. 원래 definition·부모 dataset·원시 SHA와 28/30 window를 metadata로 구분한다.

기본 `retention_mode: source`는 30일 활동확률의 logit을 이동하지 않는다. 과거 적격군 선택과 미래 활동을 구분한다. PRD 목표율을 위한 방문 보정은 upstream `config/simulator_legacy_target.yaml`의 역사적 `target` 경로이며 원천 절대 분포와 구별한다. 기본 `signup_mode: full_history_uniform`은 관측 기간 전체에서 후보 signup을 뽑고 과거 활동 적격성을 검사한다. 최종 적격 고객 signup은 조건부 분포이며 실제 signup 실측값이 아니다. 마케팅 방문 반응의 평균에는 Hillstrom의 14일 이메일 실험을 30일 쿠폰 상황으로 전이한다. 기본 구매 반응은 Tianchi O2O 시나리오(upstream `audit/simulator/tianchi_coupon_response.md`)의 과거 구매 이력별 관측 사용률·사용 지연을 사용한다. 쿠폰당 한 번 선택된 반응은 기한 내 실제 쇼핑 결정을 구매로 바꾸고 기존 checkout 회계로 이어진다. 기본 경로의 dunnhumby 사용률 gate는 대체되며 두 확률을 곱하지 않는다. 관측 사용률의 다른 서비스·행동 기회 전이는 식별된 Cosmetics 인과효과가 아니다.

small은 5,000명·6개월, full은 기본 20,000명·24개월이다. full 관측 종료는 t0 이후 12개 달력월을 모두 포함하도록 필요한 며칠을 연장한다. 미래 12개월 순매출 target은 관측이 성숙한 Control에 제공하고, 미성숙 small이나 해당 정책과 다른 Treatment target은 null이다. 이는 원가를 뺀 이익의 현재가치가 아니다.

## 출력·재현·한계

`SimulationResult.save(new_directory)`는 새 디렉토리에 `raw.sqlite`, `raw.manifest.json`, 호환 Parquet 및 `manifest.json`을 함께 게시한다. 비어 있지 않은 기존 결과를 덮어쓰지 않는다. 일곱 업무 테이블과 결과·진단을 구분하며 SQLite 상세 계약은 [SQLite 원시 업무 기록](../../docs/simulator_ERD.md)에 있다.

범위 축소 당시의 6종 smoke 경로는 `data/generated/six_event_smoke_v1/`의 200명·6개월·seed 2203 실행이다. 2,429세션·60주문·6,174이벤트를 생성했으며 `event_scope_version=shopping_coupon_review_v1`이다. 생성기 이름은 `source-shopping-v7`, raw 버전은 5로 이전 8종 실행과 같으므로 dataset ID·범위 metadata·실제 이벤트 집합으로 구분한다. 이전 `full_product_reviews_v7`의 20,000명·8종 검증을 새 실행의 증거로 재사용하지 않는다. 기존 v4/v5 PDF는 보존 문서이며, 이번 Version4 수정판은 새로운 6종·39개 계약과 재학습 결과를 별도로 정리한다.

해당 범위 축소 smoke의 주간·실험 출력은 `results/six_event_smoke_v1/`이며 export manifest의 `fingerprint.export_schema_version=2`와 `fingerprint.event_scope_version=shopping_coupon_review_v1`을 따른다. 검색·CS 컬럼을 제외하며, 이벤트 수 축소를 이유로 원시 7관계·쿠폰 회계·상품 리뷰 FK를 바꾸지 않는다.

manifest는 seed, source model SHA256, 관측 범위, 적격성, 정책·feature·schema 버전, 원천 통화·단위, 관측 이탈률과 보정값을 기록한다. 새 `generation_fingerprint`는 설정, 실제 모형·runtime 코드, 고객 수N, 실제 상품 입력, 제공 고객 입력을 반영한다. dataset ID에 `-nN-<fingerprint 일부>`를 포함해 같은seed의 다른N·입력·모형 실행이 충돌하지 않게 한다. 이전 결과의 ID는 소급 변경하지 않는다. 같은 artifact·설정·seed·N·입력의 생성 테이블은 재현된다. 최종 보고에는 실행 중 실제로 읽은 artifact SHA를 사용하며 모델이 교체된 이전 결과를 새 모형의 검증으로 재사용하지 않는다.

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_v4_source_simulator.py tests/test_v4_raw_store.py
```

테스트는 외부 원본이 없는 환경에서도 작은 결정적 관측 fixture를 적합해 실행한다. seed·가격 입력의 정책 영향·cart 재생·세션 간 보존·회계·UTC 시점·적격성과 배정·로그 label·12개월 성숙·life-cycle·안전 저장을 검증한다. 테스트 fixture는 실측 통계가 아니다. 가격을 바꾼 검증도 학습된 가격 입력의 연결을 확인하며, 현실의 가격 인과효과를 증명하지 않는다.

Cosmetics의 5개월 관측으로 24개월 장기분포가 검증되는 것은 아니다. 활동 창과 시간 배열의 경계 처리, 개인별 전체 관측 이력의 행동군, 32개 이하 경험분위수·선호 상위 상품 요약, 유지율 보정과 다른 서비스의 쿠폰 반응 전이가 남는 한계다. 개인 재구매 간격을 관측한 source 사용자는 269명으로 희소한 개인은 같은 행동군·전체 자료의 관측 묶음을 사용한다. 실제 비교에서는 checkout 단위 후보(upstream `audit/simulator/checkout_quality_decision.md`)를 채택했고, 상품 선택 후보(upstream `audit/simulator/item_choice_quality_decision.md`)는 가격·시간·재담기가 악화되어 기본값으로 채택하지 않았다. 개선된 지표와 남은 반복·재담기 과다를 함께 확인한다.
