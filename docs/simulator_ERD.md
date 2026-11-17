# 고객 행동 시뮬레이터 ERD

출처: `docs/KIT_feature_simulator_design_v5_daewon.pdf` 1–3장(여섯 이벤트, 일곱 업무 관계, 가격·장바구니·주문 회계), 11장(쿠폰 사용·별점), 15장(실험 로그). 컬럼 이름은 `simulator/src/simulate/v4/simulator/raw_store.py`의 raw5 스키마 기준.

버전: `raw_schema_version=5`, `event_scope_version=shopping_coupon_review_v1`.

```mermaid
erDiagram
    CUSTOMER ||--o{ SESSION : "starts"
    CUSTOMER ||--o{ COUPON : "receives"
    CUSTOMER ||--o{ EVENT : "performs"
    SESSION |o--o{ EVENT : "has (review는 세션 없음)"
    PRODUCT |o--o{ EVENT : "referred by"
    EVENT ||--o| ORDER : "purchase event 1:1"
    EVENT |o--o{ EVENT : "coupon_usage, review -> purchase"
    COUPON |o--o| ORDER : "used in"
    COUPON |o--o| EVENT : "coupon_usage"
    ORDER ||--|{ ORDER_ITEM : "contains"
    ORDER_ITEM |o--o{ EVENT : "discounted / reviewed line"
    PRODUCT ||--o{ ORDER_ITEM : "for"

    CUSTOMER {
        string customer_id PK
        datetime signed_up_at "생성 시계의 가입 시각. 실제 가입일 아님"
    }

    SESSION {
        string session_id PK
        string customer_id FK
        datetime started_at
        datetime ended_at "종료 전 null"
    }

    EVENT {
        int event_id PK "증가 정수, 동시각 순서 보존"
        string customer_id FK
        string session_id FK "review는 null"
        datetime event_at
        string event_type "view | cart | remove_from_cart | purchase | coupon_usage | review"
        string product_id FK "purchase는 null"
        int quantity "cart, remove는 1. coupon_usage는 1. view, purchase는 null"
        int unit_price_minor "행동 당시 표시 단가"
        string offer_product_id FK "선택. 제시 상품"
        int offer_unit_price_minor "선택. 제시 단가"
        int related_purchase_event_id FK "coupon_usage, review가 가리키는 purchase Event"
        string coupon_id FK "coupon_usage만"
        int order_line_no "coupon_usage, review의 주문행"
        int discount_minor "coupon_usage의 할인액. Order와 일치"
        int rating "review의 1-5점"
        string rating_kind "상품 별점은 product_star_scenario"
        string contact_id "과거 호환용. 항상 null"
    }

    PRODUCT {
        string product_id PK
        string category_id "원자료 제공 범위에서 기록"
        int unit_price_minor "실행 내 고정 단가"
    }

    ORDER {
        string order_id PK
        int purchase_event_id FK, UK "NOT NULL, purchase Event 1건"
        string checkout_source "cart | buy_now | mixed"
        string coupon_id FK, UK "null 또는 유일한 쿠폰"
        int gross_minor "OrderItem 단가x수량 합"
        int discount_minor "실제 적용 할인액"
        int paid_minor "gross - discount"
    }

    ORDER_ITEM {
        string order_id PK, FK
        int line_no PK "약한 개체의 부분 키"
        string product_id FK
        int quantity "0보다 큼"
        int unit_price_minor "구매 당시 할인 전 단가"
    }

    COUPON {
        string coupon_id PK
        string customer_id FK
        datetime sent_at "실제 발송된 쿠폰만 생성"
        datetime expires_at "sent_at + 28일. 만료 시각은 사용 불가"
        float discount_rate "기본 10%"
        string policy_id
        int send_cost_minor "발송비"
    }
```

## 표기와 제약

- 업무 테이블은 7개이고 Event는 17컬럼이다. 시각은 UTC로 저장하고, 주차·표시 시각은 Asia/Seoul이다.
- `*_minor`는 원천 가격 단위의 0.01인 정수다. 통화는 `SOURCE_UNSPECIFIED`이며 원화가 아니다.
- `ORDER_ITEM`은 `(order_id, line_no)`로 식별되는 약한 개체다.
- purchase Event 1건 = Order 1건 = OrderItem 1행 이상. 구매한 상품·수량은 Event가 아니라 OrderItem에서 읽는다. Order의 고객·세션·구매 시각은 `purchase_event_id`로 연결된 Event에서 얻는다.
- 금액은 Order에서 한 번만 계산한다(`paid = gross − discount`). 같은 주문의 Event·OrderItem·coupon_usage를 각각 매출로 더하지 않는다.

### 이벤트별 규칙

| event_type | 의미 | 연결 |
| --- | --- | --- |
| `view` | 상품 조회와 표시 가격 | 같은 상품의 반복 조회도 기록 |
| `cart` | 장바구니에 1개 담기 | 잔량은 다음 세션에도 유지 |
| `remove_from_cart` | 실제 보유분 1개 제거 | 잔량이 음수가 될 수 없음 |
| `purchase` | 구매 결정 1건 | Order 1개와 OrderItem 여러 행 |
| `coupon_usage` | 구매에 적용된 할인 | 같은 고객·세션·시각의 purchase. 최고 단가 품목 1개(동률이면 낮은 `line_no`)의 10%, half-up 반올림 |
| `review` | 구매 상품의 1–5점 별점 | 구매 × 상품마다 최대 1건, 구매 후 따로 도착, `session_id` null |

### 장바구니

- 장바구니는 개체가 아니다. 고객·상품별로 과거 Event를 `(event_at, event_id)` 순서로 재생해 얻는다. 빈 장바구니에서 시작한다.
- 구매 시 차감은 `checkout_source`에 따른다.

| checkout_source | 장바구니 변화 |
| --- | --- |
| `cart` | 구매한 수량만큼 차감 (구매량은 보유량 이내) |
| `buy_now` | 차감 없음. 기존 장바구니 보존 |
| `mixed` | 상품마다 `min(보유량, 구매량)` 차감 |

### 쿠폰

- 주문당 최대 1개, 쿠폰당 최대 1회 사용. 사용 여부는 `ORDER.coupon_id`로 판단한다. `coupon_usage` Event는 같은 사실을 가리키는 로그다.
- Treatment 고객만 `t0`에 1회 발송받는다. 유효기간은 `[t0, t0 + 28일)`.

### 개체가 아닌 것 (별도 파일)

실행 디렉터리에 업무 테이블과 따로 저장되며, 업무 기록에는 넣지 않는다.

| 파일 | 내용 | 용도 |
| --- | --- | --- |
| `treatments.parquet` | 고객별 고정 배정: `customer_id`, `as_of_at`(= `t0`), `treatment`, `assignment_probability` | 학습의 T |
| `labels.parquet` | 기본 이탈 정답(28일 미방문) | Retention |
| `experiment_log.parquet`, `no_purchase_labels.parquet`, `weekly_panel.parquet` | 실험 로그(결정·발송·사용), 28일 미구매 정답, 고객 × 달력 주 합계 | 대조·검증 |
| `evaluation_only/` | 발송·미발송 쌍대 rollout, 잠재 진단 | 평가 전용. 모델 입력 금지 |
| `diagnostics.parquet` | 잠재 성분, 고객별 추출값 | 비공개 |

- life-cycle(ENTRY·HOME·EXIT)은 저장된 개체가 아니라 세션에서 계산하는 상태다.
- persona·잠재 성분·반응 타입은 생성기 내부 변수이며 ER에 포함하지 않는다.
