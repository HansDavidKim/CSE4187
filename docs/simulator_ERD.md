# 고객 행동 시뮬레이터 ERD

출처: `docs/KIT_feature_simulator_design_v3.pdf` 3장(그림 1, 3.1 필드 표, 3.2 Event 로그).

```mermaid
erDiagram
    CUSTOMER ||--o{ SESSION : "starts"
    CUSTOMER ||--o{ COUPON : "receives"
    CUSTOMER ||--o{ EVENT : "performs"
    SESSION ||--o{ EVENT : "has"
    PRODUCT |o--o{ EVENT : "referred by"
    EVENT ||--o| ORDER : "purchase event 1:1"
    COUPON |o--o| ORDER : "used in"
    ORDER ||--|{ ORDER_ITEM : "contains"
    PRODUCT ||--o{ ORDER_ITEM : "for"

    CUSTOMER {
        int customer_id PK
        datetime signed_up_at "가입 시각 (별칭 signup_at)"
    }

    SESSION {
        int session_id PK
        int customer_id FK
        datetime started_at "별칭 session_start_at"
        datetime ended_at "종료 전 null"
    }

    EVENT {
        int event_id PK "데이터셋 내 증가 정수, 동시각 순서 보존"
        int customer_id FK "Session의 고객과 일치"
        int session_id FK
        datetime event_at "행동 발생 시각"
        string event_type "view | cart | remove_from_cart | purchase"
        int product_id FK "purchase는 null 가능"
        int quantity "cart·remove만 양수, view·purchase는 null"
        int unit_price_minor "행동 당시 표시 단가"
    }

    PRODUCT {
        int product_id PK
        int category_id "원자료 제공 범위에서 기록"
        int unit_price_minor "실행 내 고정 단가"
    }

    ORDER {
        int order_id PK
        int purchase_event_id FK, UK "NOT NULL, purchase Event 1건"
        string checkout_source "cart | buy_now"
        int coupon_id FK, UK "null 또는 유일한 쿠폰"
        int gross_minor "OrderItem 단가x수량 합"
        int discount_minor "실제 적용 할인액"
        int paid_minor "gross - discount"
    }

    ORDER_ITEM {
        int order_id PK, FK
        int line_no PK "약한 개체의 부분 키"
        int product_id FK
        int quantity "0보다 큼"
        int unit_price_minor "구매 당시 할인 전 단가"
    }

    COUPON {
        int coupon_id PK
        int customer_id FK
        datetime sent_at "실제 발송된 쿠폰만 생성"
        datetime expires_at "유효기간 [sent_at, expires_at)"
        float discount_rate "기본 10%"
        string policy_id
        int send_cost_minor "발송비"
    }
```

## 표기와 제약

- 속성 타입은 설계서에 명시되지 않아 추정한 값이다. `*_minor`는 통화 최소 단위(기본 KRW 1원) 정수, 시각은 UTC 저장.
- `ORDER_ITEM`은 `(order_id, line_no)`로 식별되는 약한 개체다.
- 설계서 그림 1의 Session–Order 관계(`has`)는 물리 FK가 아니다. Order의 고객·세션·구매시각은 `purchase_event_id`로 연결된 Event에서 얻으므로 위 그림에서는 Event–Order 1:1로 표현했다.
- purchase Event 1건 = Order 1건, Order 1건 = OrderItem 1행 이상.
- 쿠폰은 주문당 최대 1개, 쿠폰당 최대 1회 사용. 사용 여부는 `ORDER.coupon_id`로 판단하며 별도 Redemption 테이블은 없다.
- 장바구니, 실험 배정(`experiment_id`, `group` 등), 예측 결과는 개체가 아니다. 장바구니는 Event·OrderItem 조회로 복원하고, 실험·예측은 별도 결과 파일로 둔다.
- persona·반응 타입·life cycle은 생성기 내부 변수이며 ER에 포함하지 않는다.
