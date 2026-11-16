40_AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 시스템 구축

### 0. 요구사항 조정안

Simulator가 관측 데이터에 근거한 행동을 생성하도록 하고, 요구사항을 충족하기 위해 검증되지 않은 가정을 추가하는 문제를 줄이고자 다음 네 항목을 조정하였다.

1. **고객 Persona 정의**
    
    기존의 ‘VIP충성고객, 일반충성고객, 가격민감형 등 지정된 6가지 Persona’ 대신, 관측 로그에서 도출한 **P0·P2·P3를 기본 Persona로 사용**하도록 변경하였다. 고객의 기본 Persona와 시간에 따라 변하는 활동 상태인 lifecycle은 별도로 관리한다.
    
    여기에 쿠폰 반응 유형인 **Sure Things(ST), Persuadables(P), Sleeping Dogs(SD), Lost Causes(LC)**를 결합하여 고객을 세분화한다. 따라서 기본 Persona는 3종이지만, 쿠폰 반응까지 고려하면 최대 12개의 고객 유형을 구성한다. 반응 유형별 비율과 효과 크기는 Persona에 따라 다르며, 고객 생성 시 반응 유형을 확률적으로 선택하여 고정한다. 이 쿠폰 반응 유형은 관측 자료를 참고한 가정과 보정에 기반하며, 이후 Uplift 모델이 추정한 4분면 결과와 구분한다.
    
    이 변경은 지정된 이름에 맞춰 확인되지 않은 성향을 임의로 부여하는 대신, 관측된 행동 특성과 가정에 기반한 쿠폰 반응을 구분하여 고객의 이질성을 표현하기 위한 것이다.
    
2. **생성 이벤트의 종류**
    
    ‘최소 8종 이벤트 생성’ 조건을 변경하여, 관측 근거가 있는 **조회·장바구니 추가·장바구니 제거·구매**를 기본 행동 이벤트로 사용하도록 하였다. 세션 종료와 쿠폰 발송·사용은 별도로 기록한다.
    
    검색·리뷰·CS 문의를 추가하려면 발생 확률과 다른 행동과의 관계에 새로운 가정이 필요하다. 따라서 이벤트의 개수보다 관측된 행동 전이와 시간 구조를 얼마나 재현하는지를 평가하도록 변경하였다.
    
3. **시간에 따른 행동 변화**
    
    방문 빈도 감소, 구매 주기 증가, 세션 시간 단축 등의 변화를 요구하는 방식에서, **과거 행동과 내부상태에 따라 행동 선택 확률과 재방문 확률이 변하도록 구현하고 그 결과를 검증하는 방식**으로 변경하였다.
    
    특정 변화 양상을 이탈 고객의 필수 징후로 강제하면, 예측 모델이 생성기에 설정한 규칙을 학습하는 결과가 될 수 있다. 이를 줄이고, 생성된 고객 행동에서 어떤 변화가 실제로 나타나는지 분석하도록 하였다.
    
4. **마케팅 개입의 범위**
    
    쿠폰·푸시 알림 등 여러 개입을 포괄하던 요구사항을 **할인율 10%로 고정한 쿠폰 발송**으로 구체화하였다. Persona별 전환·무반응·역효과를 모델링하고, 효과의 시간 감쇠와 반복 노출에 따른 한계효과 감소를 반영하도록 하였다. 할인율 변화에 따른 구매 반응 모형은 추가하지 않으며, 관측 데이터에 근거한 설정과 가정에 기반한 설정도 구분하여 문서화한다.
    
    개입 종류를 늘리면서 근거가 부족한 반응 모형을 추가하기보다, 현재 Simulator에서 다루는 쿠폰 개입을 일관되게 모델링하고 평가하기 위한 변경이다.
    

위 네 항목을 중심으로 조정하고, 이에 연결된 검색·CS 분석 항목과 마케팅 전략·결과 예시도 같은 기준으로 맞추었다. 상품·가격 연결과 CLV 검증 기준, 학습 정보의 시간 경계, 공통 개발환경·공개 API 계약은 합의한 내용을 명시하였다. 원본의 목차와 최종 결과물 구성을 유지하며, ML·DL 이탈 예측, Uplift Modeling, CLV, 예산 최적화 등 핵심 모델링·평가 요구사항과 데이터 규모, Treatment/Control 규모, 이탈률 범위는 유지하였다.

### 1. 미션 소개

고객 이탈 예측은 이커머스 기업의 CRM팀과 그로스팀에서 수행하는 핵심 업무다. 신규 고객 획득 비용은 기존 고객 유지 비용의 5배 이상이며, 이탈률을 5%만 낮춰도 수익이 25% 이상 증가할 수 있다. 쿠팡, 마켓컬리, 무신사 등 국내 주요 이커머스 기업에서는 단순히 이탈을 예측하는 것을 넘어, 마케팅 개입이 효과 있는 고객을 식별하고 제한된 예산 내에서 최적의 리텐션 전략을 수립한다.
이 미션에서는 고객 행동 시뮬레이터를 직접 구현하여 다양한 고객 페르소나의 구매, 방문, 이탈 패턴을 생성한다. 코호트 분석으로 가입 시점별 이탈 패턴을 파악하고, RFM 분석과 행동 변화 피처를 설계한다. ML/DL 모델로 이탈을 예측하고, Uplift Modeling으로 마케팅 증분 효과를 측정하며, CLV 기반 예산 최적화와 A/B 테스트 분석까지 수행하는 End-to-End 리텐션 시스템을 구축한다.
단순히 모델을 만드는 것이 아니라, 누구에게 마케팅을 해야 효과가 있는지 판별하고, 제한된 예산을 어떻게 배분해야 ROI가 극대화되는지 의사결정하는 전체 프로세스를 경험한다.

---

### 2. 최종 결과물

다음 4가지 기능이 정상 동작하는 GitHub 레포지토리(Repository) URL을 제출한다.

레포지토리는 docker-compose up 명령으로 한 번에 실행되어야 한다.

1. **이탈 예측 파이프라인**
    - 입력 : python src/main.py --mode train 명령 실행
    - 출력 : models/ 폴더에 학습된 모델 파일이 저장되고, results/ 폴더에 AUC-ROC, SHAP Summary Plot이 생성된다.
2. **Uplift 기반 고객 세그먼테이션**
    - 입력 : python src/main.py --mode uplift 명령 실행
    - 출력 : results/ 폴더에 고객별 Uplift Score와 4분면 세그먼트(Persuadables, Sure Things, Lost Causes, Sleeping Dogs) 분류 결과가 CSV 파일로 저장된다.
3. **예산 최적화 및 전략 제안**
    - 입력 : python src/main.py --mode optimize --budget 50000000 명령 실행
    - 출력 : results/ 폴더에 세그먼트별 예산 배분 결과와 예상 ROI가 저장된다.
4. **통합 대시보드**
    - 입력 : 브라우저에서 http://localhost:8501 접속
    - 출력 : 이탈 위험 현황, Uplift 세그먼트 분포, 예산 최적화 시뮬레이션 화면이 표시된다.
5. **필수 산출물 체크리스트**
    
    
    | **산출물** | **파일/폴더 경로** | **확인 방법** |
    | --- | --- | --- |
    | 피처 딕셔너리 | docs/feature_dictionary.md | 30개 이상 피처 정의 |
    | 모델 리포트 | docs/model_report.md | ML/DL 비교 포함 |
    | 리텐션 전략 제안서 | docs/retention_strategy.md | 6세그먼트 전략 포함 |
    | CLV 예측 결과 | results/clv_predictions.csv | 고객별 CLV + Top-20% |
    | 6세그먼트 분류 결과 | results/segments_6plus.csv | priority score 포함 |
    | 모니터링 리포트 | results/monitoring_report.json | PSI/KS + 임계치 초과 기록 |
    | A/B 테스트 리포트 | docs/ab_test_report.md | Power 분석 + p-value |
    | Uplift 분석 문서 | docs/uplift_analysis.md | Qini Curve + 4분면 해석 |
    | SHAP Summary Plot | results/shap_summary.png | 상위 10개 피처 중요도 |

---

### 3. 과제 목표

이 과제를 마친 후, 학습자는 아래를 스스로 설명할 수 있어야 한다.

- 코호트 분석을 통해 가입 시점별 고객 이탈 패턴을 분석하고 해석할 수 있다.
- RFM 피처와 행동 변화율 피처의 설계 원리 및 비즈니스 의미를 설명할 수 있다.
- Uplift Modeling과 단순 이탈 예측의 차이를 이해하고, Persuadables와 Sleeping Dogs 식별의 비즈니스 가치를 설명할 수 있다.
- CLV와 Uplift Score를 결합하여 마케팅 예산 배분을 최적화하는 방법을 이해한다.
- A/B 테스트의 통계적 유의성 판단 방법과 Power Analysis의 역할을 설명할 수 있다.
- Treatment/Control 그룹 배정 방식과 교란 통제 전략의 타당성을 설명할 수 있다.

---

### 4. 기능 요구 사항

다음 요구사항을 모두 만족해야 한다.

1. **고객 행동 시뮬레이터**
    - config/simulator_config.yaml 파일에 관측 로그에서 도출한 P0·P2·P3 Persona를 정의하고 각각의 행동 패턴을 모델링해야 한다. 고객 생성 시 부여하는 기본 Persona(home persona)와 시간에 따라 변하는 활동 상태(lifecycle)는 별도로 관리해야 한다. 기본 Persona와 고객 생성 시 확률적으로 선택하여 고정한 쿠폰 반응 유형 ST·P·SD·LC를 결합하여 최대 12개의 고객 유형을 구성해야 한다. 신규 상태(NEW)는 별도의 고정 Persona로 세지 않는다.
    - 고객별로 관측 근거가 있는 조회(page_view), 장바구니 추가(add_to_cart), 장바구니 제거(remove_from_cart), 구매(purchase)를 기본 행동 이벤트로 생성해야 한다. 세션 종료와 쿠폰 발송·사용은 기본 행동 이벤트와 구분하여 기록해야 하며, 이벤트 종류의 개수 대신 행동 전이와 시간 구조의 재현성을 평가해야 한다.
    - 과거 행동과 내부상태에 따라 고객의 행동 선택 확률과 재방문 확률이 변하도록 구현해야 한다. 생성된 로그의 방문 빈도, 구매 주기, 세션 시간 등의 변화 양상을 분석하여 검증하되, 특정한 변화 양상을 이탈 고객의 필수 징후로 사전에 강제하지 않아야 한다.
    - 마케팅 개입은 할인율 10%로 고정한 쿠폰 발송을 대상으로 하며, 전환·무반응·역효과를 Persona별로 차등 모델링해야 한다. 쿠폰 효과의 시간에 따른 감쇠와 반복 노출에 따른 한계효과 감소를 유지하고, 관측 데이터에 근거한 설정과 가정에 기반한 설정을 구분하여 문서화해야 한다. 할인율 변화에 따른 구매 반응 모형이나 쿠폰 가정의 민감도 분석은 필수 구현 범위에 포함하지 않는다.
    - 실제 e커머스 상품·가격 데이터를 catalog로 연결하여 생성된 구매의 상품·주문금액·쿠폰 할인액을 계산해야 한다. 고객의 가격대는 관측된 구매가격 분포를 활용하고, 근거가 없는 Persona별 가격 선호를 임의로 부여하지 않아야 한다. 상품가격을 기존 행동 선택·재방문 확률에 추가로 반영하지 않는다.
    - 주문별 할인 전 금액, 쿠폰 할인액, 할인 후 금액, 통화·금액 단위와 주문·확정 시각을 기록해야 한다. 쿠폰의 발송과 사용을 구분하고, 발송비는 할인액과 별도로 기록해야 한다.
    - 최소 20,000명의 고객에 대해 12개월 이상의 행동 로그를 생성해야 한다. (small 모드: 5,000명/6개월)
    - 이탈 정의(예: 30일 미구매 또는 60일 미접속)를 설정(config) 파일에서 변경할 수 있어야 한다.
    - 동일한 고객 생성 분포에서 고객 단위로 무작위 배정한 마케팅 Treatment 그룹과 Control 그룹을 자동으로 각각 10,000명 이상 분리하여 레이블링해야 한다. 개입 시점·조건과 결과 관측 기간을 사전에 정하고 기록해야 한다.
    - 생성된 데이터의 이탈률이 15% 이상 25% 이하여야 한다.
2. **코호트 및 고객 여정 분석**
    - 가입월 기준 코호트별 M1, M3, M6, M12 리텐션 곡선(Retention Curve)을 산출해야 한다.
    - 코호트별 이탈률 차이를 분석하고 시각화해야 한다.
    - 코호트별 리텐션 곡선 시각화 파일이 results/ 폴더에 저장되어야 한다.
    - 이탈 고객의 마지막 30일 행동 시퀀스에서 공통 패턴상위 5개가 추출되어야 한다.
    - 이탈 직전 주요 행동(장바구니 포기, 장바구니 제거 등)의 빈도를 생성된 조회·장바구니·구매 기록으로 분석해야 한다.
    - 고객 여정 퍼널(가입 → 첫구매 → 재구매 → 충성 → 이탈)별 전환율과 이탈 시점을 분석해야 한다.
3. **피처 엔지니어링 (Feature Engineering)**
    - RFM 피처(Recency, Frequency, Monetary)를 산출해야 한다.
    - docs/feature_dictionary.md 파일에 30개 이상의 피처 정의가 작성되어야 한다.
    - 행동 변화율 피처를 최소 5개 이상 설계해야 한다.
        - 예시: 최근 2주 방문수 / 이전 2주 방문수, 구매 주기 변화율, 세션 시간 변화율, 장바구니 전환율 변화, 쿠폰 반응률 변화다.
    - 구매 주기 이상 피처(현재 미구매 일수 / 평균 구매 주기)를 산출해야 한다.
    - 세션 품질 피처(평균 세션 시간, 페이지뷰/세션, 조회 또는 장바구니에서 구매로 이어지는 전환율) 최소 3개 이상 산출해야 한다.
    - 시퀀스 피처를 최소 2개 이상 설계해야 한다.
        - 예시: 최근 N개 이벤트 타입 시퀀스 임베딩, 행동 패턴 클러스터 ID
    - 시간대별 행동 피처(주말/평일 구매 비율, 특정 시간대 활동 비율)를 산출해야 한다.
    - 고객 여정 단계 피처(현재 여정 단계, 단계별 체류 기간)를 산출해야 한다.
    - 모든 피처에 대해 결측치 처리와 이상치 처리 로직을 구현해야 한다.
    - 피처 산출 결과를 피처 스토어(Redis 또는 파일 기반)에 저장해야 한다.
4. **이탈 예측 모델 - ML 기반**
    - 최소 2가지 이상의 트리 기반 모델(XGBoost, LightGBM, CatBoost 중 선택)을 학습하고 성능을 비교해야 한다.
    - 클래스 불균형 처리 방법(SMOTE, class_weight, Focal Loss 중 선택)을 적용해야 한다.
    - 5-Fold Cross Validation을 수행해야 한다.
    - 테스트 데이터에 대해 AUC-ROC 0.78 이상을 달성해야 한다 (업계 평균 0.75 대비 +0.03 목표)
    - SHAP을 사용하여 전역 피처 중요도(Global Importance)와 개별 예측 해석(Local Interpretation)을 생성해야 한다.
    - SHAP Summary Plot이 results/ 폴더에 저장되어야 한다.
    - 상위 10개 피처 중요도가 출력되어야 한다.
    - 이탈 확률 임계값(threshold)에 따른 Precision-Recall Trade-off를 분석하고 비즈니스 목적에 맞는 임계값을 선정해야 한다.
    - 하이퍼파라미터 튜닝(Grid Search 또는 Optuna)을 수행해야 한다.
5. **이탈 예측 모델 - 딥러닝 기반**
    - 고객 행동 시퀀스를 입력으로 받는 LSTM 또는 Transformer 기반 모델을 구현해야 한다.
    - 시퀀스 데이터 전처리(패딩, 임베딩)를 구현해야 한다.
    - Early Stopping을 적용해야 한다.
    - ML 모델과 DL 모델의 성능을 동일한 테스트셋에서 비교해야 한다.
    - 앱상블(ML + DL 결합) 방식의 성능 향상 여부를 실험해야 한다.
    - DL 모델 파일/학습 로그/ML 대비 비교 리포트를 저장하고, 모델 선택 근거를 문서화해야 한다.
6. **Uplift Modeling**
    - 시뮬레이터에서 생성된 Treatment/Control 데이터를 활용해야 한다.
    - 최소 2가지 Uplift Modeling 방법(T-Learner, S-Learner, X-Learner, 또는 Causal Forest 중 선택)을 구현하고 비교해야 한다.
    - 고객별 Uplift Score(CATE: Conditional Average Treatment Effect)를 산출해야 한다.
    - Uplift Score를 기준으로 고객을 4분면으로 분류해야 한다.
        - Persuadables: 마케팅 효과 있음 (Uplift > 0, 기본 이탈 확률 높음)
        - Sure Things: 마케팅 없이도 유지 (Uplift ~ 0, 기본 이탈 확률 낮음)
        - Lost Causes: 마케팅해도 이탈 (Uplift ~ 0, 기본 이탈 확률 높음)
        - Sleeping Dogs: 마케팅하면 역효과 (Uplift < 0)
    - Qini Curve 또는 Uplift Curve를 그려 모델 성능을 평가하고 results/ 폴더에 저장되어야 한다.
    - Persuadables 세그먼트의 특성을 분석하여 타겟팅 기준을 도출해야 한다.
7. **CLV 예측**
    - 확률 모델(BG/NBD + Gamma-Gamma) 또는 머신러닝 기반 중 하나를 선택하여 구현해야 한다.
    - 향후 12개월 기준 고객별 예상 CLV를 산출해야 한다.
    - CLV 분포를 분석하고 상위 20% 고객을 고가치 고객으로 분류해야 한다.
    - CLV 예측의 정확도를 합성 고객의 기준일 이후 12개월 실현 구매금액과 예측값을 비교하여 검증해야 한다. 과거 입력 이력과 미래 12개월 구매 기록을 모두 확보하고, 정답은 쿠폰 할인 후 주문금액의 합계로 정의하여 통화·금액 단위를 예측값과 맞춰야 한다. 쿠폰 발송비는 이 구매금액에서 차감하지 않고 별도 비용으로 관리해야 한다.
    - 고객별 CLV 산출 결과를 results/clv_predictions.csv로 저장하고, Top-N 및 분포 리포트를 포함해야 한다.
8. **고객 세그먼테이션 및 우선순위**
    - 이탈 확률, Uplift Score, CLV를 기준으로 고객을 최소 6개 세그먼트로 분류해야 한다.
        - 예시: 고가치-Persuadables(최우선), 고가치-Sure Things(유지), 고가치-Lost Causes(심층 분석), 저가치-Persuadables(비용 효율적 개입), 저가치-Lost Causes(No Action), 신규고객(온보딩)
    - 각 세그먼트별 고객 수, 비율, 평균 CLV, 평균 이탈 확률을 산출해야 한다.
    - 리텐션 우선순위 점수(예: Uplift Score x CLV)를 산출해야 한다.
    - 세그먼트 분류 결과를 시각화해야 한다.
9. **리텐션 전략 및 예산 최적화**
    - 할인율 10%의 동일한 쿠폰 조건에서 세그먼트별 발송 대상·우선순위·발송 시점·미발송 기준을 구분한 리텐션 전략이 docs/retention_strategy.md에 정의되어야 한다. 쿠폰 금액·할인율이나 개입 종류를 늘리지 않고 대상 선정과 예산 배분을 최적화해야 한다.
    - 전략별 예상 비용과 예상 효과(Uplift 기반 전환율)를 정의해야 한다.
    - 총 마케팅 예산 제약 하에서 Expected Revenue를 최대화하는 예산 배분을 산출해야 한다.
        - 목적함수: Maximize Σ(Uplift_i × CLV_i × Action_i)
        - 제약조건: Σ(Cost_i × Action_i) ≤ Budget
    - 예산 최적화는 베이스라인(그리디 또는 선형계획 중 1개)을 먼저 구현하고, 정교한 제약/다중 목표는 확장 과제로 안내한다.
    - What-if 분석(예산 50%, 100%, 200% 시나리오)을 수행할 수 있어야 한다.
    - 최적화 결과에 따른 예상 ROI를 산출해야 한다.
10. **A/B 테스트 설계 및 분석**
    - 시뮬레이터 데이터를 활용하여 가상 A/B 테스트를 설계해야 한다.
    - 필요 샘플 사이즈를 통계적으로 산출(Power Analysis)해야 한다.
    - Treatment 그룹과 Control 그룹의 이탈률 차이에 대한 통계적 유의성을 검정(Chi-square 또는 Z-test)해야 한다.
    - 95% 신뢰구간과 p-value를 산출해야 한다.
    - 통계적 유의성 여부(p < 0.05)가 명시되어야 한다.
    - A/B 테스트 결과 해석 리포트를 생성해야 한다.
11. **통합 대시보드**
    - 전체 고객의 이탈 위험 분포를 시각화해야 한다.
    - 코호트별 리텐션 곡선을 표시해야 한다.
    - Uplift 4분면 세그먼트 분포를 시각화해야 한다.
    - CLV 분포와 상위 고객 목록을 표시해야 한다.
    - 리텐션 예산 배분 결과와 예상 ROI를 표시해야 한다.
    - A/B 테스트 결과 요약을 표시해야 한다.
    - 리텐션 대상 고객 목록(우선순위순)을 조회할 수 있어야 한다.
    - 6세그먼트 분류 결과 및 priority score 시각화를 포함해야 한다.
    - 데이터는 수동 새로고침으로 업데이트되어야 한다.
12. **모델 모니터링**
    - 피처 분포 변화(Data Drift)를 탐지(PSI 또는 KS-test 기반) 하는 로직을 구현해야 한다.
    - 모델 성능 지표(AUC, Precision, Recall)의 시간별 변화를 추적해야 한다.
    - Data Drift 또는 성능 저하 발생 시 알림을 생성해야 한다.
    - results/monitoring_report.json에 PSI/KS 계산 결과 및 임계치 초과 기록을 저장해야 한다.
13. **문서화 및 코드 품질**
    - README.md에 시스템 아키텍처 다이어그램을 포함해야 한다.
    - docs/feature_dictionary.md에 피처 설계 근거와 각 피처의 비즈니스 의미를 문서화해야 한다.
    - docs/model_report.md에 모델 학습 과정과 성능 평가 결과를 문서화해야 한다.
    - docs/retention_strategy.md에 Uplift Modeling 적용 과정과 세그먼트 해석을 문서화해야 한다.
    - 세그먼트별 리텐션 전략 제안서를 작성해야 한다.
    - A/B 테스트 결과 해석 리포트를 작성해야 한다.
    - 주요 함수에 docstring을 작성해야 한다.
    - 프로젝트 구조가 모듈별로 분리되어 있어야 한다.

---

### 5. 보너스 과제 (선택)

1. 실시간 이탈 위험 스코어링
    - 해야 할 작업: Kafka 또는 Redis Streams를 사용하여 실시간 이벤트 스트림을 처리하고, 고객별 이탈 위험 점수를 실시간으로 업데이트하는 파이프라인을 구현한다.
    - 배움 포인트: 배치 처리와 실시간 처리의 아키텍처 차이를 체감하고, 스트림 처리 프레임워크를 경험한다.
2. 이탈 시점 예측 (Survival Analysis)
    - 해야 할 작업: Cox Proportional Hazards 또는 Survival Random Forest를 사용하여 "언제" 이탈할지 시점을 예측하는 모델을 구현한다.
    - 배움 포인트: Survival Analysis 기법의 비즈니스 활용을 경험하고, 이진 분류와 생존 분석의 차이를 이해한다.
3. 개인화 추천 연동
    - 해야 할 작업: 이탈 위험이 높은 고객에게 개인화 상품 추천을 제공하는 모듈을 구현하고, 추천 상품이 대시보드에 표시되도록 연동한다.
    - 배움 포인트: 이탈 예측 모델과 추천 시스템의 연동을 경험하고, 리텐션 전략의 개인화 방안을 이해한다.
4. MLflow 기반 실험 관리
    - 해야 할 작업: MLflow로 모델 학습 실험을 추적하고, 모델 레지스트리에 버전 관리를 구축한다. 모델 서빙까지 연동한다.
    - 배움 포인트: ML 실험 관리의 중요성과 모델 버전 관리, 배포 자동화를 체감한다.

---

- 구현 가이드
    
    아래는 문제 해결에 도움이 될 수 있는 검색 키워드와 권장 순서다. 정답이 아니므로 자유롭게 접근해도 된다.
    
    ### 권장 학습 순서
    
    1. 시뮬레이터 설계 및 데이터 생성
    2. 코호트 분석 및 EDA
    3. 피처 엔지니어링
    4. ML 이탈 예측 모델 학습 및 SHAP 해석
    5. DL 시퀀스 모델 구현 및 앙상블
    6. Uplift Modeling 및 4분면 세그먼테이션
    7. CLV 예측 및 고객 우선순위 산출
    8. 예산 최적화 구현
    9. A/B 테스트 분석
    10. 통합 대시보드 개발
    11. 모니터링 및 문서화
    
    <u>동료 평가와 유지보수를 용이하게 하기 위해 다음 7가지 지침 준수를 권장</u>
    
    **G1. 노트북(Notebook)과 소스 코드(Source)의 분리**
    
    - 원칙: `.ipynb` 파일은 오직 데이터 탐색(EDA), 시각화, 간단한 실험 용도로만 사용합니다.
    - 실행: 최종 파이프라인(전처리, 학습, 추론)은 반드시 `src/` 폴더 내의 `.py` 파일로 옮겨야 합니다.
    - 이유: 노트북 파일은 버전 관리(Diff 확인)가 어렵고, 자동화 파이프라인에 부적합합니다.
    
    **G2. 모듈화 및 함수형 프로그래밍**
    
    - 원칙: 하나의 파일에 모든 코드를 작성하지 마십시오. 기능별로(데이터 로드, 전처리, 모델링 등) 파일을 분리합니다.
    - 실행: 주요 로직은 함수(Function) 또는 클래스(Class)로 감싸야 하며, 각 함수는 하나의 역할만 수행하도록 설계합니다.
        - Bad: `def do_everything(): ...`
        - Good: `def load_data(): ...`, `def calculate_rfm(): ...`, `def train_uplift_model(): ...`
    
    **G3. 설정 관리**
    
    - 원칙: 하드코딩(Hard-coding)을 금지합니다.
    - 실행: 파일 경로, 모델의 하이퍼파라미터(Learning rate 등), 마케팅 예산 범위 등 변경 가능한 값은 `config.yaml` 또는 `src/config.py`에서 관리합니다.
    - 이유: 코드를 수정하지 않고 설정만 변경하여 다양한 실험을 수행하기 위함입니다.
    
    **G4. 시뮬레이터 데이터 파이프라인**
    
    - 원칙: 데이터 생성부터 학습 데이터셋 구축까지 재현 가능해야 합니다.
    - 실행: `src/data/make_dataset.py`를 실행하면 시뮬레이터가 동작하여 데이터를 생성하고, 전처리를 거쳐 `data/processed/` 폴더에 학습용 파일(parquet/csv)을 저장하도록 스크립트를 작성합니다.
    
    **G5. 모델링 및 실험 관리**
    
    - 원칙: 모델 파일은 반드시 버전 관리되어야 합니다.
    - 실행: 모델 학습이 완료되면 `models/churn_model_v1.pkl`과 같이 버전을 명시하여 저장합니다. 학습 시 사용된 파라미터와 결과 점수는 로그로 남겨야 합니다.
    
    **G6. 코드 스타일 및 문서화**
    
    - 원칙: Python 표준 스타일 가이드(PEP8)를 준수합니다.
    - 실행:Python
        - 변수명은 직관적으로 작성합니다 (예: `df` -> `user_transaction_df`).
        - 모든 주요 함수에는 Docstring을 작성합니다.
        
        ```python
        def calculate_clv(churn_prob, average_order_value):
            """
            이탈 확률과 평균 주문 금액을 기반으로 1년 예상 CLV를 계산합니다.
        
            Args:
                churn_prob (float): 이탈 확률 (0~1)
                average_order_value (int): 평균 주문 금액
        
            Returns:
                float: 예상 CLV
            """
            ...
        ```
        
    
    **G7. 대시보드와 로직의 분리**
    
    - 원칙: 대시보드 파일(`app.py`)에는 UI 관련 코드만 남깁니다.
    - 실행: 데이터 로딩이나 복잡한 계산 로직은 `src/` 내의 모듈을 `import`하여 사용합니다. 이렇게 해야 대시보드 코드가 깔끔해지고 디버깅이 쉬워집니다.
    
    ### 참고 키워드
    
    - Customer Churn Prediction, Churn Modeling
    - Cohort Analysis, Retention Curve
    - Customer Journey Analysis, Funnel Analysis
    - RFM Analysis, Customer Segmentation
    - Feature Engineering for Churn, Behavioral Features
    - Sequence Modeling, LSTM for Churn, Transformer for Tabular Data
    - Class Imbalance, SMOTE, Focal Loss
    - SHAP, Model Interpretability, Feature Importance
    - Uplift Modeling, CATE, T-Learner, X-Learner, Causal Forest
    - CausalML, EconML, Qini Curve
    - Customer Lifetime Value, BG/NBD, Gamma-Gamma Model
    - Lifetimes library Python
    - Marketing Budget Optimization, Linear Programming, PuLP
    - A/B Testing, Statistical Significance, Power Analysis
    - Chi-square Test, Z-test for Proportions
    - Data Drift Detection, Model Monitoring
    - Streamlit Dashboard
    
    ### 참고 데이터셋 (시뮬레이터 설계 참고용)
    
    - Kaggle Telco Customer Churn
    - UCI Online Retail Dataset
    - Instacart Market Basket Analysis
    - Criteo Uplift Prediction Dataset
    - KKBox Churn Prediction (Kaggle)
    - Kaggle E-Commerce Dataset
    - Telco Customer Churn Dataset
    - CDNOW Purchase Data
    
    ### 참고 라이브러리
    
    - CausalML (Uplift Modeling)
    - EconML (Causal Inference)
    - Lifetimes (CLV 예측)
    - SHAP (모델 해석)
    - Optuna (하이퍼파라미터 튜닝)
    - PuLP, OR-Tools (최적화)
    - SciPy (통계 검정)
    - Evidently AI (Data Drift 탐지)

---

### 6. 개발 환경

- Python 3.10 이상
- PyTorch 2.0 이상
- Docker 및 Docker Compose 사용
- 전체 시스템은 docker-compose up 명령어 하나로 실행되어야 함
- CPU 환경에서 전체 시스템 동작
- 모든 파트의 의존성은 하나의 호환 가능한 공통 Python 환경으로 맞추고, 공통 lock 파일과 Docker runtime으로 버전을 관리해야 함

---

### 7. 제약 사항

- 데이터
    - 과제의 이탈 예측·Uplift·CLV 모델 학습·평가에는 시뮬레이터로 생성한 고객 데이터만 사용해야 한다. 관측 로그를 이용한 시뮬레이터의 행동 모형 학습·보정과 실제 상품·가격 catalog 연결은 허용한다.
    - 시뮬레이터의 고객 페르소나, 행동 패턴, 마케팅 반응 모델을 문서에 명시해야 한다
    - 생성 데이터의 통계적 특성(이탈률 15-25%, Uplift 분포)이 현실적인 범위여야 한다
    - 생성 데이터의 Treatment/Control 그룹은 고객 단위 무작위 배정으로 구성해야 한다
    - 참고 데이터셋(UCI Online Retail, Kaggle E-Commerce 등)은 시뮬레이터 설계·행동 모형 학습·보정 또는 상품 catalog 구성에 활용할 수 있으며, 과제의 후속 모델에 실제 고객 행을 직접 학습·평가 데이터로 제공하지 않는다.
- 모델링
    - 이탈 예측에 AutoML 도구(AutoSklearn, H2O 등)를 메인 모델로 사용하지 않는다
    - Uplift Modeling은 직접 구현하거나 CausalML, EconML 라이브러리를 사용할 수 있다
    - CLV 예측에 Lifetimes 라이브러리를 사용할 수 있다
    - SHAP 해석은 필수로 포함해야 한다
    - 딥러닝 모델(LSTM, Transformer)은 PyTorch로 직접 구현해야 한다
    - 모델 학습 코드와 평가 코드가 분리되어 있어야 한다
    - 학습 입력은 예측 기준 시각까지 확인할 수 있는 기록으로 만들고, 정답은 이후의 지정된 결과 구간에서 산출해야 한다. 시뮬레이터 내부의 쿠폰 반응 유형·미래 행동 정보를 입력에 사용하지 않으며, Uplift 입력에는 개입 이후의 정보를 포함하지 않아야 한다.
- 최적화
    - 예산 최적화는 직접 구현하거나 SciPy, PuLP, OR-Tools 등을 사용할 수 있다
    - 최적화 목적함수와 제약조건을 수식으로 명확히 정의해야 한다
- 대시보드
    - Streamlit, Dash, Gradio 중 하나를 사용해야 한다
    - 외부 유료 BI 도구는 사용하지 않는다
- 통계 검정
    - A/B 테스트 유의성 검정에 scipy.stats를 사용할 수 있다
    - 유의수준은 0.05를 기준으로 한다
    - 검정력(Power)은 0.8 이상을 목표로 한다.
- 코드 품질
    - 하드코딩된 파일 경로, 비밀번호 등은 환경 변수 또는 설정 파일(config 파일)로 분리해야 한다
    - 주요 함수에는 docstring을 작성해야 한다
    - 프로젝트 구조는 모듈별로 분리되어야 한다
    - 시뮬레이터와 각 모델은 import하여 사용할 수 있는 공개 input/output API를 정의해야 한다. 고객 ID, 기준일, 예측 기간, 금액 단위, 모델·데이터 버전 등 파트 간 공통 계약을 문서화해야 한다.
    - Git Flow 또는 GitHub Flow 브랜치 전략을 사용해야 한다
- 제출 형식
    - GitHub 레포지토리 URL만 제출한다.
    - main 브랜치에 최종 코드가 있어야 한다.
    - 제출 시점에 레포지토리가 Public 상태여야 한다.

---

### 8. 결과 예시

아래는 정답이 아니라 참고 예시다. 실제 문구, 디자인, 구현 방식은 달라도 된다.

- 시뮬레이터 실행 결과 예시
    
    ```bash
    $ python src/data/simulator.py
    [Simulator] Loading config from config/simulator_config.yaml
    [Simulator] Generating 20,000 customers over 365 days...
    [Simulator] Event types: page_view, add_to_cart, remove_from_cart, purchase
    [Simulator] Additional records: session_end, coupon_sent, coupon_use, coupon_expired
    [Simulator] Treatment: 10,000 / Control: 10,000
    [Simulator] Churn rate: 21.3%
    [Simulator] Saved to data/raw/
    ```
    
- 코호트 분석 결과 예시
    
    ```bash
    [Cohort] Monthly cohort retention analysis completed
    [Cohort] Retention rates by month:
    
    Cohort    | M0    | M1    | M3    | M6    | M12
    ----------|-------|-------|-------|-------|-------
    2024-01   | 100%  | 72.3% | 51.2% | 38.5% | 28.1%
    2024-02   | 100%  | 74.1% | 53.8% | 40.2% | 29.8%
    2024-03   | 100%  | 68.5% | 47.1% | 35.8% | 25.3%
    ...
    
    [Cohort] Key finding: M1-M2 transition shows highest drop-off (avg 15%)
    [Cohort] Critical intervention window: Day 20-40 after signup
    ```
    
- 이탈 예측 모델 성능 비교 예시
    
    ```bash
    [Model Comparison] Performance on test set (n=4,000)
    
    Model           | AUC-ROC | Precision | Recall | F1-Score
    ----------------|---------|-----------|--------|----------
    XGBoost         | 0.847   | 0.73      | 0.69   | 0.71
    LightGBM        | 0.852   | 0.74      | 0.71   | 0.72
    LSTM            | 0.831   | 0.70      | 0.72   | 0.71
    Transformer     | 0.839   | 0.71      | 0.73   | 0.72
    Ensemble(XGB+TF)| 0.861   | 0.76      | 0.74   | 0.75
    ```
    
- SHAP 상위 피처 예시
    
    ```bash
    [SHAP] Top 5 important features (XGBoost):
      1. days_since_last_purchase: 0.182
      2. purchase_frequency_change_4w: 0.156
      3. session_duration_trend: 0.134
      4. cart_abandonment_rate: 0.098
      5. coupon_response_rate_change: 0.087
    ```
    
- Uplift 세그먼트 예시
    
    ```bash
    Segment        | Criteria                           | Count  | Ratio  | Avg Uplift
    ---------------|-----------------------------------|--------|--------|------------
    Persuadables   | Uplift > 0.05, Churn Prob > 0.5   | 3,847  | 19.2%  | +12.3%
    Sure Things    | Uplift <= 0.05, Churn Prob <= 0.5 | 5,128  | 25.6%  | +0.8%
    Lost Causes    | Uplift <= 0.05, Churn Prob > 0.5  | 6,421  | 32.1%  | +0.2%
    Sleeping Dogs  | Uplift < 0, Any Churn Prob        | 4,604  | 23.0%  | -3.1%
    ```
    
- 예산 최적화 결과 예시
    
    ```bash
    [Optimization] Budget: 50,000,000 KRW
    [Optimization] Objective: Maximize Expected Revenue Saved
    
    [Optimization] Optimal allocation:
    Segment                | Strategy        | Budget     | Target | Exp. Saved | Unit Cost
    -----------------------|-----------------|------------|--------|------------|----------
    HV-Persuadables        | Target Coupon 10%  | 22,000,000 | 1,892  | 312 cust   | 70,513
    LV-Persuadables        | Auto Coupon 10% | 18,000,000 | 2,847  | 398 cust   | 45,226
    HV-Sure Things         | Limited Coupon 10% | 5,000,000  | 1,108  | 22 cust    | 227,273
    New Customer-Onboard   | Welcome Coupon 10% | 5,000,000  | 1,500  | 180 cust   | 27,778
    
    [Optimization] What-if Analysis:
    Budget Scenario | Expected Saved | ROI    | Marginal ROI
    ----------------|----------------|--------|-------------
    25M (50%)       | 412 customers  | 245%   | -
    50M (100%)      | 912 customers  | 297%   | 349%
    100M (200%)     | 1,523 customers| 248%   | 199%
    
    [Optimization] Recommendation: 50M budget achieves optimal ROI
    ```
    
- A/B 테스트 결과 예시
    
    ```bash
    [A/B Test] Campaign: "Win-back Coupon 10%" for Persuadables segment
    
    [A/B Test] Results:
    Metric          | Control (n=1,923) | Treatment (n=1,924) | Lift
    ----------------|-------------------|---------------------|-------
    Churn Rate      | 34.2%             | 22.8%               | -33.3%
    Conversion Rate | 0%                | 18.4%               | +18.4%
    Revenue/User    | 12,340 KRW        | 48,920 KRW          | +296.4%
    
    [A/B Test] Statistical Significance:
      - Chi-square statistic: 58.34
      - p-value: < 0.0001
      - 95% CI for churn rate difference: [-14.2%, -8.6%]
    
    [A/B Test] Conclusion: Campaign shows statistically significant positive effect
    [A/B Test] Recommendation: Roll out to full Persuadables segment
    ```
    

---

### 9. 동료 평가 질문 예시

기능 동작 검증

- [ ]  docker-compose up 명령으로 전체 시스템이 정상 실행되는가?
- [ ]  시뮬레이터가 20,000명 이상의 고객 데이터를 생성하는가?
- [ ]  Treatment/Control 그룹이 각각 10,000명씩 분리되어 있는가?
- [ ]  코호트 분석 결과가 M1, M3, M6, M12 리텐션율 표로 출력되는가?
- [ ]  RFM 피처, 행동 변화 피처, 시퀀스 피처가 정상 산출되는가?
- [ ]  ML/DL 기반 이탈 예측 모델이 모두 학습되고 예측 결과가 출력되는가?
- [ ]  CLV가 산출되고 results/clv_predictions.csv에 고가치 고객이 식별되는가?
- [ ]  DL 모델 파일/학습 로그/ML 대비 비교 리포트가 저장되어 있는가?
- [ ]  results/monitoring_report.json에 PSI/KS 결과 및 임계치 초과 기록이 있는가?
- [ ]  A/B 테스트에 Power 분석(목표 power ≥ 0.8) 또는 표본수 산정 근거가 리포트에 포함되는가?
- [ ]  30개 이상의 피처가 feature_dictionary.md에 정의되어 있는가?
- [ ]  ML 모델의 AUC-ROC가 0.78 이상인가?
- [ ]  SHAP Summary Plot이 results/ 폴더에 생성되는가?
- [ ]  Uplift Score와 4분면 세그먼트 분류 결과가 CSV로 출력되는가?
- [ ]  예산 입력 시 최적 배분 결과와 ROI가 산출되는가?
- [ ]  A/B 테스트 p-value와 95% 신뢰구간이 산출되는가?
- [ ]  대시보드(http://localhost:8501)에서 이탈 현황과 세그먼트 분포를 확인할 수 있는가?

코드 구조 및 설계

- [ ]  시뮬레이터, 피처, 모델, 최적화, 대시보드가 기능별로 모듈 분리되어 있는가?
- [ ]  설정값(하이퍼파라미터, 경로, 임계값 등)이 config 파일로 관리되는가?
- [ ]  새로운 피처 추가 시 기존 코드 수정이 최소화되는 구조인가?
- [ ]  모델 학습 코드와 추론 코드가 분리되어 있는가?
- [ ]  주요 함수에 docstring이 작성되어 있는가?
- [ ]  docs/ 폴더에 feature_dictionary.md, model_report.md, retention_strategy.md가 작성되어 있는가?

핵심 기술 원리 적용

- [ ]  코호트 분석이 가입월 기준으로 올바르게 수행되었는가?
- [ ]  RFM 피처가 정의(Recency, Frequency, Monetary)에 맞게 산출되었는가?
- [ ]  행동 변화율 피처가 이탈 징후를 포착하는 데 유의미한가?
- [ ]  Uplift Model이 Treatment/Control 데이터를 활용하여 CATE를 학습했는가?
- [ ]  4분면 세그먼트 분류 기준이 Uplift Score와 이탈 확률을 기반으로 하는가?
- [ ]  예산 최적화의 목적함수와 제약조건이 retention_strategy.md에 수식으로 명시되었는가?
- [ ]  A/B 테스트가 적절한 통계 검정 방법(Chi-square, Z-test 등)으로 수행되었는가?
- [ ]  고객 단위 Treatment/Control 무작위 배정과 개입 전 그룹 균형 검증(밸런스 체크) 방법이 적절한가?