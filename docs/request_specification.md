회의용 요약. 상세 기준은 Request_modified.md를 따른다.

## 1. 목표와 최종 결과물

합성 고객 데이터로 **이탈 예측 → Uplift·CLV → 대상 고객 선정·예산 최적화 → A/B 평가·대시보드**가 연결되는 시스템을 구현한다.

- 제출: 전체 시스템이 동작하는 Public GitHub 저장소의 main 브랜치.
- 실행: `docker-compose up` 한 번으로 실행. 대시보드는 `http://localhost:8501`.
- 명령: `python src/main.py --mode train`, `-mode uplift`, `-mode optimize --budget 50000000`.

## 2. 시뮬레이터와 데이터

| 항목 | 요구사항 |
| --- | --- |
| 기본 원리 | 논문의 행동 선택·재방문·쿠폰 반응 구조 유지. 과거 행동·내부상태에 따라 확률 변화 |
| 고객 유형 | P0·P2·P3 × ST·P·SD·LC = 최대 12조합. 기본 Persona·쿠폰 반응 유형은 생성 시 고정하고 lifecycle은 별도 관리. NEW는 별도 Persona가 아님 |
| 이벤트 | 조회·장바구니 추가·제거·구매. 세션 종료·쿠폰 발송·사용은 별도 기록. 행동 전이·시간 구조 재현 확인 |
| 상품 | 실제 상품·가격 catalog 연결. 관측 구매가격 분포 활용. 근거 없는 Persona별 가격 선호나 가격에 따른 행동 변화는 추가하지 않음 |
| 쿠폰·금액 | 할인율 10% 고정. 전환·무반응·역효과, 시간 감쇠·반복 노출의 한계효과 감소 유지. 주문·확정 시각, 할인 전후 금액·통화·단위 기록. 발송비와 할인액 구분 |
| 규모 | 최소 20,000명·12개월 이상. small은 5,000명·6개월 |
| 이탈 | 정의를 설정으로 변경 가능. 생성 이탈률 15~25%. 방문 감소 등의 징후를 이탈 고객에게 강제하지 않음 |
| 실험 | 같은 생성 분포에서 고객 단위 무작위 배정. Treatment/Control 각각 10,000명 이상. 개입 시점·조건·결과 관측 기간 사전 정의 |

검색·리뷰·CS 문의, 할인율에 따른 구매 반응 모형, 쿠폰 가정의 민감도 분석은 필수 구현에서 제외한다. 관련 분석은 기존 조회·장바구니·구매 기록으로 대체한다.

## 3. 분석·모델·평가

| 기능 | 필수 구현·평가 |
| --- | --- |
| 코호트·여정 | 가입월별 M1·M3·M6·M12 리텐션과 이탈률 비교. 마지막 30일 행동 패턴 상위 5개·장바구니 포기 등 분석. 가입→첫구매→재구매→충성→이탈 퍼널 |
| Feature | 30개 이상 정의. RFM, 행동 변화율 5개 이상, 세션 품질 3개 이상, 시퀀스 2개 이상, 구매 주기 이상·시간대·여정 단계 포함. 결측·이상치 처리 및 Feature store 저장 |
| 이탈 ML | 트리 모델 2종 이상. 불균형 처리·5-Fold·튜닝. 테스트 AUC-ROC ≥ 0.78. SHAP 전역·개별 해석과 상위 10개, Precision-Recall 및 임계값 분석 |
| 이탈 DL | PyTorch로 LSTM 또는 Transformer 구현. 패딩·임베딩·Early Stopping. ML과 동일 테스트셋 비교 및 ML+DL 앙상블 실험. 모델·학습 로그·선택 근거 저장 |
| Uplift | 방법 2개 이상 비교. 고객별 score·기본 결과 확률로 Persuadables·Sure Things·Lost Causes·Sleeping Dogs 분류. Qini/Uplift Curve 평가와 Persuadables 특성 분석. 생성기 내부 반응 유형과 추정 결과 구분 |
| CLV | BG/NBD+Gamma-Gamma 또는 ML. 향후 12개월 예측·상위 20% 식별·Top-N 및 분포 보고. 합성 고객의 미래 12개월 할인 후 주문금액 합계와 비교. 발송비는 별도 관리 |
| 세그먼트 | 이탈 확률·Uplift·CLV로 최소 6개 분류. 고객 수·비율·평균 CLV·평균 이탈 확률과 우선순위 점수 산출·시각화 |
| 예산 최적화 | 동일한 10% 쿠폰의 발송 대상·시점·미발송 기준 결정. 비용·효과 정의, 그리디 또는 선형계획 구현. 예산 50%·100%·200% 비교와 예상 ROI 산출 |
| A/B 테스트 | 표본수·Power 분석(목표 ≥ 0.8). 이탈률 차이 Chi-square 또는 Z-test, 95% 신뢰구간·p-value·유의성(p < 0.05) 및 해석 |
| 대시보드 | 이탈 위험·코호트·Uplift 4분면·CLV·6세그먼트·대상 고객 우선순위·예산/ROI·A/B 결과 표시. 수동 새로고침 |
| 모니터링 | PSI 또는 KS로 Feature drift 탐지. AUC·Precision·Recall의 시간별 변화 추적. 임계치 초과·성능 저하 알림 및 기록 |

예산 최적화: **Maximize Σ(Upliftᵢ × CLVᵢ × Actionᵢ)**, 제약 **Σ(Costᵢ × Actionᵢ) ≤ Budget**.

## 4. 공통 개발·학습 기준

- Python ≥ 3.10·PyTorch ≥ 2.0·CPU 동작. 모든 파트는 호환되는 공통 Python 환경·lock·Docker runtime 사용.
- 시뮬레이터·각 모델을 모듈로 분리하고 공개 API를 import하여 연결. 고객 ID·기준일·예측 기간·금액 단위·버전 계약 정의.
- 후속 모델은 합성 고객 데이터로만 학습·평가. 실제 로그는 생성기 학습·보정, 상품 데이터는 catalog 구성에 활용 가능.
- 입력은 기준 시각까지, 정답은 이후 구간에서 산출. 내부 반응 유형·미래 행동은 예측 입력에서 제외. Uplift 입력에는 개입 후 정보도 제외.
- CLV 최종 검증에는 과거 입력 이력과 미래 12개월 구매 기록 모두 확보.
- 학습·평가 코드 분리, 설정·경로 외부화, 함수 docstring·브랜치 전략 적용. 모델 버전·학습 설정·결과 기록.
- 생성 행동과 쿠폰 반응의 관측 근거·가정을 구분하여 문서화. 이탈 예측의 메인 모델로 AutoML 사용 금지.

## 5. 필수 산출물

| 위치 | 내용 |
| --- | --- |
| `models/` | 학습 모델·전처리기·버전 정보 |
| `results/` | 이탈 성능·예측, Uplift score·4분면 CSV·평가 곡선, 예산 배분·ROI, 코호트 시각화 |
| `results/clv_predictions.csv` | 고객별 CLV·상위 20% |
| `results/segments_6plus.csv` | 6개 이상 세그먼트·priority score |
| `results/shap_summary.png` | SHAP Summary Plot |
| `results/monitoring_report.json` | PSI/KS·임계치 초과 기록 |
| `docs/` | `feature_dictionary.md`, `model_report.md`, `uplift_analysis.md`, `retention_strategy.md`, `ab_test_report.md` |
| `README.md` | 아키텍처·실행 방법 |

실시간 처리·생존 분석·개인화 추천·MLflow는 선택 과제다.