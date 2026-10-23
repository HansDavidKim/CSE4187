# HeartSteps V1과 The Complete Journey: 데이터 준비·감사

## HeartSteps V1

공식 저장소 https://github.com/klasnja/HeartStepsV1 커밋 `3016391de426116bdef41880d72bc8cd4b9b2477`의 `suggestions.csv`를 로컬에서 점검했다. 저장소 라이선스는 CC BY 4.0이며, 논문에서는 저자들이 요청하는 Klasnja et al. (2019), https://doi.org/10.1093/abm/kay067 를 인용한다. 로컬 데이터는 `data/raw/heartsteps-v1`에 보관한다.

- 사용자 37명, 의사결정 8,274행, 사용자·결정 index 중복 0개. 1인당 175~309행(중앙값 215).
- 연결·가능·행동 기록이 있는 6,590행: 발송 3,915행, 미발송 2,675행. 전체 발송 여부 결측 735행; 불가능 상태인데 발송된 3행은 별도 점검이 필요하다.
- 위 6,590행 중 30분 후 Jawbone 걸음 수 결측은 789행이다. 센서 미측정은 0걸음으로 간주하지 않는다. 관측된 경우만으로 계산한 평균은 선택 편향 가능성이 있어 인과 효과로 사용하지 않는다.
- `is.randomized`는 무작위 배정 기회가 있었는지 나타내는 eligibility 지표로 해석하면 안 된다. 이 변수를 True로 필터링하면 발송 비교군이 대부분 사라진다. 공식 변수 설명 https://github.com/klasnja/HeartStepsV1/wiki/Documentation-for-suggestions.csv 및 당시 가공 설명 https://github.com/LisaGotzian/HeartSteps/blob/master/build_heartsteps_data.R 를 참고한다.
- 논문·공식 문서는 가능 시점에서 no suggestion / walking / anti-sedentary를 0.4/0.3/0.3으로 설명한다. 실제 전달·측정 문제 및 기록된 가용성 정의를 검토하기 전에는 이 확률을 모든 행의 확정 propensity로 취급하지 않는다.

의사결정 시점의 `jbsteps30pre` 등은 이전 상태 후보, `send`는 시스템의 intervention, `jbsteps30`은 이후 결과다. `response.*`와 이후 걸음 수를 발송 전 state에 넣으면 누출이다. 사용자는 시스템 알림을 선택한 것이 아니므로 무작위 `send`를 사용자의 전문가 행동으로 해석해 MaxEnt IRL에 넣으면 안 된다. 걸음 수는 개수이며 사용자의 명시적인 다항 선택 레이블이 아니다. 결측과 시점 간격을 보존해 별도 행동 정의를 정하고, 사용자·시간 분할 평가 및 무작위 개입 검증을 수행해야 한다.

## The Complete Journey

공식 https://www.dunnhumby.com/source-files/ 에서는 약 2년의 가구 구매 및 일부 가구 직접 마케팅 기록을 제공하고, 학술연구 사용을 제시한다. 다만 라이선스의 구체적인 수정·재배포 권한 및 배포 버전은 확인되지 않았다. 원본을 제3자 저장소에서 자동 취득하거나 재배포하지 않는다. 제공자의 안내에 따라 사용자가 원본 파일을 확보하면 `--journey` 옵션으로 로컬 검증을 실행할 수 있다.

감사 대상은 `transaction_data.csv`, `campaign_table.csv`, `campaign_desc.csv`의 거래, 가구·캠페인 기록, 캠페인 기간이다. 실제 파일이 없어 이번에는 실행하지 않았다. 비기록을 비접촉 대조군으로 간주하지 않으며, 캠페인 시작일은 실제 가구 접촉 시점으로 확인되지 않았다. redemption은 처치가 아니라 구매 이후 측정된 반응일 수 있고, 관측된 기록만으로 무작위 배정을 가정하지 않는다. 이용할 파일 버전과 허락 조건도 확인해야 한다.

## 재실행

```bash
PYTHONPATH=src python scripts/audit_sequential_datasets.py
PYTHONPATH=src python scripts/audit_sequential_datasets.py --journey /path/to/complete-journey
PYTHONPATH=src pytest -q
```

결과는 `artifacts/sequential-data-audit/`에 생성된다. 로컬 결정행 파일은 참가자 단위이므로 출처·개인정보·라이선스를 검토하고 무분별하게 공개하지 않는다. 이번 코드는 데이터 감사이며 NTS-NOTEARS, IRL, RL 실험은 수행하지 않았다. 기존 그룹 IRL 코드는 AmExpert의 78주·3개 행동과 특징에 고정되어 있어 두 데이터에 그대로 사용할 수 없다.
