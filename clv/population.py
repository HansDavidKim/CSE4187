"""학습 모집단 정책: 고객 분할 전에 Train/Validation/Test로 쓸 고객을 정한다(모든 모델 공통).

* ``control_no_coupon``(새 학습의 기본): Control(``treatment == 0``, 쿠폰 미발송군)이면서 52주 정답 구간
  ``[t0, t0 + 52주)``의 관측 주에 쿠폰을 받지 않은 고객. Treatment(쿠폰 발송군)와 배정 값이 없는 고객은 ``unused``다.
  52주 CLV는 '쿠폰을 보내지 않았을 때의 예상 실결제액', 4주 모델은 '쿠폰을 보내지 않았을 때 구매 고객의
  할인 전 총구매액'을 추정한다.
* ``observed_experiment_mix``: 배정과 무관하게 모든 고객(기존 전달 번들의 정책, 정답에 쿠폰 효과가 섞인다).

정책은 학습·모델 선택·평가 대상만 정한다. 추론은 정책과 무관하게 Treatment·Control 전체 고객을 예측한다.
"""
from __future__ import annotations

import pandas as pd

from .schema import ASSIGNMENT_COLUMN, CONTROL_CODE, LABEL_WEEKS

POLICY_DEFINITIONS = {
    "control_no_coupon": f"control customers ({ASSIGNMENT_COLUMN} == {CONTROL_CODE}) with no coupon sent in the "
                         f"observed weeks of [t0, t0 + {LABEL_WEEKS} weeks); treatment and unassigned customers are "
                         "unused for training, selection and evaluation",
    "observed_experiment_mix": "all customers regardless of assignment; labels include observed coupon effects",
}


def policy_mask(dataset: pd.DataFrame, policy: str) -> pd.Series:
    """``build_dataset`` 결과의 고객마다 정책상 학습 대상이면 True인 Series(같은 index)를 돌려준다."""
    if policy == "observed_experiment_mix":
        return pd.Series(True, index=dataset.index)
    if policy != "control_no_coupon":
        raise ValueError(f"unknown population policy: {policy}")
    if ASSIGNMENT_COLUMN not in dataset or dataset[ASSIGNMENT_COLUMN].isna().all():
        raise ValueError(f"population_policy control_no_coupon needs the experiment assignment column "
                         f"{ASSIGNMENT_COLUMN!r}; set population_policy='observed_experiment_mix' to train "
                         "without assignments")
    control = dataset[ASSIGNMENT_COLUMN].eq(CONTROL_CODE).fillna(False).astype(bool)
    no_coupon = dataset["coupons_in_label_window"].fillna(0).eq(0).astype(bool)
    return control & no_coupon


def population_counts(dataset: pd.DataFrame, mask: pd.Series, policy: str) -> dict:
    """정책 적용 결과(고객 수, 배정군별 수, 쿠폰을 받아 제외된 Control 수)를 registry용으로 요약한다."""
    result = {"policy": policy, "definition": POLICY_DEFINITIONS[policy], "customers": int(len(dataset)),
              "eligible": int(mask.sum()), "unused": int((~mask).sum())}
    if ASSIGNMENT_COLUMN in dataset:
        assignment = dataset[ASSIGNMENT_COLUMN]
        control = assignment.eq(CONTROL_CODE).fillna(False).astype(bool)
        result.update(control=int(control.sum()), treatment=int((assignment.notna() & ~control).sum()),
                      unassigned=int(assignment.isna().sum()))
        if policy == "control_no_coupon":
            result["control_with_coupon_excluded"] = int((control & ~mask).sum())
    return result
