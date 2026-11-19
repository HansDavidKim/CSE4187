"""WBS 4.2: 기준시점 ``t0``별 65주 구간(입력 13주 + 정답 52주), 정답 ``actual_clv_12m``, 고객 단위 분할.

정답은 ``[t0, t0 + 52주)`` 결제액(``gross_revenue - redemption_discount_cost``, 쿠폰 발송비 미차감)의 합이고,
52주가 모두 완전 관측일 때만 만든다. 시뮬레이터 ``future_revenue_12m``(달력 12개월, Control만)은 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import WEEK, build_features, fully_observed, to_cutoff
from .schema import CONTEXT_COLUMNS, INPUT_WEEKS, LABEL_COLUMN, LABEL_WEEKS, PROVENANCE_COLUMNS

SPLITS = ("train", "validation", "test")


def complete_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """패널에 없는 고객-주 행을 미관측 주로 추가한다(값은 0으로 채우지 않고 결측으로 둔다)."""
    weeks = pd.date_range(panel["week"].min(), panel["week"].max(), freq="7D")
    customers = np.sort(panel["customer_id"].unique())
    index = pd.MultiIndex.from_product([customers, weeks], names=["customer_id", "week"])
    if len(index) == len(panel):
        return panel
    attrs = dict(panel.attrs)
    result = panel.set_index(["customer_id", "week"]).reindex(index).reset_index()
    result["observed_week"] = result["observed_week"].astype("boolean").fillna(False).astype(bool)
    result["partial_coverage"] = result["partial_coverage"].astype("boolean").fillna(False).astype(bool)
    result["coverage_seconds"] = result["coverage_seconds"].fillna(0.0)
    # 고객 속성(`CONTEXT_COLUMNS`)은 그 고객의 첫 값으로, 출처 열(`PROVENANCE_COLUMNS`)은 패널 attrs 값으로 채운다.
    for name in (*PROVENANCE_COLUMNS, *CONTEXT_COLUMNS):
        if name in result:
            first = panel.groupby("customer_id")[name].first()
            result[name] = result["customer_id"].map(first) if name not in PROVENANCE_COLUMNS else attrs.get(name)
    result.attrs = attrs
    result.attrs["notes"] = [*attrs.get("notes", []), f"{len(index) - len(panel)} absent rows added as unobserved"]
    return result


def _week_matrix(panel: pd.DataFrame, values: pd.Series):
    """불리언 ``values``를 고객 × 주 행렬로 편다. 행이 없는 주는 False이며, 정렬된 주 목록도 반환한다."""
    weeks = pd.DatetimeIndex(panel["week"].unique()).sort_values()
    pivot = values.groupby([panel["customer_id"], panel["week"]]).any().unstack("week")
    pivot = pivot.reindex(columns=weeks).fillna(False).astype(bool)
    return pivot, weeks


def find_eligible_cutoffs(panel: pd.DataFrame) -> pd.DataFrame:
    """데이터에 13 + 52주가 모두 들어가는 기준시점마다 고객 수를 센다.

    ``eligible_customers``는 입력 13주와 정답 52주가 모두 완전 관측인 고객이다.
    """
    full, weeks = _week_matrix(panel, fully_observed(panel))
    seen, _ = _week_matrix(panel, panel["observed_week"].astype(bool))
    # 앞에 0 열을 붙인 누적합이라 임의 구간의 주 수 = 두 열의 차이다.
    full_sum = np.concatenate([np.zeros((len(full), 1)), full.to_numpy().cumsum(axis=1)], axis=1)
    seen_sum = np.concatenate([np.zeros((len(seen), 1)), seen.to_numpy().cumsum(axis=1)], axis=1)
    rows = []
    # position = t0의 주 위치. 입력 [position - 13, position), 정답 [position, position + 52)
    for position in range(INPUT_WEEKS, len(weeks) - LABEL_WEEKS + 1):
        before = full_sum[:, position] - full_sum[:, position - INPUT_WEEKS]
        after = full_sum[:, position + LABEL_WEEKS] - full_sum[:, position]
        # 입력 구간에 관측 주(부분 관측 포함)가 하나라도 있으면 t0 시점의 고객으로 본다.
        history = seen_sum[:, position] - seen_sum[:, position - INPUT_WEEKS]
        feature_complete, label_complete = before == INPUT_WEEKS, after == LABEL_WEEKS
        rows.append({"reference_week": weeks[position],
                     "customers_with_history": int((history > 0).sum()),
                     "feature_window_complete": int(feature_complete.sum()),
                     "label_window_complete": int((label_complete & (history > 0)).sum()),
                     "eligible_customers": int((feature_complete & label_complete).sum())})
    columns = ["reference_week", "customers_with_history", "feature_window_complete",
               "label_window_complete", "eligible_customers"]
    return pd.DataFrame(rows, columns=columns)


def select_cutoff(cutoffs: pd.DataFrame, cutoff=None) -> pd.Timestamp:
    """요청한 기준시점이 적격인지 확인하거나, 없으면 적격 고객이 가장 많은 기준시점(동률이면 최신)을 고른다."""
    usable = cutoffs.loc[cutoffs["eligible_customers"] > 0]
    if cutoff is not None:
        cutoff = to_cutoff(cutoff)
        if not usable["reference_week"].eq(cutoff).any():
            raise ValueError(f"cutoff {cutoff.isoformat()} has no customer with 13 observed input weeks "
                             f"and 52 observed label weeks")
        return cutoff
    if usable.empty:
        raise ValueError(f"no cutoff has {INPUT_WEEKS} fully observed input weeks and {LABEL_WEEKS} label weeks; "
                         "generate a longer simulation (at least 67 calendar weeks)")
    # 고객 수, 날짜 순으로 정렬하므로 마지막 행이 최다 고객 중 최신 기준시점이다.
    best = usable.sort_values(["eligible_customers", "reference_week"]).iloc[-1]
    return to_cutoff(best["reference_week"])


def customer_windows(panel: pd.DataFrame, t0) -> pd.DataFrame:
    """고객별 65주 행(long 형식). ``offset_week``는 -13..51, ``window_part``는 ``input``/``label``이다."""
    t0 = to_cutoff(t0)
    rows = panel.loc[panel["week"].ge(t0 - INPUT_WEEKS * WEEK) & panel["week"].lt(t0 + LABEL_WEEKS * WEEK)].copy()
    rows["offset_week"] = ((rows["week"] - t0) / WEEK).round().astype(int)
    rows["window_part"] = np.where(rows["offset_week"] < 0, "input", "label")
    return rows


def build_labels(panel: pd.DataFrame, t0) -> pd.DataFrame:
    """``t0``부터 52주 결제액 합(``actual_clv_12m``). 52주가 모두 완전 관측이 아니면 결측이다."""
    t0 = to_cutoff(t0)
    rows = panel.loc[panel["week"].ge(t0) & panel["week"].lt(t0 + LABEL_WEEKS * WEEK)]
    grouped = rows.assign(_full=fully_observed(rows)).groupby("customer_id")
    result = pd.DataFrame({"label_weeks_observed": grouped["_full"].sum().astype(int),
                           "label_paid_sum": grouped["paid_amount"].sum(min_count=1)})
    result["label_window_complete"] = result["label_weeks_observed"].eq(LABEL_WEEKS)
    # 미관측·부분 관측 주가 하나라도 있으면 합계가 과소 집계되므로 정답을 만들지 않는다.
    result[LABEL_COLUMN] = result["label_paid_sum"].where(result["label_window_complete"]).astype("Float64")
    result.index.name = "customer_id"
    return result.drop(columns="label_paid_sum")


def ledger_label_sums(orders: pd.DataFrame, t0) -> pd.Series:
    """주문 원장에서 ``[t0, t0 + 52주)`` 안의 고객별 결제액 합을 구한다(정답 대조용)."""
    t0 = to_cutoff(t0)
    inside = orders.loc[orders["order_at"].ge(t0) & orders["order_at"].lt(t0 + LABEL_WEEKS * WEEK)]
    return inside.groupby("customer_id")["paid_amount"].sum()


def build_dataset(panel: pd.DataFrame, t0, orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """``t0`` 이전 이력이 있는 모든 고객의 예측 변수, 정답, 상태 열을 만든다.

    ``orders``를 주면 정답을 주문 원장 합계와 고객별로 대조하고, 다르면 오류를 낸다.
    맥락 열(treatment)은 평가용으로만 남기며 예측 변수가 아니다.
    """
    t0 = to_cutoff(t0)
    # 예측 변수는 [t0 - 13주, t0), 정답은 [t0, t0 + 52주)에서 따로 계산하므로 정답이 예측 변수로 새지 않는다.
    features = build_features(panel, t0)
    labels = build_labels(panel, t0)
    result = features.join(labels, how="left")
    result["label_window_complete"] = result["label_window_complete"].astype("boolean").fillna(False).astype(bool)
    result["label_weeks_observed"] = result["label_weeks_observed"].fillna(0).astype(int)
    if orders is not None:
        # 원장 대조: 주간 패널로 만든 정답이 원장 주문 결제액 합과 정확히 같아야 한다(원장에 주문이 없으면 0).
        ledger = ledger_label_sums(orders, t0)
        labelled = result.loc[result["label_window_complete"], LABEL_COLUMN]
        expected = ledger.reindex(labelled.index).fillna(0)
        differs = labelled.astype(float).ne(expected.astype(float))
        if differs.any():
            raise ValueError(f"{int(differs.sum())} labels disagree with the order ledger, e.g. "
                             f"{list(labelled.index[differs][:5])}")
        result["ledger_checked"] = result["label_window_complete"]
    customer = panel.groupby("customer_id")
    if "treatment" in panel:
        result["treatment"] = customer["treatment"].first().reindex(result.index).astype("Int64")
    if "assigned_at" in panel:
        # 정답 구간 안에 실험 배정된 Treatment 고객 표시(정답에 쿠폰 효과가 섞인다).
        assigned = customer["assigned_at"].first().reindex(result.index)
        inside = assigned.ge(t0) & assigned.lt(t0 + LABEL_WEEKS * WEEK)
        result["treatment_in_label_window"] = (result.get("treatment", pd.Series(0, index=result.index))
                                               .fillna(0).eq(1) & inside.fillna(False))
    # 출력 `reference_week`는 예측 기준시점 t0이다(시뮬레이터 주간 행의 `reference_week`와 다른 개념).
    result.insert(0, "reference_week", t0)
    for name in ("currency", "amount_unit", "dataset_id"):
        result[name] = panel.attrs.get(name)
    return result.reset_index()


def labeled_rows(dataset: pd.DataFrame, require_complete_history: bool = True) -> pd.DataFrame:
    """학습·평가에 쓸 행: 정답이 있고, ``require_complete_history``이면 입력 13주도 완전 관측인 고객."""
    keep = dataset["label_window_complete"].astype(bool)
    if require_complete_history:
        keep = keep & dataset["feature_window_complete"].astype(bool)
    return dataset.loc[keep].reset_index(drop=True)


def split_customers(customer_ids, seed: int = 42, fractions=(0.6, 0.2, 0.2)) -> pd.Series:
    """고객 단위의 결정적 학습/검증/테스트 분할. 한 고객은 한 분할에만 속해 분할 간 누수를 막는다."""
    # 정렬한 ID에 seed 고정 순열을 적용하므로 입력 순서와 무관하게 같은 분할이 나온다.
    ids = np.sort(pd.unique(pd.Series(list(customer_ids), dtype=object)))
    if len(ids) < 3:
        raise ValueError("at least three labelled customers are needed for train/validation/test")
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = max(1, int(round(fractions[0] * len(ids))))
    n_validation = max(1, int(round(fractions[1] * len(ids))))
    # 테스트에 최소 1명이 남도록 학습 인원을 줄인다.
    n_train = min(n_train, len(ids) - n_validation - 1)
    labels = np.empty(len(ids), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_validation]] = "validation"
    labels[order[n_train + n_validation:]] = "test"
    return pd.Series(labels, index=pd.Index(ids, name="customer_id"), name="split")
