"""WBS 4.7: 학습·평가·추론을 따로 실행하고 산출물을 저장하며, 예측 CSV를 쓴다.

``output_dir``(기본 ``results/clv``) 아래 산출물::

    registry.json            기준시점, 계약, 설정, 학습 모집단 정책, 모델 상태, 최종 모델, 4주 모델 상태
    model_card.json          모델 버전, 코드·라이브러리 버전, 학습 데이터 출처, 파일 체크섬(``bundle``)
    splits.csv               customer_id -> train/validation/test/unused, 배정, 정책 대상·이력 완전·학습 대상 여부
    models/                  baseline.json, bgnbd_gg.json, xgboost/{xgboost_model,xgboost_meta}.json,
                             xgboost_two_part/{classifier,regressor,two_part_meta}.json,
                             xgboost_gross_4w/{regressor,gross_4w_meta}.json
    reports/                 입력 검증, 적격 기준시점, 평가(이력 구간별·4주 모델·배정군 진단 포함), 10분위, Top-N,
                             BG/NBD 벌점 민감도, 정답 포함 예측(52주, 4주)

학습 모집단 정책(``population``)은 고객 분할 전에 적용되어 모든 모델이 같은 고객을 쓴다. 기본
``control_no_coupon``에서는 Control 고객만 Train/Validation/Test로 나누고 Treatment 고객은 ``unused``다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .adapter import adapt_orders, adapt_weekly, load_orders, load_weekly
from .baseline import BaselineModel
from .bundle import CARD_NAME, GROSS_MODEL, GROSS_TARGET_DEFINITION_ID, gross_status, write_model_card
from .config import CLVConfig
from .dataset import SPLITS, build_dataset, complete_grid, find_eligible_cutoffs, labeled_rows, select_cutoff, split_customers
from .evaluation import (
    decile_table, evaluate_by_history, evaluate_gross_4w, evaluate_predictions, gross_by_assignment, regression_metrics,
    top_n_table,
)
from .features import WEEK, fully_observed, to_cutoff
from .io import read_json, write_json
from .population import policy_mask, population_counts
from .probabilistic import MAX_NUMERICAL_FAILURE_SHARE, ProbabilisticCLV
from .schema import (
    ASSIGNMENT_COLUMN, CONTROL_CODE, GROSS_COMPLETE_COLUMN, GROSS_LABEL_COLUMN, GROSS_PREDICTION_COLUMN,
    GROSS_PURCHASES_COLUMN, GROSS_WEEKS, HIGH_VALUE_COLUMN, INPUT_WEEKS, LABEL_COLUMN, LABEL_WEEKS, OUTPUT_COLUMNS, PREDICTOR_FEATURES,
)
from .validation import reconcile_orders_with_weekly, validate_orders, validate_weekly
from .xgboost_model import (
    GrossXGBoost4w, InsufficientBuyersError, TwoPartXGBoostCLV, XGBoostCLV, classification_metrics, xgboost_status,
)

MODEL_NAMES = ("baseline", "bgnbd_gg", "xgboost", "xgboost_two_part")
# 모델 적합(train)과 선택(validation)에 쓴 분할. 진단의 예측 요약은 이 밖의 고객으로만 낸다.
FITTED_SPLITS = ("train", "validation")
# 모델 파일을 models/<이름>/ 하위 디렉터리에 두는 모델
SUBDIRECTORY_MODELS = ("xgboost", "xgboost_two_part")
# registry.json에 그대로 기록하는 입출력 계약(구간, 정답 정의, 기준시점 의미, 금액 단위)
CONTRACT = {
    "feature_window": f"[t0 - {INPUT_WEEKS} weeks, t0)",
    "label_window": f"[t0, t0 + {LABEL_WEEKS} weeks)",
    "label_definition": "sum of gross_revenue - redemption_discount_cost (paid amount); coupon send cost not deducted",
    "reference_week": "prediction cutoff t0, Asia/Seoul Monday 00:00; not the simulator weekly row start",
    "high_value_column": f"{HIGH_VALUE_COLUMN} is the CLV-internal name; the shared name needs team agreement",
    "amounts": "source minor units with the recorded currency and amount_unit; no conversion",
    "gross_4w_window": f"[t0, t0 + {GROSS_WEEKS} weeks), same customer_id and t0 as the 52-week label",
    "gross_4w_definition": "sum of gross_revenue (before discount, repeat orders included) of customers with at least "
                           "one purchase in the window; discount and coupon send cost not deducted; non-buyers have "
                           "no amount label",
    "population": "the population policy is applied before the customer split and shared by every model; inference "
                  "scores treatment and control customers alike",
}


def _is_adapted(frame) -> bool:
    """이미 adapter를 거친 CLV 스키마 주간 frame이면 True(다시 변환하지 않는다)."""
    return isinstance(frame, pd.DataFrame) and "paid_amount" in frame and "source_format" in frame.attrs


def prepare_inputs(weekly, orders=None, *, currency=None, amount_unit=None, strict: bool = True,
                   verify_checksum: bool = True):
    """주간 행(주어지면 주문 원장도)을 CLV 스키마로 변환하고 검증한다(WBS 4.1).

    Args:
        weekly: 경로 또는 frame(시뮬레이터 ``weekly_panel`` 또는 CLV 스키마).
        orders: 선택. 경로(raw.sqlite, CSV, Parquet) 또는 frame.
        strict: True면 검증 오류가 있을 때 예외를 낸다.

    Returns:
        ``(panel, orders, reports)``.
    """
    units = {"currency": currency, "amount_unit": amount_unit}
    if _is_adapted(weekly):
        panel = weekly
    elif isinstance(weekly, pd.DataFrame):
        panel = adapt_weekly(weekly, **units)
    else:
        panel = load_weekly(weekly, verify_checksum=verify_checksum, **units)
    # 행이 없는 고객-주는 0이 아니라 미관측 주로 채운다.
    panel = complete_grid(panel)
    reports = [validate_weekly(panel)]
    if orders is not None:
        if isinstance(orders, pd.DataFrame):
            # 이미 변환된 원장(paid_amount, order_at, currency attrs 보유)은 그대로 쓴다.
            orders = orders if "paid_amount" in orders and "order_at" in orders and "currency" in orders.attrs \
                else adapt_orders(orders, **units)
        else:
            orders = load_orders(orders, verify_checksum=verify_checksum, **units)
        # 주간 패널(features manifest)과 원장이 서로 다른 raw SQLite에서 나왔으면 섞어 쓰지 않는다.
        sources = {panel.attrs.get("raw_database_sha256"), orders.attrs.get("raw_database_sha256")} - {None}
        if len(sources) > 1:
            raise ValueError("weekly panel and order ledger come from different raw databases")
        reports += [validate_orders(orders), reconcile_orders_with_weekly(panel, orders)]
    if strict:
        for report in reports:
            report.raise_if_errors()
    return panel, orders, reports


def default_reference_week(panel: pd.DataFrame) -> pd.Timestamp:
    """기본 기준시점: 실험 시점 ``as_of_at``이 하나뿐이면 그 값, 아니면 마지막 완전 관측 주의 다음 주."""
    if "as_of_at" in panel:
        values = panel["as_of_at"].dropna().unique()
        if len(values) == 1:
            try:
                return to_cutoff(values[0])
            except ValueError:
                # as_of_at이 서울 월요일 00:00이 아니면 마지막 완전 관측 주 기준으로 넘어간다.
                pass
    last = panel.loc[fully_observed(panel), "week"].max()
    if pd.isna(last):
        raise ValueError("no fully observed week to anchor a reference week")
    return to_cutoff(last + WEEK)


def _models_for(model_dir: Path, names) -> dict:
    """``model_dir/models``에 저장된 모델을 이름별로 불러온다(4주 모델 ``xgboost_gross_4w`` 포함)."""
    loaded = {}
    for name in names:
        if name == GROSS_MODEL:
            loaded[name] = GrossXGBoost4w.load(model_dir / "models" / GROSS_MODEL)
        if name == "baseline":
            loaded[name] = BaselineModel.load(model_dir / "models")
        elif name == "bgnbd_gg":
            loaded[name] = ProbabilisticCLV.load(model_dir / "models")
        elif name == "xgboost":
            loaded[name] = XGBoostCLV.load(model_dir / "models" / "xgboost")
        elif name == "xgboost_two_part":
            loaded[name] = TwoPartXGBoostCLV.load(model_dir / "models" / "xgboost_two_part")
    return loaded


def _score(model, frame: pd.DataFrame, orders, t0) -> np.ndarray:
    """``frame`` 고객 순서대로 예측값을 낸다. BG/NBD + GG는 예측 변수 대신 주문 원장을 쓴다."""
    if model.name == "bgnbd_gg":
        if orders is None:
            raise ValueError("BG/NBD + Gamma-Gamma needs the order ledger")
        return model.predict(orders, t0, frame["customer_id"])["predicted_clv_12m"].to_numpy(dtype=float)
    return model.predict(frame)


def _dataset_summary(dataset: pd.DataFrame, labelled: pd.DataFrame, cutoff) -> dict:
    """기준시점 데이터셋의 고객 수, 관측 부족으로 제외된 수, 0 정답 비율을 요약한다(registry용)."""
    return {
        "gross_4w_window_complete": int(dataset[GROSS_COMPLETE_COLUMN].sum()),
        "gross_4w_buyers": int(dataset[GROSS_LABEL_COLUMN].notna().sum()),
        "reference_week": cutoff.isoformat(),
        "customers_with_history": int(len(dataset)),
        "feature_window_complete": int(dataset["feature_window_complete"].sum()),
        "label_window_complete": int(dataset["label_window_complete"].sum()),
        "excluded_incomplete_label": int((~dataset["label_window_complete"]).sum()),
        "excluded_incomplete_history": int((dataset["label_window_complete"] & ~dataset["feature_window_complete"]).sum()),
        "labelled_customers": int(len(labelled)),
        "zero_label_share": float(labelled[LABEL_COLUMN].astype(float).eq(0).mean()) if len(labelled) else None,
        "treatment_in_label_window": int(dataset.get("treatment_in_label_window", pd.Series(dtype=bool)).sum()),
    }


def _bgnbd_sensitivity(config: CLVConfig, orders, t0, parts: dict, reference) -> pd.DataFrame:
    """BG/NBD 벌점만 바꿔 Train 고객으로 다시 적합하고 모수·Validation 지표·순위 변화를 표로 만든다.

    보고용이다. 선택 모델은 ``config.bgnbd_penalizer``로 적합한 모델 그대로 둔다.
    ``reference``는 그 선택 모델의 Validation 예측값(순위상관 비교 기준)이다.
    """
    train_ids = parts["train"]["customer_id"]
    train_labels = parts["train"].set_index("customer_id")[LABEL_COLUMN]
    validation = parts["validation"]
    rows = []
    for penalizer in config.bgnbd_sensitivity_penalizers:
        row = {"bgnbd_penalizer": penalizer, "configured": penalizer == config.bgnbd_penalizer}
        try:
            model = ProbabilisticCLV(config.calibration, config.transaction_period, config.no_purchase_fallback,
                                     penalizer, config.gg_penalizer).fit(orders, t0, train_ids, labels=train_labels)
        except (ValueError, RuntimeError) as error:
            rows.append({**row, "error": str(error)})
            continue
        diagnostics = model.diagnostics
        row.update(model.bgnbd.params)
        row.update(mean_dropout_probability=diagnostics["mean_dropout_probability"],
                   dropout_concentration=diagnostics["dropout_concentration"],
                   converged=diagnostics["checks"]["bgnbd_converged"],
                   rejected="; ".join(diagnostics["rejections"]) or None)
        if len(validation) > 1:
            predicted = model.predict(orders, t0, validation["customer_id"])["predicted_clv_12m"].to_numpy(dtype=float)
            metrics = regression_metrics(validation[LABEL_COLUMN], predicted)
            row.update(validation_mae=metrics["mae"], validation_rmse=metrics["rmse"],
                       validation_mean_predicted=metrics["mean_predicted"],
                       spearman_vs_configured=float(pd.Series(predicted).corr(pd.Series(reference), method="spearman")))
        rows.append(row)
    return pd.DataFrame(rows)


def _split_counts(frame: pd.DataFrame) -> dict:
    """분할별 고객 수(없는 분할은 0)."""
    return frame["split"].value_counts().reindex(SPLITS).fillna(0).astype(int).to_dict()


def _bundle_guard(output: Path, overwrite: bool) -> None:
    """이미 번들(``model_card.json`` 또는 ``registry.json``)이 있는 디렉터리에는 ``overwrite`` 없이 쓰지 않는다."""
    existing = [name for name in (CARD_NAME, "registry.json") if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{output} already holds a CLV bundle ({', '.join(existing)}); train into a new "
                              "directory, or pass overwrite=True (CLI --overwrite) to replace it")


def _population_frame(dataset: pd.DataFrame, config: CLVConfig) -> tuple[pd.DataFrame, pd.Series]:
    """정책 대상(``policy_eligible``), 이력 완전, 학습 대상(``training_population``) 열을 붙인다."""
    eligible = policy_mask(dataset, config.population_policy)
    frame = dataset.assign(policy_eligible=eligible.to_numpy(),
                           history_complete=dataset["feature_window_complete"].astype(bool).to_numpy())
    frame["training_population"] = frame["policy_eligible"] & (
        frame["history_complete"] if config.require_complete_history else True)
    return frame, eligible


def _gross_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """4주 관측이 완전한 고객(4주 모델의 예측·진단 대상)과 예측 변수. 금액 정답은 구매 고객만 있다."""
    keep = [*PREDICTOR_FEATURES, "customer_id", "split", "policy_eligible", "history_complete", "training_population",
            GROSS_PURCHASES_COLUMN, GROSS_LABEL_COLUMN,
            *([ASSIGNMENT_COLUMN] if ASSIGNMENT_COLUMN in frame else []),
            *(["treatment_in_gross_window"] if "treatment_in_gross_window" in frame else [])]
    rows = frame.loc[frame[GROSS_COMPLETE_COLUMN].astype(bool)]
    result = rows[keep].reset_index(drop=True)
    result[GROSS_LABEL_COLUMN] = result[GROSS_LABEL_COLUMN].astype(float)
    return result


def _assignment_diagnostic(scored: pd.DataFrame, t0, experiment) -> dict:
    """배정군 진단. ``t0``가 실험 시각(``experiment`` = 단일 ``as_of_at``)과 같고 모든 Treatment가 4주 구간 안에
    배정됐을 때만 차이 비교(``treatment_minus_control``)를 남긴다. t0가 실험 시각보다 앞서면 배정 시각이 구간 안에
    있어도 4주 구간이 쿠폰 발송 구간과 어긋나므로, 둘 중 하나라도 어긋나면 비교를 빼고 그 이유를 남긴다."""
    report = gross_by_assignment(scored)
    if not report.get("available"):
        return report
    share = None
    if "treatment_in_gross_window" in scored:
        treated = scored[ASSIGNMENT_COLUMN].notna() & scored[ASSIGNMENT_COLUMN].ne(CONTROL_CODE).fillna(False)
        share = float(scored.loc[treated, "treatment_in_gross_window"].astype(bool).mean()) if treated.any() else None
    aligned = experiment is not None and t0 == experiment
    report["treatment_assigned_in_window_share"] = share
    report["t0_equals_as_of_at"] = aligned
    reasons = []
    if not aligned:
        reasons.append(f"t0 {t0.isoformat()} is not the experiment as_of_at "
                       f"({experiment.isoformat() if experiment is not None else 'no single as_of_at'})")
    if share is not None and share < 1:
        reasons.append(f"only {share:.0%} of treatment customers were assigned inside [t0, t0 + {GROSS_WEEKS} weeks)")
    if reasons:
        report.pop("treatment_minus_control", None)
        report["comparison_suppressed"] = "; ".join(reasons) + "; use t0 = the experiment as_of_at"
    return report


def _gross_report(scored: pd.DataFrame, splits, t0, experiment) -> dict:
    """4주 모델 평가(학습 대상 구매 고객, 분할별)와 배정군 진단을 만든다."""
    primary = scored["training_population"].astype(bool)
    report = {name: evaluate_gross_4w(scored.loc[primary & scored["split"].eq(name)]) for name in splits}
    report["assignment_diagnostic"] = _assignment_diagnostic(scored, t0, experiment)
    return report


def _experiment_cutoff(panel: pd.DataFrame):
    """패널의 실험 시각 ``as_of_at``이 하나이고 서울 월요일 00:00이면 그 t0, 아니면 None."""
    if "as_of_at" not in panel:
        return None
    values = panel["as_of_at"].dropna().unique()
    try:
        return to_cutoff(values[0]) if len(values) == 1 else None
    except ValueError:
        return None


def train(weekly, orders=None, output_dir: str | Path = "results/clv", config: CLVConfig | None = None,
          cutoff=None, strict: bool = True, overwrite: bool = False, require_gross_4w: bool = False,
          **input_options) -> dict:
    """한 기준시점에서 52주 CLV 모델 4종과 4주 조건부 총구매액 모델을 학습하고 평가한다.

    학습 모집단 정책(``config.population_policy``, 기본 ``control_no_coupon``)을 고객 분할 전에 적용한다.
    정책 대상 고객만 Train/Validation/Test로 나누고 나머지(Treatment 등)는 ``unused``로 두므로, 52주 모델과
    4주 모델이 같은 모집단·같은 고객 단위 분할을 쓴다. 4주 모델(``xgboost_gross_4w``)은 그중 4주 관측이
    완전하고 4주 안에 구매한 고객만으로 학습하며, 52주 모델의 최종 모델 선택에는 들어가지 않는다.
    이미 번들이 있는 ``output_dir``에는 ``overwrite=True`` 없이 쓰지 않는다.

    ``control_no_coupon``에서 ``cutoff``를 주지 않으면 실험 시각(``as_of_at``)이 적격일 때 그 시각을 t0로 쓴다.
    4주 구간이 쿠폰 발송 구간과 같아야 Uplift와 결합할 수 있기 때문이다. 4주 모델은 쿠폰 미발송 조건으로
    정의되므로 ``observed_experiment_mix``에서는 학습하지 않는다(``skipped``). ``require_gross_4w=True``면 4주 모델을
    학습하지 못했을 때 아무것도 저장하지 않고 오류를 낸다(최종 학습용).

    실행할 수 없는 모델(원장 없음, xgboost 미설치, 구매 고객 부족)은 중단하지 않고
    사유와 함께 skipped/failed로 기록하며, 그 모델의 지표는 만들지 않는다. BG/NBD + GG가 필수
    적용 조건(수렴, ``q > 1``, 수치 안정성)을 어기면 rejected로 기록하고 저장·평가·최종 선택에서 뺀다.

    분할은 정답이 있는 모든 고객에 대해 만든다. 학습·모델 선택·기본 평가는 학습 대상
    (``require_complete_history``이면 13주 이력 완전 고객)만 쓰고, 이력 완전/미완전 구간별 평가를
    ``evaluation["by_history"]``에 따로 남긴다.

    Returns:
        저장한 registry(기준시점, 모델 상태, 최종 모델, 지표).
    """
    config = config or CLVConfig()
    output = Path(output_dir)
    _bundle_guard(output, overwrite)
    panel, orders, reports = prepare_inputs(weekly, orders, strict=strict, **input_options)
    cutoffs = find_eligible_cutoffs(panel)
    experiment = _experiment_cutoff(panel)
    usable = cutoffs.loc[cutoffs["eligible_customers"] > 0, "reference_week"]
    if cutoff is None and config.population_policy == "control_no_coupon" and experiment is not None \
            and usable.eq(experiment).any():
        cutoff = experiment
    t0 = select_cutoff(cutoffs, cutoff)
    dataset = build_dataset(panel, t0, orders)
    # 학습 모집단 정책을 분할 전에 적용한다(모든 모델 공통). 정책 대상 중 52주 또는 4주 정답 구간이 완전한 고객을
    # 고객 단위로 분할하고(4주 구간은 52주 구간 안에 있다), 정책 밖 고객은 분할하지 않고 unused로 둔다. 분할은
    # require_complete_history와 무관하므로 두 설정의 결과를 같은 Test 고객으로 비교할 수 있다.
    frame, eligible = _population_frame(dataset, config)
    population = population_counts(dataset, eligible, config.population_policy)
    candidates = frame.loc[frame["policy_eligible"] & (frame["label_window_complete"] | frame[GROSS_COMPLETE_COLUMN])]
    splits = split_customers(candidates["customer_id"], config.seed, config.split_fractions)
    frame["split"] = frame["customer_id"].map(splits).fillna("unused")
    everyone = labeled_rows(frame, require_complete_history=False)
    everyone[LABEL_COLUMN] = everyone[LABEL_COLUMN].astype(float)
    # 학습·모델 선택·기본 평가 대상(정책 대상이면서 분할된 고객)
    labelled = everyone.loc[everyone["training_population"] & everyone["split"].isin(SPLITS)].reset_index(drop=True)
    parts = {name: labelled.loc[labelled["split"].eq(name)] for name in SPLITS}
    keep = ["customer_id", "split", "policy_eligible", "history_complete", "training_population", LABEL_COLUMN,
            *(["treatment"] if "treatment" in everyone else [])]
    # 모든 모델의 예측을 같은 행(scored)에 모아 같은 고객·같은 정답으로 평가한다.
    scored = everyone[keep].copy()
    status, models, details, model_warnings = {}, {}, {}, []

    baseline = BaselineModel()
    scored["pred_baseline"] = baseline.predict(everyone)
    status["baseline"], models["baseline"] = "trained", baseline

    sensitivity = None
    if orders is None:
        status["bgnbd_gg"] = "skipped: no order ledger supplied"
    else:
        try:
            probabilistic = ProbabilisticCLV(config.calibration, config.transaction_period, config.no_purchase_fallback,
                                             config.bgnbd_penalizer, config.gg_penalizer)
            # Train 고객으로만 적합하고, 예측은 정답이 있는 전체 고객에 대해 만든다.
            train_labels = parts["train"].set_index("customer_id")[LABEL_COLUMN]
            probabilistic.fit(orders, t0, parts["train"]["customer_id"], labels=train_labels)
            output_frame = probabilistic.predict(orders, t0, everyone["customer_id"])
            counts = output_frame["status"].value_counts()
            rejections = list(probabilistic.diagnostics["rejections"])
            buyers = int(counts.get("modeled", 0) + counts.get("numerical_failure", 0))
            failure_share = float(counts.get("numerical_failure", 0)) / buyers if buyers else 0.0
            if failure_share > MAX_NUMERICAL_FAILURE_SHARE:
                rejections.append(f"numerical failures for {failure_share:.1%} of customers with a purchase")
            details["bgnbd_gg"] = {**probabilistic.diagnostics, "rejections": rejections,
                                   "status_counts": counts.to_dict(), "numerical_failure_share": failure_share}
            model_warnings += [f"bgnbd_gg: {warning}" for warning in probabilistic.diagnostics["warnings"]]
            if rejections:
                # 필수 적용 조건 위반: 진단은 남기고 저장·평가·최종 선택에서는 뺀다.
                status["bgnbd_gg"] = "rejected: " + "; ".join(rejections)
            else:
                scored["pred_bgnbd_gg"] = output_frame["predicted_clv_12m"].to_numpy(dtype=float)
                scored["bgnbd_status"] = output_frame["status"].to_numpy()
                status["bgnbd_gg"], models["bgnbd_gg"] = "trained", probabilistic
            if config.bgnbd_sensitivity_penalizers:
                validation_ids = parts["validation"]["customer_id"]
                reference = probabilistic.predict(orders, t0, validation_ids)["predicted_clv_12m"].to_numpy(dtype=float)
                sensitivity = _bgnbd_sensitivity(config, orders, t0, parts, reference)
                details["bgnbd_gg"]["penalizer_sensitivity"] = sensitivity.to_dict(orient="records")
        except (ValueError, RuntimeError) as error:
            # 적합 실패(인원 부족 등)는 전체 실행을 멈추지 않고 failed로 기록한다.
            status["bgnbd_gg"] = f"failed: {error}"

    available, version = xgboost_status()
    if not available:
        status["xgboost"] = f"skipped: {version}"
    else:
        booster = XGBoostCLV(config.seed, config.xgb_grid, config.xgb_fixed, config.selection_metric)
        booster.fit(parts["train"], parts["train"][LABEL_COLUMN], parts["validation"], parts["validation"][LABEL_COLUMN])
        scored["pred_xgboost"] = booster.predict(everyone)
        status["xgboost"], models["xgboost"] = "trained", booster
        details["xgboost"] = {"version": version, "best_params": booster.best_params, "grid_results": booster.grid_results,
                              "clipped_negative_predictions": booster.last_clipped}

    # 두 파트 XGBoost: 구매 확률 × 구매 시 기대 결제액. 같은 학습/검증 고객과 예측 변수를 쓴다.
    if not available:
        status["xgboost_two_part"] = f"skipped: {version}"
    else:
        try:
            two_part = TwoPartXGBoostCLV(config.seed, config.two_part_classifier_grid, config.two_part_regressor_grid,
                                         config.xgb_fixed)
            two_part.fit(parts["train"], parts["train"][LABEL_COLUMN], parts["validation"],
                         parts["validation"][LABEL_COLUMN])
            components = two_part.predict_parts(everyone)
            scored["pred_xgboost_two_part"] = components["predicted_clv_12m"].to_numpy(dtype=float)
            scored["two_part_purchase_probability"] = components["purchase_probability_52w"].to_numpy(dtype=float)
            scored["two_part_amount_if_purchase"] = components["expected_amount_if_purchase"].to_numpy(dtype=float)
            status["xgboost_two_part"], models["xgboost_two_part"] = "trained", two_part
            details["xgboost_two_part"] = {"version": version, **two_part.summary()}
        except ValueError as error:
            # 구매 고객이 너무 적으면 실행을 멈추지 않고 failed로 기록한다.
            status["xgboost_two_part"] = f"failed: {error}"

    # 4주 조건부 총구매액 모델: 같은 t0·같은 분할에서, 학습 대상(정책 대상) 중 4주 안에 구매한 고객만 학습한다.
    # 52주 모델의 정답이나 금액 파트를 쓰지 않는다. 예측과 진단은 4주 관측이 완전한 모든 고객에 대해 만든다.
    gross_scored = _gross_frame(frame)
    gross_buyers = gross_scored.loc[gross_scored["training_population"] & gross_scored[GROSS_LABEL_COLUMN].notna()
                                    & gross_scored["split"].isin(SPLITS)]
    gross_parts = {name: gross_buyers.loc[gross_buyers["split"].eq(name)] for name in SPLITS}
    gross_model, gross_details = None, {}
    gross_scored["out_of_sample"] = ~gross_scored["split"].isin(FITTED_SPLITS)
    if config.population_policy != "control_no_coupon":
        gross_state = "skipped: the 4-week model is defined for population_policy control_no_coupon"
    elif not available:
        gross_state = f"skipped: {version}"
    else:
        try:
            gross_model = GrossXGBoost4w(config.seed, config.gross_4w_grid, config.xgb_fixed).fit(
                gross_parts["train"], gross_parts["train"][GROSS_LABEL_COLUMN],
                gross_parts["validation"], gross_parts["validation"][GROSS_LABEL_COLUMN])
            gross_scored[GROSS_PREDICTION_COLUMN] = gross_model.predict(gross_scored)
            gross_state, gross_details = "trained", {"version": version, **gross_model.summary(),
                                                      "clipped_negative_predictions": gross_model.last_clipped}
        except InsufficientBuyersError as error:
            # 구매 고객이 너무 적을 때만 실행을 멈추지 않고 failed로 기록한다(52주 모델은 그대로 저장한다).
            gross_state = f"failed: {error}"
    if require_gross_4w and gross_model is None:
        raise ValueError(f"require_gross_4w: {GROSS_MODEL} was not trained ({gross_state}); nothing was saved")
    gross_evaluation = (_gross_report(gross_scored, ("validation", "test"), t0, experiment) if gross_model is not None
                        else {"status": gross_state,
                              "assignment_diagnostic": _assignment_diagnostic(gross_scored, t0, experiment)})
    if experiment is None and config.population_policy == "control_no_coupon" and "as_of_at" in panel:
        model_warnings.append("as_of_at is not a single Asia/Seoul Monday 00:00; t0 was not aligned to the experiment "
                              "and the 4-week window may not match the coupon window")
    if experiment is not None and t0 != experiment:
        model_warnings.append(f"t0 {t0.isoformat()} differs from the experiment as_of_at {experiment.isoformat()}; "
                              "the 4-week window does not match the coupon window")

    trained = [name for name in MODEL_NAMES if status.get(name) == "trained"]
    primary = scored["training_population"].astype(bool)
    evaluation = {name: evaluate_predictions(scored.loc[scored["split"].eq(name) & primary], trained, config.top_fraction)
                  for name in ("validation", "test")}
    # 두 파트 XGBoost 1파트(구매 여부 분류)의 성능도 같은 고객으로 남긴다.
    if "two_part_purchase_probability" in scored:
        for name in ("validation", "test"):
            part = scored.loc[scored["split"].eq(name) & primary]
            evaluation[name]["two_part_classifier"] = classification_metrics(
                part[LABEL_COLUMN].gt(0), part["two_part_purchase_probability"])
    # 같은 분할 안에서 13주 이력 완전/미완전 고객을 따로 평가한다(미완전 고객은 학습 대상이 아닐 수 있다).
    evaluation["by_history"] = {name: evaluate_by_history(scored.loc[scored["split"].eq(name)], trained,
                                                          config.top_fraction) for name in ("validation", "test")}
    # 최종 모델: Validation 지표(기본 RMSE)가 가장 낮은 모델, 동률이면 MODEL_NAMES 순서. Test는 보고에만 쓴다.
    if config.final_model == "auto":
        metric = config.selection_metric
        final = min(trained, key=lambda name: (evaluation["validation"]["models"][name][metric], MODEL_NAMES.index(name)))
    elif config.final_model in trained:
        final = config.final_model
    else:
        raise ValueError(f"final_model {config.final_model} was not trained: {status.get(config.final_model)}")

    # 모델, 보고서, 분할, registry를 단계별 파일로 저장한다.
    model_dir = output / "models"
    for name, model in models.items():
        model.save(model_dir / name if name in SUBDIRECTORY_MODELS else model_dir)
    if gross_model is not None:
        gross_model.save(model_dir / GROSS_MODEL)
    reports_dir = output / "reports"
    write_json(reports_dir / "validation.json", {report.name: report.to_dict() for report in reports})
    cutoffs.assign(reference_week=cutoffs["reference_week"].map(lambda value: value.isoformat())).to_csv(
        reports_dir / "eligible_cutoffs.csv", index=False)
    write_json(reports_dir / "evaluation.json", {**evaluation, "gross_4w": gross_evaluation})
    if sensitivity is not None:
        sensitivity.to_csv(reports_dir / "bgnbd_sensitivity.csv", index=False)
    test = scored.loc[scored["split"].eq("test") & primary]
    decile_table(test, trained).to_csv(reports_dir / "deciles_test.csv", index=False)
    pd.concat([top_n_table(test, name, 20) for name in trained]).to_csv(reports_dir / "top20_test.csv", index=False)
    scored.to_csv(reports_dir / "labelled_predictions.csv", index=False)
    gross_scored.drop(columns=list(PREDICTOR_FEATURES)).to_csv(reports_dir / "gross_4w_labelled_predictions.csv",
                                                                index=False)
    # 분할 표에는 t0 이력이 있는 모든 고객을 남긴다(정책 밖 고객과 정답이 없는 고객은 unused).
    frame[["customer_id", "split", *([ASSIGNMENT_COLUMN] if ASSIGNMENT_COLUMN in frame else []), "policy_eligible",
           "history_complete", "training_population", "label_window_complete", GROSS_COMPLETE_COLUMN]].to_csv(
        output / "splits.csv", index=False)
    summary = _dataset_summary(dataset, labelled, t0)
    by_history_test = {segment: {name: {key: values["models"][name][key] for key in ("n", "mae", "rmse")}
                                 for name in trained}
                       for segment, values in evaluation["by_history"]["test"].items() if "models" in values}
    registry = {
        "clv_version": __version__, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_week": t0.isoformat(), "horizon_weeks": LABEL_WEEKS, "input_weeks": INPUT_WEEKS,
        "contract": CONTRACT, "config": config.to_dict(), "predictors": list(PREDICTOR_FEATURES),
        "provenance": {name: panel.attrs.get(name) for name in ("dataset_id", "currency", "amount_unit", "path",
                                                                 "source_format", "coverage_source", "weekly_sha256",
                                                                 "raw_database_sha256", "seed")},
        "ledger": None if orders is None else {name: orders.attrs.get(name) for name in (
            "path", "raw_database_sha256", "observation_start_at", "observation_end_at", "source_metadata")},
        "dataset": summary, "split_counts": _split_counts(labelled), "split_counts_all_labelled": _split_counts(everyone),
        "population_policy": config.population_policy, "population": population,
        "training_population": (f"{config.population_policy} customers" + (
            " with a complete 13-week history" if config.require_complete_history else "")),
        "model_status": status, "model_details": details, "trained_models": trained, "final_model": final,
        "selection": f"lowest validation {config.selection_metric}" if config.final_model == "auto" else "configured",
        "test_metrics": {name: {key: evaluation["test"]["models"][name][key] for key in ("mae", "rmse")}
                         | {"top20_overlap_rate": evaluation["test"]["models"][name]["top"]["overlap_rate"]}
                         for name in trained},
        "test_metrics_by_history": by_history_test,
        "gross_4w": {
            "model": GROSS_MODEL, "status": gross_state, "target_definition_id": GROSS_TARGET_DEFINITION_ID,
            "training_population": f"{config.population_policy} customers with a complete {GROSS_WEEKS}-week window "
                                   f"and at least one purchase in it",
            "split_counts": _split_counts(gross_buyers), "details": gross_details,
            "test_metrics": ({key: gross_evaluation["test"].get(key) for key in
                              ("n", "mae", "rmse", "mean_actual", "mean_predicted")} if gross_model is not None
                             else None),
        },
        "warnings": [warning for report in reports for warning in report.warnings] + model_warnings,
    }
    write_json(output / "registry.json", registry)
    # 모델 카드는 registry·분할·모델 파일의 체크섬을 담으므로 모든 파일을 쓴 뒤 만든다.
    write_model_card(output, registry)
    return registry


def _labelled_frame(panel, orders, registry, cutoff, population, model_dir):
    """저장된 설정으로 ``cutoff``의 고객 전체를 만들고 정책 대상·이력 완전·학습 대상 표시를 붙인다.

    평가 대상(``population="test"``면 저장된 Test 고객, ``"all"``이면 전원)은 ``split`` 열이 ``population`` 값이고
    나머지는 ``outside``다. 배정군 진단이 Treatment를 볼 수 있도록 대상 밖 고객도 남긴다. 학습 대상 표시는 번들의
    학습 모집단 정책을 다시 적용한 것이다(기존 번들은 ``observed_experiment_mix``). 52주 정답이 없는 고객도 남기므로
    52주 평가는 ``label_window_complete``, 4주 평가는 ``gross_4w_window_complete`` 고객을 쓴다.
    """
    config = CLVConfig.from_registry(registry["config"])
    t0 = to_cutoff(cutoff or registry["reference_week"])
    frame, _ = _population_frame(build_dataset(panel, t0, orders), config)
    if population == "test":
        splits = pd.read_csv(Path(model_dir) / "splits.csv", dtype={"customer_id": str})
        selected = frame["customer_id"].isin(splits.loc[splits["split"].eq("test"), "customer_id"])
    elif population == "all":
        selected = pd.Series(True, index=frame.index)
    else:
        raise ValueError("population must be test or all")
    frame = frame.assign(split=np.where(selected, population, "outside"))
    frame = frame.reset_index(drop=True)
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(float)
    return config, t0, frame


def _check_units(panel, registry):
    """입력의 통화·금액 단위가 학습 때와 같은지 확인한다(다르면 ValueError)."""
    for name in ("currency", "amount_unit"):
        if panel.attrs.get(name) != registry["provenance"].get(name):
            raise ValueError(f"{name} {panel.attrs.get(name)!r} differs from the trained model "
                             f"{registry['provenance'].get(name)!r}")


def evaluate(weekly, orders=None, model_dir: str | Path = "results/clv", cutoff=None, population: str = "test",
             output: str | Path | None = None, strict: bool = True, **input_options) -> dict:
    """저장된 모델을 ``cutoff``(기본: 학습 기준시점)의 정답 고객으로 평가한다.

    ``population="test"``는 저장된 Test 고객만 남기므로, 다른 기준시점에서도
    학습 고객을 다시 평가에 쓰지 않는다.
    """
    model_dir = Path(model_dir)
    registry = read_json(model_dir / "registry.json")
    panel, orders, _ = prepare_inputs(weekly, orders, strict=strict, **input_options)
    _check_units(panel, registry)
    config, t0, customers = _labelled_frame(panel, orders, registry, cutoff, population, model_dir)
    # 52주 평가 대상: 평가 대상 중 52주 정답이 있고 번들의 학습 모집단 정책 대상인 고객(control_no_coupon이면 Control만)
    labelled = customers.loc[customers["split"].eq(population) & customers["label_window_complete"].astype(bool)
                             & customers["policy_eligible"].astype(bool)].reset_index(drop=True)
    if labelled.empty:
        raise ValueError("no labelled customers to evaluate at this cutoff")
    # 원장이 없으면 BG/NBD + GG는 평가에서 뺀다.
    names = [name for name in registry["trained_models"] if name != "bgnbd_gg" or orders is not None]
    frame = labelled[["customer_id", LABEL_COLUMN, "history_complete", "training_population",
                      *(["treatment"] if "treatment" in labelled else [])]].copy()
    for name, model in _models_for(model_dir, names).items():
        frame[f"pred_{name}"] = _score(model, labelled, orders, t0)
    # 기본 지표는 학습 대상 고객으로 계산하고(학습 때 Test 지표와 같다), 이력 구간별 지표를 덧붙인다.
    report = {"reference_week": t0.isoformat(), "population": population, "models_evaluated": names,
              "population_policy": config.population_policy,
              **evaluate_predictions(frame.loc[frame["training_population"].astype(bool)], names, config.top_fraction),
              "by_history": evaluate_by_history(frame, names, config.top_fraction)}
    # 4주 모델: 번들에 있으면 같은 t0의 4주 관측 완전 고객 전원에게 적용한다. MAE·RMSE는 평가 대상의 학습 대상
    # 구매 고객(Control)으로, 배정군 진단은 평가 대상과 무관하게 이 t0의 전원(Treatment 포함)으로 만든다.
    # 번들에 없으면 상태만 기록한다(대체값을 만들지 않는다).
    if gross_status(registry) == "trained":
        gross = _gross_frame(customers)
        # 학습 데이터셋이면 저장된 train·validation 고객을 진단의 예측 요약에서 뺀다(다른 데이터셋이면 전원 미사용).
        fitted = pd.Index([])
        if registry["provenance"].get("dataset_id") in (None, panel.attrs.get("dataset_id")):
            splits = pd.read_csv(model_dir / "splits.csv", dtype={"customer_id": str})
            fitted = pd.Index(splits.loc[splits["split"].isin(FITTED_SPLITS), "customer_id"])
        gross["out_of_sample"] = ~gross["customer_id"].isin(fitted)
        gross[GROSS_PREDICTION_COLUMN] = _models_for(model_dir, [GROSS_MODEL])[GROSS_MODEL].predict(
            customers.loc[customers[GROSS_COMPLETE_COLUMN].astype(bool)].reset_index(drop=True))
        report["gross_4w"] = {"status": "trained",
                              **_gross_report(gross, (population,), t0, _experiment_cutoff(panel))}
        # population="all"을 학습 데이터셋에 쓰면 적합·선택에 쓴 구매 고객도 지표에 들어간다(인원을 남긴다).
        buyers = gross["split"].eq(population) & gross["training_population"] & gross[GROSS_LABEL_COLUMN].notna()
        report["gross_4w"]["in_sample_buyers"] = int((buyers & ~gross["out_of_sample"]).sum())
    else:
        report["gross_4w"] = {"status": gross_status(registry)}
    if output is not None:
        write_json(output, report)
    return report


def predict(weekly, orders=None, model_dir: str | Path = "results/clv", reference_week=None, model: str | None = None,
            strict: bool = True, customers=None, **input_options) -> pd.DataFrame:
    """``reference_week`` 이전 이력이 있는 고객의 52주 CLV를 예측한다(``CLVModel.predict``와 같다).

    정답은 필요 없다. 13주 이력이 일부만 있는 고객도 예측하되 ``history_complete=False``로 표시한다.
    """
    from .inference import CLVModel  # inference가 pipeline을 import하므로 여기서 불러온다.

    return CLVModel(model_dir, model).predict(weekly, orders, reference_week=reference_week, customers=customers,
                                              strict=strict, **input_options)


def write_predictions(predictions: pd.DataFrame, path: str | Path = "results/clv_predictions.csv") -> Path:
    """전략 모듈용 CSV를 쓴다. 계약 열 4개(``OUTPUT_COLUMNS``)를 맨 앞에 두고 보조 열을 뒤에 붙인다."""
    missing = [name for name in OUTPUT_COLUMNS if name not in predictions]
    if missing:
        raise ValueError(f"predictions are missing contract columns: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [*OUTPUT_COLUMNS, *(name for name in predictions if name not in OUTPUT_COLUMNS)]
    predictions[ordered].to_csv(path, index=False, float_format="%.6f")
    return path


def run(weekly, orders=None, output_dir: str | Path = "results/clv",
        predictions_path: str | Path = "results/clv_predictions.csv", config: CLVConfig | None = None,
        cutoff=None, reference_week=None, strict: bool = True, overwrite: bool = False,
        require_gross_4w: bool = False, **input_options) -> dict:
    """학습 + 평가 후 최종 예측 CSV까지 한 번에 쓴다."""
    # 입력은 한 번만 변환·검증하고, 변환된 frame을 학습·예측에 그대로 넘긴다.
    panel, orders, _ = prepare_inputs(weekly, orders, strict=strict, **input_options)
    registry = train(panel, orders, output_dir, config, cutoff, strict=strict, overwrite=overwrite,
                     require_gross_4w=require_gross_4w)
    predictions = predict(panel, orders, output_dir, reference_week, strict=strict)
    path = write_predictions(predictions, predictions_path)
    return {"registry": registry, "predictions_path": str(path), "predicted_customers": int(len(predictions)),
            "prediction_reference_week": predictions["reference_week"].iloc[0]}
