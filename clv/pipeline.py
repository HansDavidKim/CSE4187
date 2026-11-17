"""WBS 4.7: 학습·평가·추론을 따로 실행하고 산출물을 저장하며, 예측 CSV를 쓴다.

``output_dir``(기본 ``results/clv``) 아래 산출물::

    registry.json            기준시점, 계약, 설정, 모델 상태, 최종 모델
    splits.csv               customer_id -> train/validation/test, 이력 완전 여부, 학습 대상 여부
    models/                  baseline.json, bgnbd_gg.json, xgboost/{xgboost_model,xgboost_meta}.json,
                             xgboost_two_part/{classifier,regressor,two_part_meta}.json
    reports/                 입력 검증, 적격 기준시점, 평가(이력 구간별 포함), 10분위, Top-N,
                             BG/NBD 벌점 민감도, 정답 포함 예측
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .adapter import adapt_orders, adapt_weekly, load_orders, load_weekly
from .baseline import BaselineModel
from .config import CLVConfig
from .dataset import SPLITS, build_dataset, complete_grid, find_eligible_cutoffs, labeled_rows, select_cutoff, split_customers
from .evaluation import (
    decile_table, evaluate_by_history, evaluate_predictions, ranks, regression_metrics, top_flags, top_n_table,
)
from .features import WEEK, build_features, fully_observed, to_cutoff
from .io import read_json, write_json
from .probabilistic import MAX_NUMERICAL_FAILURE_SHARE, ProbabilisticCLV
from .schema import (
    HIGH_VALUE_COLUMN, INPUT_WEEKS, LABEL_COLUMN, LABEL_WEEKS, OUTPUT_COLUMNS, PREDICTION_COLUMN, PREDICTOR_FEATURES,
)
from .validation import reconcile_orders_with_weekly, validate_orders, validate_weekly
from .xgboost_model import TwoPartXGBoostCLV, XGBoostCLV, classification_metrics, xgboost_status

MODEL_NAMES = ("baseline", "bgnbd_gg", "xgboost", "xgboost_two_part")
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
        panel = load_weekly(weekly, **units)
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
    """``model_dir/models``에 저장된 모델을 이름별로 불러온다."""
    loaded = {}
    for name in names:
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


def train(weekly, orders=None, output_dir: str | Path = "results/clv", config: CLVConfig | None = None,
          cutoff=None, strict: bool = True, **input_options) -> dict:
    """한 기준시점에서 Baseline, BG/NBD + Gamma-Gamma, XGBoost를 학습하고 평가한다.

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
    panel, orders, reports = prepare_inputs(weekly, orders, strict=strict, **input_options)
    cutoffs = find_eligible_cutoffs(panel)
    t0 = select_cutoff(cutoffs, cutoff)
    dataset = build_dataset(panel, t0, orders)
    # 정답이 있는 모든 고객을 먼저 고객 단위로 분할한다(한 고객은 한 분할에만 속한다). 분할이
    # require_complete_history 설정과 무관하므로 두 설정의 결과를 같은 Test 고객으로 비교할 수 있다.
    everyone = labeled_rows(dataset, require_complete_history=False)
    splits = split_customers(everyone["customer_id"], config.seed, config.split_fractions)
    everyone["split"] = everyone["customer_id"].map(splits)
    everyone[LABEL_COLUMN] = everyone[LABEL_COLUMN].astype(float)
    everyone["history_complete"] = everyone["feature_window_complete"].astype(bool)
    # 학습·모델 선택·기본 평가 대상
    everyone["training_population"] = everyone["history_complete"] if config.require_complete_history else True
    labelled = everyone.loc[everyone["training_population"]].reset_index(drop=True)
    parts = {name: labelled.loc[labelled["split"].eq(name)] for name in SPLITS}
    keep = ["customer_id", "split", "history_complete", "training_population", LABEL_COLUMN,
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
    reports_dir = output / "reports"
    write_json(reports_dir / "validation.json", {report.name: report.to_dict() for report in reports})
    cutoffs.assign(reference_week=cutoffs["reference_week"].map(lambda value: value.isoformat())).to_csv(
        reports_dir / "eligible_cutoffs.csv", index=False)
    write_json(reports_dir / "evaluation.json", evaluation)
    if sensitivity is not None:
        sensitivity.to_csv(reports_dir / "bgnbd_sensitivity.csv", index=False)
    test = scored.loc[scored["split"].eq("test") & primary]
    decile_table(test, trained).to_csv(reports_dir / "deciles_test.csv", index=False)
    pd.concat([top_n_table(test, name, 20) for name in trained]).to_csv(reports_dir / "top20_test.csv", index=False)
    scored.to_csv(reports_dir / "labelled_predictions.csv", index=False)
    everyone[["customer_id", "split", "history_complete", "training_population"]].to_csv(output / "splits.csv",
                                                                                         index=False)
    summary = _dataset_summary(dataset, labelled, t0)
    by_history_test = {segment: {name: {key: values["models"][name][key] for key in ("n", "mae", "rmse")}
                                 for name in trained}
                       for segment, values in evaluation["by_history"]["test"].items() if "models" in values}
    registry = {
        "clv_version": __version__, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_week": t0.isoformat(), "horizon_weeks": LABEL_WEEKS, "input_weeks": INPUT_WEEKS,
        "contract": CONTRACT, "config": config.to_dict(), "predictors": list(PREDICTOR_FEATURES),
        "provenance": {name: panel.attrs.get(name) for name in ("dataset_id", "currency", "amount_unit", "path",
                                                                 "source_format", "coverage_source")},
        "ledger": None if orders is None else {name: orders.attrs.get(name) for name in (
            "path", "raw_database_sha256", "observation_start_at", "observation_end_at")},
        "dataset": summary, "split_counts": _split_counts(labelled), "split_counts_all_labelled": _split_counts(everyone),
        "training_population": ("customers with a complete 13-week history" if config.require_complete_history
                                else "all labelled customers"),
        "model_status": status, "model_details": details, "trained_models": trained, "final_model": final,
        "selection": f"lowest validation {config.selection_metric}" if config.final_model == "auto" else "configured",
        "test_metrics": {name: {key: evaluation["test"]["models"][name][key] for key in ("mae", "rmse")}
                         | {"top20_overlap_rate": evaluation["test"]["models"][name]["top"]["overlap_rate"]}
                         for name in trained},
        "test_metrics_by_history": by_history_test,
        "warnings": [warning for report in reports for warning in report.warnings] + model_warnings,
    }
    write_json(output / "registry.json", registry)
    return registry


def _labelled_frame(panel, orders, registry, cutoff, population, model_dir):
    """저장된 설정으로 ``cutoff``의 정답 고객 전체를 만들고 이력 완전 여부와 학습 대상 표시를 붙인다.

    ``population="test"``면 저장된 Test 고객만 남긴다(이력 완전·미완전 모두).
    """
    config = CLVConfig.from_dict(registry["config"])
    t0 = to_cutoff(cutoff or registry["reference_week"])
    dataset = build_dataset(panel, t0, orders)
    labelled = labeled_rows(dataset, require_complete_history=False)
    if population == "test":
        splits = pd.read_csv(Path(model_dir) / "splits.csv", dtype={"customer_id": str})
        labelled = labelled.loc[labelled["customer_id"].isin(splits.loc[splits["split"].eq("test"), "customer_id"])]
    elif population != "all":
        raise ValueError("population must be test or all")
    labelled = labelled.reset_index(drop=True)
    labelled[LABEL_COLUMN] = labelled[LABEL_COLUMN].astype(float)
    labelled["history_complete"] = labelled["feature_window_complete"].astype(bool)
    labelled["training_population"] = labelled["history_complete"] if config.require_complete_history else True
    return config, t0, labelled


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
    config, t0, labelled = _labelled_frame(panel, orders, registry, cutoff, population, model_dir)
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
              **evaluate_predictions(frame.loc[frame["training_population"].astype(bool)], names, config.top_fraction),
              "by_history": evaluate_by_history(frame, names, config.top_fraction)}
    if output is not None:
        write_json(output, report)
    return report


def predict(weekly, orders=None, model_dir: str | Path = "results/clv", reference_week=None, model: str | None = None,
            strict: bool = True, **input_options) -> pd.DataFrame:
    """``reference_week`` 이전 이력이 있는 모든 고객의 52주 CLV를 예측한다.

    정답은 필요 없다. 13주 이력이 일부만 있는 고객도 예측하되 ``history_complete=False``로
    표시한다(``require_complete_history=True``로 학습한 모델이면 완전 이력 고객만 학습에 썼다).
    """
    model_dir = Path(model_dir)
    registry = read_json(model_dir / "registry.json")
    config = CLVConfig.from_dict(registry["config"])
    name = model or registry["final_model"]
    if name not in registry["trained_models"]:
        raise ValueError(f"model {name} was not trained: {registry['model_status'].get(name)}")
    panel, orders, _ = prepare_inputs(weekly, orders, strict=strict, **input_options)
    _check_units(panel, registry)
    t0 = to_cutoff(reference_week) if reference_week is not None else default_reference_week(panel)
    # 예측 변수만 만든다. t0 이후 52주 정답 구간이 데이터에 없어도 된다.
    features = build_features(panel, t0).reset_index()
    if features.empty:
        raise ValueError(f"no customer has observed history before {t0.isoformat()}")
    loaded = _models_for(model_dir, [name])[name]
    values = _score(loaded, features, orders, t0)
    # 상위 20% 표시(ceil)와 순위는 동점일 때 customer_id 오름차순으로 정한다.
    result = pd.DataFrame({
        "customer_id": features["customer_id"], "reference_week": t0.isoformat(), PREDICTION_COLUMN: values,
        HIGH_VALUE_COLUMN: top_flags(values, features["customer_id"], config.top_fraction),
        "clv_rank": ranks(values, features["customer_id"]), "model_name": name, "horizon_weeks": LABEL_WEEKS,
        "history_complete": features["feature_window_complete"].to_numpy(),
        "history_weeks_13w": features["history_weeks_13w"].to_numpy(),
        "currency": panel.attrs.get("currency"), "amount_unit": panel.attrs.get("amount_unit"),
        "dataset_id": panel.attrs.get("dataset_id"),
    })
    # 두 파트 모델이면 구매 확률과 구매 시 기대 결제액도 보조 열로 붙인다.
    if hasattr(loaded, "predict_parts"):
        components = loaded.predict_parts(features)
        for column in ("purchase_probability_52w", "expected_amount_if_purchase"):
            result[column] = components[column].to_numpy(dtype=float)
    return result.sort_values("clv_rank", kind="mergesort").reset_index(drop=True)


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
        cutoff=None, reference_week=None, strict: bool = True, **input_options) -> dict:
    """학습 + 평가 후 최종 예측 CSV까지 한 번에 쓴다."""
    # 입력은 한 번만 변환·검증하고, 변환된 frame을 학습·예측에 그대로 넘긴다.
    panel, orders, _ = prepare_inputs(weekly, orders, strict=strict, **input_options)
    registry = train(panel, orders, output_dir, config, cutoff, strict=strict)
    predictions = predict(panel, orders, output_dir, reference_week, strict=strict)
    path = write_predictions(predictions, predictions_path)
    return {"registry": registry, "predictions_path": str(path), "predicted_customers": int(len(predictions)),
            "prediction_reference_week": predictions["reference_week"].iloc[0]}
