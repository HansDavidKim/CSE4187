"""저장된 CLV 번들로 추론한다: 고객별 예측(``CLVModel.predict``)과 전략 모듈용 예측 batch(``predict_for_strategy``).

전략 모듈(``final_strategy``) 입력 계약에 맞춘 batch는 Feature Store와 같은 고객·기준 시각 키를 그대로 덮는다.
frame 열은 ``STRATEGY_COLUMNS`` 11개, 메타데이터는 ``PredictionMetadata``와 같은 이름의 5개 항목이다.
누락 예측을 0으로 채우거나 고객을 조용히 빼지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .bundle import (
    TARGET, TARGET_DEFINITION_ID, VALUE_SEMANTICS, code_version, environment, file_sha256, read_model_card,
)
from .config import CLVConfig
from .evaluation import ranks, top_flags
from .features import build_features, to_cutoff
from .io import read_json, write_json
from .pipeline import _check_units, _models_for, _score as score_model, default_reference_week, prepare_inputs
from .schema import HIGH_VALUE_COLUMN, LABEL_WEEKS, PREDICTION_COLUMN, SEOUL

# Feature Store에서 가져오는 고객·기준 시각 키와 출처 열(전략 batch에 그대로 복사한다)
STRATEGY_KEY_COLUMNS = ("dataset_id", "customer_id", "as_of_at", "reference_week", "snapshot_id", "currency",
                        "amount_unit", "raw_database_sha256")
# 전략 batch frame의 열(순서 포함). 이 밖의 열은 넣지 않는다.
STRATEGY_COLUMNS = (*STRATEGY_KEY_COLUMNS, "input_cutoff_at", "prediction_end_at", PREDICTION_COLUMN)
JOIN_KEYS = ("dataset_id", "customer_id", "as_of_at")
HORIZON = pd.Timedelta(weeks=LABEL_WEEKS)
UTC_TIME = "datetime64[ns, UTC]"
# 학습 고객 역할: train(적합), validation(모델 선택), test(보고만), unused(이 모델이 본 적 없음)
FITTED_ROLES = ("train", "validation")
ROLES = (*FITTED_ROLES, "test", "unused")


@dataclass(frozen=True)
class StrategyPredictions:
    """전략 모듈에 넘기는 CLV 예측 batch.

    Attributes:
        frame: ``STRATEGY_COLUMNS`` 순서의 DataFrame. 행 순서는 입력 Feature Store와 같다.
        metadata: ``model_version``, ``target_definition_id``, ``horizon_value``(52), ``horizon_unit``("weeks"),
            ``value_semantics``("nonnegative_amount"). ``PredictionMetadata(**metadata)``로 그대로 쓸 수 있다.
        audit: 같은 키에 이력 완전 여부, 관측 주 수, CLV 학습 고객 역할(``clv_training_role``)을 붙인 감사용 표.
        producer_metadata: 모델·학습 데이터·입력 출처·기준 시각별 고객 수·학습 고객 중복·경고.
    """

    frame: pd.DataFrame
    metadata: dict
    audit: pd.DataFrame
    producer_metadata: dict

    def save(self, path: str | Path) -> Path:
        """``<이름>.parquet``(frame), ``<이름>_metadata.json``, ``<이름>_audit.csv``를 쓰고 Parquet 경로를 돌려준다."""
        path = Path(path)
        if path.suffix != ".parquet":
            raise ValueError("strategy predictions are written as .parquet (keeps UTC timestamps and string IDs)")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(path, index=False)
        audit = self.audit.assign(as_of_at=self.audit["as_of_at"].map(lambda value: value.isoformat()))
        audit.to_csv(_sibling(path, "_audit.csv"), index=False)
        write_json(_sibling(path, "_metadata.json"), {
            "prediction_metadata": self.metadata, "producer_metadata": self.producer_metadata,
            "columns": list(self.frame.columns), "rows": int(len(self.frame)),
            "files_sha256": {path.name: file_sha256(path),
                             _sibling(path, "_audit.csv").name: file_sha256(_sibling(path, "_audit.csv"))},
        })
        return path


def _sibling(path: Path, suffix: str) -> Path:
    """``predictions.parquet`` -> ``predictions<suffix>``."""
    return path.with_name(path.stem + suffix)


def load_strategy_predictions(path: str | Path) -> StrategyPredictions:
    """``StrategyPredictions.save``로 쓴 파일 세 개를 체크섬을 대조해 다시 읽는다."""
    path = Path(path)
    meta = read_json(_sibling(path, "_metadata.json"))
    for name, digest in meta["files_sha256"].items():
        if file_sha256(path.parent / name) != digest:
            raise ValueError(f"{name} changed after it was written")
    frame = pd.read_parquet(path)
    if list(frame.columns) != meta["columns"] or len(frame) != meta["rows"]:
        raise ValueError("strategy prediction file does not match its metadata")
    audit = pd.read_csv(_sibling(path, "_audit.csv"),
                        dtype={"dataset_id": str, "customer_id": str, "clv_training_role": str})
    audit["as_of_at"] = pd.to_datetime(audit["as_of_at"], utc=True)
    return StrategyPredictions(frame, meta["prediction_metadata"], audit, meta["producer_metadata"])


def _seoul_week(value) -> str:
    """기준 시각이 속한 서울 주의 월요일 날짜(``YYYY-MM-DD``)."""
    local = pd.Timestamp(value).tz_convert(SEOUL)
    return (local.normalize() - pd.Timedelta(days=local.weekday())).date().isoformat()


def _week_label(value) -> str | None:
    """``reference_week`` 값(날짜 문자열 또는 서울 월요일 시각)을 ``YYYY-MM-DD``로 바꾼다. 해석할 수 없으면 None."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(SEOUL)
    return stamp.date().isoformat() if stamp == stamp.normalize() else None


def load_population(features, verify_checksum: bool = True) -> pd.DataFrame:
    """Feature Store(``feature_store.parquet``, 그 디렉터리, 또는 DataFrame)에서 예측할 고객·기준 시각 키를 읽는다.

    옆에 시뮬레이터 ``manifest.json``이 있으면 Parquet 체크섬을 대조한다(``verify_checksum``). 한 batch는 한 데이터셋
    (dataset_id·raw SQLite·통화·단위가 하나)이어야 하며, 키 중복·시간대 없는 시각·빈 ID와
    ``as_of_at``의 서울 주 월요일과 다른 ``reference_week``를 거절한다.
    """
    if isinstance(features, pd.DataFrame):
        frame = features
    else:
        path = Path(features)
        path = path / "feature_store.parquet" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = path.parent / "manifest.json"
        if verify_checksum and manifest_path.is_file():
            recorded = (read_json(manifest_path).get("file_sha256") or {}).get(path.name)
            if recorded is not None and recorded != file_sha256(path):
                raise ValueError(f"{path.name} checksum differs from {manifest_path}")
        frame = pd.read_parquet(path, columns=list(STRATEGY_KEY_COLUMNS))
    missing = [name for name in STRATEGY_KEY_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"feature store is missing key columns: {missing}")
    frame = frame.loc[:, list(STRATEGY_KEY_COLUMNS)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("feature store has no customers")
    for name in ("dataset_id", "customer_id", "snapshot_id", "currency", "amount_unit", "raw_database_sha256"):
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"feature store column {name} needs nonempty strings")
    times = pd.Series(frame["as_of_at"])
    if not isinstance(times.dtype, pd.DatetimeTZDtype):
        try:
            times = pd.Series([pd.Timestamp(value) for value in times], index=frame.index)
        except (TypeError, ValueError) as error:
            raise ValueError("feature store as_of_at must be timestamps") from error
        if times.map(lambda value: pd.isna(value) or value.tzinfo is None).any():
            raise ValueError("feature store as_of_at must be timezone-aware")
    # 해상도를 ns로 맞춘다(문자열에서 만든 시각은 초 단위일 수 있다).
    frame["as_of_at"] = pd.to_datetime(times, utc=True).astype(UTC_TIME)
    # 전략 모듈과 같은 규칙: reference_week = as_of_at이 속한 서울 주의 월요일
    weeks, expected = frame["reference_week"].map(_week_label), frame["as_of_at"].map(_seoul_week)
    if not weeks.eq(expected).all():
        bad = frame.loc[~weeks.eq(expected), ["customer_id", "reference_week"]].head(3).to_dict("records")
        raise ValueError(f"reference_week must be the Asia/Seoul Monday of as_of_at; e.g. {bad}")
    for name in ("dataset_id", "raw_database_sha256", "currency", "amount_unit"):
        if frame[name].nunique() != 1:
            raise ValueError(f"one batch must come from one dataset; found several {name} values")
    if frame.duplicated(list(JOIN_KEYS)).any():
        raise ValueError("feature store has duplicate (dataset_id, customer_id, as_of_at) keys")
    return frame


class CLVModel:
    """``train``이 저장한 CLV 번들(기본 ``results/clv``)의 한 모델을 불러와 추론한다.

    불러올 때 ``model_card.json``에 기록된 파일 체크섬을 대조하므로, 학습 이후 바뀐 번들은 쓰지 않는다.

    Args:
        model_dir: 번들 디렉터리(``registry.json``, ``model_card.json``, ``models/``, ``splits.csv``).
        model: 모델 이름(``baseline``, ``bgnbd_gg``, ``xgboost``, ``xgboost_two_part``). 기본은 registry 최종 모델.
        verify: False면 체크섬 대조를 건너뛴다(디버깅용).
    """

    def __init__(self, model_dir: str | Path = "results/clv", model: str | None = None, verify: bool = True):
        self.model_dir = Path(model_dir)
        self.card = read_model_card(self.model_dir, verify=verify)
        self.registry = read_json(self.model_dir / "registry.json")
        self.name = model or self.registry["final_model"]
        if self.name not in self.registry["trained_models"]:
            raise ValueError(f"model {self.name} was not trained: {self.registry['model_status'].get(self.name)}")
        self.config = CLVConfig.from_dict(self.registry["config"])
        self.model_version = self.card["model_versions"][self.name]
        self._model = _models_for(self.model_dir, [self.name])[self.name]

    @property
    def needs_ledger(self) -> bool:
        """BG/NBD + Gamma-Gamma만 예측에 주문 원장(``raw.sqlite``)이 필요하다."""
        return self.name == "bgnbd_gg"

    @property
    def training_reference_week(self) -> pd.Timestamp:
        """학습 기준시점 t0(서울 월요일 00:00)."""
        return pd.Timestamp(self.registry["reference_week"])

    @property
    def metadata(self) -> dict:
        """전략 모듈 ``PredictionMetadata``와 같은 이름의 예측 선언."""
        return {"model_version": self.model_version, "target_definition_id": TARGET_DEFINITION_ID,
                "horizon_value": LABEL_WEEKS, "horizon_unit": "weeks", "value_semantics": VALUE_SEMANTICS}

    def info(self) -> dict:
        """사람이 읽는 요약: 모델·버전·예측 대상·학습 데이터·입력 요구·Test 지표."""
        return {"model_name": self.name, "model_version": self.model_version, "bundle": str(self.model_dir),
                "final_model": self.registry["final_model"], "target": TARGET, "code": self.card["code"],
                "training_data": self.card["training_data"], "policy_condition": self.card["policy_condition"],
                "needs_ledger": self.needs_ledger, "predictors": self.card["predictors"],
                "test_metrics": self.registry["test_metrics"].get(self.name)}

    def _inputs(self, weekly, ledger, strict, input_options):
        """입력을 변환·검증하고 학습 때와 통화·단위가 같은지 확인한다."""
        panel, orders, _ = prepare_inputs(weekly, ledger, strict=strict, **input_options)
        _check_units(panel, self.registry)
        if self.needs_ledger and orders is None:
            raise ValueError("bgnbd_gg needs the order ledger (raw.sqlite)")
        return panel, orders

    def _scored(self, panel, orders, t0, customers=None) -> pd.DataFrame:
        """``t0`` 직전 13주 예측 변수로 예측한다. ``customers``를 주면 그 순서대로, 모두 있어야 한다."""
        features = build_features(panel, t0)
        if customers is not None:
            customers = pd.Index(pd.Series(list(customers), dtype=object).astype(str), name="customer_id")
            if customers.has_duplicates:
                raise ValueError("customers contains duplicate IDs")
            absent = customers.difference(features.index)
            if len(absent):
                raise ValueError(f"{len(absent)} customers have no observed week in the 13 weeks before "
                                 f"{t0.isoformat()} (e.g. {list(absent[:5])}); CLV does not invent their history")
            features = features.loc[customers]
        if features.empty:
            raise ValueError(f"no customer has observed history before {t0.isoformat()}")
        features = features.reset_index()
        scored = features[["customer_id", "feature_window_complete", "history_weeks_13w"]].rename(
            columns={"feature_window_complete": "history_complete"})
        values = score_model(self._model, features, orders, t0)
        if not (np.isfinite(values).all() and (values >= 0).all()):
            raise ValueError("CLV predictions must be finite and nonnegative")
        scored[PREDICTION_COLUMN] = values
        # 두 파트 모델이면 구매 확률과 구매 시 기대 결제액도 붙인다.
        if hasattr(self._model, "predict_parts"):
            parts = self._model.predict_parts(features)
            for column in ("purchase_probability_52w", "expected_amount_if_purchase"):
                scored[column] = parts[column].to_numpy(dtype=float)
        return scored

    def predict(self, weekly, ledger=None, *, reference_week=None, customers=None, strict: bool = True,
                **input_options) -> pd.DataFrame:
        """기준시점 직전 13주 이력이 있는 고객의 향후 52주 CLV를 예측한다(미래 정답 불필요).

        Args:
            weekly: ``weekly_panel.parquet``, 시뮬레이터 features 디렉터리, CSV 또는 DataFrame.
            ledger: ``raw.sqlite`` 또는 데이터셋 디렉터리. ``bgnbd_gg``에만 필요하고, 주면 주간 패널과 대조한다.
            reference_week: 예측 기준시점 t0(서울 월요일 00:00). 기본: 실험 ``as_of_at``, 없으면 데이터 끝.
            customers: 예측할 고객 ID 목록. 기본은 이력이 있는 모든 고객. 이력이 없는 고객이 있으면 오류다.

        Returns:
            ``clv_rank`` 순 DataFrame. 상위 20% 표시와 순위는 반환되는 고객들 안에서 정한다.
        """
        panel, orders = self._inputs(weekly, ledger, strict, input_options)
        t0 = to_cutoff(reference_week) if reference_week is not None else default_reference_week(panel)
        scored = self._scored(panel, orders, t0, customers)
        values, ids = scored[PREDICTION_COLUMN].to_numpy(dtype=float), scored["customer_id"]
        result = pd.DataFrame({
            "customer_id": ids, "reference_week": t0.isoformat(), PREDICTION_COLUMN: values,
            HIGH_VALUE_COLUMN: top_flags(values, ids, self.config.top_fraction), "clv_rank": ranks(values, ids),
            "model_name": self.name, "model_version": self.model_version, "horizon_weeks": LABEL_WEEKS,
            "prediction_end_at": (t0 + HORIZON).isoformat(),
            "history_complete": scored["history_complete"].to_numpy(dtype=bool),
            "history_weeks_13w": scored["history_weeks_13w"].to_numpy(dtype=float),
            "currency": panel.attrs.get("currency"), "amount_unit": panel.attrs.get("amount_unit"),
            "dataset_id": panel.attrs.get("dataset_id"),
        })
        for column in ("purchase_probability_52w", "expected_amount_if_purchase"):
            if column in scored:
                result[column] = scored[column].to_numpy(dtype=float)
        return result.sort_values("clv_rank", kind="mergesort").reset_index(drop=True)

    def same_dataset(self, dataset_id: str) -> bool:
        """``dataset_id``가 학습 데이터셋일 수 있으면 True(학습 데이터셋 ID를 모르면 True로 본다)."""
        trained_on = self.registry["provenance"].get("dataset_id")
        return trained_on is None or trained_on == dataset_id

    def training_roles(self, dataset_id: str, customer_ids) -> pd.Series:
        """고객별 CLV 학습 역할(``ROLES``). 학습 데이터셋과 다른 데이터셋의 고객은 모두 ``unused``다."""
        if not self.same_dataset(dataset_id):
            return pd.Series("unused", index=pd.Index(customer_ids), dtype=object)
        splits = pd.read_csv(self.model_dir / "splits.csv", dtype={"customer_id": str})
        roles = splits.set_index("customer_id")["split"]
        ids = pd.Index(np.asarray(customer_ids, dtype=object))
        return pd.Series(ids.map(roles), index=ids, dtype=object).fillna("unused")

    def _version_warnings(self, dataset_id, panel, orders) -> list[str]:
        """학습 때와 다른 CLV·xgboost 버전, 학습 데이터와 같은 seed로 다시 만든 데이터셋을 경고한다."""
        warnings = []
        trained = self.card["code"].get("clv_version")
        if trained != __version__:
            warnings.append(f"bundle was trained with clv {trained}, inference runs clv {__version__}")
        library = self.card["environment"].get("xgboost")
        if self.name.startswith("xgboost") and library != environment()["xgboost"]:
            warnings.append(f"bundle was trained with xgboost {library}, inference runs xgboost {environment()['xgboost']}")
        if not self.same_dataset(dataset_id):
            source = orders.attrs.get("source_metadata") if orders is not None else None
            seed = panel.attrs.get("seed") if panel.attrs.get("seed") is not None else (source or {}).get("seed")
            training_seed = self.registry["provenance"].get("seed")
            if training_seed is None:
                training_seed = (self.card["training_data"].get("simulator") or {}).get("seed")
            if seed is not None and seed == training_seed:
                warnings.append(f"input dataset uses the same simulator seed ({seed}) as the CLV training data; "
                                "it may regenerate the training customers under another configuration — "
                                "prefer a dataset with a different seed")
        return warnings

    def predict_for_strategy(self, features, weekly=None, ledger=None, *, allow_training_customers: bool = False,
                             strict: bool = True, **input_options) -> StrategyPredictions:
        """Feature Store의 모든 (고객, 기준 시각)에 대해 전략 모듈 입력 batch를 만든다.

        Args:
            features: ``feature_store.parquet``, 그 디렉터리, 또는 ``STRATEGY_KEY_COLUMNS``가 있는 DataFrame.
            weekly: 같은 데이터셋의 주간 패널. 기본: ``features``가 경로면 그 디렉터리(시뮬레이터 features 출력).
            ledger: ``raw.sqlite`` 또는 데이터셋 디렉터리. 주면 Feature Store의 ``raw_database_sha256``과 대조한다.
            allow_training_customers: 기본 False면 CLV 학습(train)·모델 선택(validation)에 쓴 고객이 하나라도 있을 때
                오류를 낸다. 최종 전략 평가는 모델이 보지 않은 고객으로 해야 한다. 개발용 확인에만 True로 둔다.

        Returns:
            ``StrategyPredictions``. ``as_of_at``이 서울 월요일 00:00이 아니면(주 단위 입력과 어긋나면) 오류다.
        """
        checked = bool(input_options.get("verify_checksum", True))
        population = load_population(features, verify_checksum=checked)
        if weekly is None:
            if isinstance(features, pd.DataFrame):
                raise ValueError("pass weekly when features is a DataFrame")
            path = Path(features)
            weekly = path if path.is_dir() else path.parent
        panel, orders = self._inputs(weekly, ledger, strict, input_options)
        first = population.iloc[0]
        dataset_id, raw_sha = first["dataset_id"], first["raw_database_sha256"]
        # 같은 데이터셋인지 확인: dataset_id, 통화·단위, 원천 raw SQLite 체크섬(알 수 있는 입력마다)
        if panel.attrs.get("dataset_id") is not None and panel.attrs["dataset_id"] != dataset_id:
            raise ValueError(f"weekly input dataset_id {panel.attrs['dataset_id']!r} differs from the feature store "
                             f"{dataset_id!r}")
        for name in ("currency", "amount_unit"):
            if str(panel.attrs.get(name)) != str(first[name]):
                raise ValueError(f"feature store {name} {first[name]!r} differs from the weekly input "
                                 f"{panel.attrs.get(name)!r}")
        # manifest에 적힌 raw SQLite 체크섬이 Feature Store와 같은지 본다. 파일 해시를 건너뛰었으면(verify_checksum=False)
        # 값은 비교하되 검증된 출처로는 기록하지 않는다.
        matched = []
        for source, digest in (("weekly_manifest", panel.attrs.get("raw_database_sha256")),
                               ("ledger_manifest", None if orders is None else orders.attrs.get("raw_database_sha256"))):
            if digest is None:
                continue
            if digest != raw_sha:
                raise ValueError(f"{source} raw_database_sha256 {digest} differs from the feature store {raw_sha}")
            matched.append(source)
        verified_by = matched if checked else []

        frames, snapshots = [], []
        for as_of_at in sorted(population["as_of_at"].unique()):
            as_of_at = pd.Timestamp(as_of_at)
            try:
                t0 = to_cutoff(as_of_at)
            except ValueError as error:
                raise ValueError(f"as_of_at {as_of_at.isoformat()} is not an Asia/Seoul Monday 00:00; CLV inputs "
                                 "are whole Seoul weeks, so the input cutoff would not equal as_of_at") from error
            part = population.loc[population["as_of_at"].eq(as_of_at)]
            scored = self._scored(panel, orders, t0, part["customer_id"])
            scored.index = part.index
            frames.append(part.join(scored.drop(columns="customer_id")).assign(input_cutoff_at=t0.tz_convert("UTC")))
            snapshots.append({"as_of_at": as_of_at.isoformat(), "input_cutoff_at": t0.tz_convert("UTC").isoformat(),
                              "prediction_end_at": (as_of_at + HORIZON).isoformat(), "customers": int(len(part))})
        joined = pd.concat(frames).loc[population.index]
        joined["prediction_end_at"] = joined["as_of_at"] + HORIZON
        for column in ("as_of_at", "input_cutoff_at", "prediction_end_at"):
            joined[column] = joined[column].astype(UTC_TIME)
        joined["clv_training_role"] = self.training_roles(dataset_id, joined["customer_id"]).to_numpy()

        roles = joined["clv_training_role"].value_counts().reindex(ROLES).fillna(0).astype(int).to_dict()
        fitted = sum(roles[name] for name in FITTED_ROLES)
        warnings = []
        if fitted:
            message = (f"{fitted} of {len(joined)} customers were used to fit (train) or select (validation) this CLV "
                       "model; final strategy evaluation needs customers the model never saw — use another "
                       "simulator dataset (e.g. a different seed) or only clv_training_role == 'test'")
            if not allow_training_customers:
                raise ValueError(message + "; pass allow_training_customers=True only for development checks")
            warnings.append(message)
        if not checked:
            warnings.append("file checksums were skipped (verify_checksum=False); input provenance is unverified")
        elif not verified_by:
            warnings.append("raw_database_sha256 could not be checked against the weekly or ledger inputs")
        warnings += self._version_warnings(dataset_id, panel, orders)

        frame = joined.loc[:, list(STRATEGY_COLUMNS)].reset_index(drop=True)
        frame[PREDICTION_COLUMN] = frame[PREDICTION_COLUMN].astype("float64")
        extra = [name for name in ("purchase_probability_52w", "expected_amount_if_purchase") if name in joined]
        audit = joined.loc[:, [*JOIN_KEYS, "history_complete", "history_weeks_13w", "clv_training_role", *extra]]
        audit = audit.reset_index(drop=True)
        training = self.card["training_data"]
        producer = {
            "producer": "clv", "model_name": self.name, "model_version": self.model_version,
            "bundle": str(self.model_dir), "code": self.card["code"], "inference_code": code_version(),
            "training_environment": self.card["environment"], "inference_environment": environment(), "target": TARGET,
            "policy_condition": self.card["policy_condition"],
            "training": {"reference_week": training["reference_week"], "dataset_id": training["dataset_id"],
                         "raw_database_sha256": training["raw_database_sha256"],
                         "label_window_end": (self.training_reference_week + HORIZON).isoformat()},
            "inputs": {"dataset_id": dataset_id, "raw_database_sha256": raw_sha,
                       "raw_database_sha256_verified_by": verified_by, "raw_database_sha256_matched": matched,
                       "checksums_skipped": not checked, "weekly_path": panel.attrs.get("path"),
                       "weekly_sha256": panel.attrs.get("weekly_sha256"),
                       "ledger_path": None if orders is None else orders.attrs.get("path")},
            "snapshots": snapshots, "customers": int(len(frame)),
            "history_incomplete_customers": int((~audit["history_complete"].astype(bool)).sum()),
            "training_overlap": {"same_dataset": self.same_dataset(dataset_id), "roles": roles,
                                 "allowed": bool(allow_training_customers)},
            "warnings": warnings, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return StrategyPredictions(frame, self.metadata, audit, producer)
