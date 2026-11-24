"""시뮬레이터 v5 출력이나 합의 스키마 입력을 통화·단위 변경 없이 CLV 프레임으로 읽는다 (WBS 4.1).

* 주간: 시뮬레이터 ``weekly_panel.parquet``(또는 그 CSV) 또는 CLV 스키마 CSV. 이름은 ``schema.SIMULATOR_TO_CLV``로 매핑.
* 주문 원장: ``raw.sqlite``(orders + 구매 events 조인) 또는 주문 CSV/Parquet. BG/NBD는 이 실제 주문을 쓴다.
* 금액은 원천 최소 단위 그대로이고, ``SOURCE_UNSPECIFIED`` 통화는 KRW로 간주하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .schema import (
    AMOUNT_COLUMNS, ASSIGNMENT_COLUMN, CHECK_COLUMNS, CONTEXT_COLUMNS, CONTROL_CODE, CONTROL_GROUP_LABEL,
    COUNT_COLUMNS, COVERAGE_COLUMNS, ORDER_ALIASES, REQUIRED_WEEKLY, SEOUL, SIMULATOR_TO_CLV, WEEK_SECONDS,
)

# 모델 카드에 학습 데이터 출처로 옮겨 적는 시뮬레이터 manifest 항목(재생성·대조용).
SIMULATOR_METADATA_KEYS = (
    "dataset_id", "seed", "months", "eligible_customers", "simulator_version", "configuration_sha256",
    "simulator_code_sha256", "as_of_at", "observation_start_at", "observation_end_at", "retention_mode",
    "distribution", "min_customer_history_days", "eligibility_window_days", "coupon_validity_days",
)
# 문자열 끝의 시간대 표기(`Z`, `+09:00`, `+0900`)를 찾는 패턴.
_ZONE = r"(?:Z|[+-]\d{2}:?\d{2})$"


def read_table(path: str | Path) -> pd.DataFrame:
    """Parquet 또는 CSV 파일을 읽는다. CSV의 `customer_id`·`order_id`는 문자열로 유지한다."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"customer_id": str, "order_id": str}, keep_default_na=True)
    raise ValueError(f"unsupported table format: {path}")


def resolve_weekly_path(path: str | Path) -> Path:
    """주간 파일 경로 또는 시뮬레이터 features 디렉터리를 받아 실제 주간 파일 경로를 돌려준다.

    디렉터리면 `weekly_panel.parquet`를 먼저, 없으면 `weekly_panel.csv`를 찾는다.
    """
    path = Path(path)
    if path.is_dir():
        for name in ("weekly_panel.parquet", "weekly_panel.csv"):
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f"no weekly_panel.parquet or weekly_panel.csv in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_ledger_path(path: str | Path) -> Path:
    """`raw.sqlite`, 주문 CSV/Parquet, 또는 시뮬레이터 데이터셋 디렉터리를 받아 주문 원장 경로를 돌려준다."""
    path = Path(path)
    if path.is_dir():
        if (path / "raw.sqlite").is_file():
            return path / "raw.sqlite"
        raise FileNotFoundError(f"no raw.sqlite in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def to_seoul(values: pd.Series, name: str, notes: list[str] | None = None) -> pd.Series:
    """타임스탬프를 Asia/Seoul 시각으로 변환한다. 시간대 없는 값은 서울 현지 시각으로 읽는다.

    그 가정은 ``notes``에 기록하고, 시간대 있는 값과 없는 값이 섞이면 오류를 낸다.
    """
    notes = notes if notes is not None else []
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert(SEOUL)
    if pd.api.types.is_datetime64_dtype(values):
        notes.append(f"{name}: naive timestamps read as {SEOUL} local time")
        return values.dt.tz_localize(SEOUL)
    # 문자열이면 결측이 아닌 값 전체가 시간대 표기를 가졌는지(또는 전혀 없는지) 본다.
    text = values.astype("string")
    present = text.notna()
    zoned = text.str.contains(_ZONE, regex=True, na=False)
    if zoned[present].all():
        return pd.to_datetime(text, utc=True, format="ISO8601").dt.tz_convert(SEOUL)
    if not zoned[present].any():
        notes.append(f"{name}: naive timestamps read as {SEOUL} local time")
        return pd.to_datetime(text, format="ISO8601").dt.tz_localize(SEOUL)
    raise ValueError(f"{name} mixes timezone-aware and naive values")


def _whole_numbers(values: pd.Series, name: str) -> pd.Series:
    """값을 결측 허용 정수(`Int64`)로 바꾼다. 소수·무한대가 있으면 오류(단위 환산 없이 원천 최소 단위)."""
    numeric = pd.to_numeric(values, errors="raise").astype("Float64")
    present = numeric.dropna()
    if not np.isfinite(present.to_numpy(dtype=float)).all() or (present % 1 != 0).any():
        raise ValueError(f"{name} must hold whole numbers in the source unit")
    return numeric.round().astype("Int64")


def _booleans(values: pd.Series, name: str) -> pd.Series:
    """bool 또는 `true/false/1/0` 문자열을 bool로 바꾼다. 결측이나 다른 값이 있으면 오류."""
    if pd.api.types.is_bool_dtype(values) and not values.isna().any():
        return values.astype(bool)
    mapped = values.map(lambda value: {"true": True, "false": False, "1": True, "0": False}.get(str(value).strip().lower()))
    if mapped.isna().any():
        raise ValueError(f"{name} must be boolean")
    return mapped.astype(bool)


def _provenance(frame: pd.DataFrame, name: str, given) -> str | None:
    """출처 값(`currency`·`amount_unit`·`dataset_id`)을 하나로 정한다.

    데이터 안의 값은 하나여야 하고, 인자 ``given``과 다르면 오류다. 데이터에 없을 때만 ``given``을 쓴다.
    """
    values = frame[name].dropna().astype(str).unique() if name in frame else []
    if len(values) > 1:
        raise ValueError(f"one input must use one {name}: {sorted(values)}")
    found = values[0] if len(values) else None
    if given is not None and found is not None and str(given) != found:
        raise ValueError(f"{name} argument {given!r} disagrees with the data {found!r}")
    value = found if found is not None else given
    return None if value is None else str(value)


def _require_units(currency, amount_unit, source: str) -> None:
    """통화와 금액 단위가 모두 명시돼야 한다. 기본 통화(KRW 등)를 가정하지 않는다."""
    if not currency or not str(currency).strip() or not amount_unit or not str(amount_unit).strip():
        raise ValueError(f"{source} requires explicit currency and amount_unit; no default currency is assumed")


def _check_assignment(result: pd.DataFrame) -> None:
    """실험 배정(``ASSIGNMENT_COLUMN``)이 고객마다 하나이고, ``group``이 있으면 Control 표기와 일치하는지 본다.

    Control-only 학습(``population.control_no_coupon``)이 이 열로 모집단을 고르므로, 규약이 어긋나면 조용히
    섞지 않고 중단한다. 배정 값이 없는 고객(결측)은 허용하며 학습 정책에서 미배정으로 다룬다.
    """
    if ASSIGNMENT_COLUMN not in result:
        return
    assignment = result[ASSIGNMENT_COLUMN]
    if assignment.groupby(result["customer_id"]).nunique().gt(1).any():
        raise ValueError(f"{ASSIGNMENT_COLUMN} must be constant per customer")
    if "group" in result:
        labelled = assignment.notna() & result["group"].notna()
        control_code = assignment.eq(CONTROL_CODE).fillna(False)
        control_label = result["group"].astype("string").eq(CONTROL_GROUP_LABEL).fillna(False)
        if (labelled & control_code.ne(control_label)).any():
            raise ValueError(f"group {CONTROL_GROUP_LABEL!r} must match {ASSIGNMENT_COLUMN} == {CONTROL_CODE}")


def adapt_weekly(frame: pd.DataFrame, *, currency: str | None = None, amount_unit: str | None = None,
                 dataset_id: str | None = None) -> pd.DataFrame:
    """주간 프레임을 CLV 스키마로 매핑하고 결제액을 만든다.

    결제액 ``paid_amount = gross_revenue - redemption_discount_cost``(총매출 - 할인액)다.
    쿠폰 발송비(``marketing_send_cost_minor``)는 차감하지 않고 가져오지도 않는다.
    미관측 주는 결측으로 두며 0으로 채우지 않는다.

    Args:
        frame: 시뮬레이터 ``weekly_panel`` 또는 합의 스키마의 주간 행.
        currency: 프레임에 ``currency`` 컬럼이 없을 때 쓸 통화.
        amount_unit: 프레임에 ``amount_unit`` 컬럼이 없을 때 쓸 금액 단위.
        dataset_id: 프레임에 없을 때 쓸 데이터셋 식별자.

    Returns:
        입력된 고객-주마다 한 행. CLV 컬럼 이름, 관측 정보, 출처(컬럼과 ``attrs``)를 담는다.
    """
    notes: list[str] = []
    columns = set(frame)
    # 컬럼 이름으로 입력 형식(시뮬레이터 패널 / CLV 스키마)을 판별한다.
    if {"reference_week", "gross_revenue_minor"} <= columns:
        source, mapping = "simulator_v5_weekly_panel", SIMULATOR_TO_CLV
    elif {"week", "gross_revenue"} <= columns:
        source, mapping = "clv_schema", {name: name for name in SIMULATOR_TO_CLV.values()}
    else:
        raise ValueError("weekly input needs simulator columns (reference_week, gross_revenue_minor, ...) "
                         "or CLV columns (week, gross_revenue, ...)")
    missing = [src for src, dst in mapping.items() if dst in REQUIRED_WEEKLY and src not in frame]
    if missing:
        raise ValueError(f"weekly input is missing required columns: {missing}")
    if frame["customer_id"].isna().any():
        raise ValueError("customer_id cannot be missing")
    result = pd.DataFrame({dst: frame[src] for src, dst in mapping.items() if src in frame})
    result["customer_id"] = frame["customer_id"].map(str)
    result["week"] = to_seoul(result["week"], "week", notes)
    if "coupon_action" not in result:
        result["coupon_action"] = pd.NA
        notes.append("coupon_action absent; kept as missing")
    for column in COUNT_COLUMNS + AMOUNT_COLUMNS:
        result[column] = _whole_numbers(result[column], column)
    # `net_revenue_minor`는 검증 단계의 금액 대조에만 쓴다.
    for src, dst in CHECK_COLUMNS.items():
        if src in frame:
            result[dst] = _whole_numbers(frame[src], src)
    present = [name for name in COVERAGE_COLUMNS if name in frame]
    if len(present) == len(COVERAGE_COLUMNS):
        result["observed_week"] = _booleans(frame["observed_week"], "observed_week")
        result["partial_coverage"] = _booleans(frame["partial_coverage"], "partial_coverage")
        result["coverage_seconds"] = pd.to_numeric(frame["coverage_seconds"], errors="raise").astype(float)
        coverage_source = "supplied"
    elif not present:
        # 관측 컬럼이 없으면 값이 모두 있는 행만 완전 관측으로 보고, 그 가정을 notes에 남긴다.
        observed = result[list(REQUIRED_WEEKLY[2:])].notna().all(axis=1)
        result["observed_week"] = observed
        result["partial_coverage"] = False
        result["coverage_seconds"] = np.where(observed, float(WEEK_SECONDS), 0.0)
        coverage_source = "assumed_from_rows"
        notes.append("coverage not supplied: rows with values are full observed weeks; absent rows are unobserved")
    else:
        raise ValueError(f"coverage columns must be supplied together: {list(COVERAGE_COLUMNS)}")
    # 미관측 주의 값은 0이 아니라 결측(NA)으로 덮어쓴다. 미관측 주를 활동 없음과 구분하기 위함이다.
    unobserved = ~result["observed_week"]
    value_columns = [*COUNT_COLUMNS, *AMOUNT_COLUMNS, *([v for v in CHECK_COLUMNS.values() if v in result])]
    masked = int(result.loc[unobserved, value_columns].notna().any(axis=1).sum())
    if masked:
        notes.append(f"{masked} unobserved rows carried values; they were set to missing")
    result.loc[unobserved, value_columns] = pd.NA
    # 결제액 = 총매출 - 할인액. 마케팅 발송비는 빼지 않는다. 미관측 주는 NA가 그대로 전파된다.
    result["paid_amount"] = (result["gross_revenue"] - result["redemption_discount_cost"]).astype("Int64")
    for name in CONTEXT_COLUMNS:
        if name in frame:
            values = frame[name]
            if name in ("assigned_at", "as_of_at", "signed_up_at"):
                values = to_seoul(values, name, notes)
            elif name == "treatment":
                values = _whole_numbers(values, name)
            result[name] = values.to_numpy()
    _check_assignment(result)
    # 통화·단위는 입력 그대로 전달한다(`SOURCE_UNSPECIFIED`도 KRW로 바꾸지 않고 환산하지 않음).
    currency = _provenance(frame, "currency", currency)
    amount_unit = _provenance(frame, "amount_unit", amount_unit)
    dataset_id = _provenance(frame, "dataset_id", dataset_id)
    _require_units(currency, amount_unit, "weekly input")
    result["currency"], result["amount_unit"] = currency, amount_unit
    result["dataset_id"] = dataset_id
    result = result.sort_values(["customer_id", "week"], kind="mergesort").reset_index(drop=True)
    # 시뮬레이터 seed(값이 하나일 때만). 학습 데이터셋과 같은 seed로 다시 만든 데이터셋인지 경고하는 데 쓴다.
    seeds = pd.Series(frame["seed"]).dropna().unique() if "seed" in frame else []
    result.attrs = {"source_format": source, "currency": currency, "amount_unit": amount_unit,
                    "dataset_id": dataset_id, "coverage_source": coverage_source, "notes": notes,
                    "seed": int(seeds[0]) if len(seeds) == 1 else None}
    return result


def _features_manifest(file: Path, verify_checksum: bool) -> dict:
    """시뮬레이터 features 디렉터리의 ``manifest.json``에서 주간 파일 체크섬과 원천 raw SQLite 체크섬을 읽는다.

    manifest가 없거나(직접 만든 CSV 등) 이 파일의 체크섬이 기록돼 있지 않으면 계산한 체크섬만 돌려준다.
    """
    digest = _sha256(file)
    result = {"weekly_sha256": digest, "raw_database_sha256": None}
    manifest_path = file.parent / "manifest.json"
    if not manifest_path.is_file():
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        return result
    recorded = (manifest.get("file_sha256") or {}).get(file.name) if isinstance(manifest, dict) else None
    if recorded is None:
        return result
    # 주간 파일이 features manifest 생성 이후 바뀌었으면 출처(raw SQLite 체크섬)를 믿을 수 없으므로 중단한다.
    if verify_checksum and recorded != digest:
        raise ValueError(f"{file.name} checksum differs from {manifest_path}")
    result["raw_database_sha256"] = manifest.get("raw_database_sha256")
    return result


def load_weekly(path: str | Path, *, verify_checksum: bool = True, **units) -> pd.DataFrame:
    """주간 패널 파일(또는 디렉터리)을 읽어 ``adapt_weekly``로 변환한다. ``units``는 통화·단위·dataset_id.

    시뮬레이터 features 디렉터리의 파일이면 옆 ``manifest.json``의 체크섬과 대조하고(``verify_checksum``),
    원천 ``raw_database_sha256``을 ``attrs``에 남긴다.
    """
    file = resolve_weekly_path(path)
    frame = adapt_weekly(read_table(file), **units)
    frame.attrs.update(path=str(file), **_features_manifest(file, verify_checksum))
    return frame


def adapt_orders(frame: pd.DataFrame, *, currency: str | None = None, amount_unit: str | None = None,
                 dataset_id: str | None = None) -> pd.DataFrame:
    """주문 원장을 ``order_id, customer_id, order_at (UTC), gross/discount/paid_amount``로 변환한다.

    금액은 원천 최소 단위 정수로 두며, 통화·단위가 명시되지 않으면 오류다.
    """
    notes: list[str] = []
    frame = frame.rename(columns={old: new for old, new in ORDER_ALIASES.items() if old in frame and new not in frame})
    required = ["customer_id", "order_at", "gross_amount", "discount_amount", "paid_amount"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"order ledger is missing required columns: {missing}")
    result = pd.DataFrame({"order_id": frame["order_id"].map(str) if "order_id" in frame
                           else pd.Series(np.arange(len(frame)), index=frame.index).map(lambda i: f"row-{i}")})
    if "order_id" not in frame:
        notes.append("order_id absent; row numbers used")
    if frame["customer_id"].isna().any() or frame["order_at"].isna().any():
        raise ValueError("orders require customer_id and order_at")
    result["customer_id"] = frame["customer_id"].map(str)
    # 시간대 없는 시각은 서울 시각으로 읽은 뒤 UTC로 저장한다.
    result["order_at"] = to_seoul(frame["order_at"], "order_at", notes).dt.tz_convert("UTC")
    for name in ("gross_amount", "discount_amount", "paid_amount"):
        result[name] = _whole_numbers(frame[name], name)
    if "coupon_id" in frame:
        result["coupon_id"] = frame["coupon_id"].to_numpy()
    currency = _provenance(frame, "currency", currency)
    amount_unit = _provenance(frame, "amount_unit", amount_unit)
    dataset_id = _provenance(frame, "dataset_id", dataset_id)
    _require_units(currency, amount_unit, "order ledger")
    result = result.sort_values(["customer_id", "order_at", "order_id"], kind="mergesort").reset_index(drop=True)
    result.attrs = {"currency": currency, "amount_unit": amount_unit, "dataset_id": dataset_id, "notes": notes}
    return result


def _sha256(path: Path) -> str:
    """파일을 1 MiB씩 읽어 SHA-256 16진 문자열을 돌려준다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_orders_sqlite(path: str | Path, verify_checksum: bool = True) -> pd.DataFrame:
    """시뮬레이터 ``raw.sqlite``에서 실제 주문을 읽기 전용으로 읽는다.

    통화·단위·dataset_id는 옆의 manifest에서 가져오고, ``verify_checksum``이면 manifest 체크섬과 대조한다.
    """
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"raw SQLite manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    # 원장 파일이 manifest 생성 이후 바뀌었으면 단위·출처를 믿을 수 없으므로 중단한다.
    if verify_checksum and _sha256(path) != manifest.get("database_sha256"):
        raise ValueError("raw.sqlite checksum differs from its manifest")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        # 주문 시각은 연결된 구매 이벤트의 `event_at`이다. 조인에서 빠지는 주문이 없는지 총 건수로 확인한다.
        total = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        frame = pd.read_sql_query(
            "SELECT o.order_id, e.customer_id, e.event_at AS order_at, o.gross_minor AS gross_amount, "
            "o.discount_minor AS discount_amount, o.paid_minor AS paid_amount, o.coupon_id "
            "FROM orders o JOIN events e ON e.event_id = o.purchase_event_id AND e.event_type = 'purchase'",
            connection)
    finally:
        connection.close()
    if len(frame) != total:
        raise ValueError(f"{total - len(frame)} orders have no purchase event")
    result = adapt_orders(frame, currency=metadata.get("currency"), amount_unit=metadata.get("amount_unit"),
                          dataset_id=metadata.get("dataset_id"))
    result.attrs.update(path=str(path), observation_start_at=metadata.get("observation_start_at"),
                        observation_end_at=metadata.get("observation_end_at"),
                        raw_database_sha256=manifest.get("database_sha256"),
                        source_metadata={key: metadata[key] for key in SIMULATOR_METADATA_KEYS if key in metadata})
    return result


def load_orders(path: str | Path, *, verify_checksum: bool = True, **units) -> pd.DataFrame:
    """`raw.sqlite`, 데이터셋 디렉터리, CSV 또는 Parquet에서 주문 원장을 읽는다.

    SQLite면 manifest의 통화·단위를 쓰고, CSV/Parquet이면 파일 안 값이나 ``units`` 인자를 쓴다.
    """
    file = resolve_ledger_path(path)
    if file.suffix in (".sqlite", ".db"):
        return load_orders_sqlite(file, verify_checksum=verify_checksum)
    result = adapt_orders(read_table(file), **units)
    result.attrs["path"] = str(file)
    return result
