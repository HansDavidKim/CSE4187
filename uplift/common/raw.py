"""시뮬레이터 실행 디렉터리에서 원시 로그와 실험 배정을 읽는다.

T/X/H-learner가 공통으로 쓰는 입력 단계다. 읽은 결과는 common.features가 feature로 바꾼다.

    raw = load_raw("simulator/artifacts/v5_raw")

읽는 파일 (ecommerce-sim v4 generate 출력):
    raw.sqlite            업무 테이블 7개
    raw.manifest.json     관측 범위, dataset_id, seed, SQLite 체크섬
    treatments.parquet    고객별 고정 배정 (customer_id, as_of_at, treatment, assignment_probability)

평가 전용 출력(쌍대 rollout, 잠재 진단)은 여기서 읽지 않는다. 모델 입력에 섞이지 않게 한다.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import schema as S


@dataclass
class RawData:
    tables: dict  # 테이블 이름 -> DataFrame. 시각 컬럼은 UTC Timestamp
    assignments: pd.DataFrame  # customer_id, t0, treatment, assignment_probability
    observation_start: pd.Timestamp | None
    observation_end: pd.Timestamp
    metadata: dict


def to_utc(values):
    return pd.to_datetime(values, utc=True, format="ISO8601")


def normalize_tables(tables: dict) -> dict:
    """필요한 컬럼만 남기고 시각을 UTC로 바꾼다. raw4처럼 리뷰 컬럼이 없으면 NULL로 채운다."""
    out = {}
    for name, columns in S.RAW_COLUMNS.items():
        frame = tables[name].reindex(columns=columns).copy()
        for column in S.TIME_COLUMNS.intersection(columns):
            frame[column] = to_utc(frame[column])
        out[name] = frame
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(run_dir, verify_checksum: bool = True) -> RawData:
    run_dir = Path(run_dir)
    database = run_dir / S.RAW_DATABASE
    manifest = json.loads((run_dir / S.RAW_MANIFEST).read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    if verify_checksum and _sha256(database) != manifest["database_sha256"]:
        raise ValueError(f"{database}: manifest의 체크섬과 다르다")

    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {name: pd.read_sql_query(f'SELECT * FROM "{name}"', connection)
                  for name in S.RAW_COLUMNS}
    finally:
        connection.close()

    assigned = pd.read_parquet(run_dir / S.ASSIGNMENT_FILE)
    assignments = pd.DataFrame({
        S.CUSTOMER_ID: assigned["customer_id"],
        S.T0: to_utc(assigned["as_of_at"]),
        S.TREATMENT: assigned["treatment"].astype(int),
        "assignment_probability": assigned["assignment_probability"].astype(float),
    })
    if assignments[S.CUSTOMER_ID].duplicated().any():
        raise ValueError("고객당 배정은 한 번이어야 한다")
    if not assignments[S.TREATMENT].isin([0, 1]).all():
        raise ValueError("treatment는 0 또는 1이어야 한다")

    start = metadata.get("observation_start_at")
    return RawData(
        tables=normalize_tables(tables),
        assignments=assignments,
        observation_start=to_utc(start) if start else None,
        observation_end=to_utc(metadata["observation_end_at"]),
        metadata=metadata,
    )
