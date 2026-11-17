"""모델 메타데이터·보고서용 엄격한 JSON 입출력 (NaN은 null로 기록)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def jsonable(value):
    """numpy/pandas 스칼라·타임스탬프·NaN을 JSON에 쓸 수 있는 값으로 재귀 변환한다.

    NaN·inf·``pd.NA``·``pd.NaT``는 ``None``(JSON null)이 된다.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def write_json(path: str | Path, value) -> Path:
    """``value``를 UTF-8 JSON으로 저장하고 경로를 돌려준다(상위 디렉터리 자동 생성).

    ``allow_nan=False``라서 변환되지 않은 NaN이 남아 있으면 오류가 난다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path):
    """UTF-8 JSON 파일을 읽어 파이썬 객체로 돌려준다."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
