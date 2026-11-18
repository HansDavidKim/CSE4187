"""WBS 4.4 Baseline: 과거 주당 평균 결제액 × 52 (완전 관측이면 ``(net_spend_13w / 13) * 52``).

부분 이력 고객(기본은 추론만 해당)은 관측 주 수로 나눠 미관측 주를 무구매 주로 세지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import LABEL_WEEKS

NAME = "baseline"


class BaselineModel:
    """평균 결제액 외삽 모델. 학습하는 모수가 없다."""

    name = NAME
    formula = "(net_spend_13w / history_weeks_13w) * 52; history_weeks_13w = 13 for complete history"

    def fit(self, frame: pd.DataFrame | None = None, y=None) -> "BaselineModel":
        """다른 모델과 인터페이스를 맞추기 위한 자리로, 아무것도 하지 않고 자신을 반환한다."""
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """``net_spend_13w``와 ``history_weeks_13w``로 52주 예측값을 낸다. 관측 주가 없으면 0, 음수는 0으로 자른다."""
        weeks = frame["history_weeks_13w"].to_numpy(dtype=float)
        spend = frame["net_spend_13w"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(weeks > 0, spend / weeks, 0.0)
        return np.clip(rate * LABEL_WEEKS, 0.0, None)

    def save(self, directory: str | Path) -> Path:
        """``baseline.json``에 수식과 예측 기간을 저장하고 그 경로를 반환한다."""
        path = Path(directory) / "baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": NAME, "formula": self.formula, "horizon_weeks": LABEL_WEEKS}, indent=2))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "BaselineModel":
        """``baseline.json``이 Baseline 모델 파일인지 확인하고 모델을 만든다."""
        values = json.loads((Path(directory) / "baseline.json").read_text())
        if values.get("model") != NAME:
            raise ValueError("not a baseline model file")
        return cls()
