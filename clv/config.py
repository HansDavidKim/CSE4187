"""CLV 실행 설정(``CLVConfig``)과 JSON/YAML 로더. 구간 길이(13주·52주)는 ``schema`` 계약으로 고정한다."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

# Validation 지표로 고르는 XGBoost 탐색 그리드
DEFAULT_XGB_GRID = {
    "objective": ["reg:squarederror", "reg:tweedie"],
    "max_depth": [3, 5],
    "min_child_weight": [1, 5],
    "learning_rate": [0.05],
}
# 두 파트 XGBoost: 1파트(구매 여부 분류, 검증 로그 손실로 선택)와 2파트(구매 고객 결제액 회귀, 검증 RMSE로 선택)
DEFAULT_TWO_PART_CLASSIFIER_GRID = {"max_depth": [3, 5], "min_child_weight": [1, 5], "learning_rate": [0.05]}
DEFAULT_TWO_PART_REGRESSOR_GRID = {
    "objective": ["reg:squarederror", "reg:gamma"],
    "max_depth": [3, 5],
    "min_child_weight": [1, 5],
    "learning_rate": [0.05],
}
# 4주 조건부 총구매액 XGBoost(Control 구매 고객 회귀, 검증 RMSE로 선택). 금액이 양수라 gamma도 후보다.
DEFAULT_GROSS_4W_GRID = {
    "objective": ["reg:squarederror", "reg:gamma"],
    "max_depth": [3, 5],
    "min_child_weight": [1, 5],
    "learning_rate": [0.05],
}
# 학습 모집단 정책(``population`` 모듈). 새 학습의 기본은 Control-only다.
POPULATION_POLICIES = ("control_no_coupon", "observed_experiment_mix")
# 이 설정 항목이 생기기 전 registry(기존 번들)에 적용하는 값. 기존 번들은 배정군을 섞어 학습했다.
LEGACY_REGISTRY_DEFAULTS = {"population_policy": "observed_experiment_mix"}
# 모든 그리드 점에 공통인 설정. n_jobs=1로 고정해 같은 입력이면 같은 예측이 나오게 한다.
DEFAULT_XGB_FIXED = {
    "n_estimators": 1000, "early_stopping_rounds": 50, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0, "tree_method": "hist", "n_jobs": 1,
}


@dataclass
class CLVConfig:
    """학습·평가·추론이 함께 쓰는 설정.

    Attributes:
        seed: 고객 분할과 XGBoost의 난수 seed.
        population_policy: 학습 모집단 정책. ``control_no_coupon``(기본)은 Control(쿠폰 미발송군) 중 정답 구간에
            쿠폰을 받지 않은 고객만 Train/Validation/Test로 나누고 Treatment는 ``unused``로 둔다.
            ``observed_experiment_mix``는 배정과 무관하게 모두 쓴다(기존 전달 번들의 정책). 분할 전에 적용되어
            모든 모델(52주 4종, 4주 총구매액)이 같은 모집단을 쓴다. 추론은 정책과 무관하게 전체 고객을 예측한다.
        split_fractions: 학습/검증/테스트 분할의 고객 비율.
        top_fraction: 고가치 고객으로 표시할 상위 비율(기본 상위 20% 고객).
        require_complete_history: True면 13주 입력이 모두 완전 관측인 고객만 학습·평가한다. 기본 False는
            정답(52주 완전 관측)이 있는 고객 전체로 학습한다. 같은 Test 고객 비교에서 두 이력 구간 모두
            전체 학습이 나았다(readme 참조). 어느 쪽이든 추론은 부분 이력 고객도 예측한다.
        calibration: BG/NBD 보정 기간. ``feature_window``(다른 모델과 같은 13주) 또는
            ``full_history``(t0 이전 전체 주문).
        transaction_period: BG/NBD 거래 단위. ``D``는 같은 서울 날짜의 주문을 하나로 묶는다.
        no_purchase_fallback: 보정 기간 구매가 없는 고객의 BG/NBD 값. ``zero`` 또는
            ``train_mean``(같은 처지인 Train 고객의 평균 정답).
        bgnbd_penalizer: BG/NBD 로그 모수의 L2 벌점.
        gg_penalizer: Gamma-Gamma 로그 모수의 L2 벌점.
        bgnbd_sensitivity_penalizers: 민감도 보고용으로 다시 적합할 BG/NBD 벌점 목록(빈 목록이면 생략).
            보고서만 만들고 선택 모델(``bgnbd_penalizer``)은 바꾸지 않는다.
        selection_metric: XGBoost 설정과 최종 모델을 고르는 Validation 지표.
        final_model: ``auto``(Validation 지표가 가장 좋은 모델) 또는 모델 이름 하나.
        xgb_grid: Validation으로 탐색할 하이퍼파라미터 그리드.
        xgb_fixed: 모든 그리드 점에 공통인 XGBoost 설정(두 파트 XGBoost도 같이 쓴다).
        two_part_classifier_grid: 두 파트 XGBoost 1파트(구매 여부 분류) 탐색 그리드.
        two_part_regressor_grid: 두 파트 XGBoost 2파트(구매 고객 결제액 회귀) 탐색 그리드.
        gross_4w_grid: 4주 조건부 총구매액 XGBoost(``xgboost_gross_4w``) 탐색 그리드.
    """

    seed: int = 42
    population_policy: str = "control_no_coupon"
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    top_fraction: float = 0.2
    require_complete_history: bool = False
    calibration: str = "feature_window"
    transaction_period: str = "D"
    no_purchase_fallback: str = "zero"
    bgnbd_penalizer: float = 0.0
    gg_penalizer: float = 0.0
    bgnbd_sensitivity_penalizers: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1)
    selection_metric: str = "rmse"
    final_model: str = "auto"
    xgb_grid: dict = field(default_factory=lambda: {key: list(value) for key, value in DEFAULT_XGB_GRID.items()})
    xgb_fixed: dict = field(default_factory=lambda: dict(DEFAULT_XGB_FIXED))
    two_part_classifier_grid: dict = field(
        default_factory=lambda: {key: list(value) for key, value in DEFAULT_TWO_PART_CLASSIFIER_GRID.items()})
    two_part_regressor_grid: dict = field(
        default_factory=lambda: {key: list(value) for key, value in DEFAULT_TWO_PART_REGRESSOR_GRID.items()})
    gross_4w_grid: dict = field(
        default_factory=lambda: {key: list(value) for key, value in DEFAULT_GROSS_4W_GRID.items()})

    def __post_init__(self):
        """비율 범위와 허용 선택지를 검사한다(잘못되면 ValueError)."""
        self.split_fractions = tuple(float(value) for value in self.split_fractions)
        if len(self.split_fractions) != 3 or min(self.split_fractions) <= 0 or abs(sum(self.split_fractions) - 1) > 1e-9:
            raise ValueError("split_fractions must be three positive shares summing to one")
        self.bgnbd_sensitivity_penalizers = tuple(float(value) for value in self.bgnbd_sensitivity_penalizers)
        if any(not value >= 0 for value in (self.bgnbd_penalizer, self.gg_penalizer, *self.bgnbd_sensitivity_penalizers)):
            raise ValueError("penalizers must be non-negative")
        if not 0 < self.top_fraction < 1:
            raise ValueError("top_fraction must be between zero and one")
        choices = {"calibration": ("feature_window", "full_history"), "transaction_period": ("D", "order"),
                   "no_purchase_fallback": ("zero", "train_mean"), "selection_metric": ("rmse", "mae"),
                   "population_policy": POPULATION_POLICIES}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.final_model not in ("auto", "baseline", "bgnbd_gg", "xgboost", "xgboost_two_part"):
            raise ValueError("final_model must be auto, baseline, bgnbd_gg, xgboost or xgboost_two_part")

    def to_dict(self) -> dict:
        """JSON(registry) 저장용 dict. ``split_fractions``는 list로 바꾼다."""
        result = asdict(self)
        result["split_fractions"] = list(self.split_fractions)
        result["bgnbd_sensitivity_penalizers"] = list(self.bgnbd_sensitivity_penalizers)
        return result

    @classmethod
    def from_dict(cls, values: dict | None) -> "CLVConfig":
        """dict에서 설정을 만든다. 모르는 키가 있으면 ValueError."""
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown CLV settings: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_registry(cls, values: dict | None) -> "CLVConfig":
        """번들 registry에 저장된 설정을 읽는다. 나중에 생긴 항목은 기존 번들의 값(``LEGACY_REGISTRY_DEFAULTS``)으로 채운다."""
        return cls.from_dict({**LEGACY_REGISTRY_DEFAULTS, **dict(values or {})})


def load_config(path: str | Path | None = None) -> CLVConfig:
    """JSON 또는 YAML(``clv:`` 섹션 허용)에서 설정을 읽는다. 경로가 없거나 파일이 비면 기본값."""
    if path is None:
        return CLVConfig()
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        values = yaml.safe_load(text) or {}
    else:
        values = json.loads(text) if text.strip() else {}
    if isinstance(values, dict) and isinstance(values.get("clv"), dict):
        values = values["clv"]
    return CLVConfig.from_dict(values)
