"""WBS 4.4: 주문 원장(``adapter.load_orders``)의 실제 주문 시각·결제액으로 BG/NBD + Gamma-Gamma CLV를 추정한다.

주간 집계는 쓰지 않는다. 시간 단위는 주, 거래는 서울 날짜별 주문 묶음(``period="D"``) 또는 주문 1건(``"order"``).
References: Fader, Hardie & Lee (2005) "Counting Your Customers the Easy Way";
Fader & Hardie (2013) "The Gamma-Gamma Model of Monetary Value".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

from .features import WEEK, to_cutoff
from .io import read_json, write_json
from .schema import INPUT_WEEKS, LABEL_WEEKS, SEOUL

NAME = "bgnbd_gg"
MIN_BGNBD_CUSTOMERS = 10
MIN_GG_CUSTOMERS = 5
CORRELATION_LIMIT = 0.3


def rfm_summary(orders: pd.DataFrame, t0, calibration: str = "feature_window", period: str = "D") -> pd.DataFrame:
    """보정 기간에 주문이 있는 고객별 RFM 요약(BG/NBD·Gamma-Gamma 입력)을 만든다.

    ``frequency`` = 반복 거래 수(거래 수 - 1), ``recency`` = 첫 거래부터 마지막 거래까지 주,
    ``T`` = 첫 거래부터 ``t0``까지 주, ``monetary_value`` = 반복 거래의 평균 결제액(첫 거래 제외).
    보정 기간은 ``feature_window``(``t0`` 직전 13주) 또는 ``full_history``(``t0`` 이전 전체)다.
    """
    t0 = to_cutoff(t0)
    rows = orders.loc[orders["order_at"].lt(t0)]
    if calibration == "feature_window":
        rows = rows.loc[rows["order_at"].ge(t0 - INPUT_WEEKS * WEEK)]
    elif calibration != "full_history":
        raise ValueError("calibration must be feature_window or full_history")
    columns = ["frequency", "recency", "T", "monetary_value", "calibration_orders", "calibration_paid"]
    if rows.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="customer_id"), dtype=float)
    local = rows["order_at"].dt.tz_convert(SEOUL)
    # 거래 = 서울 날짜별 주문 묶음(Lifetimes 관례). 다른 값이면 주문 시각마다 1건이다.
    key = local.dt.normalize() if period == "D" else local
    transactions = rows.assign(_at=key).groupby(["customer_id", "_at"])["paid_amount"].sum().astype(float)
    transactions = transactions.reset_index().sort_values(["customer_id", "_at"], kind="mergesort")
    grouped = transactions.groupby("customer_id")
    first, last = grouped["_at"].min(), grouped["_at"].max()
    count, total = grouped.size(), grouped["paid_amount"].sum()
    # Gamma-Gamma 관례대로 첫 거래 금액은 monetary_value에서 뺀다.
    first_value = grouped["paid_amount"].first()
    frequency = (count - 1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        monetary = ((total - first_value) / frequency).where(frequency > 0)
    # 시간은 모두 주 단위(Timedelta / 1주)다.
    result = pd.DataFrame({
        "frequency": frequency,
        "recency": (last - first) / WEEK,
        "T": (t0 - first) / WEEK,
        "monetary_value": monetary,
        "calibration_orders": rows.groupby("customer_id").size().reindex(count.index).astype(float),
        "calibration_paid": total,
    })
    result.index.name = "customer_id"
    return result


def _fit(objective, start, data) -> tuple[np.ndarray, dict]:
    """로그 모수 공간에서 음의 로그우도를 최소화하고 (로그 모수, 수렴 정보)를 반환한다.

    L-BFGS-B가 수렴하지 않으면 Nelder-Mead로 다시 풀고, 유한한 결과 중 목적함수가 가장 낮은 것을 쓴다.
    모수를 로그로 두어 양수 제약 없이 최적화한다.
    """
    best = None
    for method in ("L-BFGS-B", "Nelder-Mead"):
        options = {"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12} if method == "Nelder-Mead" else {"maxiter": 5000}
        result = minimize(objective, start, args=data, method=method, options=options)
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and (best is None or result.fun < best.fun - 1e-12):
            best = result
        if result.success:
            break
    if best is None:
        raise RuntimeError("likelihood optimisation failed")
    return best.x, {"converged": bool(best.success), "method": best.message if isinstance(best.message, str)
                    else str(best.message), "objective": float(best.fun), "iterations": int(best.nit)}


def _bgnbd_negll(log_params, x, tx, T, penalizer):
    """BG/NBD 고객당 평균 음의 로그우도(Fader, Hardie & Lee 2005). 인자는 log(r, alpha, a, b), L2 벌점 포함."""
    r, alpha, a, b = np.exp(log_params)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    # x = 0이면 A4 항이 없으므로 log 값을 -inf로 두어 logaddexp에서 사라지게 한다.
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + tx), -np.inf)
    # log(A3 + A4)를 넘침 없이 계산한다.
    ll = a1 + a2 + np.logaddexp(a3, a4)
    return -ll.mean() + penalizer * np.sum(log_params ** 2)


def _gg_negll(log_params, x, m, penalizer):
    """Gamma-Gamma 고객당 평균 음의 로그우도(Fader & Hardie 2013). 인자는 log(p, q, v), ``m``은 반복 거래 평균액."""
    p, q, v = np.exp(log_params)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v) + (p * x - 1) * np.log(m)
          + (p * x) * np.log(x) - (p * x + q) * np.log(x * m + v))
    return -ll.mean() + penalizer * np.sum(log_params ** 2)


class BGNBD:
    """BG/NBD(Beta-Geometric/NBD) 구매 횟수 모델(시간 단위: 주)."""

    def __init__(self, params: dict | None = None):
        self.params = params

    def fit(self, frequency, recency, T, penalizer: float = 0.0) -> dict:
        """r, alpha, a, b를 최우추정해 ``params``에 저장하고 수렴 정보를 반환한다."""
        x, tx, T = (np.asarray(values, dtype=float) for values in (frequency, recency, T))
        start = np.log([1.0, max(float(np.mean(T)), 1e-3), 1.0, 1.0])
        log_params, info = _fit(_bgnbd_negll, start, (x, tx, T, penalizer))
        self.params = dict(zip(("r", "alpha", "a", "b"), map(float, np.exp(log_params))))
        return info

    def expected_purchases(self, t: float, frequency, recency, T) -> np.ndarray:
        """조건부 기대 거래 수 E[transactions in (T, T + t] | x, t_x, T] (Fader, Hardie & Lee 2005, eq. 10)."""
        r, alpha, a, b = (self.params[name] for name in ("r", "alpha", "a", "b"))
        x, tx, T = (np.asarray(values, dtype=float) for values in (frequency, recency, T))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
            ratio = np.exp((r + x) * np.log((alpha + T) / (alpha + T + t)))
            # a <= 1이면 생애 전체 기대 거래 수는 발산하지만, 유한 기간 t의 조건부 기대값은 유한하고
            # 이 식은 a < 1에서도 그대로 성립한다(Monte Carlo로 확인).
            numerator = (a + b + x - 1) / (a - 1) * (1 - ratio * hyp)
            # 분모의 생존 odds 항은 로그로 계산해 넘침을 막고, x = 0이면 항이 없다(분모 1).
            log_alive_odds = np.log(a) - np.log(np.maximum(b + x - 1, 1e-300)) + (r + x) * np.log((alpha + T) / (alpha + tx))
            denominator = 1 + np.where(x > 0, np.exp(log_alive_odds), 0.0)
            return numerator / denominator

    def probability_alive(self, frequency, recency, T) -> np.ndarray:
        """``t0``에 아직 이탈하지 않았을 확률. BG/NBD는 반복 거래 직후에만 이탈하므로 x = 0이면 1이다."""
        r, alpha, a, b = (self.params[name] for name in ("r", "alpha", "a", "b"))
        x, tx, T = (np.asarray(values, dtype=float) for values in (frequency, recency, T))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_odds = np.log(a) - np.log(np.maximum(b + x - 1, 1e-300)) + (r + x) * np.log((alpha + T) / (alpha + tx))
            return np.where(x > 0, 1 / (1 + np.exp(log_odds)), 1.0)


class GammaGamma:
    """Gamma-Gamma 거래당 결제액 모델(금액은 원천 최소 단위)."""

    def __init__(self, params: dict | None = None):
        self.params = params

    def fit(self, frequency, monetary_value, penalizer: float = 0.0) -> dict:
        """반복 거래 고객의 ``frequency``, ``monetary_value``로 p, q, v를 최우추정하고 수렴 정보를 반환한다."""
        x, m = np.asarray(frequency, dtype=float), np.asarray(monetary_value, dtype=float)
        # 금액을 평균으로 나눠 수치를 안정시킨 뒤 적합하고, 척도 모수 v만 원래 단위로 되돌린다.
        scale = float(np.mean(m))
        log_params, info = _fit(_gg_negll, np.log([1.0, 2.0, 1.0]), (x, m / scale, penalizer))
        p, q, v = map(float, np.exp(log_params))
        self.params = {"p": p, "q": q, "v": v * scale}
        info["amount_scale"] = scale
        return info

    def population_mean(self) -> float:
        """모집단 평균 거래 금액 ``p * v / (q - 1)``. ``q <= 1``이면 평균이 유한하지 않아 NaN이다."""
        p, q, v = (self.params[name] for name in ("p", "q", "v"))
        return p * v / (q - 1) if q > 1 else float("nan")

    def expected_value(self, frequency, monetary_value) -> np.ndarray:
        """조건부 기대 거래 금액 E[M | x, m] = p (v + x m) / (p x + q - 1). x = 0이면 모집단 평균이다."""
        p, q, v = (self.params[name] for name in ("p", "q", "v"))
        x = np.asarray(frequency, dtype=float)
        m = np.nan_to_num(np.asarray(monetary_value, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, p * (v + x * m) / (p * x + q - 1), self.population_mean())


class ProbabilisticCLV:
    """BG/NBD 기대 거래 수 × Gamma-Gamma 기대 금액으로 52주 CLV(할인 없음)를 낸다."""

    name = NAME

    def __init__(self, calibration: str = "feature_window", period: str = "D", no_purchase_fallback: str = "zero",
                 bgnbd_penalizer: float = 0.0, gg_penalizer: float = 0.0):
        self.calibration, self.period, self.no_purchase_fallback = calibration, period, no_purchase_fallback
        self.bgnbd_penalizer, self.gg_penalizer = float(bgnbd_penalizer), float(gg_penalizer)
        self.bgnbd, self.gamma_gamma = BGNBD(), GammaGamma()
        self.fallback_value = 0.0
        self.diagnostics: dict = {}

    def fit(self, orders: pd.DataFrame, t0, customers, labels: pd.Series | None = None) -> "ProbabilisticCLV":
        """``customers``(학습 분할)의 보정 기간 주문으로 적합한다. ``labels``는 ``train_mean`` 대체값에만 쓴다.

        적용 조건은 ``diagnostics["checks"]``에 남긴다: 최적화 수렴, ``q > 1``(Gamma-Gamma 평균 유한, 필수),
        ``a > 1``(참고용), |corr(frequency, monetary)| <= 0.3(Gamma-Gamma 독립 가정). 최소 인원 미달이면 오류다.
        """
        customers = pd.Index(customers, name="customer_id")
        summary = rfm_summary(orders, t0, self.calibration, self.period).reindex(customers)
        # 보정 기간에 첫 거래(출생)가 없는 고객은 BG/NBD가 정의되지 않으므로 적합에서 뺀다.
        buyers = summary.dropna(subset=["frequency"])
        if len(buyers) < MIN_BGNBD_CUSTOMERS:
            raise ValueError(f"BG/NBD needs at least {MIN_BGNBD_CUSTOMERS} customers with a calibration purchase")
        bgnbd_info = self.bgnbd.fit(buyers["frequency"], buyers["recency"], buyers["T"], self.bgnbd_penalizer)
        # Gamma-Gamma는 반복 거래가 있고 금액이 양수인 고객으로만 적합한다.
        repeat = buyers.loc[(buyers["frequency"] > 0) & (buyers["monetary_value"] > 0)]
        if len(repeat) < MIN_GG_CUSTOMERS:
            raise ValueError(f"Gamma-Gamma needs at least {MIN_GG_CUSTOMERS} repeat customers with positive spend")
        gg_info = self.gamma_gamma.fit(repeat["frequency"], repeat["monetary_value"], self.gg_penalizer)
        non_buyers = summary.index[summary["frequency"].isna()]
        # train_mean: 보정 기간 구매가 없는 학습 고객들의 평균 정답을 대체값으로 쓴다.
        if self.no_purchase_fallback == "train_mean":
            if labels is None:
                raise ValueError("train_mean fallback needs train labels")
            values = labels.reindex(non_buyers).dropna().astype(float)
            self.fallback_value = float(values.mean()) if len(values) else 0.0
        # 표본이 적거나 값이 모두 같으면 상관계수가 정의되지 않아 None으로 둔다.
        correlation =(float(np.corrcoef(repeat["frequency"], repeat["monetary_value"])[0, 1])
                       if len(repeat) > 2 and repeat["frequency"].nunique() > 1 and repeat["monetary_value"].nunique() > 1
                       else None)
        self.diagnostics = {
            "cutoff": to_cutoff(t0).isoformat(), "calibration": self.calibration, "transaction_period": self.period,
            "fit_customers": int(len(customers)), "customers_with_purchase": int(len(buyers)),
            "customers_without_purchase": int(len(non_buyers)), "repeat_customers": int(len(repeat)),
            "bgnbd": {"params": self.bgnbd.params, **bgnbd_info},
            "gamma_gamma": {"params": self.gamma_gamma.params, **gg_info,
                            "population_mean_value": self.gamma_gamma.population_mean()},
            "frequency_monetary_correlation": correlation,
            # a_gt_1은 참고용이다(a <= 1이어도 52주 조건부 기대값은 유한). q_gt_1은 필수 조건이다.
            "checks": {
                "bgnbd_converged": bgnbd_info["converged"], "gamma_gamma_converged": gg_info["converged"],
                "a_gt_1": self.bgnbd.params["a"] > 1, "q_gt_1": self.gamma_gamma.params["q"] > 1,
                "independence_ok": None if correlation is None else abs(correlation) <= CORRELATION_LIMIT,
                "correlation_limit": CORRELATION_LIMIT,
            },
            "no_purchase_fallback": self.no_purchase_fallback, "fallback_value": self.fallback_value,
        }
        return self

    def predict(self, orders: pd.DataFrame, t0, customers) -> pd.DataFrame:
        """고객별 52주 예측값과 중간 값(기대 거래 수, 기대 금액, 생존 확률), ``status``를 반환한다."""
        customers = pd.Index(customers, name="customer_id")
        summary = rfm_summary(orders, t0, self.calibration, self.period).reindex(customers)
        bought = summary["frequency"].notna().to_numpy()
        # 구매 없는 고객은 계산용 자리값(x = 0, T = 1)으로 채우고, 결과는 아래에서 NaN·대체값으로 덮는다.
        x = summary["frequency"].fillna(0).to_numpy()
        tx = summary["recency"].fillna(0).to_numpy()
        T = summary["T"].fillna(1).to_numpy()
        transactions = np.where(bought, self.bgnbd.expected_purchases(LABEL_WEEKS, x, tx, T), np.nan)
        value = np.where(bought, self.gamma_gamma.expected_value(x, summary["monetary_value"]), np.nan)
        alive = np.where(bought, self.bgnbd.probability_alive(x, tx, T), np.nan)
        clv = transactions * value
        valid = np.isfinite(clv) & (clv >= 0)
        # 보정 기간 구매 없음은 대체값, 수치 오류(무한대·음수)는 0으로 예측한다.
        status = np.where(~bought, "no_purchase_in_calibration", np.where(valid, "modeled", "numerical_failure"))
        prediction = np.where(status == "modeled", clv, np.where(status == "no_purchase_in_calibration",
                                                                 self.fallback_value, 0.0))
        result = summary.assign(expected_transactions_52w=transactions, expected_transaction_value=value,
                                probability_alive=alive, predicted_clv_12m=prediction, status=status)
        return result

    def to_dict(self) -> dict:
        """설정, 모수, 진단 정보를 JSON으로 저장할 dict로 반환한다."""
        return {"model": NAME, "calibration": self.calibration, "transaction_period": self.period,
                "no_purchase_fallback": self.no_purchase_fallback, "fallback_value": self.fallback_value,
                "bgnbd_penalizer": self.bgnbd_penalizer, "gg_penalizer": self.gg_penalizer,
                "bgnbd_params": self.bgnbd.params, "gamma_gamma_params": self.gamma_gamma.params,
                "horizon_weeks": LABEL_WEEKS, "time_unit": "week", "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, values: dict) -> "ProbabilisticCLV":
        """``to_dict`` 결과로 모델을 복원한다."""
        if values.get("model") != NAME:
            raise ValueError("not a BG/NBD + Gamma-Gamma model file")
        model = cls(values["calibration"], values["transaction_period"], values["no_purchase_fallback"],
                    values["bgnbd_penalizer"], values["gg_penalizer"])
        model.bgnbd, model.gamma_gamma = BGNBD(values["bgnbd_params"]), GammaGamma(values["gamma_gamma_params"])
        model.fallback_value, model.diagnostics = float(values["fallback_value"]), values.get("diagnostics", {})
        return model

    def save(self, directory: str | Path) -> Path:
        """``bgnbd_gg.json``으로 저장하고 그 경로를 반환한다."""
        return write_json(Path(directory) / "bgnbd_gg.json", self.to_dict())

    @classmethod
    def load(cls, directory: str | Path) -> "ProbabilisticCLV":
        """``bgnbd_gg.json``에서 모델을 불러온다."""
        return cls.from_dict(read_json(Path(directory) / "bgnbd_gg.json"))
