"""Uplift 모델 평가 (UC-03): Qini curve, Qini 계수, 분위별 실제 uplift."""

import numpy as np
import pandas as pd


def qini_curve(score, t, y) -> pd.DataFrame:
    """점수가 높은 순으로 타기팅할 때의 누적 증분 유지 고객 비율.

    x: 타기팅한 고객 비율, qini: (처치군 유지 수 - 대조군 유지 수 x 처치/대조 인원비) / 전체 처치군 수,
    random: 무작위 타기팅의 기준선.
    """
    score, t, y = (np.asarray(a, dtype=float) for a in (score, t, y))
    order = np.argsort(-score, kind="stable")
    t, y = t[order], y[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    qini = np.concatenate([[0.0], (y_t - y_c * ratio) / t.sum()])
    x = np.arange(len(y) + 1) / len(y)
    return pd.DataFrame({"x": x, "qini": qini, "random": x * qini[-1]})


def qini_coefficient(score, t, y) -> float:
    """Qini curve와 무작위 기준선 사이의 넓이. 클수록 좋고 0이면 무작위와 같다."""
    c = qini_curve(score, t, y)
    return float(np.trapezoid(c["qini"] - c["random"], c["x"]))


def uplift_by_quantile(score, t, y, q: int = 10) -> pd.DataFrame:
    """점수 분위별(1 = 상위) 인원, 평균 예측 점수, 실제 uplift(처치군 유지율 - 대조군 유지율)."""
    df = pd.DataFrame({"score": np.asarray(score, dtype=float), "t": np.asarray(t), "y": np.asarray(y)})
    df["quantile"] = q - pd.qcut(df["score"].rank(method="first"), q, labels=False)
    rows = []
    for k, g in df.groupby("quantile"):
        rows.append({"quantile": k, "n": len(g), "mean_score": g["score"].mean(),
                     "observed_uplift": g.loc[g.t == 1, "y"].mean() - g.loc[g.t == 0, "y"].mean()})
    return pd.DataFrame(rows)


def ate(t, y) -> float:
    """전체 평균 효과: 처치군 유지율 - 대조군 유지율."""
    t, y = np.asarray(t), np.asarray(y, dtype=float)
    return float(y[t == 1].mean() - y[t == 0].mean())


def against_truth(score, true_uplift) -> dict:
    """쌍대 실행 등으로 고객별 실제 uplift를 알 때만 쓰는 지표."""
    score, true_uplift = np.asarray(score, dtype=float), np.asarray(true_uplift, dtype=float)
    rank = pd.Series(score).corr(pd.Series(true_uplift), method="spearman")
    return {"rmse": float(np.sqrt(np.mean((score - true_uplift) ** 2))), "spearman": float(rank)}
