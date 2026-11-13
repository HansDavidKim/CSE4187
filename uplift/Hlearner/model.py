"""H-learner (Liang, van der Laan, Alaa, AISTATS 2026; uplift/H_learner.pdf).

1단계: T-learner로 mu0, mu1을 학습하고 가상 효과 Y_phi를 만든다 (논문 Table 1).
    X 방식   Y_phi = T (Y - mu0(X)) + (1 - T) (mu1(X) - Y)
    DR 방식  Y_phi = (T - e) / (e (1 - e)) * (Y - mu_T(X)) + mu1(X) - mu0(X)
2단계: 두 함수 f0, f1을 식 (7)의 손실로 함께 학습한다.
    sum_i (1 - lam) * (Y_i - f_{T_i}(X_i))^2 + lam * (f1(X_i) - f0(X_i) - Y_phi,i)^2
    lam = 0이면 T-learner(간접), lam = 1이면 가상 효과 회귀(직접)와 같다.
    tau(x) = f1(x) - f0(x).
lam 선택(논문 A.3): 검증 데이터에서 따로 학습한 결과 모델로 X 방식 가상 효과를
    만들고, f1 - f0와의 제곱 오차가 가장 작은 lam과 반복 횟수를 고른다.

논문과 다른 점: 논문은 신경망(TARNet)으로 f0, f1을 학습한다. 여기서는 결측을
그대로 처리하도록, 같은 손실을 두 함수에 대한 gradient boosting(회귀 트리)으로
최소화한다. 논문과 같이 1·2단계 모두 전체 학습 데이터를 쓴다(표본 분할 없음).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ..common.base import BaseUpliftLearner, as_array, fit_outcome_models, make_classifier


def pseudo_outcome(t, y, mu0, mu1, kind="x", propensity=0.5):
    if kind == "x":
        return t * (y - mu0) + (1 - t) * (mu1 - y)
    if kind == "dr":
        e = propensity
        return (t - e) / (e * (1 - e)) * (y - np.where(t == 1, mu1, mu0)) + mu1 - mu0
    raise ValueError(f"unknown pseudo-outcome: {kind}")


class _JointBoosting:
    """식 (7)을 최소화하는 f0, f1을 Newton boosting으로 함께 학습한다."""

    def __init__(self, lam, n_rounds, learning_rate, max_depth, min_samples_leaf, seed):
        self.lam, self.n_rounds, self.learning_rate = lam, n_rounds, learning_rate
        self.tree_args = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.seed = seed

    def fit(self, X, t, y, y_phi, on_round=None):
        lam = self.lam
        self.init_ = (y[t == 0].mean(), y[t == 1].mean())
        f0, f1 = np.full(len(y), self.init_[0]), np.full(len(y), self.init_[1])
        self.trees_ = []
        # 2차 미분(상수): 자기 군에서는 결과 손실과 효과 손실이 모두, 반대 군에서는 효과 손실만 걸린다.
        h0, h1 = (1 - lam) * (t == 0) + lam, (1 - lam) * (t == 1) + lam
        for r in range(self.n_rounds):
            gap = f1 - f0 - y_phi
            g0 = (1 - lam) * (t == 0) * (f0 - y) - lam * gap
            g1 = (1 - lam) * (t == 1) * (f1 - y) + lam * gap
            pair = []
            for g, h in ((g0, h0), (g1, h1)):
                step = np.divide(-g, h, out=np.zeros_like(g), where=h > 0)
                tree = DecisionTreeRegressor(random_state=self.seed + r, **self.tree_args)
                pair.append(tree.fit(X, step, sample_weight=h))
            f0 += self.learning_rate * pair[0].predict(X)
            f1 += self.learning_rate * pair[1].predict(X)
            self.trees_.append(pair)
            if on_round is not None:
                on_round(r, pair)
        return self

    def predict(self, X, n_rounds=None):
        f0, f1 = np.full(len(X), self.init_[0]), np.full(len(X), self.init_[1])
        for t0, t1 in self.trees_[:n_rounds]:
            f0 += self.learning_rate * t0.predict(X)
            f1 += self.learning_rate * t1.predict(X)
        return f0, f1


class HLearner(BaseUpliftLearner):
    name = "H-learner"

    def __init__(self, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), pseudo="x", n_rounds=200,
                 learning_rate=0.1, max_depth=3, min_samples_leaf=50, val_size=0.25,
                 make_model=make_classifier, random_state=0):
        self.lambdas, self.pseudo = tuple(lambdas), pseudo
        self.n_rounds, self.learning_rate = n_rounds, learning_rate
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.val_size, self.make_model, self.random_state = val_size, make_model, random_state

    def _booster(self, lam, n_rounds):
        return _JointBoosting(lam, n_rounds, self.learning_rate, self.max_depth,
                              self.min_samples_leaf, self.random_state)

    def _stage1(self, X, t, y, propensity):
        mu0, mu1 = fit_outcome_models(X, t, y, self.make_model)
        y_phi = pseudo_outcome(t, y, mu0.predict_proba(X)[:, 1], mu1.predict_proba(X)[:, 1],
                               self.pseudo, propensity)
        return mu0, mu1, y_phi

    def _select(self, X, t, y, propensity):
        """검증 분할에서 lam과 반복 횟수를 고른다."""
        tr, va = train_test_split(np.arange(len(y)), test_size=self.val_size, stratify=t,
                                  random_state=self.random_state)
        _, _, y_phi = self._stage1(X[tr], t[tr], y[tr], propensity)
        # 검증 기준은 검증 데이터에서 따로 학습한 결과 모델로 만든다.
        # 1단계 모델로 만들면 같은 가상 효과를 맞히는 lam = 1이 항상 유리해진다.
        nu0, nu1 = fit_outcome_models(X[va], t[va], y[va], self.make_model)
        target = pseudo_outcome(t[va], y[va], nu0.predict_proba(X[va])[:, 1],
                                nu1.predict_proba(X[va])[:, 1], "x")
        base = y[tr][t[tr] == 1].mean() - y[tr][t[tr] == 0].mean()  # f1 - f0의 초기값

        best = (np.inf, None, None)
        self.validation_ = {}
        for lam in self.lambdas:
            booster = self._booster(lam, self.n_rounds)
            tau = np.full(len(va), base)
            losses = []

            def on_round(r, pair, tau=tau, losses=losses):
                tau += self.learning_rate * (pair[1].predict(X[va]) - pair[0].predict(X[va]))
                losses.append(np.mean((tau - target) ** 2))

            booster.fit(X[tr], t[tr], y[tr], y_phi, on_round)
            r = int(np.argmin(losses))
            self.validation_[lam] = (losses[r], r + 1)
            if losses[r] < best[0]:
                best = (losses[r], lam, r + 1)
        return best[1], best[2]

    def fit(self, X, t, y, propensity: float = 0.5):
        X, t, y = as_array(X), as_array(t), as_array(y)
        if len(self.lambdas) > 1:
            self.lambda_, self.n_rounds_ = self._select(X, t, y, propensity)
        else:
            self.lambda_, self.n_rounds_ = self.lambdas[0], self.n_rounds
        # 고른 설정으로 전체 학습 데이터에 다시 학습한다.
        self.mu0_, self.mu1_, y_phi = self._stage1(X, t, y, propensity)
        self.booster_ = self._booster(self.lambda_, self.n_rounds_).fit(X, t, y, y_phi)
        return self

    def predict(self, X):
        A = as_array(X)
        f0, f1 = self.booster_.predict(A)
        # f0, f1은 차이가 정확하도록 학습된 중간 함수라 각각은 유지 확률의 좋은 추정이
        # 아닐 수 있다(논문 3절). 그래서 p0는 1단계 mu0로 두고 p1 = p0 + (f1 - f0)로 맞춘다.
        p0 = self.mu0_.predict_proba(A)[:, 1]
        return self._frame(p0, p0 + (f1 - f0), X)
