"""X-learner (Künzel et al., 2019).

1단계: 결과 모델 mu0, mu1 학습 (T-learner와 동일).
2단계: 가상 효과를 만들고 각 군에서 회귀한다.
    처치군  D1 = Y - mu0(X)   -> tau1
    대조군  D0 = mu1(X) - Y   -> tau0
3단계: tau(x) = e * tau0(x) + (1 - e) * tau1(x),  e = 배정 확률.
"""

from ..common.base import (BaseUpliftLearner, as_array, fit_outcome_models, make_classifier,
                           make_regressor)


class XLearner(BaseUpliftLearner):
    name = "X-learner"

    def __init__(self, make_model=make_classifier, make_effect_model=make_regressor):
        self.make_model = make_model
        self.make_effect_model = make_effect_model

    def fit(self, X, t, y, propensity: float = 0.5):
        X, t, y = as_array(X), as_array(t), as_array(y)
        self.propensity_ = propensity
        self.mu0_, self.mu1_ = fit_outcome_models(X, t, y, self.make_model)

        X0, X1 = X[t == 0], X[t == 1]
        d1 = y[t == 1] - self.mu0_.predict_proba(X1)[:, 1]
        d0 = self.mu1_.predict_proba(X0)[:, 1] - y[t == 0]
        self.tau1_ = self.make_effect_model().fit(X1, d1)
        self.tau0_ = self.make_effect_model().fit(X0, d0)
        return self

    def predict(self, X):
        A = as_array(X)
        e = self.propensity_
        tau = e * self.tau0_.predict(A) + (1 - e) * self.tau1_.predict(A)
        # tau는 직접 추정한 값이므로, p0는 mu0로 두고 p1 = p0 + tau로 맞춘다.
        p0 = self.mu0_.predict_proba(A)[:, 1]
        return self._frame(p0, p0 + tau, X)
