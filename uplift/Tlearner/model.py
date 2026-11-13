"""T-learner: 대조군과 처치군의 결과 모델을 따로 학습하고 그 차이를 uplift로 쓴다."""

from ..common.base import BaseUpliftLearner, as_array, fit_outcome_models, make_classifier


class TLearner(BaseUpliftLearner):
    name = "T-learner"

    def __init__(self, make_model=make_classifier):
        self.make_model = make_model

    def fit(self, X, t, y, propensity: float = 0.5):
        X, t, y = as_array(X), as_array(t), as_array(y)
        self.mu0_, self.mu1_ = fit_outcome_models(X, t, y, self.make_model)
        return self

    def predict(self, X):
        A = as_array(X)
        p0 = self.mu0_.predict_proba(A)[:, 1]
        p1 = self.mu1_.predict_proba(A)[:, 1]
        return self._frame(p0, p1, X)
