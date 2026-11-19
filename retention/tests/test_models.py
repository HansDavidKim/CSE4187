import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from retention.models import make_model, fit_model, predict
from retention.sequence import RecurrentEncoder, SequenceModel


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_padding_is_not_a_real_event(kind):
    torch.manual_seed(7)
    model = RecurrentEncoder(hidden=8, kind=kind).eval()
    short, long = torch.randn(2, 8), torch.randn(6, 8)
    alone = model(short[None], torch.tensor([2]))
    together = model(pad_sequence([short, long], batch_first=True), torch.tensor([2, 6]))
    torch.testing.assert_close(alone[0], together[0])


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_sequence_training_and_train_only_scaler(kind, tmp_path):
    x = [np.ones((2, 8), dtype=np.float32) * i for i in range(1, 9)]
    y = np.array([0, 1] * 4)
    validation = [np.ones((3, 8), dtype=np.float32) * 100 for _ in range(4)]
    model = SequenceModel(kind=kind, hidden=8, batch_size=4).fit(x, y, (validation, np.array([0, 1, 0, 1])), epochs=2)
    np.testing.assert_allclose(model.scaler.mean_, 4.5)
    p = model.predict_proba(x)
    assert p.shape == (8,) and np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    path = tmp_path / f"{kind}.pt"
    model.save(path)
    restored = SequenceModel.load(path)
    np.testing.assert_allclose(restored.predict_proba(x), p, atol=1e-7)


@pytest.mark.parametrize("name", ["lightgbm", "catboost", "xgboost"])
def test_library_models_fit_with_missing_values(name):
    x = pd.DataFrame({"value": np.arange(40, dtype=float), "missing": [np.nan] * 40})
    y = np.array([0, 0, 0, 1] * 10)
    model = fit_model(make_model(name, {"iterations": 5}), x, y, name)
    p = predict(model, x)
    assert len(p) == 40 and np.isfinite(p).all()
