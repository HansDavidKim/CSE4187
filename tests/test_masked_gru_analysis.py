import pandas as pd
import torch

from analysis.persona.sequence import MaskedGRU, padded, rows_from_events
from utils.masking import mask_tokens


def test_sequence_boundaries_masking_and_embedding():
    events = pd.DataFrame({
        "user": [1, 1, 1],
        "session": [10, 10, 11],
        "action": [0, 1, 2],
    })
    rows = rows_from_events(events, max_users=3, max_tokens=8, n_actions=3, seed=42)
    assert rows == [(1, [0, 1, 3, 2, 3], 2)]
    tokens, lengths = padded([rows[0][1]], 8, pad=4)
    masked, selected = mask_tokens(tokens, lengths, 0.15, torch.Generator().manual_seed(42))
    assert selected[:, :5].any()
    assert not selected[:, 5:].any()
    assert torch.equal(tokens[~selected], masked[~selected])
    model = MaskedGRU(vocab_size=6, pad=4)
    assert model.encode(tokens, lengths).shape == (1, 48)
    assert model(tokens, lengths).shape == (1, 8, 4)
    masked[masked.eq(6)] = 5
    loss = torch.nn.functional.cross_entropy(model(masked, lengths)[selected], tokens[selected])
    loss.backward()
    assert model.readout.weight.grad is not None
