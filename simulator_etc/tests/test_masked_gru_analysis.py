import pandas as pd
import torch

from analysis.persona.sequence import MaskedGRU, padded, rows_from_events, source_data
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


def test_tianchi_uses_daily_combinations_including_coupon_receipts(tmp_path):
    source = tmp_path / 'online.csv'
    pd.DataFrame({
        'User_id': [10, 10, 10, 10, 20],
        'Action': [0, 0, 1, 2, 1],
        'Date_received': [None, None, None, '20160101', None],
        'Date': ['20160101', '20160101', '20160101', None, '20160102'],
    }).to_csv(source, index=False)

    events, quality, actions = source_data('tianchi', source, sample_modulus=1)

    assert actions == [
        'click_only_day', 'purchase_only_day', 'click_purchase_day',
        'coupon_receive_only_day', 'click_receive_day', 'purchase_receive_day',
        'click_purchase_receive_day',
    ]
    assert events[['user', 'action']].values.tolist() == [[10, 6], [20, 1]]
    assert quality['raw_action_counts']['coupon_receive'] == 1
    assert quality['sample_user_days'] == 2


def test_electronics_drop_removes_cosmetics_only_action(tmp_path):
    source = tmp_path / 'cosmetics'
    source.mkdir()
    pd.DataFrame({
        'user_id': [10, 10, 10, 10],
        'event_time': ['2026-01-01 00:00:00 UTC'] * 4,
        'event_type': ['view', 'cart', 'remove_from_cart', 'purchase'],
        'user_session': ['s1'] * 4,
        'product_id': [1, 1, 1, 1],
    }).to_csv(source / '2026-01.csv', index=False)

    events, quality, actions = source_data('electronics_drop', source, sample_modulus=1)

    assert actions == ['view', 'cart', 'purchase']
    assert events.action.tolist() == [0, 1, 2]
    assert quality['dropped_actions'] == ['remove_from_cart']


def test_retail_combined_uses_one_shared_action_vocabulary_and_namespaces_users(tmp_path):
    cosmetics = tmp_path / 'cosmetics'
    electronics = tmp_path / 'electronics'
    cosmetics.mkdir()
    electronics.mkdir()
    columns = {
        'user_id': [10, 10, 10, 10],
        'event_time': ['2026-01-01 00:00:00 UTC'] * 4,
        'event_type': ['view', 'cart', 'remove_from_cart', 'purchase'],
        'user_session': ['s1'] * 4,
        'product_id': [1, 1, 1, 1],
    }
    pd.DataFrame(columns).to_csv(cosmetics / '2026-01.csv', index=False)
    pd.DataFrame({key: value[:3] for key, value in {
        **columns, 'event_type': ['view', 'cart', 'purchase', 'purchase']
    }.items()}).to_csv(electronics / 'events.csv', index=False)

    events, quality, actions = source_data('retail_combined', tmp_path, sample_modulus=1)

    assert actions == ['view', 'cart', 'purchase']
    assert sorted(events.user.unique().tolist()) == [20, 21]
    assert events.groupby('user').action.apply(list).to_dict() == {20: [0, 1, 2], 21: [0, 1, 2]}
    assert quality['user_namespace'].startswith('cosmetics_drop=')
