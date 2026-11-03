import numpy as np
import pandas as pd

from analysis.feature_selection import make_panel, order_events, split_panel


def test_features_are_past_only_and_end_is_censored():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({
        'user_id': ['u', 'u', 'u'], 'user_session': ['s1', 's1', 's2'],
        'event_time': [t, t + pd.Timedelta(minutes=5), t + pd.Timedelta(hours=2)],
        'event_type': ['view', 'cart', 'purchase'], 'price': [10., 20., 30.],
        'product_id': ['a', 'b', 'c'], 'category_id': ['x', 'x', 'y'], 'brand': ['b', 'b', 'c'],
    })
    panel, _ = make_panel(events, t + pd.Timedelta(hours=2, minutes=10))
    assert not any('unique_brand' in col for col in panel.columns)
    assert panel.target.tolist() == ['cart', 'end']
    cart, end = [row for _, row in panel.iterrows()]
    assert cart.lag1_view == 1 and cart.previous_price == 10 and cart.session_cart_count == 0
    assert cart.session_view_unique_product_id == 1 and cart.session_cart_unique_product_id == 0
    assert cart.session_view_event_price_mean == 10
    assert end.lag1_cart == 1 and end.session_cart_count == 1
    assert end.session_cart_unique_product_id == 1 and end.session_cart_event_price_sum == 20
    assert cart['at'] == t and end['at'] == t + pd.Timedelta(minutes=5)
    assert end.session_elapsed_minutes == 5


def test_short_inter_session_gap_does_not_create_end():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({'user_id': ['u', 'u'], 'user_session': ['s1', 's2'],
        'event_time': [t, t + pd.Timedelta(minutes=10)], 'event_type': ['view', 'cart'],
        'price': [1., 2.], 'product_id': ['a', 'b'], 'category_id': ['x', 'x'], 'brand': ['b', 'b']})
    panel, _ = make_panel(events, t + pd.Timedelta(hours=1))
    assert panel.target.tolist() == ['end']


def test_split_users_and_time_are_disjoint():
    users = [f'user-{i}' for i in range(100)]
    panel = pd.DataFrame({'user_id': users * 3,
        'at': pd.to_datetime(['2020-01-01'] * 100 + ['2020-01-14'] * 100 + ['2020-01-20'] * 100, utc=True)})
    panel['outcome_at'] = panel['at']
    masks, _ = split_panel(panel)
    sets = [set(panel.loc[m, 'user_id']) for m in masks]
    assert all(sets)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_next_event_time_and_attributes_do_not_change_current_state():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({'user_id': ['u', 'u'], 'user_session': ['s', 's'],
        'event_time': [t, t + pd.Timedelta(minutes=5)], 'event_type': ['view', 'cart'],
        'price': [1., 2.], 'product_id': ['a', 'b'], 'category_id': ['x', 'y'], 'brand': ['b', 'c']})
    first, _ = make_panel(events, t + pd.Timedelta(hours=2))
    events.loc[1, ['event_time', 'event_type', 'price']] = [t + pd.Timedelta(minutes=10), 'purchase', 99.]
    second, _ = make_panel(events, t + pd.Timedelta(hours=2))
    excluded = ['target', 'outcome_at']
    pd.testing.assert_series_equal(first.iloc[0].drop(excluded), second.iloc[0].drop(excluded))


def test_ambiguous_timestamp_barrier_does_not_create_transition_or_end():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({'user_id': ['u', 'u'], 'user_session': ['s', 's'],
        'event_time': [t, t + pd.Timedelta(minutes=5)], 'event_type': ['view', 'cart'],
        'price': [1., 2.], 'product_id': ['a', 'b'], 'category_id': ['x', 'y'], 'brand': ['b', 'c'],
        'order_block': [0, 2], 'safe_start': [True, False], 'safe_end': [False, True]})
    panel, _ = make_panel(events, t + pd.Timedelta(hours=1))
    assert panel.target.tolist() == ['end']
    assert panel.iloc[0].lag1_cart == 1
    assert panel.iloc[0].previous_session_exists == 0
    assert panel.iloc[0].history_truncated == 1 and panel.iloc[0].session_left_truncated == 1


def test_same_action_ties_collapse_but_mixed_ties_cut_history():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({'user_id': ['u'] * 6, 'user_session': ['s'] * 6,
        'event_time': [t, t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=1),
                       t + pd.Timedelta(minutes=2), t + pd.Timedelta(minutes=3)],
        'event_type': ['purchase', 'purchase', 'view', 'cart', 'view', 'cart'],
        'price': [1.] * 6, 'product_id': ['a'] * 6, 'category_id': ['x'] * 6, 'brand': ['b'] * 6})
    ordered, collapsed, removed = order_events(events)
    assert collapsed == 1 and removed == {'view': 1, 'cart': 1}
    assert ordered.event_type.tolist() == ['purchase', 'view', 'cart']
    assert ordered.iloc[0].batch_size == 2
    assert pd.isna(ordered.iloc[0].price)
    panel, _ = make_panel(ordered, t + pd.Timedelta(hours=1))
    assert panel.target.tolist() == ['cart', 'end']
    assert panel.iloc[0].lag1_view == 1 and panel.iloc[0].previous_session_exists == 0


def test_purchase_batch_size_is_visible_only_after_the_batch():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({'user_id': ['u'] * 3, 'user_session': ['s'] * 3,
        'event_time': [t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=1)],
        'event_type': ['view', 'purchase', 'purchase'], 'price': [1.] * 3,
        'product_id': ['a', 'b', 'c'], 'category_id': ['x'] * 3, 'brand': ['b'] * 3})
    ordered, _, _ = order_events(events)
    panel, _ = make_panel(ordered, t + pd.Timedelta(hours=1))
    before, after = panel.iloc[0], panel.iloc[1]
    assert before.target == 'purchase' and before.session_purchase_event_count == 0
    assert after.target == 'end' and after.session_purchase_event_count == 2
    assert after.previous_purchase_batch_size == 2


def test_sequence_history_uses_only_observed_events():
    t = pd.Timestamp('2020-01-01', tz='UTC')
    events = pd.DataFrame({'user_id': ['u'] * 4, 'user_session': ['s'] * 4,
        'event_time': [t + pd.Timedelta(minutes=i) for i in (0, 2, 5, 9)],
        'event_type': ['view', 'cart', 'view', 'purchase'], 'price': [10., 10., 12., 99.],
        'product_id': ['a', 'a', 'a', 'b'], 'category_id': ['x', 'x', 'y', 'z'], 'brand': ['b'] * 4})
    panel, _ = make_panel(events, t + pd.Timedelta(hours=1))
    row = panel.loc[panel.target.eq('purchase')].iloc[0]
    assert row.prior_pair_cart_view == 1
    assert row.previous_inter_event_gap_minutes == 3
    assert row.previous_item_prior_count == 2 and row.previous_item_recency_minutes == 3
    assert row.previous_item_prior_view_count == 1 and row.previous_item_prior_purchase_count == 0
    assert pd.isna(row.session_past_purchase_minus_view_price)
    assert row.previous_category_switch == 1
    assert row.observed_cart_event_balance == 1
    assert not any('unique_brand' in col for col in panel.columns)
