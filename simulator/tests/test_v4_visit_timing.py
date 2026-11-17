"""Native schedule support, strict training clocks and analytic/runtime agreement."""
from collections import Counter
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl.fit_visit_timing import (
    CUTOFF, HORIZON_SECONDS, _pack, fit_visit_timing, first_session_share,
    sample_session_offsets, validate_model,
)


def _users(train, count):
    return [str(i) for i in range(1000) if (hashlib.sha256(f'42:{i}'.encode()).digest()[0] < 205) == train][:count]


@pytest.fixture
def source():
    a, b, c, d = _users(True, 4)
    heldout = _users(False, 1)[0]
    sequences = {
        a: ['2019-10-01T09:00Z', '2019-10-01T09:00Z', '2019-10-01T10:00Z',
            '2019-11-05T08:00Z', '2019-11-05T08:00Z', '2019-11-10T16:00Z', '2019-12-01T16:00Z'],
        b: ['2019-10-10T18:00Z', '2019-11-01T18:00Z', '2019-12-29T20:00Z'],
        c: ['2019-10-01T20:00Z', '2019-12-05T15:00Z'],
        d: ['2019-12-30T05:00Z', '2020-02-01T05:00Z'],
        heldout: ['2019-10-02T23:00Z', '2019-10-31T17:00Z', '2019-11-30T16:00Z'],
    }
    rows = [dict(user_id=uid, session=f'{uid}:{i}', at=pd.Timestamp(at))
            for uid, times in sequences.items() for i, at in enumerate(times)]
    return pd.DataFrame(rows), (a, b, c, d, heldout)


def _fit(rows):
    return fit_visit_timing(rows, '2019-10-01T00:00:00Z', '2020-03-01T00:00:00Z')


def _pid(uid):
    return hashlib.sha256(f'cosmetics:{uid}'.encode()).hexdigest()[:16]


def test_fit_uses_native_joint_count_time_ties_and_strict_available_training_users(source):
    rows, (a, b, c, d, heldout) = source
    model = _fit(rows)
    validate_model(model)
    assert model['regular_anchors'] == ['2019-10-31T15:00:00+00:00', '2019-11-30T15:00:00+00:00']
    assert _pid(d) not in model['entry_profile_banks']  # Entry window ends exactly at cutoff.
    assert _pid(heldout) not in model['entry_profile_banks']
    assert _pid(heldout) not in model['regular_profile_banks']
    entry = model['entry_profile_banks'][_pid(a)]['offsets_seconds']
    assert entry == [[9 * 3600., 9 * 3600., 10 * 3600.]]  # Native day phase and ties survive.
    assert set(model['regular_profile_banks'][_pid(c)]) == {'0'}
    assert pd.Timestamp(model['training']['max_window_end_at']) < pd.Timestamp(CUTOFF)
    assert model['training']['entry_duplicate_start_clocks'] == 1
    assert model['training']['regular_duplicate_start_clocks'] == 1
    assert model['training']['source_training_users'] == 3


def test_future_post_cutoff_changes_cannot_change_fitted_bank(source):
    rows, (a, *_) = source
    original = _fit(rows)
    future = pd.DataFrame([dict(user_id=a, session='future', at=pd.Timestamp('2020-02-10T00:00:00Z'))])
    assert _fit(pd.concat([rows, future], ignore_index=True)) == original


class ExactDraw:
    def __init__(self, value):
        self.value = value
    def integers(self, high):
        assert 0 <= self.value < high
        return self.value


def test_sampler_preserves_empirical_frequency_and_correlated_whole_schedule(source):
    model = _fit(source[0])
    bank = _pack(Counter({(0., 10., 10.): 1, (28 * 86400.,): 2}))
    model['regular_profile_banks']['p'] = {'1': bank}
    validate_model(model)
    draws = [sample_session_offsets(model, 'p', False, True, ExactDraw(i)) for i in range(3)]
    assert Counter(tuple(offsets) for offsets, _ in draws) == Counter({(0., 10., 10.): 1, (28 * 86400.,): 2})
    assert all(scope == 'regular_profile_active30_1' for _, scope in draws)
    assert first_session_share(model, 'p', True, 28) == pytest.approx(1 / 3)
    assert sum(offsets[0] < 28 * 86400 for offsets, _ in draws) / 3 == first_session_share(model, 'p', True, 28)
    assert first_session_share(model, 'p', True, 30) == 1
    assert first_session_share(model, 'p', True, 0) == 0
    draws[0][0].clear()
    assert bank['offsets_seconds'][0] == [0., 10., 10.]  # Caller cannot mutate trained evidence.


def test_fallback_uses_only_matching_mode_and_observed_prior_activity(source):
    model = _fit(source[0])
    assert sample_session_offsets(model, 'unknown', True, False, np.random.default_rng(3))[1] == 'entry_pool'
    assert sample_session_offsets(model, 'unknown', False, False, np.random.default_rng(3))[1] == 'regular_active30_0_pool'
    del model['regular_stratum_banks']['0']
    assert sample_session_offsets(model, 'unknown', False, False, np.random.default_rng(3))[1] == 'regular_pool'


def test_population_schedules_drop_donor_keys_and_preserve_visit_counts(source):
    from simulate.v4.simulator.population_timing import aggregate_visit_timing, materialize_visit_timing

    original = _fit(source[0])
    calibration = aggregate_visit_timing(original)
    assert not {"entry_profile_banks", "regular_profile_banks", "entry_pool", "regular_pool"} & calibration.keys()
    source_metadata = {"dataset": "fixture"}
    generated = materialize_visit_timing(calibration, source_metadata)
    assert generated == materialize_visit_timing(calibration, source_metadata)
    assert not generated["entry_profile_banks"] and not generated["regular_profile_banks"]
    expected = sum(len(offsets) * n for offsets, n in zip(original["entry_pool"]["offsets_seconds"],
                                                       original["entry_pool"]["frequencies"]))
    result = sum(len(offsets) * n for offsets, n in zip(generated["entry_pool"]["offsets_seconds"],
                                                     generated["entry_pool"]["frequencies"]))
    assert (result / generated["entry_pool"]["observed_windows"]
            == pytest.approx(expected / original["entry_pool"]["observed_windows"]))

    profiles = [{"home_group": "group", "profile_id": _pid(uid)} for uid in source[1]]
    by_group = materialize_visit_timing(aggregate_visit_timing(original, profiles), source_metadata)
    assert by_group["profile_key_scope"] == "synthetic_home_group"
    assert set(by_group["entry_profile_banks"]) == set(by_group["regular_profile_banks"]) == {"group"}


@pytest.mark.parametrize('bad', ['frequency', 'empty', 'order', 'nan', 'horizon', 'total', 'maturity', 'schema'])
def test_invalid_source_bank_is_rejected(source, bad):
    model = deepcopy(_fit(source[0])); bank = model['entry_pool']
    if bad == 'frequency':bank['frequencies'][0] = True
    elif bad == 'empty':bank['offsets_seconds'][0] = []
    elif bad == 'order':bank['offsets_seconds'][0] = [10., 1.]
    elif bad == 'nan':bank['offsets_seconds'][0] = [float('nan')]
    elif bad == 'horizon':bank['offsets_seconds'][0] = [float(HORIZON_SECONDS)]
    elif bad == 'total':bank['observed_windows'] = True
    elif bad == 'maturity':model['training']['max_window_end_at'] = CUTOFF
    else:model['schema_version'] = True
    with pytest.raises(ValueError):validate_model(model)


@pytest.mark.parametrize('days', [True, -1, 31, float('nan')])
def test_invalid_analytic_horizon_is_rejected(source, days):
    with pytest.raises(ValueError, match='outcome_days'):
        first_session_share(_fit(source[0]), 'unknown', True, days)


def test_shorter_outcome_uses_exact_sampler_bank_and_preserves_factual_records(source):
    from simulate.v4.simulator.outcomes import redefine_outcomes
    from test_v4_outcome_definitions import outcome_fixture
    tables, metadata, old_model = outcome_fixture()
    timing = _fit(source[0])
    timing['regular_profile_banks']['pA'] = {'1': _pack(Counter({(0.,): 1, (28 * 86400.,): 2}))}
    timing['regular_profile_banks']['pB'] = {'1': _pack(Counter({(28 * 86400.,): 1}))}
    timing['regular_stratum_banks']['1'] = _pack(Counter({(1.,): 1, (29 * 86400.,): 1}))
    before = tables['diagnostics'].copy(deep=True)
    view, info = redefine_outcomes(tables, metadata, shopping_model=old_model, visit_timing_model=timing)
    shares = np.asarray([1 / 3, 0, .5, .5])
    np.testing.assert_allclose(view['diagnostics'].first_session_within_28d_share, shares)
    np.testing.assert_allclose(view['diagnostics'].no_coupon_return_probability_28d, shares * before.no_coupon_return_probability_30d)
    assert 'first_native_session_offset' in info['analytic_outcome_view']['formula']
    assert 'one_second_fallback_customers' not in info['analytic_outcome_view']
    pd.testing.assert_frame_equal(tables['diagnostics'], before)
    for name in ('customers', 'products', 'sessions', 'events', 'orders', 'order_items', 'coupons', 'treatments'):
        assert view[name] is tables[name]
    assert view['labels'].churn.tolist() == [0, 1, 1, 0]
    legacy, original = redefine_outcomes(tables, metadata, shopping_model=old_model)
    assert 'positive_idle_gap' in original['analytic_outcome_view']['formula']
    assert legacy['diagnostics'].first_session_within_28d_share.tolist() != shares.tolist()
