"""Native schedules preserve ties, clipping and raw/accounting clocks."""
import copy
import json

import pandas as pd
import pytest

from test_v4_checkout_simulator import checkout_fixture, _customers
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS, validate_raw_dataset


@pytest.mark.parametrize('ends', [True, False])
def test_joint_timing_keeps_visits_without_resampling_or_overlap(checkout_fixture, tmp_path, monkeypatch, ends):
    from simulate.v4.irl import fit_latent_shopping as latent, fit_visit_timing as timing
    config, model = copy.deepcopy(checkout_fixture)
    model['latent_shopping'] = dict(type='fixture', discount=.95, regularization=.01, state_count=1,
        mixture_weights=[1.], feature_names=[])
    for holder in [model['distributions'], *model['home_groups'].values(), *model['profiles']]:
        holder['session_counts_active_30d'] = ['must_not_sample']
        holder['idle_gap_seconds'] = ['must_not_sample']
        holder['choice_gap_seconds'] = [5.]
    path, bank = tmp_path/'actor.json', tmp_path/'timing.json'
    path.write_text(json.dumps(model))
    bank.write_text(json.dumps({'source': model['source'], 'type': 'fixture'}))
    config['source_simulation'].update(model_path=str(path), visit_timing_model_path=str(bank), max_events_per_session=4)
    calls, contexts = [], []
    monkeypatch.setattr(latent, 'validate_model', lambda m: None)
    monkeypatch.setattr(latent, 'sample_component', lambda m, p, rng: (0, 'population_prior'))
    def policy(m, ctx, component):
        contexts.append(ctx.copy())
        return {a:float(a == ('end' if ends and ctx['session_decision_count'] else 'view')) for a in ['view','cart','remove','purchase','end']}
    monkeypatch.setattr(latent, 'policy', policy)
    monkeypatch.setattr(timing, 'validate_model', lambda m: None)
    def schedule(m, profile_id, entry, active_previous30, rng):
        calls.append((profile_id, entry, active_previous30))
        return [18000., 18000., 18005., 29*86400.], 'fixture'
    monkeypatch.setattr(timing, 'sample_session_offsets', schedule)
    monkeypatch.setattr(timing, 'first_session_share', lambda *args: 1.)
    result = EcommerceSimulator(config).generate(months=3, seed=17, customers=_customers())
    sessions = result.tables['sessions']
    for _, group in sessions.groupby('customer_id', sort=False):
        ordered = group.sort_values(['started_at', 'session_id'])
        first = ordered.head(3)
        assert first.started_at.tolist() == [pd.Timestamp('2024-01-10T05:00:00Z')]*2 + [pd.Timestamp('2024-01-10T05:00:05Z')]
        assert (ordered.started_at.iloc[1:].to_numpy() >= ordered.ended_at.iloc[:-1].to_numpy()).all()
    assert sum(entry for _, entry, _ in calls) == 8
    diagnostic = result.tables['session_diagnostics']
    assert diagnostic.termination_reason.eq('causal_policy_end').all() if ends else diagnostic.termination_reason.eq('scheduled_next_start').any()
    history = [c['mdp_history_completed_session_count_30d'] for c in contexts]
    assert max(history) > 0 if ends else max(history) == 0
    assert result.tables['diagnostics'].first_session_within_28d_share.eq(1.).all()
    assert '-joint-timing-' in result.metadata['dataset_id']
    assert result.metadata['visit_timing']['scope_counts_all_attempts']['fixture'] == len(calls)
    validate_raw_dataset({k:result.tables[k] for k in RAW_COLUMNS}, result.metadata)


def test_visit_timing_rejects_a_different_native_source(checkout_fixture, tmp_path, monkeypatch):
    from simulate.v4.irl import fit_visit_timing as timing
    config, _ = copy.deepcopy(checkout_fixture)
    path = tmp_path/'wrong_timing.json'
    path.write_text(json.dumps({'source': {'sample_modulus': -1}, 'type':'fixture'}))
    config['source_simulation']['visit_timing_model_path'] = str(path)
    monkeypatch.setattr(timing, 'validate_model', lambda m: None)
    with pytest.raises(ValueError, match='exact native source sample'):
        EcommerceSimulator(config).generate(months=3, customers=_customers())


def test_generation_identity_covers_customer_count_and_supplied_input(checkout_fixture):
    config, model = copy.deepcopy(checkout_fixture)
    sim = EcommerceSimulator(config)
    first = sim.generate(months=3, seed=17, customers=_customers())
    repeat = sim.generate(months=3, seed=17, customers=_customers())
    larger = sim.generate(months=3, seed=17, customers=_customers(10))
    changed = _customers(); changed.loc[0, 'customer_id'] = 'different'
    other = sim.generate(months=3, seed=17, customers=changed)
    products = pd.DataFrame(model['catalog'])
    products.loc[products.product_id.eq('A'), 'unit_price_minor'] += 1
    repriced = sim.generate(months=3, seed=17, customers=_customers(), products=products)
    assert first.metadata['generation_fingerprint'] == repeat.metadata['generation_fingerprint']
    assert len({x.metadata['dataset_id'] for x in [first, larger, other, repriced]}) == 4
