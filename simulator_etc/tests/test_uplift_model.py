import numpy as np
import pandas as pd

from analysis.uplift.model import calibrate, fit_latent_canonical_assignment, maximum_entropy_types


def test_maximum_entropy_types_recover_both_marginals():
    types = maximum_entropy_types(.2, .4)
    assert np.isclose(sum(types.values()), 1)
    assert np.isclose(types['sure_thing'] + types['sleeping_dog'], .2)
    assert np.isclose(types['sure_thing'] + types['persuadable'], .4)


def test_calibration_maps_personas_and_reactivation_groups():
    proxies = []
    for name in ('P4_purchase_path_proxy', 'P2_deep_consideration_proxy',
                 'P3_cart_revision_proxy'):
        proxies.append({
            'proxy_persona': name, 'purchase_14d_control': .1,
            'purchase_14d_treated': .2,
        })
    returns = []
    for group in ('purchase_only', 'click_purchase', 'click_only'):
        returns.extend([
            {'group': group, 'anchor_receipt': 0, 'return_7d': 10, 'n': 100},
            {'group': group, 'anchor_receipt': 1, 'return_7d': 20, 'n': 100},
        ])
    model = calibrate({'results': proxies}, returns)
    assert set(model['personas']) == {'0', '2', '3'}
    assert all(row['purchase_logit_shift'] > 0 for row in model['personas'].values())
    assert all(row['return_logit_shift'] > 0 for row in model['personas'].values())


def test_latent_assignment_matches_persona_targets_with_regularized_matrix():
    rng = np.random.default_rng(3)
    frames = []
    for persona, centre in zip((0, 2, 3), (-2., 0., 2.)):
        for user in range(40):
            frames.append({
                'user_id': persona * 100 + user, 'cluster': persona,
                'z_0': centre + rng.normal(scale=.2), 'z_1': rng.normal(),
            })
    proxies = []
    for index, name in enumerate(('P4_purchase_path_proxy', 'P2_deep_consideration_proxy',
                                  'P3_cart_revision_proxy')):
        proxies.append({
            'proxy_persona': name, 'purchase_14d_control': .1 + .05 * index,
            'purchase_14d_treated': .2 + .1 * index,
        })
    returns = []
    for group in ('purchase_only', 'click_purchase', 'click_only'):
        returns.extend([
            {'group': group, 'anchor_receipt': 0, 'return_7d': 10, 'n': 100},
            {'group': group, 'anchor_receipt': 1, 'return_7d': 20, 'n': 100},
        ])
    model = fit_latent_canonical_assignment(
        calibrate({'results': proxies}, returns), pd.DataFrame(frames)
    )
    assignment = model['canonical_assignment']
    matrix = np.asarray(assignment['matrix'])
    assert matrix.shape == (3, 4)
    assert np.allclose(matrix.sum(axis=1), 0)
    assert assignment['l1'] > 0 and assignment['l2'] > 0
    assert assignment['final_persona_distribution_rmse'] < .03


def test_latent_assignment_calibrates_regularized_subgroup_offset():
    rng = np.random.default_rng(4)
    frames = []
    for persona, centre in zip((0, 2, 3), (-2., 0., 2.)):
        for user in range(40):
            frames.append({
                'user_id': persona * 100 + user, 'cluster': persona,
                'z_0': centre + rng.normal(scale=.2), 'z_1': rng.normal(),
            })
    proxies = [{
        'proxy_persona': name, 'purchase_14d_control': .1,
        'purchase_14d_treated': .2,
    } for name in ('P4_purchase_path_proxy', 'P2_deep_consideration_proxy',
                   'P3_cart_revision_proxy')]
    returns = []
    for group in ('purchase_only', 'click_purchase', 'click_only'):
        returns.extend([
            {'group': group, 'anchor_receipt': 0, 'return_7d': 10, 'n': 100},
            {'group': group, 'anchor_receipt': 1, 'return_7d': 20, 'n': 100},
        ])
    model = calibrate({'results': proxies}, returns)
    model['canonical_subgroups'] = {
        'P_LT': {'canonical_type_probabilities': maximum_entropy_types(.3, .4)}
    }
    model = fit_latent_canonical_assignment(
        model, pd.DataFrame(frames),
        subgroup_users={'P_LT': set(range(200, 220))},
    )
    assignment = model['canonical_assignment']
    assert len(assignment['group_logit_offsets']['P_LT']) == 4
    assert assignment['subgroup_fit'][0]['users'] == 20
    assert assignment['subgroup_fit'][0]['rmse'] < .03
