from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from simulate.v4.irl.fit_latent_shopping import (context_vector, policy_likelihood_gradient,
    weighted_q_objective, mixture_marginal, build_source, fit_encoder, encode,
    _reward, sample_component, prequential_metrics, SHOPPING_HISTORY_FEATURES)
from simulate.v4.irl.fit_latent_shopping import family_marginal,family_q_objective,centered_offsets


def mdp():
    matrices = [sparse.csr_matrix((3, 3)) for _ in range(5)]
    for a, source, target in [(0, 0, 1), (1, 0, 2), (3, 1, 2)]:
        p = matrices[a].tolil();p[source, target] = 1;matrices[a] = p.tocsr()
    mask = np.zeros((3, 5), dtype=bool);mask[0, :2] = True;mask[1, [3, 4]] = True;mask[2, 4] = True
    return matrices, mask


def test_exact_bellman_likelihood_adjoint_gradient():
    p, mask = mdp();reward = np.zeros((3, 5));reward[1, 3] = 1
    counts = np.zeros((3, 5));counts[0, 0] = 2;counts[0, 1] = 1;counts[1, 3] = 1;counts[1, 4] = 1;counts[2, 4] = 2
    _, gradient, _ = policy_likelihood_gradient(reward, counts, p, mask)
    for s, a in [(0, 0), (0, 1), (1, 3)]:
        delta = np.zeros_like(reward);delta[s, a] = 1e-5
        upper = policy_likelihood_gradient(reward + delta, counts, p, mask)[0]
        lower = policy_likelihood_gradient(reward - delta, counts, p, mask)[0]
        assert gradient[s, a] == pytest.approx((upper-lower)/2e-5, abs=1e-6)


def test_mixture_reward_q_and_marginal_gradients_agree_with_finite_difference():
    p, mask = mdp();counts = np.zeros((2, 15));counts[0, [0, 8, 14]] = [3, 2, 1];counts[1, [1, 9, 14]] = [3, 1, 2]
    users = sparse.csr_matrix(counts);base = np.arange(12).reshape(3, 4)*.01;delta = np.asarray([.2, -.1, .05, .3]);weights=np.asarray([.6, .4])
    _, _, q, _ = mixture_marginal(base, delta, weights, users, p, mask, .01)
    component = np.asarray([(users.T @ q[:, z]).reshape(3, 5) for z in range(2)])
    flat=np.concatenate([base.ravel(),delta]);normalizer=counts.sum()
    _, gradient=weighted_q_objective(flat,component,p,mask,normalizer,.01)
    def observed_loss(x):return -mixture_marginal(x[:12].reshape(3,4),x[12:],weights,users,p,mask,.01)[0]
    for i in [0,1,7,12,13,15]:
        change=np.zeros_like(flat);change[i]=1e-5
        assert gradient[i] == pytest.approx((observed_loss(flat+change)-observed_loss(flat-change))/2e-5,abs=2e-6)


def test_source_history_is_delayed_and_unknown_sku_dynamics_not_fabricated():
    rows=[]
    for i,(sid,action,product,at) in enumerate([
        (1,'view','A','2019-12-01T00:00:00Z'),(1,'cart','A','2019-12-01T00:01:00Z'),
        (1,'remove','B','2019-12-01T00:02:00Z'),
        (2,'view','A','2019-12-01T00:10:00Z'),(2,'view','A','2019-12-01T00:40:00Z')]):
        rows.append(dict(user_id='100',session=sid,action=action,product_id=product,at=pd.Timestamp(at),source_event_id=str(i),price_minor=100))
    data=build_source(pd.DataFrame(rows),'2020-01-01T00:00:00Z')
    remove=np.flatnonzero(data['records'].action.eq(2))[0]
    assert not data['dynamics_usable'][remove]
    assert data['features'][remove,0] == data['features'][remove+1,0] == pytest.approx(np.log1p(1))
    assert data['audit']['unmatched_remove_transition_pairs_excluded']==1
    # A session ending at 00:02 becomes available strictly after 00:32.
    second_start=data['records'].index[(data['records'].session==2)&(data['records'].step==0)][0]
    assert data['features'][second_start,-9]==0  # cart-history observed indicator is absent
    assert data['features'][second_start+2,-9]==1


def test_fine_progress_run_length_and_history_are_actual_encoder_inputs():
    start=dict(previous_action='start',session_decision_count=0,previous_action_run_length=0)
    one=dict(previous_action='view',session_decision_count=1,previous_action_run_length=1)
    five=dict(previous_action='view',session_decision_count=5,previous_action_run_length=5)
    assert not np.array_equal(context_vector(one),context_vector(five))
    observed=deepcopy(one);observed[SHOPPING_HISTORY_FEATURES[0]]=3
    assert not np.array_equal(context_vector(one),context_vector(observed))
    assert context_vector(start).shape==context_vector(one).shape
    with pytest.raises(ValueError):context_vector(dict(previous_action='view',session_decision_count=1,previous_action_run_length=0))


def test_component_draw_uses_only_fitted_prefix_bank_or_population_prior():
    model={'mixture_weights':[0.,1.],'profile_posteriors':{'known':[1.,0.]}}
    assert sample_component(model,'known',np.random.default_rng(2))==(0,'train_prefix_profile_posterior')
    assert sample_component(model,'unknown',np.random.default_rng(2))==(1,'population_prior')


def test_prequential_posterior_waits_for_end_target_and_same_clock_evidence():
    rows=pd.DataFrame([dict(user_id='100',session=i+1,action='view',product_id='A',
        at=pd.Timestamp(at),source_event_id=str(i),price_minor=100)
        for i,at in enumerate(['2019-12-01T00:00:00Z','2019-12-01T00:10:00Z'])])
    data=build_source(rows,'2020-01-01T00:00:00Z')
    from simulate.v4.irl.fit_latent_shopping import select_panel,action_mask
    encoder=fit_encoder(select_panel(data,~data['sessions'].is_censored),minimum_leaf=1)
    mask=action_mask(encoder);n=len(mask);policies=np.zeros((2,n,5))
    groups=np.asarray(encoder['state_keys'])[:,1]
    for s,g in enumerate(groups):
        if g%2:
            policies[0,s,[0,1]]=[.9,.1];policies[1,s,[0,1]]=[.1,.9]
        else:
            policies[0,s,[0,4]]=[.9,.1];policies[1,s,[0,4]]=[.1,.9]
    metrics=prequential_metrics(data,encoder,policies,np.asarray([.5,.5]),{'second':data['sessions'].session.eq(2)})['second']
    assert metrics['median_available_evidence_at_first_record']==1
    # First view has arrived, first END (00:30) has not, and second view is tied with its END context.
    assert metrics['prequential_action_nll']==pytest.approx((-np.log(.82)-np.log(.18))/2)


@pytest.mark.parametrize('components',[3,4])
def test_centered_style_family_gradient_matches_marginal_finite_difference(components):
    p,mask=mdp();counts=np.zeros((2,15));counts[0,[0,8,14]]=[3,2,1];counts[1,[1,9,14]]=[3,1,2]
    users=sparse.csr_matrix(counts);base=np.arange(12).reshape(3,4)*.01
    free=np.arange((components-1)*4).reshape(components-1,4)*.015-.05
    offsets=centered_offsets(free);weights=np.arange(1,components+1,dtype=float);weights/=weights.sum()
    np.testing.assert_allclose(offsets.sum(axis=0),0,atol=1e-14)
    _,_,q,_=family_marginal(base,offsets,weights,users,p,mask,.01)
    c=np.asarray([(users.T@q[:,z]).reshape(3,5) for z in range(components)])
    flat=np.concatenate([base.ravel(),free.ravel()]);_,gradient=family_q_objective(flat,c,p,mask,counts.sum(),.01)
    def loss(x):return -family_marginal(x[:12].reshape(3,4),centered_offsets(x[12:].reshape(components-1,4)),weights,users,p,mask,.01)[0]
    for i in [0,1,7,12,13,len(flat)-1]:
        shift=np.zeros_like(flat);shift[i]=1e-5
        assert gradient[i]==pytest.approx((loss(flat+shift)-loss(flat-shift))/2e-5,abs=2e-6)
    target=components-1
    block={'mixture_weights':[float(i==target) for i in range(components)],'profile_posteriors':{}}
    assert sample_component(block,'unseen',np.random.default_rng(1))==(target,'population_prior')
