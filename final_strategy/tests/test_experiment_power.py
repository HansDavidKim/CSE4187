"""Explicit hypothetical rates; no real experimental results or fitted models."""
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
import json
import math
from statistics import NormalDist

import numpy as np
import pytest

from final_strategy.experiment_spec import ExperimentSpec
from final_strategy.experiment_power import PowerScenario, approximate_power, plan_sample_size
from final_strategy.validation import InputValidationError


def scenarios(mde=.02, censor=0.):
    return tuple(PowerScenario((arm, "D2"), .1, mde, "increase", censor,
                               "hypothetical unit-test scenario") for arm in ("B", "C"))


def independent_power(p0, p1, n0, n1, alpha):
    # Independent stdlib normal implementation and algebraically equivalent atan transform.
    h = 2 * (math.atan2(math.sqrt(p1), math.sqrt(1-p1))
             - math.atan2(math.sqrt(p0), math.sqrt(1-p0)))
    z = NormalDist().inv_cdf(1-alpha/2)
    mu = abs(h) * math.sqrt(n0*n1/(n0+n1))
    return NormalDist(mu=mu).cdf(-z) + 1 - NormalDist(mu=mu).cdf(z)


@pytest.mark.parametrize("p0,p1,n0,n1", [(.1,.12,5000,5000), (.4,.3,200,350), (0,.1,500,500), (1,.9,500,500)])
def test_power_matches_independent_equation(p0,p1,n0,n1):
    assert approximate_power(p0,p1,n0,n1,.025) == pytest.approx(independent_power(p0,p1,n0,n1,.025), abs=1e-12)


def test_null_power_is_alpha():
    assert approximate_power(.3,.3,100,200,.05) == pytest.approx(.05)


@pytest.mark.parametrize("weights", [(1,1,1,1,1), (1,2,2,1,3)])
def test_integer_plan_minimality_allocation_and_rounding(weights):
    spec=ExperimentSpec("dummy", "v1", weights)
    source=scenarios(censor=.13)
    plan=plan_sample_size(spec,source)
    analyzed=dict(plan.minimum_analyzable_by_arm); assigned=dict(plan.required_assigned_by_arm)
    assert list(analyzed)==["A","B","C","D1","D2"]
    k=analyzed['A']//weights[0]
    assert all(analyzed[a]==k*w for a,w in zip(analyzed,weights))
    for s in source:
        a,b=s.comparison
        assert independent_power(s.baseline_rate,s.alternative_rate,analyzed[a],analyzed[b],.025)>=.8
        assert Decimal(assigned[a]) * Decimal('.87') >= analyzed[a]
    assert any(independent_power(s.baseline_rate,s.alternative_rate,
               (k-1)*dict(zip(analyzed,weights))[s.comparison[0]],(k-1)*weights[-1],.025)<.8 for s in source)
    data=plan.to_dict();json.dumps(data,allow_nan=False)
    assert data['total_required_evaluation_customers']==sum(assigned.values())
    assert data['comparison_alpha']==.025 and data['effect_is_scenario_assumption']
    assert data['power_scope']=='per_comparison_not_joint'
    assert plan==plan_sample_size(spec,source)
    assert source==scenarios(censor=.13)
    assert all(p>=.8 for _,p in plan.achieved_power)


def test_sample_size_monotonicity():
    spec=ExperimentSpec('dummy','v1',(1,)*5)
    total=lambda sp,sc: sum(n for _,n in plan_sample_size(sp,sc).required_assigned_by_arm)
    n=total(spec,scenarios())
    assert total(spec,scenarios(mde=.01))>n
    assert total(replace(spec,target_power=.9),scenarios())>n
    assert total(replace(spec,family_alpha=.01),scenarios())>n
    assert total(spec,scenarios(censor=.2))>n


@pytest.mark.parametrize('field,value',[
    ('baseline_rate',-.1),('baseline_rate',1.1),('baseline_rate',float('nan')),
    ('baseline_rate',True),('mde_absolute',0),('mde_absolute',-.01),
    ('mde_absolute',float('inf')),('mde_absolute',.99),('mde_absolute',1e-30),
    ('censoring_rate',-1),('censoring_rate',1),('censoring_rate',float('nan')),
    ('direction','unknown'),('assumption_source',''),('baseline_is_assumed','yes'),
    ('comparison',('A','D2')),
])
def test_invalid_scenario(field,value):
    with pytest.raises(InputValidationError):replace(scenarios()[0],**{field:value})


@pytest.mark.parametrize('args', [(-.1,.2,10,10,.05),(.1,1.1,10,10,.05),
    (.1,.2,0,10,.05),(.1,.2,True,10,.05),(.1,.2,10,10,0),(.1,.2,10,10,1),
    (.1,.2,10,10,float('nan')),(.1,.2,10,10,5e-324),
    (10**400,.2,10,10,.05),(.1,.2,10**400,10,.05)])
def test_invalid_power_inputs(args):
    with pytest.raises(InputValidationError):approximate_power(*args)


def test_boundaries_and_impossible_plans():
    spec=ExperimentSpec('dummy','v1',(1,)*5)
    with pytest.raises(InputValidationError):plan_sample_size(spec,scenarios(),max_total=5)
    with pytest.raises(InputValidationError):plan_sample_size(spec,scenarios(censor=.999999),max_total=100000)
    with pytest.raises(InputValidationError):plan_sample_size(spec,(scenarios()[0],scenarios()[0]))
    with pytest.raises(InputValidationError):plan_sample_size(spec,(scenarios()[0],replace(scenarios()[1],censoring_rate=.1)))
    with pytest.raises(InputValidationError):plan_sample_size(spec,list(scenarios()))
    with pytest.raises(InputValidationError):replace(scenarios()[0],direction='decrease',mde_absolute=.2)
    decrease=tuple(replace(s,baseline_rate=.3,direction='decrease') for s in scenarios())
    assert all(p>=.8 for _,p in plan_sample_size(spec,decrease).achieved_power)


def assert_native_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_native_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_native_tree(item)
    else:
        assert type(value) in (str, int, float, bool, type(None))


@pytest.mark.parametrize("scalar", [np.float32, np.float64])
def test_numpy_plan_strict_json_and_calculation_preserved(scalar):
    spec = ExperimentSpec("dummy", "v1", (1,)*5, family_alpha=scalar(.05),
                          target_power=scalar(.8))
    source = tuple(PowerScenario((arm, "D2"), scalar(.1), scalar(.02), "increase",
                                scalar(.1), "hypothetical") for arm in ("B", "C"))
    originals = tuple((s.baseline_rate, s.mde_absolute, s.censoring_rate) for s in source)
    plan = plan_sample_size(spec, source)
    data = plan.to_dict()
    json.dumps(data, allow_nan=False)
    assert_native_tree(data)
    assert type(data["scenarios"]) is tuple
    assert data["scenarios"][0]["baseline_rate"] == float(source[0].baseline_rate)
    assert plan == plan_sample_size(spec, source)
    analyzed = dict(plan.minimum_analyzable_by_arm)
    for s, original, (_, achieved) in zip(source, originals, plan.achieved_power):
        assert all(a is b for a, b in zip((s.baseline_rate, s.mde_absolute, s.censoring_rate), original))
        a, b = s.comparison
        expected = independent_power(float(s.baseline_rate), s.alternative_rate,
                                     analyzed[a], analyzed[b], float(plan.comparison_alpha))
        assert achieved == pytest.approx(expected, abs=1e-12)
    with pytest.raises(FrozenInstanceError):
        source[0].baseline_rate = .2
    with pytest.raises(FrozenInstanceError):
        plan.target_power = .9


def test_numpy_integer_scenario_fields_and_python_example():
    spec = ExperimentSpec("dummy", "v1", (1,)*5)
    source = tuple(replace(s, baseline_rate=np.int64(0), censoring_rate=np.int32(0))
                   for s in scenarios())
    data = plan_sample_size(spec, source).to_dict()
    json.dumps(data, allow_nan=False)
    assert_native_tree(data)
    assert type(data["scenarios"][0]["baseline_rate"]) is int
    plan = plan_sample_size(spec, scenarios(censor=.1))
    assert set(dict(plan.minimum_analyzable_by_arm).values()) == {4644}
    assert set(dict(plan.required_assigned_by_arm).values()) == {5160}
    assert plan.to_dict()["total_required_evaluation_customers"] == 25800


@pytest.mark.parametrize("field,value", [
    ("baseline_rate", np.float32("nan")), ("baseline_rate", np.float64("inf")),
    ("baseline_rate", True), ("mde_absolute", np.float32("inf")),
    ("mde_absolute", np.bool_(True)), ("censoring_rate", np.float64("nan")),
    ("censoring_rate", True),
])
def test_numpy_invalid_scenario_still_rejected(field, value):
    with pytest.raises(InputValidationError):
        replace(scenarios()[0], **{field: value})
