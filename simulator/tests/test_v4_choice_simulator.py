import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_v4_gaussian_simulator import gaussian_fixture, checkout_fixture, _customers
from simulate.v4.irl import choice_gmm as choice
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS


def choice_config(gaussian_fixture):
    config,_,path=gaussian_fixture
    model_path=Path(config['source_simulation']['model_path'])
    model=json.loads(model_path.read_text())
    latent=model['latent_shopping'];count=latent['component_count']
    block={'schema_version':1,'type':choice.TYPE,'backend':choice.BACKEND,'purpose':'synthesis_only',
        'parameter_names':choice.PARAMETERS,'parameter_units':choice.UNITS,
        'fitted_parameter_names':choice.PARAMETERS,'component_count':count,
        'persona_labels':[str(k) for k in range(count)],'component_model_indices':list(range(count)),
        'mixture_weights':latent['mixture_weights'],'condition_on_profile_components':True,
        'means':[[0.,0.,0.,0.,r] for r in latent['return_activity']['component_logit_offsets']],
        'covariances':np.tile(.01*np.eye(5),(count,1,1)).tolist(),'dispersion_scales':[0.]*count,
        'base_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'return_model_sha256':latent['return_activity']['base_model_sha256']}
    path.write_text(json.dumps(block))
    result=copy.deepcopy(config);result['source_simulation']['choice_gmm_model_path']=str(path)
    return result,path


def test_zero_choice_taste_keeps_discrete_irl_raw_business_records_and_reports_separate_units(gaussian_fixture):
    config,_,_=gaussian_fixture
    discrete=EcommerceSimulator(config).generate(months=3,seed=17,customers=_customers())
    configured,_=choice_config(gaussian_fixture)
    result=EcommerceSimulator(configured).generate(months=3,seed=17,customers=_customers())
    for name in RAW_COLUMNS:
        pd.testing.assert_frame_equal(result.tables[name],discrete.tables[name])
    assert result.metadata['simulator_version']=='source-shopping-v13'
    assert result.metadata['choice_gmm']['parameter_names']==choice.PARAMETERS
    assert 'reward_gmm' not in result.metadata
    assert 'gaussian_choice_logit_offset_cart' in result.tables['diagnostics']
    assert 'gaussian_reward_offset_cart' not in result.tables['diagnostics']


def test_choice_and_reward_sidecars_cannot_share_configuration_or_artifact_units(gaussian_fixture):
    config,path=choice_config(gaussian_fixture)
    config['source_simulation']['reward_gmm_model_path']=str(path)
    with pytest.raises(ValueError,match='one continuous'):
        EcommerceSimulator(config).generate(months=3,seed=17,customers=_customers())
    del config['source_simulation']['reward_gmm_model_path']
    block=json.loads(path.read_text());block['parameter_units']=['reward']*5
    path.write_text(json.dumps(block))
    with pytest.raises(ValueError,match='incompatible choice-logit'):
        EcommerceSimulator(config).generate(months=3,seed=17,customers=_customers())
