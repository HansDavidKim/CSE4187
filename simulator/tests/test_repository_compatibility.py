"""Public package imports and configured data paths."""
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


@pytest.mark.parametrize(('module', 'class_name'), [
    ('models.choice', 'CalibratedMultinomialLogit'),
    ('simulate.latent_session_model', 'LatentSessionUser'),
    ('models.lightgcn', 'LightGCN'),
    ('simulate.pipeline', 'EcommercePipeline'),
    ('simulate.uplift', 'StarbucksUpliftSimulator'),
    ('models.user_choice', 'UserMaxEnt'),
])
def test_class_pickle_references(module, class_name):
    reference = f'c{module}\n{class_name}\n.'.encode('ascii')
    assert pickle.loads(reference) is getattr(importlib.import_module(module), class_name)


def test_external_project_location_is_configurable(tmp_path):
    env = {**os.environ, 'ECOMMERCE_CAPSTONE_ROOT': str(tmp_path)}
    result = subprocess.run(
        [sys.executable, '-c',
         'from utils.paths import CAPSTONE_ROOT; print(CAPSTONE_ROOT)'],
        env=env, check=True, capture_output=True, text=True,
    )
    assert Path(result.stdout.strip()) == tmp_path.resolve()


def test_analysis_imports_need_no_external_code_and_keep_torch_dtype():
    code = '''
import importlib
import torch
dtype = torch.get_default_dtype()
for name in (
    'analysis.persona.cluster',
    'analysis.persona.behavior',
    'analysis.irl.history', 'analysis.irl.history',
    'analysis.persona.behavior',
    'analysis.persona.behavior',
):
    importlib.import_module(name)
assert torch.get_default_dtype() == dtype
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
