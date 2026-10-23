"""Public import paths and local script helpers survive the layout cleanup."""
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

import ecommerce_sim


@pytest.mark.parametrize('role', ['preprocessing', 'analysis', 'models', 'simulate'])
def test_old_module_paths_resolve_to_the_same_implementations(role):
    directory = Path(ecommerce_sim.__file__).parent / role
    for path in directory.glob('*.py'):
        if path.stem == '__init__':
            continue
        current = importlib.import_module(f'ecommerce_sim.{role}.{path.stem}')
        previous = importlib.import_module(f'ecommerce_sim.{path.stem}')
        assert previous is current
        assert getattr(ecommerce_sim, path.stem) is current


@pytest.mark.parametrize(('old_module', 'class_name', 'new_module'), [
    ('choice', 'CalibratedMultinomialLogit', 'models.choice'),
    ('latent_session_model', 'LatentSessionUser', 'simulate.latent_session_model'),
    ('lightgcn', 'LightGCN', 'models.lightgcn'),
    ('pipeline', 'EcommercePipeline', 'simulate.pipeline'),
    ('uplift', 'StarbucksUpliftSimulator', 'simulate.uplift'),
    ('user_choice', 'UserMaxEnt', 'models.user_choice'),
])
def test_serialized_class_references_still_resolve(old_module, class_name, new_module):
    reference = f'cecommerce_sim.{old_module}\n{class_name}\n.'.encode('ascii')
    assert pickle.loads(reference) is getattr(importlib.import_module(f'ecommerce_sim.{new_module}'), class_name)


def test_external_project_location_is_configurable(tmp_path):
    env = {**os.environ, 'ECOMMERCE_CAPSTONE_ROOT': str(tmp_path)}
    result = subprocess.run(
        [sys.executable, '-c',
         'from ecommerce_sim.paths import CAPSTONE_ROOT; print(CAPSTONE_ROOT)'],
        env=env, check=True, capture_output=True, text=True,
    )
    assert Path(result.stdout.strip()) == tmp_path.resolve()


def test_analysis_imports_need_no_external_code_and_keep_torch_dtype():
    code = '''
import importlib
import torch
dtype = torch.get_default_dtype()
for name in (
    'compare_masked_gru_clusters', 'compare_early_later_behavior',
    'sweep_history_temperature', 'run_history_trajectory_irl',
    'analyze_persona_delays', 'analyze_persona_heterogeneity',
    'tsboss_lingam_diagnostic',
):
    importlib.import_module('scripts.' + name)
assert torch.get_default_dtype() == dtype
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('command', [
    ['scripts/run_conditional_trajectory_irl.py'],
    ['-m', 'scripts.run_conditional_trajectory_irl'],
])
def test_existing_conditional_script_supports_direct_and_module_execution(command):
    result = subprocess.run(
        [sys.executable, *command, '--help'],
        check=True, capture_output=True, text=True,
    )
    assert '--sequences' in result.stdout
