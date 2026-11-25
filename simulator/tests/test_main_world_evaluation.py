"""Only immutable copy assets may be shared by the eval-only driver."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import main_world_benchmark_utils as utils
from scripts import run_main_world_evaluation as driver
from test_main_world_benchmark import ToyWeeklyEnv


def test_clone_shares_only_fixed_templates_and_isolates_customer_graph_and_reset():
    env=ToyWeeklyEnv();env.reset(9)
    baseline=utils.clone_episode(env);optimized=driver.clone_episode(env)
    assert optimized.home_groups is env.home_groups
    assert optimized.records['products'] is not env.records['products']
    assert optimized.records['products'][0] is env.records['products'][0]
    assert optimized.state is not env.state and optimized.rng is not env.rng
    assert optimized.state['rng'] is optimized.rng
    assert optimized._active['row'] is optimized.records['sessions'][0]
    for action in (0,1,1):
        left=baseline.advance(action);right=optimized.advance(action)
        np.testing.assert_array_equal(left[0],right[0]);assert left[1:]==right[1:]
    assert baseline.records['sessions']==optimized.records['sessions']
    frozen=copy.deepcopy(env.home_groups);optimized.reset(31)
    assert env.home_groups==frozen and optimized.home_groups is not env.home_groups


def amendment(tmp_path,monkeypatch):
    world=tmp_path/'world.json';world.write_text('{}')
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'status':'exact_match','source_world_files_changed':0}))
    block={'schema_version':1,'type':'main_eval_immutable_copy_amendment','status':'approved',
        'base_frozen_manifest':str(world),'base_frozen_manifest_sha256':utils.sha256(world),
        'driver_file':driver.__file__,'driver_file_sha256':utils.sha256(driver.__file__),
        'equivalence_proof_file':str(proof),'equivalence_proof_file_sha256':utils.sha256(proof)}
    path=tmp_path/'amend.json';path.write_text(json.dumps(block))
    monkeypatch.setenv(driver._AMENDMENT_ENV,str(path))
    return block,path,proof


def test_amendment_is_explicit_approved_and_rejects_changed_proof(tmp_path,monkeypatch):
    block,path,proof=amendment(tmp_path,monkeypatch)
    assert driver.runtime_amendment()['driver_sha256']==utils.sha256(driver.__file__)
    proof.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='input changed'):driver.runtime_amendment()
    block['status']='candidate';path.write_text(json.dumps(block))
    with pytest.raises(ValueError,match='approved'):driver.runtime_amendment()


def test_shard_provenance_binds_driver_amendment_and_original_physical_world(tmp_path,monkeypatch):
    block,path,proof=amendment(tmp_path,monkeypatch)
    shard=tmp_path/'shard';shard.mkdir()
    protocol=shard/'protocol.json'
    def completed(*args):
        utils.write_json(protocol,{'frozen_world':{'manifest_sha256':block['base_frozen_manifest_sha256']},'protocol':{}})
        return {'directory':str(shard),'protocol_sha256':utils.sha256(protocol)}
    monkeypatch.setattr(driver,'_ORIGINAL_SHARD',completed)
    original=utils.clone_episode
    result=driver.evaluate_domain_shard('unused','cosmetics',0,64)
    saved=json.loads(protocol.read_text())
    assert saved['evaluation_runtime_amendment']==saved['protocol']['evaluation_runtime_amendment']
    assert result['protocol_sha256']==utils.sha256(protocol)
    assert result['evaluation_runtime_amendment']['amendment_sha256']==utils.sha256(path)
    assert utils.clone_episode is original


def test_eval_only_main_keeps_driver_binding_in_final_domain_and_cell_protocol(tmp_path,monkeypatch):
    import sys
    block,path,proof=amendment(tmp_path,monkeypatch)
    output=tmp_path/'benchmark';output.mkdir()
    domain=output/'shared_evaluation/cosmetics';domain.mkdir(parents=True)
    cell=output/'cosmetics_S_UCB_seed42';cell.mkdir()
    # Record undo points before main installs its in-memory-only functions.
    monkeypatch.setattr(utils,'clone_episode',utils.clone_episode)
    monkeypatch.setattr(driver.runner,'evaluate_domain_shard',driver.runner.evaluate_domain_shard)
    def evaluated():
        assert sys.argv[-2:]==['--stage','evaluate']
        for directory in (domain,cell):utils.write_json(directory/'summary.json',{'evaluation_protocol':{}})
    monkeypatch.setattr(driver.runner,'main',evaluated)
    monkeypatch.setattr(sys,'argv',['driver','--copy-amendment',str(path),'--output',str(output),
        '--frozen-manifest',block['base_frozen_manifest']])
    driver.main()
    for directory in (domain,cell):
        saved=json.loads((directory/'summary.json').read_text())
        assert saved['evaluation_protocol']['evaluation_runtime_amendment']==saved['evaluation_runtime_amendment']
        assert saved['evaluation_runtime_amendment']['driver_sha256']==utils.sha256(driver.__file__)
    assert json.loads((output/'evaluation_copy_runtime.json').read_text())['status']=='complete'


def test_driver_rejects_training_stage_before_installing_overrides(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(sys,'argv',['driver','--copy-amendment','unused','--stage','train'])
    original=utils.clone_episode
    with pytest.raises(SystemExit):driver.main()
    assert utils.clone_episode is original
