"""Evaluate existing42 checkpoints with immutable catalog/template copy reuse.

The physical192-file world, trained policies, seeds, cache and inference batches
stay under the original freeze. A separately approved, hashed amendment binds
this driver and its equivalence proof. No training is allowed through this CLI.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from scripts import main_world_benchmark_utils as utils
from scripts import run_main_world_benchmark as runner

_ORIGINAL_SHARD=runner.evaluate_domain_shard
_AMENDMENT_ENV='MAIN_EVAL_COPY_AMENDMENT'


def clone_episode(env):
    memo={id(env._basket_banks):env._basket_banks} if hasattr(env,'_basket_banks') else {}
    if hasattr(env,'home_groups'):memo[id(env.home_groups)]=env.home_groups
    products=getattr(env,'records',{}).get('products')
    if products is not None:memo[id(products)]=list(products)
    return deepcopy(env,memo)


def runtime_amendment():
    path=Path(os.environ[_AMENDMENT_ENV]).resolve();block=json.loads(path.read_text())
    if (block.get('schema_version')!=1 or block.get('type')!='main_eval_immutable_copy_amendment'
            or block.get('status')!='approved'):
        raise ValueError('evaluation copy reuse requires an approved runtime amendment')
    for stem in ('base_frozen_manifest','driver_file','equivalence_proof_file'):
        target=Path(block[stem]).resolve()
        if utils.sha256(target)!=block[stem+'_sha256']:raise ValueError(f'evaluation amendment input changed: {stem}')
    if Path(block['driver_file']).resolve()!=Path(__file__).resolve():raise ValueError('amendment names a different evaluation driver')
    proof=json.loads(Path(block['equivalence_proof_file']).read_text())
    if proof.get('status')!='exact_match' or proof.get('source_world_files_changed')!=0:
        raise ValueError('evaluation copy reuse lacks its unchanged-world equivalence proof')
    return {'driver_file':str(Path(__file__).resolve()),'driver_sha256':utils.sha256(__file__),
        'amendment_file':str(path),'amendment_sha256':utils.sha256(path),
        'base_frozen_manifest':str(Path(block['base_frozen_manifest']).resolve()),
        'base_frozen_manifest_sha256':block['base_frozen_manifest_sha256'],
        'equivalence_proof_file':str(Path(block['equivalence_proof_file']).resolve()),
        'equivalence_proof_sha256':block['equivalence_proof_file_sha256'],
        'shared_immutable_assets':['record product dictionaries (fresh list per clone)','customer home-group templates'],
        'unchanged':'physical world, policies, mutable customer state/RNG, original cache,5000customers,52weeks,64batch'}


def evaluate_domain_shard(matrix_dir,domain,user_start,user_stop):
    amendment=runtime_amendment()
    result=_ORIGINAL_SHARD(matrix_dir,domain,user_start,user_stop)
    if runtime_amendment()!=amendment:raise ValueError('evaluation copy runtime changed during its shard')
    path=Path(result['directory'])/'protocol.json';block=json.loads(path.read_text())
    if block['frozen_world']['manifest_sha256']!=amendment['base_frozen_manifest_sha256']:
        raise ValueError('copy amendment differs from the physical training/evaluation world')
    block['evaluation_runtime_amendment']=amendment
    block['protocol']['evaluation_runtime_amendment']=amendment
    utils.write_json(path,block)
    return {**result,'protocol_sha256':utils.sha256(path),'evaluation_runtime_amendment':amendment}


def install():
    utils.clone_episode=clone_episode
    runner.evaluate_domain_shard=evaluate_domain_shard


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--copy-amendment',type=Path,required=True)
    parser.add_argument('--stage',choices=('evaluate',),default='evaluate')
    args,base_args=parser.parse_known_args()
    locations=argparse.ArgumentParser(add_help=False)
    locations.add_argument('--output',type=Path,required=True)
    locations.add_argument('--frozen-manifest',type=Path,required=True)
    paths,_=locations.parse_known_args(base_args)
    os.environ[_AMENDMENT_ENV]=str(args.copy_amendment.resolve())
    amendment=runtime_amendment()
    if paths.frozen_manifest.resolve()!=Path(amendment['base_frozen_manifest']):
        raise ValueError('driver must retain the original physical-world freeze')
    install();sys.argv=[sys.argv[0],*base_args,'--stage','evaluate']
    utils.write_json(paths.output/'evaluation_copy_runtime.json',{'status':'running',**amendment})
    runner.main()
    for path in (paths.output/'shared_evaluation').glob('*/summary.json'):
        block=json.loads(path.read_text());block['evaluation_runtime_amendment']=amendment
        block['evaluation_protocol']['evaluation_runtime_amendment']=amendment
        utils.write_json(path,block)
    for path in paths.output.glob('*/summary.json'):
        block=json.loads(path.read_text());block['evaluation_runtime_amendment']=amendment
        block['evaluation_protocol']['evaluation_runtime_amendment']=amendment
        utils.write_json(path,block)
    utils.write_json(paths.output/'evaluation_copy_runtime.json',{'status':'complete',**amendment})


if __name__=='__mp_main__':install()
if __name__=='__main__':main()
