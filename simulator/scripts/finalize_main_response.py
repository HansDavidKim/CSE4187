"""Publish frozen bounded-calibration coefficients, retaining every residual and code amendment."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
from simulate.v4.configuration import load_config
from simulate.v4.irl.source_contract import file_sha
from simulate.v4.simulator.canonical_campaign import make_campaign_env
from simulate.v4.simulator.canonical_response import validate_adapter


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def finalize(config_path,results,output,*,amendments=(),engineering_adoption=False):
    config=load_config(config_path);source=Path(config['source_simulation']['canonical_response_model_path'])
    original=json.loads(source.read_text());model=copy.deepcopy(original)
    results,output=Path(results),Path(output)
    if output.exists():raise FileExistsError(output)
    reports={};strengths={};differences={};binding={}
    allowed={'main_campaign.py','population_profiles.py','canonical_campaign.py','calibrate_main_response_role.py'}
    for role,spec in sorted(model['personas'].items()):
        fit_path=results/f'{role}_fit/result.json';val_path=results/f'{role}_validation/result.json'
        fit=json.loads(fit_path.read_text());checked=json.loads(val_path.read_text())
        target=spec['purchase_14d_receipt']-spec['purchase_14d_control'];chosen=fit['selected']
        if (chosen['strength']!=checked['selected_strength'] or target!=fit['target_purchase14_gain']
                or fit.get('validation_used_for_selection',False) or checked.get('validation_used_for_selection',False)):
            raise ValueError('fixed strength/target or separation from validation changed')
        selected=min(fit['trials'],key=lambda row:(abs(row['difference']-target),row['strength']))
        if chosen!=selected:raise ValueError('selected coefficient differs from original bounded finite-MC choice')
        zero=pd.read_parquet(results/f'{role}_fit/zero.parquet')
        calibrated=pd.read_parquet(results/f'{role}_fit/calibration.parquet')
        validation=pd.read_parquet(results/f'{role}_validation/validation.parquet')
        if (len(zero)!=20 or len(calibrated)!=256 or len(validation)!=512 or not zero.p0.eq(zero.p1).all()
                or not zero.return0.eq(zero.return1).all() or fit['zero_strength_exact'] is not True):
            raise ValueError('required zero20/cal256/independentvalidation512 contract failed')
        for frame in (zero,calibrated,validation):
            if not frame.static_gmm_draw_persistent.all() or not frame.same_initial_prefix.all():
                raise ValueError('source traits or paired prefix changed')
        if not set(calibrated.seed).isdisjoint(set(validation.seed)):
            raise ValueError('validation must use independently keyed synthetic seeds')
        # Inspect recorded inputs and retain actual differences, never rewrite their old hashes.
        for phase,record in (('fit',fit),('validation',checked)):
            for path,before in record.get('input_sha256',{}).items():
                after=file_sha(path)
                if before!=after:
                    if Path(path).name not in allowed:raise ValueError('frozen model/source/runtime changed without allowed independent amendment: '+path)
                    differences[path]={'recorded_sha256':before,'current_sha256':after}
        strengths[role]=chosen['strength'];summary=checked['validation']
        reports[role]={'target_purchase14_gain':target,'selected':chosen,'trials':fit['trials'],'zero_strength_exact':True,
            'validation':summary,'purchase14_validation_error':summary['difference']-target,
            'quiet14_return7_difference':summary['quiet14_return1']-summary['quiet14_return0'] if summary['quiet14_users'] else None,
            'return7_prior_contrast':spec['return_7d_receipt']-spec['return_7d_control'],
            'return7_used_for_selection':False,'validation_used_for_selection':False,
            'source_fit_result':str(fit_path.resolve()),'source_validation_result':str(val_path.resolve())}
        for path in (fit_path,val_path,results/f'{role}_fit/zero.parquet',results/f'{role}_fit/calibration.parquet',results/f'{role}_validation/validation.parquet'):
            binding[str(path.resolve())]=file_sha(path)
    amendment_binding={}
    for path in map(Path,amendments):
        if not path.is_file():raise ValueError('missing declared amendment proof: '+str(path))
        amendment_binding[str(path.resolve())]=file_sha(path)
    if differences and not amendment_binding:raise ValueError('recorded code SHA changes require explicit proof/amendment binding')
    fit_pass=all(abs(v['selected']['error'])<=.02 for v in reports.values())
    validation_pass=all(abs(v['purchase14_validation_error'])<=.02 for v in reports.values())
    if not fit_pass and not engineering_adoption:raise ValueError('bounded fit has residuals; explicit engineering adoption required')
    model['strength_by_persona']=strengths
    model['new_world_calibrated']=True
    model['strength_status']=('fresh_world_purchase_gain_calibrated' if fit_pass and validation_pass else
        'fresh_world_bounded_calibration_engineering_selected_with_residual')
    model['fresh_world_calibration']={'reports':reports,'all_purchase_calibration_errors_le_0_02':fit_pass,
        'all_purchase_validation_errors_le_0_02':validation_pass,'engineering_adoption':engineering_adoption,
        'new_world_calibrated_meaning':'bounded fresh-world coefficient estimation completed and fixed; target/validation tolerance is reported separately',
        'validation_used_for_selection':False,'return7_used_for_selection':False,'result_sha256':binding,
        'source_model_sha256':file_sha(source),'recorded_code_sha256_differences':differences,'amendment_sha256':amendment_binding,
        'source_coefficients_changed':False,'real_causal_effect_identified':False,
        'scope':'independent synthetic seeds in the same frozen scenario population; not independent real-world causal or source validation'}
    for key in ('canonical_assignment','personas','original_matrix_json_base64','original_matrix_float64_sha256','response_kernel'):
        if model[key]!=original[key]:raise ValueError('fixed canonical matrix/target/utility/kernel changed')
    bank_path=source.parent/model['latent_bank_file'];model['latent_bank_file']=str(bank_path.resolve())
    validate_adapter(model,pd.read_parquet(bank_path))
    output.mkdir(parents=True,exist_ok=False);write(output/'model.json',model)
    config['source_simulation']['canonical_response_model_path']=str((output/'model.json').resolve())
    write(output/'config.json',config)
    env=make_campaign_env(config,domain=model['domain'],require_calibrated=True,horizon_days=28)
    verification={'domain':model['domain'],'model_sha256':file_sha(output/'model.json'),'config_sha256':file_sha(output/'config.json'),
        'production_factory_require_calibrated_accepted':True,'all_zero20_pairs_exact':True,
        'all_purchase_calibration_errors_le_0_02':fit_pass,'all_purchase_validation_errors_le_0_02':validation_pass,
        'selected_strengths':strengths,'recorded_code_sha256_differences':differences,'amendment_sha256':amendment_binding,
        'source_model_shas':env.model_shas,'original49x4_matrix_and_utilityratio_kernel_exact':True}
    write(output/'verification.json',verification)
    return verification


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--amendments',type=Path,nargs='*',default=[]);p.add_argument('--engineering-adoption',action='store_true')
    a=p.parse_args();print(json.dumps(finalize(a.config,a.results,a.output,amendments=a.amendments,engineering_adoption=a.engineering_adoption),indent=2))
