"""Freeze completed calibration coefficients before reading independent validation outcomes."""
import argparse,copy,json
from pathlib import Path
import pandas as pd
from simulate.v4.configuration import load_config
from simulate.v4.irl.source_contract import file_sha
from simulate.v4.simulator.canonical_campaign import make_campaign_env
from simulate.v4.simulator.canonical_response import validate_adapter


def publish(config_path,results,output,*,amendments=()):
    config=load_config(config_path);source=Path(config['source_simulation']['canonical_response_model_path'])
    model=json.loads(source.read_text());original=copy.deepcopy(model);results,output=Path(results),Path(output)
    if output.exists():raise FileExistsError(output)
    reports={};strengths={};bindings={};changes={}
    allowed={'main_campaign.py','population_profiles.py','canonical_campaign.py','calibrate_main_response_role.py'}
    for role,spec in sorted(model['personas'].items()):
        path=results/f'{role}_fit/result.json';fit=json.loads(path.read_text())
        target=spec['purchase_14d_receipt']-spec['purchase_14d_control']
        selected=min(fit['trials'],key=lambda r:(abs(r['difference']-target),r['strength']))
        if selected!=fit['selected'] or abs(selected['difference']-target)>.02 or fit['zero_strength_exact'] is not True:
            raise ValueError('all original supported-role calibration/zero gates must pass before publication')
        zero=pd.read_parquet(results/f'{role}_fit/zero.parquet');cal=pd.read_parquet(results/f'{role}_fit/calibration.parquet')
        if len(zero)!=20 or len(cal)!=256 or not zero.p0.eq(zero.p1).all() or not zero.return0.eq(zero.return1).all():
            raise ValueError('registered sample or zero-effect gate failed')
        for frame in (zero,cal):
            if not frame.static_gmm_draw_persistent.all() or not frame.same_initial_prefix.all():raise ValueError('initial prefix/staticdraw changed')
        for name,before in fit.get('input_sha256',{}).items():
            after=file_sha(name)
            if before!=after:
                if Path(name).name not in allowed:raise ValueError('unamended frozen input changed: '+name)
                changes[name]={'recorded_sha256':before,'current_sha256':after}
        strengths[role]=selected['strength'];reports[role]={'target_purchase14_gain':target,'selected':selected,
            'trials':fit['trials'],'zero_strength_exact':True,'validation_used_for_selection':False,'return7_used_for_selection':False}
        for p in (path,results/f'{role}_fit/zero.parquet',results/f'{role}_fit/calibration.parquet'):bindings[str(p.resolve())]=file_sha(p)
    amendment_binding={str(Path(p).resolve()):file_sha(p) for p in amendments}
    if changes and not amendment_binding:raise ValueError('actual code SHA differences require a bound exact-path amendment')
    model.update(strength_by_persona=strengths,new_world_calibrated=True,strength_status='fresh_world_coefficients_selected_on_calibration_only')
    model['fresh_world_calibration']={'reports':reports,'all_purchase_calibration_errors_le_0_02':True,
        'all_zero20_pairs_exact':True,'coefficient_selection_complete':True,'validation_used_for_selection':False,
        'validation_evidence_policy':'independent512-pair reports are immutable sidecars bound to this already frozen model SHA; they never change model bytes or coefficients',
        'result_sha256':bindings,'recorded_code_sha256_differences':changes,'amendment_sha256':amendment_binding,
        'source_coefficients_changed':False,'real_causal_effect_identified':False,'return7_used_for_selection':False}
    for key in ('canonical_assignment','personas','original_matrix_json_base64','original_matrix_float64_sha256','response_kernel'):
        assert model[key]==original[key]
    bank=source.parent/model['latent_bank_file'];model['latent_bank_file']=str(bank.resolve());validate_adapter(model,pd.read_parquet(bank))
    output.mkdir(parents=True,exist_ok=False)
    def write(path,value):
        with path.open('x') as s:json.dump(value,s,indent=2,allow_nan=False);s.write('\n')
    write(output/'model.json',model);config['source_simulation']['canonical_response_model_path']=str((output/'model.json').resolve())
    write(output/'config.json',config);env=make_campaign_env(config,domain=model['domain'],require_calibrated=True,horizon_days=28)
    proof={'domain':model['domain'],'model_sha256':file_sha(output/'model.json'),'config_sha256':file_sha(output/'config.json'),
        'production_factory_require_calibrated_accepted':True,'all_calibration_target_errors_le_0_02':True,'all_zero20_pairs_exact':True,
        'selected_strengths':strengths,'original49x4_ratio_kernel_exact':True,'validation_read_for_publication':False,
        'validation_outcomes_will_not_change_model':True,'source_model_shas':env.model_shas,'recorded_code_sha256_differences':changes,
        'amendment_sha256':amendment_binding}
    write(output/'verification.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--amendments',nargs='*',default=[])
    a=p.parse_args();print(json.dumps(publish(a.config,a.results,a.output,amendments=a.amendments),indent=2))
