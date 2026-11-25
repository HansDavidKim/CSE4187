"""Attach independent MC evidence to already frozen coefficients without rewriting the model."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from simulate.v4.irl.source_contract import file_sha


def attach(production,results):
    production,results=Path(production),Path(results);model_path=production/'model.json'
    before=file_sha(model_path);model=json.loads(model_path.read_text());reports={};files={}
    for role,spec in sorted(model['personas'].items()):
        path=results/f'{role}_validation/result.json';data=json.loads(path.read_text())
        frame_path=results/f'{role}_validation/validation.parquet';frame=pd.read_parquet(frame_path)
        fixed=model['strength_by_persona'][role]
        if data['selected_strength']!=fixed or len(frame)!=512 or data['validation_used_for_selection'] is not False:
            raise ValueError('frozen coefficients or independent512 validation protocol changed')
        if not frame.static_gmm_draw_persistent.all() or not frame.same_initial_prefix.all():raise ValueError('paired/source state changed')
        summary=data['validation'];target=spec['purchase_14d_receipt']-spec['purchase_14d_control']
        reports[role]={'target_purchase14_gain':target,'fixed_strength':fixed,'validation':summary,
            'purchase14_error':summary['difference']-target,
            'quiet14_return7_difference':summary['quiet14_return1']-summary['quiet14_return0'] if summary['quiet14_users'] else None,
            'return7_prior_contrast':spec['return_7d_receipt']-spec['return_7d_control'],'return7_used_for_selection':False}
        files[str(path.resolve())]=file_sha(path);files[str(frame_path.resolve())]=file_sha(frame_path)
    result={'domain':model['domain'],'model_sha256':before,'coefficient_selection_unchanged':True,
        'validation_complete':True,'validation_used_for_selection':False,'all_purchase_validation_errors_le_0_02':
        all(abs(r['purchase14_error'])<=.02 for r in reports.values()),'reports':reports,'input_sha256':files,
        'scope':'independent synthetic validation seeds under the same frozen scenario population; not real-world causal validation',
        'adoption':'engineering decision may retain disclosed finite-MC residuals; no grid extension or coefficient reselection'}
    with (production/'validation.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    if file_sha(model_path)!=before:raise RuntimeError('independent validation changed frozen model bytes')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--production',required=True);p.add_argument('--results',required=True)
    a=p.parse_args();print(json.dumps(attach(a.production,a.results),indent=2))
