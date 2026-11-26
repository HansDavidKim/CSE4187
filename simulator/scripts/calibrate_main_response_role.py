"""One canonical persona fit/validation job, so the shared second CPU slot can pause between roles."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import calibrate_main_response as common


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',required=True);p.add_argument('--stage',choices=('fit','validation'),required=True)
    p.add_argument('--domain',required=True);args=p.parse_args()
    config=common.load_config(args.config);source=Path(config['source_simulation']['canonical_response_model_path'])
    model=json.loads(source.read_text());role=args.role;spec=model['personas'][role]
    root=args.output;root.mkdir(parents=True,exist_ok=True);output=root/f'{role}_{args.stage}';output.mkdir(exist_ok=False)
    helper=Path(__file__).resolve().parents[1]/'../../v6-paper/simulator/scripts/calibrate_v6_paper_response.py'
    select_trial,next_strength=common.legacy_search(helper);locks=common.lock_inputs(config,[args.config,helper,Path(__file__)])
    # The daily execution path never calls retail/population/Main history code.
    if args.domain=='tianchi':
        unused={'main_campaign.py','population_profiles.py','population_timing.py','canonical_actor_mapping.py'}
        locks={path:sha for path,sha in locks.items() if Path(path).name not in unused}
    began=time.monotonic();target=spec['purchase_14d_receipt']-spec['purchase_14d_control']
    common.write(output/'protocol.json',{'input_sha256':locks,'role':role,'stage':args.stage,'target':target,
        'calibration_users':256,'validation_users':512,'maximum_trials':8,'strength_bounds':[.01,128.],'gain_tolerance':.02,
        'seed':20261125 if args.stage=='fit' else 20261126,'one_strength_shared_purchase_return':True,
        'original49x4_type_ratios_utilityratio_kernel_fixed':True,'future_validation_used_for_selection':False,
        'role_conditioning':'original joint source prior conditional on frozen BiGRU persona, before warmup',
        'frozen_source_primary_failure':str((root.parent/'coupon_source_primary/result.json').resolve())})
    with threadpool_limits(limits=1):
        env=common.make_campaign_env(config,domain=args.domain,calibration_persona=role,require_calibrated=False,
            warmup_days=364,min_history_days=91,horizon_days=14 if args.stage=='fit' else 28,feature_builder=lambda _:np.zeros(39))
        if args.stage=='fit':
            initial=[];zero,zsummary=common.cohort(env,role,20261125,20,0.,'canonical-calibration',invariance=True,
                initial_states=initial,raw_prefix=output/'zero_raw')
            zero.to_parquet(output/'zero.parquet',index=False)
            strength=model.get('strength_by_persona',{}).get(role,model['default_strength']);trials=[];best=None
            for iteration in range(8):
                frame,summary=common.cohort(env,role,20261125,256,strength,'canonical-calibration',
                    initial_states=initial,raw_prefix=output/f'trial{iteration}_raw')
                summary.update(iteration=iteration,target=target,error=summary['difference']-target);trials.append(summary)
                if select_trial(trials,target) is summary:best=frame
                common.write(output/f'trial{iteration}.json',summary)
                print(json.dumps({'domain':args.domain,'role':role,**summary}),flush=True)
                if abs(summary['error'])<=.02:break
                proposed=next_strength(trials,target)
                if any(np.isclose(proposed,t['strength']) for t in trials):break
                strength=proposed
            selected=select_trial(trials,target);best.to_parquet(output/'calibration.parquet',index=False)
            result={'target_purchase14_gain':target,'selected':selected,'trials':trials,'zero_strength_exact':True,'zero_summary':zsummary,
                'validation_used_for_selection':False,'return7_used_for_selection':False}
        else:
            # Every role must be selected before any synthetic validation outcome is generated.
            for name in model['personas']:
                if not (root/f'{name}_fit/result.json').is_file():raise ValueError('all supported roles must finish fit before validation')
            fitted=json.loads((root/f'{role}_fit/result.json').read_text());strength=fitted['selected']['strength']
            frame,summary=common.cohort(env,role,20261126,512,strength,'canonical-validation',raw_prefix=output/'validation_raw')
            frame.to_parquet(output/'validation.parquet',index=False)
            summary['purchase_gain_ci95']=[summary['difference']+sign*1.96*summary['paired_se'] for sign in (-1,1)]
            summary['purchase_4w_control']=float(frame.purchase_4w_control.mean());summary['purchase_4w_receipt']=float(frame.purchase_4w_receipt.mean())
            result={'target_purchase14_gain':target,'selected_strength':strength,'validation':summary,'validation_used_for_selection':False}
        common.verify(locks);result.update(seconds=time.monotonic()-began,input_sha256=locks,changed_inputs=[])
        common.write(output/'result.json',result)
        print(json.dumps({'stage':'role_complete','domain':args.domain,'role':role,'mode':args.stage,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
