"""Export a calibrated Main or daily-Tianchi world without replacing prior data."""
import argparse
from threadpoolctl import threadpool_limits

from simulate.v4.data.campaign_export import export_campaign,load_campaign_inputs
from simulate.v4.simulator.canonical_campaign import make_campaign_env
from simulate.v4.simulator.coupon_assignment import add_coupon_assignment_arguments,coupon_assignment_from_args


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--domain',required=True,choices=['cosmetics','electronics','tianchi'])
    parser.add_argument('--output',required=True);parser.add_argument('--customers',type=int,default=128)
    parser.add_argument('--seed',type=int,default=20261127)
    add_coupon_assignment_arguments(parser)
    args=parser.parse_args()
    assignment=coupon_assignment_from_args(args)
    with threadpool_limits(limits=1):
        env=make_campaign_env(args.config,domain=args.domain,require_calibrated=True,warmup_days=364,horizon_days=364)
        result=export_campaign(env,args.output,domain=args.domain,customers=args.customers,seed=args.seed,
            coupon_assignment=assignment)
        print(result,flush=True)
        for task in ('retention','uplift',*(['clv'] if result['assignment_counts']['0'] else [])):
            _,features,_=load_campaign_inputs(args.output,task)
            assert features.shape[1]==39
