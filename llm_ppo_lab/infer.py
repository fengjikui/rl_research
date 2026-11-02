"""Read a before/after checkpoint and infer; never takes an optimizer step."""
import argparse
from pathlib import Path

import torch

from models import Bundle
from train import make_data, evaluate, save_json

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grpo_lab.guard import snapshot,pressure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--local-only',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Use a fresh inference output file')
    resource=snapshot()
    if pressure(resource):raise RuntimeError(f'Resource pressure: {resource}')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=saved['config']
    bundle=Bundle(config['backend'],config['model'],config['revision'],args.device,args.local_only or config['local_only'])
    bundle.policy.load_trainable_state(saved['policy']);bundle.policy.eval()
    result=evaluate(bundle,make_data(config['backend'])[args.split],args.device,config['max_new_tokens'])
    result.update(checkpoint=str(args.checkpoint),split=args.split,rollout_batches=saved['update'],
                  policy_updates_during_inference=0)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.out,result)
    print({key:value for key,value in result.items() if key!='records'})


if __name__=='__main__':main()
