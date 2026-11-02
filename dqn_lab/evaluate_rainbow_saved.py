"""Evaluate a saved Rainbow stage using reproducible, epsilon-greedy holdouts."""
import argparse
from pathlib import Path
import sys

import cv2
import torch

from atari_evaluation import evaluate_stage
from rainbow_core import RainbowQNet

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grpo_lab.guard import snapshot, pressure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed-start',type=int,default=30000)
    p.add_argument('--epsilon',type=float,default=.05)
    p.add_argument('--max-steps',type=int,default=4500)
    args=p.parse_args()
    if args.episodes<1 or args.max_steps<1 or not 0<=args.epsilon<=1:p.error('Invalid evaluation options')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    resource=snapshot()
    if pressure(resource,2):raise RuntimeError(f'Resource pressure: {resource}')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    if args.out.exists():raise RuntimeError('Use a fresh output directory')
    saved=torch.load(args.model,map_location='cpu',weights_only=True)
    atoms=len(saved['support']);actions=saved['advantage.2.bias_mu'].numel()//atoms
    model=RainbowQNet(actions,atoms,float(saved['support'][0]),float(saved['support'][-1])).to(args.device)
    model.load_state_dict(saved)
    result=evaluate_stage(model,args.out,range(args.seed_start,args.seed_start+args.episodes),
        args.max_steps,args.epsilon)
    print('RESULT',result,flush=True)


if __name__=='__main__':main()
