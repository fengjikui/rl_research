"""Independent milestone check: held-out seeds and modest action randomness."""
import argparse
from pathlib import Path
import shutil

import cv2
import torch

from atari_evaluation import evaluate_stage
from breakout import NatureQNet


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--epsilon',type=float,default=.05)
    p.add_argument('--max-steps',type=int,default=4500)
    args=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.model,args.out/'model.pt')
    model=NatureQNet(4).to('mps')
    model.load_state_dict(torch.load(args.model,map_location='cpu',weights_only=True))
    result=evaluate_stage(model,args.out,list(range(30000,30000+args.episodes)),args.max_steps,args.epsilon)
    print(result,flush=True)


if __name__=='__main__':main()
