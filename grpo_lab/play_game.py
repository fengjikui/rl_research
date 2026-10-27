"""Reload a trained policy and create a fresh game replay without training."""
import argparse
from pathlib import Path
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tutorials/code'))
from common import Backbone, replay_catch
from game import exhaustive


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='grpo_lab/runs/game/model.pt')
    p.add_argument('--seed',type=int,default=456)
    p.add_argument('--games',type=int,default=6)
    p.add_argument('--out',default='grpo_lab/runs/game/reloaded.gif')
    a=p.parse_args()
    torch.set_num_threads(1)
    model=nn.Sequential(Backbone('mlp'),nn.Linear(64,3))
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    model.eval()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    replay_catch(model,a.out,seed=a.seed,games=a.games)
    print({'exhaustive_success':exhaustive(model)[0],'replay':a.out})


if __name__=='__main__': main()
