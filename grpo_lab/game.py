"""Hand-written group-relative policy optimization for the Catch game."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tutorials/code'))
from common import Catch, Backbone, evaluate_catch, replay_catch, setup, tensor, save_run
from core import group_advantages, clipped_loss


@torch.no_grad()
def exhaustive(model):
    env = Catch(25, 999)
    env.ball[:] = np.repeat(np.arange(5), 5)
    env.player[:] = np.tile(np.arange(5), 5)
    actions = []
    for _ in range(5):
        a = model(tensor(env.obs())).argmax(-1).numpy()
        actions.append(a.tolist())
        _, reward, _, _ = env.step(a)
    return float((reward > 0).mean()), actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=250)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='grpo_lab/runs/game')
    args = p.parse_args()
    setup(args.seed); torch.set_num_threads(2)
    started = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    policy = nn.Sequential(Backbone('mlp'), nn.Linear(64, 3))
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
    initial, initial_actions = exhaustive(policy)
    replay_catch(policy, out/'before.gif')
    torch.save(policy.state_dict(), out/'before.pt')
    history = [dict(step=0, success=initial)]
    tasks, group = 16, 16
    rng = np.random.default_rng(args.seed)
    passed = 0
    for update in range(1, args.updates+1):
        env = Catch(tasks*group, args.seed+update)
        # Within a group, exactly the same game initial state; independent actions.
        env.ball[:] = np.repeat(rng.integers(0, 5, tasks), group)
        env.player[:] = np.repeat(rng.integers(0, 5, tasks), group)
        states, actions, old = [], [], []
        with torch.no_grad():
            for _ in range(5):
                state = tensor(env.obs())
                dist = Categorical(logits=policy(state))
                action = dist.sample()
                _, reward, _, _ = env.step(action.numpy())
                states.append(state); actions.append(action); old.append(dist.log_prob(action))
            states = torch.stack(states, 1)  # [B,T,2,H,W]
            actions, old = torch.stack(actions, 1), torch.stack(old, 1)
            advantage, informative = group_advantages(tensor(reward), group)
        for _ in range(4):
            dist = Categorical(logits=policy(states.flatten(0, 1)))
            new = dist.log_prob(actions.flatten()).reshape(tasks*group, 5)
            loss = clipped_loss(new, old, advantage, torch.ones_like(new)) - .01*dist.entropy().mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
        if update % 10 == 0:
            success, _ = exhaustive(policy)
            history.append(dict(step=update*tasks*group*5, success=success))
            print(json.dumps(dict(update=update, success=success, informative_groups=informative)), flush=True)
            torch.save(policy.state_dict(), out/'model.pt')
            passed = passed+1 if success >= .95 else 0
            if passed >= 2: break
    final, final_actions = exhaustive(policy)
    save_run(out, args, dict(initial_success=initial, final_success=final,
             random_success_500=evaluate_catch(), target=.95, target_met=final>=.95,
             evaluation='all 25 possible initial game states, greedy actions, five-step budget',
             updates_completed=update, initial_actions=initial_actions, final_actions=final_actions,
             algorithm='GRPO; G=16; no critic; clip=.2; beta=0; entropy=.01'), history, policy, started)
    replay_catch(policy, out/'after.gif')


if __name__ == '__main__': main()
