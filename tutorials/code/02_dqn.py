"""DQN on Catch: replay buffer + detached target network + TD regression."""
import argparse
from collections import deque
from copy import deepcopy
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import Catch, QNet, setup, tensor, evaluate_catch, replay_catch, save_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--encoder', choices=['mlp', 'cnn'], default='mlp')
    p.add_argument('--out', default='tutorials/runs/dqn')
    args = p.parse_args()
    setup(args.seed)
    started = time.time()
    env = Catch(1, args.seed)
    model = QNet(args.encoder)
    target = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay = deque(maxlen=10000)
    initial = evaluate_catch(model)
    history = [dict(step=0, success=initial)]
    for step in range(1, args.steps+1):
        obs = env.obs()
        epsilon = max(.05, 1. - step / (args.steps * .65))
        with torch.no_grad():
            action = random.randrange(3) if random.random() < epsilon else int(model(tensor(obs)).argmax(-1)[0])
        next_obs, reward, done, _ = env.step([action])
        replay.append((obs[0], action, reward[0], next_obs[0], done[0]))
        if len(replay) >= 128 and step % 4 == 0:
            states, actions, rewards, next_states, dones = map(np.array, zip(*random.sample(replay, 64)))
            q = model(tensor(states)).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
            with torch.no_grad():
                # Auto-reset next_obs is harmless here ONLY because done masks it.
                td_target = tensor(rewards) + .99 * (1-tensor(dones)) * target(tensor(next_states)).max(-1).values
            loss = F.smooth_l1_loss(q, td_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if step % 200 == 0: target.load_state_dict(model.state_dict())
        if step % 1000 == 0 or step == args.steps:
            success = evaluate_catch(model)
            history.append(dict(step=step, success=success))
            print(f'step={step} success={success:.3f}', flush=True)
    save_run(args.out, args, dict(initial_success=initial, final_success=evaluate_catch(model),
             random_success=evaluate_catch(), evaluation_episodes=500,
             evaluation_scope='fresh draws from the same finite game'), history, model, started)
    replay_catch(model, Path(args.out)/'trained.gif')
    replay_catch(None, Path(args.out)/'random.gif')


if __name__ == '__main__': main()
