"""Same environment and backbone, three policy optimization algorithms."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from common import Catch, ActorCritic, setup, tensor, evaluate_catch, replay_catch, save_run


def gae(rewards, values, dones, next_value, gamma=.99, lam=.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards)-1 else values[t+1]
        mask = 1 - dones[t]
        delta = rewards[t] + gamma * mask * following - values[t]
        running = delta + gamma * lam * mask * running
        advantages[t] = running
    return advantages, advantages + values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['reinforce', 'ac', 'ppo'], default='ppo')
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='tutorials/runs/ppo')
    args = p.parse_args()
    setup(args.seed)
    started = time.time()
    n, horizon = 64, 5  # complete episodes, identical transition budget per update
    env = Catch(n, args.seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    initial = evaluate_catch(model)
    history = [dict(step=0, success=initial)]
    for update in range(1, args.updates+1):
        states, actions, rewards, dones, values, old_logp = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(horizon):
                obs = tensor(env.obs())
                logits, value = model(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                _, reward, done, _ = env.step(action.numpy())
                states.append(obs); actions.append(action); values.append(value)
                rewards.append(tensor(reward)); dones.append(tensor(done))
                old_logp.append(dist.log_prob(action))
            values, rewards, dones = map(torch.stack, (values, rewards, dones))
            if args.algo == 'reinforce':
                # gamma=1: each complete-episode reward is shared by all its actions.
                advantages = rewards.sum(0).expand(horizon, -1)
                returns = advantages
            else:
                advantages, returns = gae(rewards, values, dones, model(tensor(env.obs()))[1])
            states = torch.stack(states).flatten(0, 1)
            actions = torch.stack(actions).flatten()
            old_logp = torch.stack(old_logp).flatten()
            advantages, returns = advantages.flatten(), returns.flatten()
        # PPO reuses frozen rollout data; REINFORCE/AC update it only once.
        epochs = 4 if args.algo == 'ppo' else 1
        for _ in range(epochs):
            logits, predicted_values = model(states)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions)
            if args.algo == 'ppo':
                ratio = (logp-old_logp).exp()
                actor_loss = -torch.minimum(ratio*advantages,
                             ratio.clamp(.8, 1.2)*advantages).mean()
            else:
                actor_loss = -(logp*advantages).mean()
            critic_loss = F.mse_loss(predicted_values, returns) if args.algo != 'reinforce' else torch.tensor(0.)
            loss = actor_loss + .5*critic_loss - .01*dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
        if update % 20 == 0 or update == args.updates:
            success = evaluate_catch(model)
            history.append(dict(step=update*n*horizon, success=success))
            print(f'update={update} success={success:.3f}', flush=True)
    save_run(args.out, args, dict(initial_success=initial, final_success=evaluate_catch(model),
             random_success=evaluate_catch(), environment_steps=args.updates*n*horizon,
             evaluation_episodes=500), history, model, started)
    replay_catch(model, Path(args.out)/'trained.gif')


if __name__ == '__main__': main()
