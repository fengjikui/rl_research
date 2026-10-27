"""An information ablation: identical terminal observations, different histories."""
import argparse
import time
import torch
from torch import nn
from torch.distributions import Categorical
from common import setup, save_run


class MemoryPolicy(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.encoder = nn.GRU(3, 32, batch_first=True) if kind == 'gru' else nn.Linear(3, 32)
        self.head = nn.Linear(32, 2)

    def forward(self, observations):
        if self.kind == 'gru':
            features, _ = self.encoder(observations)
            return self.head(features[:, -1])
        return self.head(self.encoder(observations[:, -1]).tanh())


def make_batch(n, delay, balanced=False):
    labels = torch.arange(n) % 2 if balanced else torch.randint(2, (n,))
    obs = torch.zeros(n, delay+2, 3)
    obs[torch.arange(n), 0, labels] = 1  # cue seen only at the start
    obs[:, -1, 2] = 1  # identical 'answer now' observation for everyone
    return obs, labels


@torch.no_grad()
def evaluate(model, delay):
    obs, labels = make_batch(1000, delay, balanced=True)
    return float((model(obs).argmax(-1) == labels).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', choices=['mlp', 'gru'], default='gru')
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--delay', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='tutorials/runs/memory_gru')
    args = p.parse_args()
    setup(args.seed)
    started = time.time()
    model = MemoryPolicy(args.encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = evaluate(model, args.delay)
    history = [dict(step=0, success=initial)]
    for step in range(1, args.updates+1):
        obs, hidden_answer = make_batch(128, args.delay)
        dist = Categorical(logits=model(obs))
        action = dist.sample()
        reward = (action == hidden_answer).float()
        # hidden_answer is used by the environment to reward, never fed to model.
        loss = -((reward-.5).detach()*dist.log_prob(action)).mean() - .01*dist.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 20 == 0 or step == args.updates:
            history.append(dict(step=step, success=evaluate(model, args.delay)))
    save_run(args.out, args, dict(initial_success=initial, final_success=evaluate(model, args.delay),
             longer_delay_success=evaluate(model, args.delay*2),
             evaluation_scope='balanced two-cue distribution; longer delay is a separate probe'),
             history, model, started)


if __name__ == '__main__': main()
