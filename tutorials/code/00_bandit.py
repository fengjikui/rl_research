"""Sample -> reward -> log_prob -> backward, with no hidden trainer."""
import argparse
import time
import torch
from torch.distributions import Categorical
from common import setup, save_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='tutorials/runs/bandit')
    args = p.parse_args()
    setup(args.seed)
    started = time.time()
    logits = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.SGD([logits], lr=.15)
    reward_means = torch.tensor([.1, .4, .9])
    history = []
    for step in range(args.steps+1):
        if step % 10 == 0 or step == args.steps:
            probs = logits.softmax(-1).detach()
            history.append(dict(step=step, expected_reward=float(probs @ reward_means),
                                best_action_probability=float(probs[2])))
        if step == args.steps: break
        dist = Categorical(logits=logits)
        actions = dist.sample((64,))
        rewards = torch.bernoulli(reward_means[actions])
        # The oracle means are used ONLY for plotting; the trainer sees samples.
        advantage = rewards - .5  # fixed, action-independent baseline
        loss = -(advantage.detach() * dist.log_prob(actions)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    save_run(args.out, args, {'initial_expected_reward': float(reward_means.mean()),
             'final_expected_reward': history[-1]['expected_reward'],
             'probabilities': logits.softmax(-1).detach().tolist()}, history, started=started)


if __name__ == '__main__': main()
