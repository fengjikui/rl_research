"""Two-call tool agent, trained with trajectory-level GRPO.

The environment hides a bit. Agent can inspect it, or submit answer 0/1.
Tool observation is input context, never an action target. No arbitrary code exec.
"""
import argparse
import time
import torch
from torch.distributions import Categorical
from common import setup, save_run
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('tiny_lesson', Path(__file__).with_name('05_tiny_lm.py'))
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)


class ToolPolicy(tiny.TinyLM):
    def __init__(self):
        super().__init__()
        self.token = torch.nn.Embedding(6, 32)
        self.head = torch.nn.Linear(32, 6)

    def action_logits(self, context):
        # Schema-constrained actions: token 3=inspect, 4=submit0, 5=submit1.
        # Normalize over the SAME allowed actions in rollout and training.
        return self(context)[:, -1, 3:6]


def contexts(bits):
    first = torch.full((len(bits), 1), 2, dtype=torch.long)
    second = torch.stack([torch.full_like(bits, 2), torch.full_like(bits, 3), bits], 1)
    return first, second


@torch.no_grad()
def evaluate(model):
    bits = torch.tensor([0, 1])
    first, second = contexts(bits)
    a = model.action_logits(first).argmax(-1)
    b = model.action_logits(second).argmax(-1)
    chosen = torch.where(a == 0, b, a)-1
    return dict(success=float((chosen == bits).float().mean()),
                inspect_rate=float((a == 0).float().mean()),
                trace=[{'hidden_bit': int(bit), 'first_action': int(x),
                        'second_action': int(y) if x == 0 else None}
                       for bit, x, y in zip(bits, a, b)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--group-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='tutorials/runs/tool_agent')
    args = p.parse_args()
    if args.group_size < 2: p.error('group-size must be at least 2')
    setup(args.seed)
    started = time.time()
    model = ToolPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = evaluate(model)
    history = [dict(step=0, success=initial['success'], inspect_rate=initial['inspect_rate'])]
    bits = torch.tensor([0, 1]).repeat_interleave(args.group_size)
    first, second = contexts(bits)
    for step in range(1, args.updates+1):
        with torch.no_grad():
            d1, d2 = Categorical(logits=model.action_logits(first)), Categorical(logits=model.action_logits(second))
            a, b = d1.sample(), d2.sample()
            inspected = (a == 0).float()
            answer = torch.where(a == 0, b, a)-1
            reward = (answer == bits).float() - .05*inspected
            grouped = reward.reshape(2, args.group_size)
            advantages = ((grouped-grouped.mean(1, keepdim=True)) /
                          (grouped.std(1, unbiased=False, keepdim=True)+1e-4)).flatten()
            old = torch.stack([d1.log_prob(a), d2.log_prob(b)], 1)
            # A second call exists only after inspect; hypothetical unused calls
            # are fully masked and never counted as executed environment steps.
            action_mask = torch.stack([torch.ones_like(inspected), inspected], 1)
        for _ in range(3):
            d1, d2 = Categorical(logits=model.action_logits(first)), Categorical(logits=model.action_logits(second))
            logp = torch.stack([d1.log_prob(a), d2.log_prob(b)], 1)
            ratio = (logp-old).exp()
            objective = torch.minimum(ratio*advantages[:, None],
                                      ratio.clamp(.8, 1.2)*advantages[:, None])
            # Sum over actual calls, average over trajectories: no length division.
            loss = -(objective*action_mask).sum(1).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if step % 10 == 0 or step == args.updates:
            scores = evaluate(model)
            history.append(dict(step=step, success=scores['success'], inspect_rate=scores['inspect_rate']))
    final = evaluate(model)
    save_run(args.out, args, dict(initial=initial, final=final,
             evaluation_scope='exhaustive two hidden states; at most TWO calls, not a long-horizon benchmark'),
             history, model, started)


if __name__ == '__main__': main()
