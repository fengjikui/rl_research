"""Tiny causal language model: compare SFT, DPO and sampled GRPO.

Task: reverse two characters, e.g. 12>21. All 16 prompts are training support;
evaluation enumerates that support. This is NOT held-out language generalization.
"""
import argparse
from copy import deepcopy
import itertools
import time
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from common import setup, save_run


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(5, 32)
        self.position = nn.Embedding(8, 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 5)

    def forward(self, ids):
        length = ids.shape[1]
        h = self.token(ids) + self.position(torch.arange(length))
        # The PyTorch module is called Encoder, but this mask makes it CAUSAL.
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        return self.head(self.norm(self.blocks(h, mask=mask)))


def prompts():
    return torch.tensor([[a, b, 4] for a, b in itertools.product(range(4), repeat=2)])


def completion_logp(model, prompt, answer):
    sequence = torch.cat([prompt, answer], dim=1)
    logits = model(sequence[:, :-1])[:, prompt.shape[1]-1:]
    return logits.log_softmax(-1).gather(-1, answer.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate(model, prompt, greedy=False):
    sequence = prompt.clone()
    for _ in range(2):
        logits = model(sequence)[:, -1]
        action = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
        sequence = torch.cat([sequence, action[:, None]], 1)
    return sequence[:, -2:]


@torch.no_grad()
def evaluate(model):
    p = prompts()
    answer = generate(model, p, greedy=True)
    correct = p[:, :2].flip(1)
    probability = completion_logp(model, p, correct).sum(-1).exp().mean()
    return float((answer == correct).all(-1).float().mean()), float(probability)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['sft', 'dpo', 'grpo'], default='grpo')
    p.add_argument('--updates', type=int, default=160)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--group-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='tutorials/runs/tiny_grpo')
    args = p.parse_args()
    if args.group_size < 2: p.error('group-size must be at least 2')
    setup(args.seed)
    started = time.time()
    model = TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    base = prompts()
    correct = base[:, :2].flip(1)
    for _ in range(args.warmup):
        loss = -completion_logp(model, base, correct).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    reference = deepcopy(model).requires_grad_(False)
    # Reset optimizer so each algorithm starts with identical parameters AND state.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial, initial_prob = evaluate(model)
    history = [dict(step=0, greedy_success=initial, exact_success_probability=initial_prob)]
    informative = 0.
    for update in range(1, args.updates+1):
        if args.algo == 'sft':
            loss = -completion_logp(model, base, correct).mean()
        elif args.algo == 'dpo':
            # Exact verifier constructs synthetic preferences; not human labels.
            rejected = correct.clone()
            rejected[:, 0] = (rejected[:, 0]+1) % 4
            pi_margin = completion_logp(model, base, correct).sum(-1) - completion_logp(model, base, rejected).sum(-1)
            with torch.no_grad():
                ref_margin = completion_logp(reference, base, correct).sum(-1) - completion_logp(reference, base, rejected).sum(-1)
            loss = -F.logsigmoid(.2 * (pi_margin-ref_margin)).mean()
        else:
            group = args.group_size
            repeated = base.repeat_interleave(group, 0)
            answer = generate(model, repeated)
            with torch.no_grad():
                reward = (answer == repeated[:, :2].flip(1)).all(-1).float().reshape(-1, group)
                std = reward.std(-1, unbiased=False, keepdim=True)
                advantage = ((reward-reward.mean(-1, keepdim=True))/(std+1e-4)).flatten()
                informative = float((std > 0).float().mean())
                old_logp = completion_logp(model, repeated, answer)
            # 3 updates on a frozen group. Each answer has exactly two action tokens.
            for _ in range(3):
                logp = completion_logp(model, repeated, answer)
                ratio = (logp-old_logp).exp()
                weighted = ratio * advantage[:, None]
                clipped = ratio.clamp(.8, 1.2) * advantage[:, None]
                loss = -torch.minimum(weighted, clipped).mean()  # beta=0, no KL
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        if args.algo != 'grpo':
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if update % 10 == 0 or update == args.updates:
            score, probability = evaluate(model)
            history.append(dict(step=update, greedy_success=score, exact_success_probability=probability))
            print(f'update={update} greedy={score:.3f} p(success)={probability:.3f}', flush=True)
    score, probability = evaluate(model)
    outputs = generate(model, base, greedy=True)
    alphabet = '0123>'
    samples = [{'prompt': ''.join(alphabet[t] for t in row),
                'answer': ''.join(alphabet[t] for t in output)} for row, output in zip(base, outputs)]
    save_run(args.out, args, dict(initial_success=initial, final_success=score,
             initial_exact_success_probability=initial_prob, final_exact_success_probability=probability,
             last_informative_group_fraction=informative if args.algo == 'grpo' else None,
             evaluation_scope='all 16 TRAINING prompts; no held-out generalization claim',
             samples=samples), history, model, started)


if __name__ == '__main__': main()
