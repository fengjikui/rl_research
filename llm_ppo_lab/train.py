"""Readable LLM PPO: sample -> token rewards -> GAE -> clipped actor/value update.

Default tiny backend needs no download. HF is an explicit Qwen+LoRA entry point,
not a claim that pretrained-model PPO was trained on this computer.
"""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import torch

from core import (response_mask, token_statistics, build_rewards, finite_episode_gae,
                  normalize_advantages, ppo_loss, masked_mean)
from models import Bundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_lab.guard import snapshot, pressure


def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return value


def save_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def save_torch(path, value):
    tmp = path.with_name(path.name + '.tmp'); torch.save(cpu_copy(value), tmp); tmp.replace(path)


def update_directory(out, number):
    """Preserve an interrupted attempt; resume retries from last full checkpoint."""
    directory = out/f'update_{number:04d}'
    if directory.exists():
        archive = directory.with_name(directory.name + '.interrupted_' + str(time.time_ns()))
        directory.rename(archive)
        save_json(archive/'interruption.json', dict(reason='attempt is newer than last complete checkpoint; preserved before retry'))
    directory.mkdir(exist_ok=False)
    return directory


def make_data(backend='tiny'):
    width = 2 if backend == 'tiny' else 4
    numbers = list(range(100)) if backend == 'tiny' else list(range(1000,10000))
    excluded = []
    if backend == 'hf':
        # Fixed, committed manifest: a clean checkout and this workstation must
        # never silently select different training data based on local runs/.
        path = Path(__file__).parent/'data/reserved_grpo_eval_ids.json'
        excluded = json.loads(path.read_text())['ids']
        if len(excluded) != 512 or len(set(excluded)) != 512:
            raise ValueError('Reserved GRPO evaluation manifest must have 512 unique IDs')
        excluded_set = {int(number) for number in excluded}
        numbers = [number for number in numbers if number not in excluded_set]
    rows = [dict(id=f'reverse{width}-{n:0{width}d}', digits=f'{n:0{width}d}',
                 prompt=f'Reverse the digits of {n:0{width}d}. Reply with only the reversed digits.',
                 answer=f'{n:0{width}d}'[::-1]) for n in numbers]
    random.Random(731).shuffle(rows)
    if backend == 'tiny':return dict(train=rows[:64], validation=rows[64:80], test=rows[80:])
    return dict(train=rows[:1024], validation=rows[1024:1152], test=rows[1152:1408],
                excluded_prior_grpo_evaluation_digits=[str(n) for n in excluded])


@torch.no_grad()
def generate(bundle, rows, device, max_tokens, greedy=False):
    prompt, prompt_mask = bundle.tokenizer.encode_prompts(rows, device)
    sequence, attention = prompt.clone(), prompt_mask.clone()
    finished = torch.zeros(len(rows), dtype=torch.bool, device=device)
    answers, masks = [], []
    for _ in range(max_tokens):
        logits, _ = bundle.policy(sequence, attention)
        # No temperature/top-k/top-p warping: rollout and learner use SAME p.
        p = logits[:, -1].float().softmax(-1)
        action = p.argmax(-1) if greedy else torch.multinomial(p, 1).squeeze(-1)
        valid = ~finished
        action = torch.where(valid, action, torch.full_like(action, bundle.tokenizer.pad_token_id))
        answers.append(action); masks.append(valid)
        sequence = torch.cat([sequence, action[:, None]], -1)
        attention = torch.cat([attention, valid.long()[:, None]], -1)
        finished |= action.eq(bundle.tokenizer.eos_token_id) & valid
        if finished.all(): break
    answer, mask = torch.stack(answers, 1), torch.stack(masks, 1)
    if not torch.equal(mask, response_mask(answer, bundle.tokenizer.eos_token_id)):
        raise RuntimeError('Generated mask differs from first-EOS mask')
    texts = [bundle.tokenizer.decode(tokens) for tokens in answer.cpu().tolist()]
    return prompt, prompt_mask, answer, mask, texts, finished


@torch.no_grad()
def collect(bundle, rows, device, max_tokens, beta, gamma, lam, missing_eos_penalty=0.):
    prompt, prompt_mask, answer, mask, texts, finished = generate(bundle, rows, device, max_tokens)
    old_logp, old_values, _ = token_statistics(bundle.policy, prompt, prompt_mask, answer, mask)
    with bundle.reference_model() as reference:
        reference_logp, _, _ = token_statistics(reference, prompt, prompt_mask, answer, mask)
    exact = torch.tensor([float(text.strip() == row['answer']) for text,row in zip(texts,rows)], device=device)
    scores = exact - (~finished).float() * missing_eos_penalty
    rewards, sampled_kl = build_rewards(scores, old_logp, reference_logp, mask, beta)
    advantages, returns = finite_episode_gae(rewards, old_values, mask, gamma, lam)
    batch = dict(prompt=prompt, prompt_mask=prompt_mask, response=answer, mask=mask,
                 old_logp=old_logp.detach(), old_values=old_values.detach(),
                 reference_logp=reference_logp.detach(), sampled_kl=sampled_kl,
                 task_scores=scores, exact_scores=exact, rewards=rewards,
                 advantages=advantages, returns=returns, finished_eos=finished)
    records = [dict(row, completion=text, exact_match=bool(score), task_score=float(task_score),
                    ended_by='eos' if eos else 'max_new_tokens_finite_horizon')
               for row,text,score,task_score,eos in zip(rows,texts,exact.cpu(),scores.cpu(),finished.cpu())]
    return batch, records


def update(bundle, optimizer, batch, epochs, clip, value_coef, entropy_coef, normalize):
    advantage = normalize_advantages(batch['advantages'], batch['mask']) if normalize else batch['advantages']
    metrics = []
    parameters = [p for p in bundle.policy.parameters() if p.requires_grad]
    old_snapshot = batch['old_logp'].clone()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total_tokens = batch['mask'].sum().item()
        # Microbatch = one completion. Accumulation is weighted by token count
        # so it is exactly the same valid-token MEAN as a full-batch loss.
        for i in range(len(batch['prompt'])):
            select = slice(i, i + 1)
            logp, values, entropy = token_statistics(bundle.policy, batch['prompt'][select],
                batch['prompt_mask'][select], batch['response'][select], batch['mask'][select])
            loss, row = ppo_loss(logp, values, entropy, batch['old_logp'][select],
                batch['old_values'][select], advantage[select], batch['returns'][select],
                batch['mask'][select], clip=clip, value_coef=value_coef, entropy_coef=entropy_coef)
            weight = batch['mask'][select].sum().item() / total_tokens
            if not torch.isfinite(loss): raise RuntimeError('Non-finite PPO loss')
            (loss * weight).backward()
            metrics.append(dict(epoch=epoch, completion=i, weight=weight, **row))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient norm')
        optimizer.step()
    if not torch.equal(batch['old_logp'], old_snapshot): raise RuntimeError('Old policy changed within PPO batch')
    return metrics


@torch.no_grad()
def evaluate(bundle, rows, device, max_tokens):
    _, _, _, mask, texts, finished = generate(bundle, rows, device, max_tokens, greedy=True)
    records = [dict(row, completion=text, correct=text.strip() == row['answer'],
                    ended_by='eos' if bool(eos) else 'max_new_tokens_finite_horizon',
                    generated_tokens=int(length))
               for row,text,eos,length in zip(rows,texts,finished,mask.sum(-1))]
    return dict(n=len(rows), success=sum(row['correct'] for row in records) / len(rows),
                policy='greedy', records=records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['tiny','hf'], default='tiny')
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--revision', default='7ae557604adf67be50417f59c2c2f167def9a775')
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=None)
    p.add_argument('--eval-n', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=.05)
    p.add_argument('--gamma', type=float, default=1.)
    p.add_argument('--lam', type=float, default=.95)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--value-coef', type=float, default=.5)
    p.add_argument('--entropy-coef', type=float, default=0.)
    p.add_argument('--missing-eos-penalty', type=float, default=0.)
    p.add_argument('--normalize-advantages', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.max_new_tokens is None:args.max_new_tokens = 4 if args.backend == 'tiny' else 8
    if min(args.updates, args.batch_size, args.epochs, args.max_new_tokens, args.eval_n) < 1:
        p.error('Updates and batch/token/evaluation sizes must be positive')
    if args.batch_size > 64 or args.eval_n > (16 if args.backend == 'tiny' else 128):
        p.error('Batch<=64; validation eval<=16 for tiny or <=128 for HF')
    if not 0 < args.gamma <= 1 or not 0 <= args.lam <= 1 or args.beta < 0 or args.missing_eos_penalty < 0:
        p.error('Invalid discount, lambda, KL coefficient or penalty')
    if args.lr <= 0 or not 0 < args.clip < 1 or min(args.value_coef,args.entropy_coef) < 0:
        p.error('Learning rate and clip must be positive (clip<1); loss coefficients must be nonnegative')
    if args.backend == 'tiny' and args.max_new_tokens > 32: p.error('Tiny max-new-tokens <=32')
    resource = snapshot()
    if pressure(resource): raise RuntimeError(f'Resource pressure: {resource}')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    if args.resume:
        if not (args.out/'latest.pt').is_file(): raise RuntimeError('No latest.pt to resume')
    else: args.out.mkdir(parents=True, exist_ok=False)
    bundle = Bundle(args.backend, args.model, args.revision, device, args.local_only)
    bundle.policy.eval()  # Dropout OFF does not disable gradients in learner updates.
    parameters = [parameter for parameter in bundle.policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    config = {**vars(args), 'out':str(args.out), 'torch':str(torch.__version__),
              'resolved_revision':getattr(bundle.policy, 'resolved_revision', None),
              'reference':'fixed initial model; frozen tiny copy or frozen HF base with adapter disabled',
              'horizon':'first EOS or max_new_tokens; both treated as terminal for this finite task',
              'loss_reduction':'mean over valid response tokens; not per-sequence mean',
              'dtype':'float32', 'trainable_parameters':sum(p.numel() for p in parameters)}
    step0 = optimizer_steps = 0
    if args.resume:
        saved = torch.load(args.out/'latest.pt', map_location='cpu', weights_only=True)
        for key in ('backend','model','revision','seed','max_new_tokens','beta','gamma','lam','lr',
                    'clip','value_coef','entropy_coef','missing_eos_penalty','normalize_advantages'):
            if saved['config'][key] != config[key]: raise RuntimeError(f'Resume differs: {key}')
        bundle.policy.load_trainable_state(saved['policy']); optimizer.load_state_dict(saved['optimizer'])
        if args.backend == 'tiny': bundle.reference.load_state_dict(saved['reference'])
        torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
        if args.device == 'cuda' and saved.get('cuda_rng') is not None: torch.cuda.set_rng_state(saved['cuda_rng'])
        step0 = saved['update']
        optimizer_steps = saved.get('optimizer_steps', step0 * saved['config']['epochs'])
    save_json(args.out/'config.json', config); save_json(args.out/'manifest.json', make_data(args.backend))
    source = args.out/'source'; source.mkdir(exist_ok=True)
    for name in ('train.py','core.py','models.py'): shutil.copy2(Path(__file__).parent/name, source/name)
    shutil.copytree(Path(__file__).parent/'data', source/'data', dirs_exist_ok=True)
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    def save(step, label=None):
        state = dict(policy=bundle.policy.trainable_state(), reference=bundle.reference.state_dict()
                     if args.backend == 'tiny' else None, optimizer=optimizer.state_dict(), update=step,
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state() if args.device == 'cuda' else None,
                     optimizer_steps=optimizer_steps, config=config)
        save_torch(args.out/'latest.pt', state)
        if label: save_torch(args.out/f'{label}.pt', state)

    rows = make_data(args.backend)
    save(step0, 'before' if not args.resume else f'resumed_{step0:04d}')
    save_json(args.out/f'validation_{step0:04d}.json', evaluate(bundle, rows['validation'][:args.eval_n], device, args.max_new_tokens))
    step = step0
    try:
        for next_step in range(step0 + 1, step0 + args.updates + 1):
            if stop_requested: break
            resource = snapshot()
            if pressure(resource): raise RuntimeError(f'Resource pressure; previous checkpoint preserved: {resource}')
            selected = random.sample(rows['train'], args.batch_size)
            batch, records = collect(bundle, selected, device, args.max_new_tokens,
                args.beta, args.gamma, args.lam, args.missing_eos_penalty)
            directory = update_directory(args.out, next_step)
            save_torch(directory/'rollout.pt', batch)
            # JSON carries original numbers so readers need not unpickle a tensor
            # file merely to inspect rewards/returns/advantages.
            audit = {key:value.detach().cpu().tolist() for key,value in batch.items()}
            save_json(directory/'rollout.json', dict(samples=records, tensors=audit,
                      shapes={key:list(value.shape) for key,value in batch.items()}))
            metrics = update(bundle, optimizer, batch, args.epochs, args.clip,
                             args.value_coef, args.entropy_coef, args.normalize_advantages)
            step = next_step
            optimizer_steps += args.epochs
            save_json(directory/'learner.json', metrics); save(step)
            record = dict(update=step, rollout_batches=step, optimizer_steps=optimizer_steps,
                mean_task_score=float(batch['task_scores'].mean()),
                exact_match=float(batch['exact_scores'].mean()),
                mean_sampled_kl=float(masked_mean(batch['sampled_kl'], batch['mask'])), resource=resource)
            with (args.out/'updates.jsonl').open('a') as log: log.write(json.dumps(record)+'\n')
            save_json(args.out/'status.json', dict(status='training', **record))
            print('UPDATE', json.dumps(record), flush=True)
        save(step, 'after')
        save_json(args.out/f'validation_{step:04d}.json', evaluate(bundle, rows['validation'][:args.eval_n], device, args.max_new_tokens))
        save_json(args.out/'status.json', dict(status='stopped' if stop_requested else 'completed',
                    update=step, rollout_batches=step, optimizer_steps=optimizer_steps))
    except BaseException as error:
        save_json(args.out/'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__': main()
