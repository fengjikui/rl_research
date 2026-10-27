"""Explicit, single-GPU LoRA GRPO teaching loop (no Trainer abstraction).

Defaults intentionally small. No API calls, no remote code, no arbitrary execution.
Run baseline first. Downloads the selected Hugging Face model when not cached.
"""
import argparse
import json
from pathlib import Path
import random
import time
import torch


def make_data(task, seed=2026):
    if task == 'reverse':
        data = [{'id': str(n), 'prompt': f'Reverse the digits of {n}. Reply with only the reversed digits.',
                 'answer': str(n)[::-1]} for n in range(1000, 10000)]
    else:
        data = [{'id': f'{a}+{b}', 'prompt': f'Compute {a} + {b}. Reply with only the integer answer.',
                 'answer': str(a+b)} for a in range(10, 100) for b in range(a, 100)]
    random.Random(seed).shuffle(data)
    return {'train': data[:1024], 'validation': data[1024:1152], 'test': data[1152:1280]}


def exact_reward(completion, answer):
    return float(completion.strip() == answer)


def completion_mask(ids, eos_id):
    """Include the FIRST EOS, exclude later padding even when PAD == EOS."""
    eos = (ids == eos_id).long()
    return ((eos.cumsum(-1)-eos) == 0).float()


def action_logp(model, prompt_ids, prompt_attention, completion, mask):
    ids = torch.cat([prompt_ids, completion], 1)
    attention = torch.cat([prompt_attention, mask.long()], 1)
    positions = attention.cumsum(-1)-1
    positions.masked_fill_(attention == 0, 0)
    logits = model(input_ids=ids, attention_mask=attention,
                   position_ids=positions, use_cache=False).logits
    logits = logits[:, prompt_ids.shape[1]-1:-1].float()
    return logits.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['baseline', 'train', 'evaluate'], default='baseline')
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--revision', default='main', help='Pin a model commit for reproducible runs')
    p.add_argument('--adapter', help='Saved LoRA directory, for evaluation')
    p.add_argument('--task', choices=['reverse', 'add'], default='reverse')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--prompts-per-step', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    p.add_argument('--out', default='tutorials/runs/llm_reverse')
    args = p.parse_args()
    if args.group_size < 2: p.error('group-size must be at least 2')
    if args.adapter and args.mode != 'evaluate': p.error('--adapter is supported for evaluate only')
    if args.mode == 'train' and args.split == 'test': p.error('Train uses validation; reserve test for final evaluation')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Use --device cpu for a slow smoke run, or run on an NVIDIA GPU.')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    random.seed(args.seed); torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
            torch_dtype=dtype, trust_remote_code=False, attn_implementation='eager').to(args.device)
    model.config.use_cache = False
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    elif args.mode == 'train':
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    # eval disables dropout, NOT gradients. Rollout and optimization must match.
    model.eval()
    data = make_data(args.task)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/'data_manifest.json').write_text(json.dumps(data, indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(dict(vars(args), torch_version=torch.__version__,
             resolved_model_commit=getattr(model.config, '_commit_hash', None)), indent=2)+'\n')

    def encode(rows):
        strings = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                   tokenize=False, add_generation_prompt=True) for r in rows]
        return tokenizer(strings, padding=True, add_special_tokens=False, return_tensors='pt').to(args.device)

    @torch.no_grad()
    def rollout(rows, sample):
        batch = encode(rows)
        generation = dict(max_new_tokens=args.max_new_tokens, do_sample=sample,
                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                          repetition_penalty=1., num_beams=1)
        if sample: generation.update(temperature=1., top_p=1., top_k=0)
        # These settings make sampled probabilities match unwarped model logp.
        all_ids = model.generate(**batch, **generation)
        completion = all_ids[:, batch['input_ids'].shape[1]:]
        mask = completion_mask(completion, tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(completion, skip_special_tokens=True)
        rewards = torch.tensor([exact_reward(t, r['answer']) for t, r in zip(texts, rows)], device=args.device)
        return batch, completion, mask, rewards, texts

    @torch.no_grad()
    def evaluate():
        records = []
        for row in data[args.split]:
            _, _, _, rewards, texts = rollout([row], False)
            records.append(dict(row, completion=texts[0], reward=float(rewards[0])))
        return dict(success=sum(r['reward'] for r in records)/len(records), n=len(records), records=records)

    started = time.time()
    before = evaluate()
    (out/f'{args.mode}_{args.split}_before.json').write_text(json.dumps(before, indent=2)+'\n')
    print('before', before['success'], flush=True)
    if args.mode == 'train':
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        log = []
        for step in range(1, args.steps+1):
            chosen = random.sample(data['train'], args.prompts_per_step)
            rows = [r for r in chosen for _ in range(args.group_size)]
            batch, answer, mask, reward, _ = rollout(rows, True)
            grouped = reward.reshape(-1, args.group_size)
            std = grouped.std(1, unbiased=False, keepdim=True)
            advantage = ((grouped-grouped.mean(1, keepdim=True))/(std+1e-4)).flatten()
            with torch.no_grad():
                old = torch.cat([action_logp(model, batch['input_ids'][i:i+1],
                                 batch['attention_mask'][i:i+1], answer[i:i+1], mask[i:i+1])
                                 for i in range(len(rows))])
            for _ in range(2):
                optimizer.zero_grad()
                for i in range(len(rows)):
                    lp = action_logp(model, batch['input_ids'][i:i+1], batch['attention_mask'][i:i+1],
                                     answer[i:i+1], mask[i:i+1])[0]
                    ratio = (lp-old[i]).exp()
                    objective = torch.minimum(ratio*advantage[i], ratio.clamp(.8, 1.2)*advantage[i])
                    # Per-sequence token mean, then batch mean; beta=0 (no KL).
                    loss = -(objective*mask[i]).sum()/mask[i].sum().clamp_min(1)/len(rows)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
            record = dict(step=step, reward=float(reward.mean()),
                          informative_group_fraction=float((std>0).float().mean()))
            log.append(record)
            with (out/'train_log.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(record, flush=True)
        model.save_pretrained(out/'adapter')
        tokenizer.save_pretrained(out/'adapter')
        after = evaluate()
        (out/f'train_{args.split}_after.json').write_text(json.dumps(after, indent=2)+'\n')
        print('after', after['success'], flush=True)
    print('seconds', round(time.time()-started, 1))


if __name__ == '__main__': main()
