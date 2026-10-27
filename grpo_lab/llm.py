"""Pretrained Qwen + hand-written GRPO, bounded local CPU/MPS stages.

No SFT updates. Train only LoRA adapters, keeping rollout logp and reward fixed
through each update. Checkpoint optimizer and RNG for honest continuation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from core import group_advantages, clipped_loss

spec = importlib.util.spec_from_file_location('lesson', Path(__file__).resolve().parents[1]/'tutorials/code/07_pretrained_grpo.py')
lesson = importlib.util.module_from_spec(spec); spec.loader.exec_module(lesson)


def make_lab_data(task):
    data = lesson.make_data(task)
    if task == 'reverse':
        full = [{'id':str(n),'prompt':f'Reverse the digits of {n}. Reply with only the reversed digits.',
                 'answer':str(n)[::-1]} for n in range(1000,10000)]
    else:
        full = [{'id':f'{a}+{b}','prompt':f'Compute {a} + {b}. Reply with only the integer answer.',
                 'answer':str(a+b)} for a in range(10,100) for b in range(a,100)]
    random.Random(2026).shuffle(full)
    # Reserved before stage 2. The initial 128 test items have already been seen.
    data['final_test'] = full[1280:1536]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='grpo_lab/.cache/qwen')
    parser.add_argument('--out', default='grpo_lab/runs/llm')
    parser.add_argument('--task', choices=['reverse', 'add'], default='reverse')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--split', choices=['validation', 'test', 'final_test'], default='validation')
    parser.add_argument('--eval-n', type=int, default=32)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.steps and args.split != 'validation': parser.error('No training with test evaluation')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed); random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'; tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
              torch_dtype=dtype, attn_implementation='sdpa').to(args.device)
    model = (PeftModel.from_pretrained(base, out/'adapter', is_trainable=True) if args.resume else
             get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
               target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')))
    model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
    step0 = 0
    if args.resume:
        state = torch.load(out/'training_state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups: group['lr'] = args.lr
        random.setstate(state['random']); torch.set_rng_state(state['torch_rng'])
        if args.device == 'mps' and 'mps_rng' in state: torch.mps.set_rng_state(state['mps_rng'])
        step0 = state['step']
    data = make_lab_data(args.task)
    (out/'final_test_manifest.json').write_text(json.dumps(data['final_test'],indent=2)+'\n')
    if not args.resume:
        (out/'manifest.json').write_text(json.dumps(data, indent=2)+'\n')
        (out/'config.json').write_text(json.dumps(dict(vars(args), model_parameters=sum(p.numel() for p in model.parameters()),
              trainable_parameters=sum(p.numel() for p in parameters), dtype=str(dtype),
              goal='held-out greedy >= 0.80 and >= 0.20 absolute gain; no SFT; no KL'), indent=2)+'\n')

    @torch.no_grad()
    def rollout(rows, sample):
        prompts = [tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],
                    tokenize=False, add_generation_prompt=True) for r in rows]
        batch = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(args.device)
        kwargs = dict(do_sample=sample, max_new_tokens=args.max_tokens, use_cache=True,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                      repetition_penalty=1., num_beams=1)
        if sample: kwargs.update(temperature=1., top_k=0, top_p=1.)
        result = model.generate(**batch, **kwargs)
        answer = result[:, batch['input_ids'].shape[1]:]
        mask = lesson.completion_mask(answer, tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(answer, skip_special_tokens=True)
        reward = torch.tensor([lesson.exact_reward(t,r['answer']) for t,r in zip(texts,rows)], device=args.device)
        return batch, answer, mask, reward, texts

    def logp(batch, answer, mask):
        return lesson.action_logp(model, batch['input_ids'], batch['attention_mask'], answer, mask)

    @torch.no_grad()
    def evaluate(step):
        rows = data[args.split][:args.eval_n]
        records = []
        for offset in range(0, len(rows), 4):
            subset = rows[offset:offset+4]
            _, _, _, rewards, texts = rollout(subset, False)
            records.extend([dict(row, completion=text, correct=float(r)) for row,text,r in zip(subset,texts,rewards)])
            if offset % 16 == 0: print(f'eval step={step} {offset+len(subset)}/{len(rows)}', flush=True)
        report = dict(step=step, split=args.split, n=len(rows), success=sum(r['correct'] for r in records)/len(rows), records=records)
        (out/f'{args.split}_{step:04d}_{len(rows)}.json').write_text(json.dumps(report, indent=2)+'\n')
        print('EVAL', json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)
        return report

    def save(step):
        model.save_pretrained(out/'adapter')
        state = dict(step=step, optimizer=optimizer.state_dict(), random=random.getstate(), torch_rng=torch.get_rng_state())
        if args.device == 'mps': state['mps_rng'] = torch.mps.get_rng_state()
        torch.save(state, out/'training_state.pt')

    if args.steps == 0:
        evaluate(step0)
        return
    if step0 == 0: evaluate(0)
    for step in range(step0+1, step0+args.steps+1):
        started = time.monotonic()
        chosen = random.sample(data['train'], 2)
        rows = [r for r in chosen for _ in range(args.group)]
        batch, answer, mask, rewards, texts = rollout(rows, True)
        advantage, informative = group_advantages(rewards, args.group)
        # Exact behavior logp under the same unwarped, dropout-free policy.
        with torch.no_grad():
            old = torch.cat([logp({k:v[i:i+1] for k,v in batch.items()}, answer[i:i+1], mask[i:i+1])
                             for i in range(len(rows))])
        total_loss = 0.
        if informative > 0:
            for epoch in range(2):
                optimizer.zero_grad()
                for i in range(len(rows)):
                    new = logp({k:v[i:i+1] for k,v in batch.items()}, answer[i:i+1], mask[i:i+1])
                    loss = clipped_loss(new, old[i:i+1], advantage[i:i+1], mask[i:i+1])/len(rows)
                    loss.backward(); total_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
        save(step)  # stage interruption loses at most the active update
        record = dict(step=step, mean_reward=float(rewards.mean()), informative=informative,
                      loss=total_loss, seconds=round(time.monotonic()-started,2),
                      sample_prompt=rows[0]['prompt'], sample_completion=texts[0])
        with (out/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print('TRAIN',json.dumps(record),flush=True)
        time.sleep(1)  # short duty-cycle break; independent guard monitors thermal state
    evaluate(step0+args.steps)


if __name__ == '__main__': main()
