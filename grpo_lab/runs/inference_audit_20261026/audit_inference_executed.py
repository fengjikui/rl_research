"""Read-only, reproducible base/LoRA inference with token-level raw evidence.

Run under guard.py. Does not import training code, create an optimizer, or
modify model checkpoints. Creates a new directory and refuses to overwrite it.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'grpo_lab'
BASE = LAB / '.cache/qwen'
ADAPTER = LAB / 'runs/llm_reverse/adapter'
HISTORY = {
    'before': LAB / 'runs/llm_reverse_base_final/final_test_0000_256.json',
    'after': LAB / 'runs/llm_reverse/final_test_0020_256.json',
}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def infer(model, tokenizer, rows, variant, out):
    started = time.monotonic()
    records = []
    raw_path = out / f'{variant}.jsonl'
    with raw_path.open('x') as raw:
        for offset in range(0, len(rows), 4):
            subset = rows[offset:offset+4]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['prompt']}],
                tokenize=False, add_generation_prompt=True) for row in subset]
            batch = tokenizer(prompts, padding=True, add_special_tokens=False,
                              return_tensors='pt')
            begin = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **batch, do_sample=False, max_new_tokens=8, use_cache=True,
                    num_beams=1, repetition_penalty=1.,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            elapsed = time.monotonic()-begin
            answers = generated[:, batch['input_ids'].shape[1]:]
            texts = tokenizer.batch_decode(answers, skip_special_tokens=True)
            for j, (row, text) in enumerate(zip(subset, texts)):
                # Remove only batch padding after the first EOS, retaining EOS.
                ids = answers[j].tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)+1]
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                assert decoded == text
                record = dict(index=offset+j, variant=variant, **row,
                    completion=text, normalized_completion=text.strip(),
                    correct=(text.strip() == row['answer']),
                    prompt_token_ids=batch['input_ids'][j][batch['attention_mask'][j].bool()].tolist(),
                    generated_token_ids=ids,
                    completion_with_special_tokens=tokenizer.decode(ids, skip_special_tokens=False),
                    generated_tokens=len(ids),
                    ended_with_eos=bool(ids and ids[-1] == tokenizer.eos_token_id),
                    batch_generation_seconds=elapsed)
                records.append(record)
                raw.write(json.dumps(record, ensure_ascii=False)+'\n')
            raw.flush()
            if offset % 16 == 0 or len(records) == len(rows):
                print(f'{variant} {len(records)}/{len(rows)} correct={sum(r["correct"] for r in records)}', flush=True)
            time.sleep(.15)
    historical = json.loads(HISTORY[variant].read_text())['records']
    assert [r['id'] for r in historical] == [r['id'] for r in records]
    differences = [dict(id=new['id'], historical=old['completion'],
                        rerun=new['completion']) for old, new in zip(historical, records)
                   if old['completion'] != new['completion']]
    summary = dict(variant=variant, n=len(records),
        correct=sum(r['correct'] for r in records),
        accuracy=sum(r['correct'] for r in records)/len(records),
        elapsed_seconds=time.monotonic()-started,
        historical_file=str(HISTORY[variant]),
        historical_file_sha256=sha256(HISTORY[variant]),
        historical_output_mismatches=differences,
        raw_file_sha256=sha256(raw_path))
    write_json(out / f'{variant}_summary.json', summary)
    return records, summary


def cell(value):
    return json.dumps(value, ensure_ascii=False).replace('|', '\\|').replace('`', '\\`')


def make_report(variant, records, summary, provenance, out):
    label = '训练前：原始指令模型' if variant == 'before' else '训练后：原始模型＋第 20 步 LoRA'
    title = '01_训练前推理记录.md' if variant == 'before' else '02_训练后推理记录.md'
    command = provenance['command']
    model_line = ('直接加载基础模型，没有挂载任何 adapter。' if variant == 'before' else
                  '加载同一份基础模型，再用 PeftModel.from_pretrained 挂载保存的 LoRA；不合并权重，不更新参数。')
    lines = [f'# {label}', '', f'本次重新推理时间：{provenance["started_at"]}。', '',
        '## 本次实测统计', '',
        f'- 总题数：**{summary["n"]}**。正确：**{summary["correct"]}**；错误：**{summary["n"]-summary["correct"]}**。',
        f'- 准确率：**{summary["correct"]}/{summary["n"]} = {100*summary["accuracy"]:.5f}%**。',
        f'- 本阶段推理与落盘耗时：{summary["elapsed_seconds"]:.2f} 秒，不包含模型首次加载。',
        '- 判分：`completion.strip() == answer`；按字符串反转，保留前导零；不从输出中额外提取答案。',
        '- 固定 256 道原最终测试题，贪心解码，一题一个答案，CPU FP32，batch=4，max_new_tokens=8，repetition_penalty=1。',
        '- 去除 special tokens 后按上述规则判分；原始 token ID、含 special tokens 文本及 EOS 状态全部保存。', '',
        '## 加载了什么', '',
        '- 模型：`Qwen/Qwen2.5-0.5B-Instruct`。',
        f'- 原下载记录中的 revision：`{provenance["model_source"]["revision"]}`。',
        f'- 基础模型文件：`{BASE / "model.safetensors"}`。',
        f'- 基础权重 SHA-256：`{provenance["base_sha256_before"]}`。',
        f'- 加载方式：{model_line}',
        f'- 本阶段参数总数：{summary["parameter_count"]:,}；本次推理可训练参数：0。', '']
    if variant == 'after':
        lines += [f'- LoRA 文件：`{ADAPTER / "adapter_model.safetensors"}`。',
            f'- LoRA SHA-256：`{provenance["adapter_sha256_before"]}`。',
            '- LoRA：rank=8，alpha=16，作用于 q_proj/k_proj/v_proj/o_proj；训练时 1,081,344 个可训练参数。',
            '- 文件哈希与原 selected_step=20 的冻结选取记录相符。', '']
    lines += ['## 与历史记录核对', '',
        f'- 原记录：`{HISTORY[variant]}`。',
        f'- 本次生成文本与历史记录不同的题数：**{len(summary["historical_output_mismatches"])}**。',
        '- 这次是同一固定测试集的重复推理核验，不是新增独立测试集，也不是重新训练。', '',
        '## 原始证据与重跑', '',
        f'- [本阶段逐题原始 JSONL]({variant}.jsonl)：包含提示词、标准答案、原始输出和 token ID。',
        f'- [本阶段统计]({variant}_summary.json)。',
        '- [固定题单](test_manifest.json) · [两模型逐题配对](paired_results.json) · [综合统计](comparison.json)。',
        '- [来源、配置、环境与文件哈希](provenance.json) · [负载监控](resources.jsonl) · [实际运行输出](console.log)。',
        f'- [实际执行的推理代码](audit_inference_executed.py)，SHA-256：`{provenance["script_sha256"]}`。',
        '- 完整命令（在仓库根目录执行；新路径必须不存在，避免覆盖证据）：', '', '```bash',
        command.replace(str(out), str(out)+'_rerun'), '```', '',
        '脚本先直接推理基础模型，再挂载保存的 LoRA，在相同题单、模板和解码设置下推理。无 optimizer、backward 或参数更新。', '',
        '## 全部 256 题：未筛选', '',
        '输出列使用 JSON 字符串形式，保留引号、空格与转义；完整 token 细节见 JSONL。', '',
        '| 序号 | 输入 | 正确答案 | 模型原始输出 | 判定 |',
        '|---:|---|---|---|---|']
    for row in records:
        lines.append(f'| {row["index"]+1} | {row["id"]} | {row["answer"]} | {cell(row["completion"])} | {"正确" if row["correct"] else "错误"} |')
    lines += ['', '## 结果适用范围', '',
        '只评价同分布四位数字字符串反转；不证明一般数学、代码或长程 Agent 能力。原训练选取过程中曾查看第一批 128 题测试并继续训练；本批 256 题是随后预留且不参与选取的最终集，详见原 RESULTS.md。', '']
    (out/title).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    script = Path(__file__).resolve()
    (out/'audit_inference_executed.py').write_bytes(script.read_bytes())
    manifest_path = LAB/'runs/llm_reverse/final_test_manifest.json'
    (out/'test_manifest.json').write_bytes(manifest_path.read_bytes())
    rows = json.loads(manifest_path.read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 256
    for row in rows:
        assert row['answer'] == row['id'][::-1]
    import random
    pool = list(range(1000, 10000)); random.Random(2026).shuffle(pool)
    assert [r['id'] for r in rows] == [str(n) for n in pool[1280:1536]]
    assert not set(r['id'] for r in rows).intersection(str(n) for n in pool[:1280])
    selection = json.loads((LAB/'runs/llm_reverse/selection.json').read_text())
    for path, name in [(LAB/'model_source.json', 'original_model_source.json'),
                       (LAB/'runs/llm_reverse/selection.json', 'original_selection.json'),
                       (ADAPTER/'adapter_config.json', 'original_adapter_config.json')]:
        (out/name).write_bytes(path.read_bytes())
    provenance = dict(started_at=started, device='cpu', dtype='float32', batch_size=4,
        intraop_threads=2, interop_threads=1, seed=0,
        model_source=json.loads((LAB/'model_source.json').read_text()),
        base_sha256_before=sha256(BASE/'model.safetensors'),
        adapter_sha256_before=sha256(ADAPTER/'adapter_model.safetensors'),
        script_sha256=sha256(script), manifest_sha256=sha256(manifest_path),
        packages={name: importlib.metadata.version(name) for name in ['torch','transformers','peft','psutil']},
        python=platform.python_version(), platform=platform.platform(),
        decoding=dict(do_sample=False, max_new_tokens=8, num_beams=1,
                      repetition_penalty=1., use_cache=True),
        source_files={str(path.relative_to(ROOT)):sha256(path) for path in
            [BASE/'config.json', BASE/'generation_config.json', BASE/'tokenizer_config.json',
             BASE/'tokenizer.json', ADAPTER/'adapter_config.json']},
        command=f'.venv/bin/python grpo_lab/guard.py --log {out}_resources.jsonl --max-seconds 900 -- .venv/bin/python grpo_lab/audit_inference.py --out {out}')
    assert provenance['adapter_sha256_before'] == selection['adapter_sha256']
    write_json(out/'provenance.json', provenance)
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.padding_side='left'; tokenizer.pad_token=tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa').to('cpu')
    model.eval(); model.requires_grad_(False)
    print('Loaded original base model; no adapter', flush=True)
    base_count = sum(p.numel() for p in model.parameters())
    before, before_summary = infer(model, tokenizer, rows, 'before', out)
    before_summary['parameter_count'] = base_count
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval(); model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    print('Loaded saved step-20 LoRA on the same base', flush=True)
    after, after_summary = infer(model, tokenizer, rows, 'after', out)
    after_summary['parameter_count'] = sum(p.numel() for p in model.parameters())
    for variant, summary in [('before',before_summary),('after',after_summary)]:
        write_json(out/f'{variant}_summary.json',summary)
    paired = [dict(index=a['index'], id=a['id'], prompt=a['prompt'], answer=a['answer'],
        before=a['completion'], after=b['completion'], before_correct=a['correct'],
        after_correct=b['correct']) for a,b in zip(before,after)]
    counts = {key:0 for key in ['both_correct','fixed','regressed','both_wrong']}
    for row in paired:
        key = ('both_correct' if row['after_correct'] else 'regressed') if row['before_correct'] else ('fixed' if row['after_correct'] else 'both_wrong')
        counts[key] += 1
    comparison = dict(n=len(rows), before_correct=before_summary['correct'],
        after_correct=after_summary['correct'], before_accuracy=before_summary['accuracy'],
        after_accuracy=after_summary['accuracy'],
        absolute_gain_percentage_points=100*(after_summary['accuracy']-before_summary['accuracy']),
        paired_counts=counts,
        historical_output_mismatches={v:len(s['historical_output_mismatches']) for v,s in [('before',before_summary),('after',after_summary)]})
    write_json(out/'paired_results.json', paired)
    write_json(out/'comparison.json', comparison)
    provenance.update(completed_at=now(), base_sha256_after=sha256(BASE/'model.safetensors'),
        adapter_sha256_after=sha256(ADAPTER/'adapter_model.safetensors'))
    assert provenance['base_sha256_after'] == provenance['base_sha256_before']
    assert provenance['adapter_sha256_after'] == provenance['adapter_sha256_before']
    provenance['inference_only_weights_unchanged'] = True
    provenance['evidence_sha256'] = {p.name:sha256(p) for p in out.glob('*.json*') if p.name!='provenance.json'}
    write_json(out/'provenance.json', provenance)
    make_report('before', before, before_summary, provenance, out)
    make_report('after', after, after_summary, provenance, out)
    print('COMPLETE', json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
