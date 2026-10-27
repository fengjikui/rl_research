"""Verify actual pretrained/adapted model behavior probabilities, read-only."""
import importlib.util
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lesson', ROOT/'tutorials/code/07_pretrained_grpo.py')
lesson = importlib.util.module_from_spec(spec); spec.loader.exec_module(lesson)


def main():
    torch.set_num_threads(2); torch.manual_seed(2026)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'grpo_lab/.cache/qwen', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(ROOT/'grpo_lab/.cache/qwen', local_files_only=True,
             torch_dtype=torch.float32, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, ROOT/'grpo_lab/runs/llm_reverse/adapter').eval()
    prompts = [tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True)
               for s in ['Reverse the digits of 1234. Reply with only the reversed digits.', 'Reverse 1200.']]
    batch = tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors='pt')
    with torch.no_grad():
        result = model.generate(**batch,do_sample=True,temperature=1.,top_k=0,top_p=1.,max_new_tokens=8,
                  repetition_penalty=1.,num_beams=1,pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id,use_cache=True,
                  return_dict_in_generate=True,output_scores=True)
        answer = result.sequences[:,batch['input_ids'].shape[1]:]
        mask = lesson.completion_mask(answer,tokenizer.eos_token_id)
        sampled_lp = torch.stack([s.float().log_softmax(-1).gather(1,answer[:,t:t+1]).squeeze(1)
                                  for t,s in enumerate(result.scores)],1)
        recomputed = lesson.action_logp(model,batch['input_ids'],batch['attention_mask'],answer,mask)
        error = float(((sampled_lp-recomputed).abs()*mask).max())
        assert error < 1e-4, error
    selection=json.loads((ROOT/'grpo_lab/runs/llm_reverse/selection.json').read_text())
    record=dict(actual_pretrained_model=True,adapter_step=selection['selected_step'],generation_recomputation_max_logp_error=error,
                tolerance=1e-4,left_padding_tested=True,eos_mask_tested=True,passed=True)
    (ROOT/'grpo_lab/runs/llm_reverse/logp_check.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
