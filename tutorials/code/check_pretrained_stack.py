"""No-weight-download smoke check for the GPU lesson's HF/LoRA code path.

Uses a RANDOM, tiny Qwen2 configuration on CPU. This is not pretrained training.
"""
import importlib.util
import json
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

spec = importlib.util.spec_from_file_location('pretrained', Path(__file__).with_name('07_pretrained_grpo.py'))
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


def main():
    torch.manual_seed(0); torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                         attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(r=4, lora_alpha=8,
                           lora_dropout=0., target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    inputs = torch.tensor([[0, 0, 1, 5], [1, 6, 7, 8]])
    attention = (inputs != 0).long()
    with torch.no_grad():
        output = model.generate(input_ids=inputs, attention_mask=attention, max_new_tokens=4,
                                do_sample=True, temperature=1., top_k=0, top_p=1.,
                                repetition_penalty=1., return_dict_in_generate=True, output_scores=True)
        answer = output.sequences[:, inputs.shape[1]:]
        mask = lesson.completion_mask(answer, 2)
        rollout_lp = torch.stack([scores.log_softmax(-1).gather(1, answer[:, t:t+1]).squeeze(1)
                                 for t, scores in enumerate(output.scores)], 1)
        old = lesson.action_logp(model, inputs, attention, answer, mask)
        error = float(((old-rollout_lp).abs()*mask).max())
        assert error < 1e-5, f'Generation/recomputation distribution mismatch: {error}'
    lp = lesson.action_logp(model, inputs, attention, answer, mask)
    advantage = torch.tensor([1., -1.])[:, None]
    ratio = (lp-old).exp()
    loss = -(torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)*mask).sum()/mask.sum()
    loss.backward()
    trainable_grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert trainable_grad > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    result = {'status': 'passed', 'scope': 'random tiny Qwen2 on CPU, no pretrained weights',
              'generation_logp_max_error': error, 'lora_gradient_l1': trainable_grad,
              'torch': torch.__version__}
    out = Path('tutorials/runs/pretrained_stack_check')
    out.mkdir(parents=True, exist_ok=True)
    (out/'check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
