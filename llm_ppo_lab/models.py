"""A causal tiny LM and a real Qwen+LoRA backend with the same teaching API."""
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn


class TinyTokenizer:
    pad_token_id, bos_token_id, sep_token_id, eos_token_id = 0, 1, 2, 3
    vocab_size = 14

    def encode_prompts(self, rows, device):
        ids = torch.tensor([[1, int(r['digits'][0]) + 4, int(r['digits'][1]) + 4, 2]
                            for r in rows], device=device)
        return ids, torch.ones_like(ids)

    def decode(self, tokens):
        # Specials other than EOS remain visible so a malformed response cannot
        # receive a reward merely because a decoder silently dropped bad tokens.
        out = []
        for token in tokens:
            token = int(token)
            if token == self.eos_token_id:
                break
            out.append(str(token - 4) if 4 <= token < 14 else f'<{token}>')
        return ''.join(out)


class TinyActorCritic(nn.Module):
    def __init__(self, width=32, max_length=64):
        super().__init__()
        self.embedding = nn.Embedding(14, width)
        self.position = nn.Embedding(max_length, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0., batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, 14)
        self.value_head = nn.Linear(width, 1)

    def forward(self, ids, attention):
        positions = attention.cumsum(-1) - 1
        positions = positions.masked_fill(attention.eq(0), 0)
        hidden = self.embedding(ids) + self.position(positions)
        causal = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).triu(1)
        hidden = self.norm(self.transformer(hidden, mask=causal, src_key_padding_mask=attention.eq(0)))
        return self.lm_head(hidden), self.value_head(hidden).squeeze(-1)

    def trainable_state(self):
        return self.state_dict()

    def load_trainable_state(self, state):
        self.load_state_dict(state)


class HFTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if self.eos_token_id is None:
            raise ValueError('This teaching backend requires an EOS token')

    def encode_prompts(self, rows, device):
        texts = [self.tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])],
                 tokenize=False, add_generation_prompt=True) for row in rows]
        batch = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt')
        return batch['input_ids'].to(device), batch['attention_mask'].to(device)

    def decode(self, tokens):
        tokens = [int(token) for token in tokens]
        if self.eos_token_id in tokens:
            tokens = tokens[:tokens.index(self.eos_token_id)]
        # Do not hide arbitrary non-EOS special tokens from the verifier.
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


class HFActorCritic(nn.Module):
    def __init__(self, model_name, revision, device, local_only=False):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
            local_files_only=local_only, dtype=torch.float32,
            attn_implementation='eager', trust_remote_code=False)
        self.resolved_revision = getattr(base.config, '_commit_hash', None)
        self.lm = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
        self.value_head = nn.Linear(base.config.hidden_size, 1)
        self.to(device)

    def forward(self, ids, attention):
        positions = (attention.cumsum(-1) - 1).masked_fill(attention.eq(0), 0)
        output = self.lm(input_ids=ids, attention_mask=attention, position_ids=positions,
                         output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1]
        return output.logits, self.value_head(hidden.float()).squeeze(-1)

    def trainable_state(self):
        from peft import get_peft_model_state_dict
        return dict(adapter=get_peft_model_state_dict(self.lm), value_head=self.value_head.state_dict())

    def load_trainable_state(self, state):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        expected = get_peft_model_state_dict(self.lm)
        supplied = state['adapter']
        if set(supplied) != set(expected):
            raise ValueError(f'Incomplete adapter state: missing={set(expected)-set(supplied)}, '
                             f'unexpected={set(supplied)-set(expected)}')
        for key, value in expected.items():
            if not isinstance(supplied[key], torch.Tensor) or supplied[key].shape != value.shape:
                raise ValueError(f'Adapter tensor shape differs: {key}')
        result = set_peft_model_state_dict(self.lm, state['adapter'])
        if result.unexpected_keys:
            raise ValueError(f'Unexpected adapter keys: {result.unexpected_keys}')
        self.value_head.load_state_dict(state['value_head'])


class Bundle:
    def __init__(self, backend, model_name, revision, device, local_only=False):
        self.backend = backend
        if backend == 'tiny':
            self.tokenizer = TinyTokenizer()
            self.policy = TinyActorCritic().to(device).eval()
            self.reference = deepcopy(self.policy).requires_grad_(False).eval()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                local_files_only=local_only, trust_remote_code=False)
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = HFTokenizer(tokenizer)
            self.policy = HFActorCritic(model_name, revision, device, local_only).eval()
            self.reference = None

    @contextmanager
    def reference_model(self):
        if self.backend == 'tiny':
            yield self.reference
        else:
            # Base weights are frozen. Disabling LoRA is the unchanged initial
            # policy; this is NOT the rolling old policy that sampled the batch.
            with self.policy.lm.disable_adapter():
                yield self.policy
