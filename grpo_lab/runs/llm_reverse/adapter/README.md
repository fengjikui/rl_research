---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - lora
  - grpo
  - transformers
---

# 四位数字反转 GRPO LoRA

此目录提供本项目训练后选定的第 20 步 LoRA adapter，需与官方 Qwen/Qwen2.5-0.5B-Instruct 基础权重组合使用。

- 基础模型原 revision：7ae557604adf67be50417f59c2c2f167def9a775。
- 训练方式：手写 GRPO，无额外 SFT 热身，无 Critic，无参考 KL。
- LoRA：rank=8，alpha=16，q_proj/k_proj/v_proj/o_proj；训练时 1,081,344 个可训练参数。
- adapter 权重 SHA-256：c4df15cb4db074b3e329ba4f38c66bc6fa755fdb279f383e868902312994e0a3。
- 固定 256 题测试：基础模型 113/256，adapter 后 236/256；2026-10-26 重新推理一致。完整逐题输出、失败记录与数据选取说明见 [实验报告](../../../RESULTS.md)。

本项目修改了基础模型的行为，修改内容以 LoRA 增量参数存储；未修改或再分发完整基础权重。模型仅在四位数字字符串反转的同分布任务上得到评价，不能据此宣称通用能力提高。

基础模型由 Qwen 团队发布，原始许可为 Apache License 2.0；本 adapter 按同一许可发布，许可副本见 [LICENSE](LICENSE)。[基础模型及版权来源](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。

从仓库根目录显式加载基础模型，再挂载本目录，避免依赖配置中原机器的缓存路径：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("grpo_lab/.cache/qwen")
model = PeftModel.from_pretrained(
    base, "grpo_lab/runs/llm_reverse/adapter", is_trainable=False
).eval()
```

基础权重下载与完整推理命令见 [GRPO 使用说明](../../../README.md#从公开仓库复现推理)。
