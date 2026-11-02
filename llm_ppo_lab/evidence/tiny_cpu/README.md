# 微型语言模型 PPO 的原始验证证据

这里公开的是小型因果 Transformer 的 CPU 流程验证，不是预训练大语言模型能力提升实验。

实际执行了 fresh 1 批、恢复后再训练 1 批，共 **2 批 rollout、4 次 optimizer.step**。随后用保存的训练前后检查点分别独立推理，同一组16道验证题，结果均为 **0/16**。这次没有成功的任务提升，不能从参数变化或 loss 推断学会数字反转。

| 文件 | 可以核对什么 |
|---|---|
| [config.json](config.json) | 实际训练参数与固定参考模型定义 |
| [rollout_0001.json](rollout_0001.json)、[rollout_0002.json](rollout_0002.json) | 实际题目、回答、token、mask、log-prob、KL、奖励、优势与价值目标 |
| [learner_0001.json](learner_0001.json)、[learner_0002.json](learner_0002.json) | 各epoch与微批的实际损失和更新幅度指标 |
| [before_inference.json](before_inference.json)、[after_inference.json](after_inference.json) | 同样验证输入、仅替换保存的模型，逐题查看真实输出 |
| [verification.json](verification.json) | 奖励和GAE独立复算、固定reference未变化、检查点恢复、11项测试与实际分数 |

主代理也另从这两份 rollout JSON 逐项重算了奖励、优势和价值目标，最大绝对误差小于 `1e-5`。原始检查点保存在本机 `llm_ppo_lab/runs/final_tiny_v2_20261103/`，未随仓库发布；读者可按[运行说明](../../README.md)重新生成自己的训练前后权重。

本地随机微型 Qwen2＋PEFT 的测试用于核验真实 Transformers/LoRA 接口、左padding和固定参考策略。它与预训练 Qwen2.5-0.5B 的语言能力不是一回事。后者只提供代码入口，未在本机或4090上做本次PPO实测。
