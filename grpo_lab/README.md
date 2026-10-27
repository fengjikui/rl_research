# 手写 GRPO：游戏与预训练语言模型的本机训练

这是在已有教程之后增加的实际训练项目。两个任务共用 [core.py](core.py) 中的组内优势与裁剪损失，没有使用 GRPOTrainer，也没有 Critic。

**已训练达标并停止：** 游戏 20%→96%；真实 Qwen 0.5B 在保留的 256 道数字反转题上 44.1%→92.2%。完整经过、初次未达标结果、检查点与资源记录见 [训练完成报告](RESULTS.md)。

**2026-10-26 已重新推理核验：** 同一固定 256 题，基础模型 113/256（44.140625%），挂载保存的第 20 步 LoRA 后 236/256（92.1875%）；512 个回答逐题均与历史记录一致。两份完整文档：[训练前推理记录](runs/inference_audit_20261026/01_训练前推理记录.md)、[训练后推理记录](runs/inference_audit_20261026/02_训练后推理记录.md)。每份列出全部 256 题，并链接原始输出和统计；[原始配对结果](runs/inference_audit_20261026/paired_results.json)、[独立核验结果](runs/inference_audit_20261026/verification.json)、[实际执行的推理脚本快照](runs/inference_audit_20261026/audit_inference_executed.py)。本次只推理，基础权重与 LoRA 文件的 SHA-256 前后未变。

## 先确定什么叫成功

- 游戏：Catch 的全部 25 种初始球／挡板位置，固定五步预算，贪心成功率至少 95%；连续两次评估达到阈值。
- 预训练 LLM：先用验证集选定有提升空间的任务，再以固定解码设置在留出任务上达到 80%，且较基础模型提高至少 20 个百分点。验证集用于开发，最终测试集单独报告；不把训练集成绩代替泛化。
- 每个任务使用同一任务与相同评估口径比较训练前后；奖励、模型、预算变化须记录。没有达到目标时保留真实结果，不隐藏失败运行。

第一阶段 10 步后，32 题验证集达到 90.6%，但首次 128 题测试只有 78.1%，未达目标，因此继续训练。开发评价扩大为完整 128 题验证集；该轮测试已查看，后续不能再称为盲测。第二阶段开始前固定原始打乱池中 `[1280:1536]` 的 256 道题为 `final_test`，与所有旧数据不重叠。决策与清单保留在运行目录中，最终只用验证集选择下一检查点。

## 结构、目标和代码

| 部分 | 游戏 | LLM |
|---|---|---|
| 网络 | 60→64→64→3 的 MLP 策略 | Qwen2.5-0.5B-Instruct＋LoRA |
| 一组 | 相同初始位置的 16 次完整游戏 | 同 prompt 的 8 个回答 |
| 动作 | 左／不动／右 | 词表 token |
| 奖励 | 接到 +1，没接到 −1 | 完整输出精确匹配 1／0 |
| 优势 | 组内均值与标准差归一化 | 相同 |
| 目标 | token／动作级概率比裁剪 | 相同，并处理 EOS、padding |
| 正则 | 熵系数 0.01；无 reference KL | 无 reference KL；只训练 LoRA |
| 监督训练 | 无 | 无 SFT 热身，直接从预训练指令模型开始 RL |

核心只有两件事：`group_advantages(rewards, G)` 构造冻结的相对评价；`clipped_loss(new_logp, old_logp, advantage, mask)` 把评价变成可微策略损失。这里按序列的实际动作数取均值，再按序列平均。两份训练脚本分别负责环境、生成、评分与检查点。

模型来自 [Qwen 官方仓库](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，精确 revision 记录在 [model_source.json](model_source.json)。默认 LLM 任务为四位数字反转，另支持两位数加法；最终采用配置与结果以实验报告为准。

## 运行方式

在仓库根目录使用已有 `.venv`。本机训练通过独立监护进程启动：

```bash
.venv/bin/python grpo_lab/check.py
.venv/bin/python grpo_lab/play_game.py
.venv/bin/python grpo_lab/guard.py --log grpo_lab/runs/game_resources.jsonl -- .venv/bin/python grpo_lab/game.py
.venv/bin/python grpo_lab/guard.py --log grpo_lab/runs/llm_probe_resources.jsonl -- .venv/bin/python grpo_lab/llm.py --eval-n 32
.venv/bin/python grpo_lab/guard.py --log grpo_lab/runs/llm_stage1_resources.jsonl -- .venv/bin/python grpo_lab/llm.py --steps 10
.venv/bin/python grpo_lab/guard.py --log grpo_lab/runs/llm_stage2_resources.jsonl -- .venv/bin/python grpo_lab/llm.py --resume --steps 10
```

实际训练输出目录为 `grpo_lab/runs/llm_reverse`，所以复现当前检查点时需加 `--out grpo_lab/runs/llm_reverse`。新的默认目录 `grpo_lab/runs/llm` 用于独立重跑。最终测试命令使用 `--resume --steps 0 --split final_test --eval-n 256`，基础模型对照省略 resume 并使用另一个输出目录。

LLM 默认本地权重路径 `grpo_lab/.cache/qwen`，该目录需先下载。每次 LLM 更新保存 adapter、优化器与随机状态；`--resume` 接着训练。未指定 resume 会从原模型重新初始化，因此新实验使用新输出目录。只加载本项目自己生成的 `training_state.pt`。

## 从公开仓库复现推理

最终 `runs/llm_reverse/adapter/` 已随仓库提供，包含第 20 步的 LoRA 权重与配置；基础模型从官方源按原 revision 下载。以下下载约 1 GB，需自行执行。命令均在仓库根目录运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r grpo_lab/requirements.txt
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", local_dir="grpo_lab/.cache/qwen", allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE"])'
.venv/bin/python grpo_lab/guard.py --log grpo_lab/runs/my_audit_resources.jsonl --max-seconds 900 -- .venv/bin/python grpo_lab/audit_inference.py --out grpo_lab/runs/my_audit
```

`my_audit` 必须是新目录。推理脚本只读取基础模型和 adapter，不需要本地未发布的 `training_state.pt`，也不会重新训练。它先推理基础模型，再挂载同一个保存的 LoRA；输出两份全部 256 题的文档、JSONL 和统计。跨依赖版本或硬件的输出可能有数值差异；原始实测版本见 `requirements-tested.txt`。

历史核验文档保留原机器绝对路径与哈希；公开复现采用上面的相对路径。生成的报告会链接本次运行日志；命令中的资源日志实际写在 `my_audit_resources.jsonl`，可在进程完成后复制到 `my_audit/resources.jsonl`，控制台输出同样可保存为 `my_audit/console.log`。其他平台若无法读取 macOS 热状态，监控值为 null，仍检查 CPU 与内存。

## 本机负载与热状态

本机为 Apple M2 Ultra、64GB。默认 CPU 两线程、nice 10、单个训练任务；GPU/MPS 只在确认负载后单独试验。训练期间每五秒读取系统热状态、CPU 总体使用率、可用内存、任务 RSS 和 load average，写入 JSONL。

[guard.py](guard.py) 在热状态达到 fair 或更高、CPU 总体超过 75%、可用内存不足 6GiB、任务 RSS 超过 9GiB 时暂停本任务进程组；连续四次恢复到阈值内才继续。各阶段还有时间上限，结束或异常时只清理自己启动的进程。

热状态来自 [Apple ProcessInfo](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum)，nominal=0、fair=1、serious=2、critical=3。它不是温度计，不提供具体摄氏度；不能据此保证绝对不会过热。若系统无法提供该信号，日志记录 null，不伪造温度。

用户要求已记入全局 `~/.codex/AGENTS.md`、记忆摘要与来源笔记。其跨项目含义是高负载工作要优先保守并行并监控、降载或暂停；不是授权自动购买云算力。

## 已验证的机制

`check.py` 核对不同任务不混组、相同奖励组优势为零、old log-prob 与 advantage 不被反向传播、mask 屏蔽动作，以及热状态触发保护。已有教程测试另覆盖因果 mask、EOS、PPO 正负优势裁剪。

实验结果、检查点和资源日志位于 `runs/`。公开版本保留最终 LoRA、结果与评测日志；基础权重缓存、其他中间权重及优化器状态由 `.gitignore` 排除。
