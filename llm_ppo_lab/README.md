# 从游戏 PPO 走到逐 token 的语言模型 PPO

这份代码让已经读懂游戏 Actor–Critic/PPO 的读者，看到同一个训练循环怎样用于语言模型。它不调用 PPOTrainer。默认运行一个从随机参数开始的小型因果 Transformer；显式 `--backend hf` 才加载 Qwen2.5-0.5B-Instruct 和 LoRA。后者是可执行入口，不代表已在本机或 4090 上完成这个预训练模型的 PPO 训练。

## 先读这些文件

| 文件 | 看什么 |
|---|---|
| [models.py](models.py) | 状态是 token 前缀，Actor 输出词表 logits，Critic 输出前缀价值；Tiny 和 HF 的接口相同 |
| [train.py](train.py) 的 `generate` | 从当前概率中实际采样 token，何时 EOS、何时达到预算 |
| [core.py](core.py) 的 `token_statistics` | 为什么第一个回答 token 的概率与价值都从 prompt 最后一个位置取 |
| [core.py](core.py) 的 `build_rewards` | 每个有效 token 加 KL 成本，只在最后一个有效动作加一次任务分 |
| [core.py](core.py) 的 `finite_episode_gae` | 如何从末尾向前计算优势与 Critic 回归目标 |
| [train.py](train.py) 的 `update` | 固定旧概率、旧价值、优势和目标；多轮重算当前模型再反传 |
| [infer.py](infer.py) | 只替换检查点推理，不进行训练 |

详细概念请配合仓库主教程阅读。这里只说明实现与实验范围。

初学者先读[逐步讲解](逐步讲解_LLM的奖励回报与PPO.md)，再看上面的源码顺序。[本次公开原始证据](evidence/tiny_cpu/README.md)保留真实输入输出、奖励与优势：微型模型前后都是0/16，明确只是流程验证。大型检查点留在本机，公开复现需要自行运行示例。

## 一批数据怎样对应模型输出

`prompt` 是 `[B,P]`，左侧 padding 由 `prompt_mask` 排除；`response` 是 `[B,T]`。合并后的 LM 输出为 `[B,P+T,V]`，价值输出为 `[B,P+T]`。第 t 个动作的状态是 **生成它之前的全部有效前缀**，因此从合并输出中的 `P-1+t` 位置取得该动作 log-prob 和价值，不让价值估计看到自己要评价的动作或未来 token。

`mask` 为 `[B,T]`：第一个 EOS 算有效动作，EOS 后 padding 不算，即使 PAD 与 EOS 使用同一个编号。prompt 自身不进入策略或价值损失。当前实现对全部有效回答 token 求平均，较长回答贡献更多 token；这不是先对每个回答求平均再对回答求平均。

采样使用原始模型的概率，temperature=1，没有 top-k/top-p 等截断。因此训练重算的 log-prob 与实际行为策略一致。训练时仍保持 `model.eval()` 关闭 dropout；它不会阻止 autograd 求参数梯度。

## 奖励 回报和优势

Tiny 使用两位数字字符串反转，HF 使用四位数字字符串反转，均保留答案中的前导零。两个后端用于不同尺度的教学，不能直接比较成功率。任务分 `score` 是完整回答精确匹配的 0/1，可选 `--missing-eos-penalty` 再扣缺失 EOS 的分数。默认惩罚为 0，达到长度上限的答案若恰好匹配仍可得到任务分；结束原因在每条原始记录中公开。

每个有效回答动作有：

```text
sampled_kl[t] = old_logp[t] - reference_logp[t]
reward[t] = -beta * sampled_kl[t]
reward[last_valid_action] += task_score
```

任务分只加一次。KL 样本在个别 token 上可为负，但在相应策略分布下求期望得到非负 KL。奖励随后进入 GAE；它不是把同一个 0/1 直接当所有 token 的价值回归标签。`returns = advantages + old_values` 是冻结的 Critic 目标。优势可以在有效 token 上标准化，但价值目标不跟着标准化。

本代码把 **第一个 EOS 或指定 max_new_tokens 视为有限任务终点**，都令最后 bootstrap 价值为 0；后者是一项明确的任务定义，不声称任何意外截断都应该这样处理。它与游戏中“任务继续但采样窗口停止，仍要 bootstrap”不同。默认 γ=1；GAE λ 默认 .95，所以回归目标不总是纯蒙特卡洛奖励总和。

旧策略是采集当前这批数据的策略快照，通过保存的旧 log-prob 表示；参考策略是固定初始模型。Tiny 使用冻结副本。HF 从零初始化 LoRA 开始、不做 SFT 热身，基础模型权重冻结，禁用 adapter 就得到不变的初始参考策略，避免再复制一整个 Qwen。共享 LM 特征上额外加 value head；LoRA 与 value head 一起更新。

KL 已放进奖励里，本实现没有再额外叠同一个 reference KL loss。PPO 的新旧概率比约束与参考模型 KL 也不是同一个东西。GRPO 的差别见既有 [GRPO 实战](../grpo_lab/README.md)：那里以同题多回答的相对分数构造优势，没有单独学习的 Critic；其实际训练配方没有 reference KL，不能混用成绩。

## 默认无下载的小模型短跑

在仓库根目录使用已有 `.venv`。保持本机长训练停止；以下仅是几秒级流程验证示例。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python grpo_lab/guard.py \
  --log llm_ppo_lab/runs/my_tiny_resources.jsonl --max-seconds 60 -- \
  .venv/bin/python llm_ppo_lab/train.py \
  --backend tiny --device cpu --updates 2 \
  --out llm_ppo_lab/runs/my_tiny
```

同样命令添加 `--resume` 可继续。`--updates` 是本次追加多少批 rollout，`rollout_batches` 是累计批数；每批做 `--epochs` 次实际 `optimizer.step`，另用 `optimizer_steps` 计数。每轮内部按一份回答做微批，依据其有效 token 数加权累积梯度，等价于整批有效 token 平均的损失。

Tiny 的训练集有 64 个两位数字任务，验证集 16 个，测试集 20 个；默认回答最多 4 token。HF 的新四位任务划分是训练 1024、验证 128、测试 256；默认回答最多 8 token，且先排除旧 GRPO 实验中 validation、test、final_test 共 512 个评估输入。固定排除清单及来源随仓库提交在 [data/reserved_grpo_eval_ids.json](data/reserved_grpo_eval_ids.json)，不依赖本机是否保留旧 runs。各自固定划分互不重叠。默认训练只评验证集；最终测试由独立推理入口显式选择，不能反复看测试分来挑检查点。

## NVIDIA 上的真实预训练模型入口

先按官方 PyTorch 安装方式选 CUDA wheel，再安装本目录依赖。使用新输出目录：

```bash
python llm_ppo_lab/train.py \
  --backend hf --device cuda \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --batch-size 4 --max-new-tokens 8 --updates 20 \
  --out llm_ppo_lab/runs/qwen_ppo
```

HF 分支显式加载模型，未缓存时会下载；`--local-only` 可禁止联网下载。FP32、LoRA r=8、微批一份短回答是便于理解和检查的保守配方，未测量 4090 训练时长或显存峰值。先跑 1–2 批、检查 `nvidia-smi`、奖励与长度，再决定加预算。没有默认 A3 适配，也不声称 NPU 验证通过。

## 保存与只推理

- `before.pt`、`after.pt`：真正保存的学习状态。HF 保存 LoRA 与 value head，不重复写被冻结的基础模型；恢复时仍需同一基础模型版本。
- `latest.pt`：在线模型、优化器、随机数状态、完整批次计数与配置。Tiny 还保存固定参考模型。只使用自己生成且信任的文件，加载使用 `weights_only=True`。
- `update_*/rollout.json`：原始题目和回答、token ids、mask、旧/参考 log-prob、旧价值、逐 token 奖励、优势和回归目标，均可直接读 JSON。
- `update_*/learner.json`：各 epoch 的策略损失、价值损失、熵、近似新旧 KL 和裁剪比例；`updates.jsonl` 保存每批概要。
- `validation_*.json`：固定验证题的全部实际输出；`manifest.json`、`config.json`、`source/` 保存任务划分和本段源码。

如果一次更新中途被打断，只从上一个完整检查点恢复。较新的未完成 `update_*` 目录会改名成 `.interrupted_*` 保留，不覆盖证据，再重新采集和训练。未保存的梯度更新不算恢复成功；不承诺任意断电位级复现。

```bash
python llm_ppo_lab/infer.py --checkpoint llm_ppo_lab/runs/my_tiny/before.pt \
  --out llm_ppo_lab/runs/before_inference.json
python llm_ppo_lab/infer.py --checkpoint llm_ppo_lab/runs/my_tiny/after.pt \
  --out llm_ppo_lab/runs/after_inference.json
```

两个命令只换检查点，评估任务和解码保持一致。它们不更新参数。加 `--split test` 才查看留出的最终测试。

## 已验证与尚未验证

2026-11-03 的最终 CPU 短测采集 2 批 rollout、执行 4 次优化器更新，并通过保存后继续学习和独立前后推理；两次验证均为 **0/16**，没有能力提升结论。两批原始数字、独立 NumPy 重算的逐 token 奖励/GAE/首轮策略损失核验，以及不变 reference 的检查均见 [小型公开证据](evidence/tiny_cpu/verification.json)。[训练前推理](evidence/tiny_cpu/before_inference.json)、[训练后推理](evidence/tiny_cpu/after_inference.json)、[第一批原始数据](evidence/tiny_cpu/rollout_0001.json) 可直接阅读；权重和大型运行文件留在本地，按上述命令重新运行可生成自己的检查点。

本机仅做 CPU 小型验证，没有启动 Qwen 0.5B PPO 长训练或 NVIDIA 训练。单位测试覆盖 EOS/PAD、一次终局奖励、signed KL、变长 GAE、动作前价值与因果对齐、padding 梯度、旧数据和参考模型冻结、token 平均，以及本地随机微型 Qwen2＋PEFT 的 LoRA/value 更新、固定 base reference 和 adapter 恢复；随机小 Qwen2 不等于预训练 Qwen 能力验证。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python -m unittest discover -s llm_ppo_lab -p test_ppo.py -v
```

几秒的随机 TinyLM 短跑只证明训练管道运行。若任务奖励全为零，也不能从 loss 或参数变化声称语言能力提高；必须用独立评估和足够预算检验。
