# 强化学习实践教程：从第一行梯度到语言模型与工具智能体

目标是亲手回答五个问题：模型吃什么、输出什么、数据怎么来、损失怎么算、更新以后行为是否改变。每课都包含完整代码入口、张量形状、关键训练语句、观察指标和修改实验。建议按课完成，不要求在一次 90 分钟报告中全部讲完。

## 课程导航

| 课 | 实践与问题 | 代码 |
|---|---|---|
| [00｜建立训练的整体认识](lessons/00_训练的共同骨架.md) | 网络结构、目标、采样、优化和部署如何分工 | 阅读＋前一版梯度核验 |
| [01｜从老虎机到 Q 表](lessons/01_老虎机与Q表.md) | 第一个真实采样更新；RL 不一定需要神经网络 | `00_bandit.py`、`01_q_table.py` |
| [02｜DQN 学会接球](lessons/02_DQN接球.md) | 自写环境、经验回放、目标网络与回放动画 | `02_dqn.py` |
| [03｜直接训练动作概率](lessons/03_策略梯度与ActorCritic.md) | REINFORCE → Actor–Critic → GAE | `03_policy_gradient.py` |
| [04｜把它变成 PPO](lessons/04_PPO逐行拆解.md) | 固定旧策略数据，多次更新与裁剪 | 同一脚本的 `--algo ppo` |
| [05｜结构为什么重要](lessons/05_模型结构的对照实验.md) | MLP／CNN 与记忆 MLP／GRU 对照 | `02_dqn.py`、`04_memory.py` |
| [06｜微型语言模型的三种训练](lessons/06_微型语言模型SFT_DPO_GRPO.md) | 同一个因果 Transformer，替换 SFT／DPO／GRPO | `05_tiny_lm.py` |
| [07｜开源预训练 LLM 的 RL](lessons/07_预训练语言模型实战.md) | LoRA、分组生成、奖励、EOS mask、留出评估 | `07_pretrained_grpo.py` |
| [08｜工具调用与长程任务](lessons/08_从工具调用到长程Agent.md) | 先跑两次调用，再设计真实代码修复训练 | `06_tool_agent.py` |
| [09｜迁移到更复杂的游戏](lessons/09_贪吃蛇与Mario扩展.md) | 贪吃蛇、飞机游戏、Mario 的具体迁移点 | 扩展指南；未实现这些游戏适配 |
| [专题｜如何设计一个新结构](lessons/A_如何设计新的模型结构.md) | 信息、归纳偏置、优化、成本、消融与设计流程 | 结合第 05、06 课 |

## 先运行一个实验

以下命令在仓库根目录执行。Python 3.11 是本次实测环境；普通 CPU 即可完成第 01–06、08 课。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r tutorials/requirements.txt
python tutorials/code/00_bandit.py
python tutorials/code/02_dqn.py --out tutorials/runs/my_dqn
```

当前工作目录已有隔离环境 `.venv`，可直接激活；不必覆盖它。`requirements-tested.txt` 记录本机实测的完整版本；它不锁定 GPU 驱动或 CUDA。NVIDIA 机器先按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/) 安装适配驱动的 PyTorch，再安装其余依赖。GPU 课另装 `requirements-llm.txt`。

每次实验在 `--out` 目录保存 `results.json`、`curve.png`，有网络的课程还保存 `model.pt`。接球实验保存实际轨迹的 GIF。配置、参数量、网络打印、运行时长、Python/PyTorch 版本与指标都在 JSON 中。重复使用同一个输出目录会覆盖这些产物，比较实验请换目录。

核心实测环境为 Python 3.11、PyTorch 2.14；GPU 课相关依赖另记录在 [requirements-llm-tested.txt](requirements-llm-tested.txt)。完整版本锁定不保证跨设备浮点结果逐位一致。

```bash
python tutorials/code/check_invariants.py
python tutorials/code/run_cpu_suite.py
```

第二条运行全部 13 个小实验组合，并覆盖相应默认输出目录；不下载预训练权重。核心代码使用 PyTorch 自动微分，没有调用高级 RL Trainer。

## 如何判断自己学会了

每课完成三件事：先描述一批数据的形状；再指出哪项停止梯度、哪项重新前向；最后只改一个变量并解释曲线。能跑到高分只是开始，能解释为什么不学习、为何指标失真、哪种改动不公平，才是完整实践。

建议分四次实验室活动：第 01–02 课一次，第 03–05 课一次，第 06–07 课一次，第 08 课与结果复盘一次。90 分钟报告选接球回放、记忆对照、微型语言模型三个演示，训练过程提前完成。

## 证据与范围

实测结果见 [实验记录](RESULTS.md)。小型实验的成功不等于任意游戏、通用语言能力或小时级任务的提升。微型 LM 枚举的是训练支持集；预训练 LLM 脚本另设互不重叠的训练／验证／测试输入。

后续已在本机 M2 Ultra 上开展真实 Qwen2.5-0.5B-Instruct 的 GRPO＋LoRA 训练，见 [手写 GRPO 实战](../grpo_lab/README.md)。这与本目录初版的 13 个小型实验分别记录。Mario、贪吃蛇和真实仓库 Agent 仍属于明确标注的后续扩展。

机制推导可配合 [已有梯度专题](../docs/06_为什么奖励能通过梯度改进行为.md)，长程论文可配合 [已有 Agent 调研](../docs/07_长程Agent强化学习怎样实际训练.md)。
