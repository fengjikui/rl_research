# Rainbow 本机打砖块实验

这份程序在我们已读懂的 DQN 上加入完整的六项 Rainbow 组件，仍然是自写 PyTorch 更新，没有调用高级 RL Trainer。游戏沿用 Atari 的原版 Breakout；不是重新编写一个打砖块游戏。所有结果写入新的运行目录，旧 DQN 权重和录像保持独立。

## 六项组件在哪里

| 组件 | 实现位置 | 在解决什么问题 |
|---|---|---|
| Double DQN | `rainbow_core.update` | 在线网络选择下一动作，目标网络评估它，减少最大值选择带来的高估 |
| Dueling | `RainbowQNet.logits` | 分别建模局面的价值和各动作相对优势，再逐原子合并 |
| C51 | `categorical_projection`、`RainbowQNet.distribution` | 为每个动作预测 51 个回报原子的概率分布；期望仍是普通 Q 值 |
| 优先经验回放 PER | `RainbowReplay`、`SumTree` | 按优先级采样，以 importance weights 修正损失；学习后更新优先级 |
| 多步回报 | `RainbowReplay.transition` | 累积最多三步实际奖励，再接折扣后的后续价值分布 |
| NoisyNet | `NoisyLinear` | 训练时使用可学习的参数噪声探索；评估时 `eval()` 关闭噪声 |

训练预热阶段仍均匀随机选动作，之后不额外使用 epsilon 贪心，而是用 NoisyNet 探索。噪声在每次学习和每四个交互决策处刷新。评估关闭参数噪声；独立复测另以 5% 概率随机选择动作。

## 输入、输出和损失

输入是 `[B,4,84,84]` 的 uint8 连续四帧。卷积变换依次为 `[B,32,20,20]`、`[B,64,9,9]`、`[B,64,7,7]`，展平后 `[B,3136]`。Dueling 的两路 Noisy 全连接输出 `[B,1,51]` 和 `[B,A,51]`，合并后沿原子维 softmax 得到 `[B,A,51]`。

`forward()` 返回分布的期望 `[B,A]`，所以环境交互和现有录像评估仍然能按最大的 Q 值选择动作。训练时按实际执行的动作取出 `[B,51]` 的 log probabilities；目标是停止梯度的 C51 投影分布。逐条交叉熵是 `[B]`，乘 PER 的 importance weights 后求平均得到一个标量，再反向传播、裁剪梯度、Adam 更新。目标网络不反向传播。

## 回放怎样省内存并处理边界

每条只保存当前观察最新一帧、下一观察最新一帧、动作、训练奖励、真实终止标记、轨迹边界、所在局起点和优先级。十万条的两组 uint8 图片总计约 1.41 GB，放在磁盘映射数组；需要时才取四帧。

抽样先由 SumTree 按优先级选择起点，随后累计最多三步奖励。真实 `terminated` 会令 bootstrap 折扣为零；`truncated` 只切断这条轨迹，仍用真实最后观察 bootstrap。新一局不能与上一局拼接。环形覆盖后最旧的三条暂不抽样，避免四帧重建依赖已覆盖的数据；最新不足三步且尚未结束的起点也不抽样。

续训会重置模拟器，新写入第一条前将上一段未结束的采集轨迹切断，以其保存的最后下一观察 bootstrap。它不伪造游戏失败，也不会把两个独立游戏接起来。

PER 使用分层随机采样，优先级为目标分布与预测分布的逐条 KL 散度加一个小正数的 alpha 次方（损失仍是交叉熵）；beta 从 0.4 逐渐增加到 1。importance weights 以本批最大值归一化。

## 与旧 DQN 的配方差异

本次目的是比较两套实际训练程序，不是声称只更换一个变量的消融实验。除了六项组件外，默认学习率由 `1e-4` 改为 `6.25e-5`，Adam epsilon 为 `1.5e-4`，目标网络同步间隔为 8000 决策步，梯度范数阈值为 10。C51 支撑集是 -10 到 10，共 51 个原子。网络的参数量和每次更新计算量也更大。

两者共用的游戏规则包括四帧堆叠、每动作执行四个原始帧、训练奖励取符号、完整游戏终止、无 sticky actions、重置或丢球后自动发球。动作记录保存真实执行的发球操作。评估和录像累计原始游戏奖励。

这是完整六组件的教学实现，但预热、回放容量、优化器、同步周期、完整游戏终止等不完全复刻原论文。Rainbow 不保证每个随机种子都比 DQN 高分，也不能用单次固定开局成绩宣称稳定学会。

## 启动与继续

先确认旧任务已经停止，再在仓库根目录运行。本次已完成六项 CPU 检查、本机 MPS 短训与恢复续训，并用独立 NumPy 投影/损失复算和 CPU 重载验证实际保存的模型输出。短测仅验证训练流程，不能用截短评估分数推断学习效果。

2026-11-02 的正式实验由 `continue_rainbow.py` 启动，输出为 `dqn_lab/runs/rainbow_mps_20261102/`。它将每段半小时的训练依次串行运行，每五分钟保留阶段模型、分数和录像，每段末尾用十局 epsilon=0.05 复测。200/300/400 分需三个不同且相隔至少一万交互步的检查点复测达标；400 分确认后结束。热状态、CPU、内存和磁盘压力会阻止启动或暂停本任务。旧模型按两小时完整保留、之后每小时代表节点及历史最佳/里程碑保留的规则精简，原始分数与动作日志始终保留。

修改该实验 `continuous_control.json` 的 `enabled=false` 会阻止下一段启动；当前段正常完成。立即停训应先关闭续训开关，再对 `run/status.json` 中经命令行核实的本任务训练 PID 发 SIGTERM，以保存最新学习状态。

下面是单段手动运行示例，请使用新目录，不要与已经运行的任务重叠。

```bash
.venv/bin/python grpo_lab/guard.py \
  --log dqn_lab/runs/rainbow_breakout/resources.jsonl \
  --max-seconds 2700 --thermal-pause-at 2 -- \
  .venv/bin/python dqn_lab/train_rainbow.py \
  --out dqn_lab/runs/rainbow_breakout/run --device mps \
  --seconds 1800 --stage-seconds 300
```

再次运行时添加 `--resume`。每段默认训练窗口半小时，包括中间评估，训练前、最后和独立复测会增加额外时间。更长预算可修改 `--seconds`，保护程序的最大时间也要相应留出评估余量。保护程序只暂停本任务自己的进程组，严重热状态、内存或负载压力触发暂停；未取得摄氏温度读数时不能报告具体温度。

## 结果与恢复文件

- `status.json`：实时交互步、更新次数、已完成游戏、PER beta、回放量和设备。
- `stages.json`、`REPORT.md`、`evaluation_curve.png`：训练前、每五分钟、最后的固定种子评估；三种子可能走出同一轨迹，不能当作三份独立泛化证据。
- 每阶段目录：实际评估的 `model.pt`、`evaluation.json`、逐步 `episode_*.jsonl`、第一局全程 `play.mp4` 和短 GIF。达到单局步数上限会明确标为 capped。
- 最后阶段的 `heldout/`、根目录 `latest_heldout.json`：另外十个种子、epsilon=0.05 的复测，独立记录。
- `updates.jsonl`：逐次损失、更新编号、PER beta。`episodes.jsonl`：每个完成的训练游戏原始分数。
- `first_batch/`：第一批输入、C51 目标投影、实际动作 log probabilities、Double 下一动作、IS 权重和实际 loss，便于独立复算。
- `latest.pt` 加 `replay/`：完整恢复材料；每分钟与每次阶段评估后保存。包含两个网络、优化器、随机数状态、累计步数、回放计数和配置；回放保留原始帧和优先级。`source/` 留存本段实际源码。

阶段 `model.pt` 只供推理，不能与最新回放配对来恢复训练。恢复使用 `latest.pt` 和同一目录下最新 `replay/`。突然断电时磁盘映射数据与检查点可能不同步；程序检查计数，不匹配就拒绝恢复，不能声称可任意断电无损恢复。正常 SIGTERM 会结束循环并保存学习状态。

可单独复测一个模型：

```bash
.venv/bin/python dqn_lab/evaluate_rainbow_saved.py \
  --model dqn_lab/runs/rainbow_breakout/run/某阶段/model.pt \
  --out dqn_lab/runs/rainbow_holdout_new --device mps
```

复测也应放在 guard 之下，避免与训练抢占算力。

## 核验

```bash
.venv/bin/python dqn_lab/test_rainbow.py
```

这些 CPU 检查覆盖 C51 整数原子和截断端点的概率守恒、Noisy 训练与评估、PER 树、n-step 跨终止/截断/环形覆盖、续训边界、Double 选动作、IS 加权损失以及目标停止梯度。它们验证关键语义，不代替真正的 MPS 环境短测和最终独立游戏评估。

资料：[Rainbow 原论文](https://arxiv.org/abs/1710.02298)、[CleanRL Rainbow 实现说明](https://docs.cleanrl.dev/rl-algorithms/rainbow/)。
