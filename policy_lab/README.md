# 用现成视觉游戏学习策略梯度

这里把 **REINFORCE → Actor-Critic → PPO** 放在同一套可读 PyTorch 代码里。游戏由 Gymnasium 或 Craftium 提供，代码负责采样、计算训练信号和更新网络，没有使用高级 RL Trainer。

**实现完成不等于已经学会游戏。** 本目录的 CPU 单元测试验证公式和边界；真实游戏成绩必须以某次运行保存的评估、原始轨迹和录像为准。Craftium 插件接口不代表本机已安装或训练成功；现有 Catch 教程的成绩也不能迁移为这里的成绩。

## 本机实际验证范围

已在真实 `CarRacing-v3` 上完成三个算法的 CPU 流程短测。初次独立核验快照为：

| 算法 | 采集决策步 | 已优化的采样批数 | 实际参数更新次数 |
|---|---:|---:|---:|
| PPO | 541 | 67 | 134 |
| Actor-Critic | 414 | 51 | 51 |
| REINFORCE | 429 | 8 | 8 |

历史日志里的 `updates` 是“有多少批采样数据进入了学习函数”，不是 `optimizer.step()` 次数。这个 PPO 短测每批做 2 个 epoch，每个 epoch 只有 1 个 minibatch，所以 67 批对应 134 次参数更新；原始记录不改写，核验文件另加派生计数解释。默认配置一批有 256×4=1024 个动作，每个 minibatch 64 个动作、做 4 个 epoch，一批会更新参数 4×16=64 次。比较算法的计算量时要同时看采集步、采样批数和参数更新次数。

这些运行保存了训练前后权重和实际 MP4；独立重载后的动作概率、逐步原始奖励之和和 NumPy 重算的 GAE 均通过核验，见 [原始核验记录](runs/cpu_smoke_20261102/verification.json)。测试刻意采用很短训练预算、24 步训练回合上限和 12 步评估上限，**只证明采样、更新、保存、推理与录像流程跑通，不证明学会赛车或提高得分**。后续续训会增加相应目录的累计步数。

Craftium 尚未在本机安装或测试；CUDA 尚未在 4090 上测试；当前不投入 A3 部署与适配。不要把通用后端分支当作上述硬件已验证的证据。

## 先看哪几份代码

| 文件 | 先找什么 |
|---|---|
| `envs.py` | 原始 HWC uint8 图像如何变成四张连续灰度图；真实终局观察在 reset 前返回 |
| `policy_core.py` | CNN 的 actor 输出动作 logits，critic 输出 V(s)；三种损失与 GAE |
| `rollouts.py` | 动作从概率分布中采样，记录实际动作的 old log probability；完整回合与窗口的区别 |
| `train_visual_policy.py` | 环境、采样、更新、阶段保存和评估怎样串起来 |
| `evaluation.py` | 固定种子真实推理、逐步原始奖励、第一局完整 MP4 |
| `runtime.py` | 显式设备选择和可恢复学习状态 |

输入处理为 `[H,W,3] uint8 → 灰度缩放 → [4,64,64]`，四帧提供运动线索。批量进入 CNN 是 `[B,4,64,64]`，卷积依次得到 `[B,32,15,15]`、`[B,64,6,6]`、`[B,64,4,4]`，展平后映射为 256 维共享特征。actor 输出 `[B,A]` 的 logits，critic 输出 `[B]` 的状态价值。模型不再给每个动作预测 Q 值；动作概率由 `Categorical(logits=...)` 得到。

## 三种训练方式

**REINFORCE** 等完整回合结束，计算真实奖励的 return-to-go，再用 `-log π(a|s) × G` 形成损失。代码另外使用 `γ^t`，并按完整回合的梯度和取平均，避免把变长回合直接拼接后按随机总长度缩放。这里不使用 critic 基线。为易读只使用单环境；末尾因墙钟预算中断的半条轨迹不拿来假装完整回报。

本实现把环境的 TimeLimit 视为有限任务的回合终点，REINFORCE 不在这里 bootstrap；它不声称这是无限时域任务在任意截断下的无偏梯度。

**Actor-Critic** 按采样窗口计算 GAE，使用固定优势乘 log probability 更新 actor，用固定 return target 拟合 critic。一批数据只做一次累计梯度更新；小批次只是节省视觉网络的反向图内存。

**PPO** 使用同样的窗口和 GAE，但保留采样时的 `old_logp`，在多个 learner epoch 内保持它不变。新策略重新前向计算 log probability，构成新旧概率比值，再计算 clipped surrogate。优势和价值标签停止梯度。PPO 之外不复用旧窗口，也没有经验回放池。

AC/PPO 在真实 `terminated` 后不 bootstrap；`truncated` 使用 reset **之前**真实末帧的 V(s') bootstrap，但同时切断跨回合的 GAE 递推。窗口末尾未结束时仍 bootstrap。由于这个处理与上述有限任务 REINFORCE 的 TimeLimit 目标不同，跨算法比较时需要明确这个目标差别。默认还有 `entropy_coef=0.01` 的探索正则；设为 0 才是不带此项的纯损失。

## 可选现成游戏

- `CarRacing-v3`：默认实战后备环境，强制 `continuous=False`，离散动作；需要 Box2D。图像来自实际游戏，不是手写的模拟赛道。
- `Craftium/SmallRoom-v0`：较小的 Minecraft 风格房间，适合作为 Craftium 起步；找到目标越快，累计负奖励越少。
- `Craftium/Room-v0`：较大的随机起点/目标房间，3D 视觉寻路。
- `Craftium/SpidersAttack-v0`：第一人称持剑与蜘蛛战斗，杀敌获得奖励，学习难度更高。

Craftium 环境在首次创建前自动 `import craftium`，其现成任务已提供离散动作。安装可能涉及引擎编译与 OpenGL；本脚本不会替你安装、绕过缺依赖，或自动切换成别的游戏。官方资料：[环境规则](https://craftium.readthedocs.io/en/latest/environments/)、[项目及安装入口](https://github.com/mikelma/craftium)。

## 运行与预算

在仓库根目录运行；先停止其他高负载训练，避免抢占本机 GPU。需要兼容的 `torch`、`numpy`、`gymnasium`、`pillow`、`psutil` 和外部 `ffmpeg`；CarRacing 另需 Gymnasium 的 Box2D 依赖，Craftium 另按其官方安装文档准备环境。不要覆盖已有正在训练的 Python 环境。

新机器先创建独立虚拟环境，按该机器选择 PyTorch wheel，然后安装 `python -m pip install -r policy_lab/requirements.txt`。`ffmpeg` 通过系统包管理器安装。本机历史短测把 Box2D 2.3.10 wheel 放在隔离路径中使用，不表示原 `.venv` 已全局装好 Box2D；公开读者按本依赖文件准备即可。大型本机检查点不随 Git 发布，运行脚本会重新保存本次训练前后权重。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python grpo_lab/guard.py \
  --log policy_lab/runs/car_ppo/resources_guard.jsonl \
  --max-seconds 3000 --thermal-pause-at 2 -- \
  .venv/bin/python policy_lab/train_visual_policy.py \
  --algo ppo --env CarRacing-v3 --device cpu \
  --out policy_lab/runs/car_ppo/run \
  --seconds 1800 --stage-seconds 300 --num-envs 4
```

使用现成 CUDA 服务器时将 `--device cpu` 改为 `--device cuda`。可用 `--device mps` 指定苹果 GPU；**不可用就报错，不会静默回退 CPU**。通用 `npu` 分支仅做显式选择，尚未验证，当前阶段不开展 A3 适配。

用 REINFORCE 时指定 `--algo reinforce --num-envs 1`；用 Actor-Critic 时指定 `--algo ac`。使用 Craftium 时替换 `--env Craftium/SmallRoom-v0`。不同算法需要独立输出目录。

默认 30 分钟是训练窗口，包含中间评估，训练前和最终评估另计时间；单个引擎 step、更新或视频编码也可能增加少量尾部耗时。`--eval-max-steps` 是**评估决策步数上限**，不是分数。`--max-episode-steps` 会更改任务时间限制，比较时必须相同并记录。`--action-repeat` 的默认值为 1，累计原始奖励并保留实际执行的原始 step 数。

`--num-envs` 默认 4，使用同步 CPU 游戏列表，策略一次批量前向。它是容易读懂的初版，不是吞吐最优的异步采样系统；4090 的利用率应当实测，不能用这份默认参数直接宣称硬件性能上限。先观察环境每秒 step 数、采样与学习耗时，再决定是否加环境。

## 保存、复核与继续

- `before_*/model.pt`、`final_*/model.pt`：实际训练前后权重；中间默认每五分钟也保存和评估。
- 各阶段 `evaluation/evaluation.json`：固定种子各局分数、均分、标准差和 capped 标记。
- 各阶段 `evaluation/episode_*.jsonl`：真实动作、概率、单步奖励、累计奖励、终止标记和观察哈希；`play.mp4` 是第一局完整录像或明确上限内的完整片段，未挑选最高分录像。
- `episodes.jsonl`：训练时每个实际完成的游戏原始分数。
- `updates.jsonl`：actor/critic 损失、熵、PPO 近似 KL/裁剪比例、采样步数、固定 old_logp 的哈希。
- `first_batch/`：第一批实际观察、动作、old log probability、优势、标签；AC/PPO 另存 GAE 的奖励/真实终止/截断/价值原始输入。
- `config.json`、`source/`：实际配置与源码；`status.json`：当前设备、累计决策、原始 step 数、更新和完成回合数；`REPORT.md`：阶段对照。
- 新记录同时保存 `rollout_updates`（已优化的采样批数）、`optimizer_steps`（累计实际 `optimizer.step()` 次数）和 `optimizer_steps_this_update`（本批的参数更新次数，位于逐批日志）。旧 `updates` 保留为 `rollout_updates` 的兼容别名。`optimizer_steps_since_resume` 另记本次进程段实际更新次数。
- `latest.pt`：在一次学习结束的边界，若距上次保存已满一分钟就保存；阶段评估后也保存。它包含模型、优化器、随机数状态和上述计数，同样命令添加 `--resume` 可继续。较慢的游戏回合或采样窗口可能超过一分钟，不能理解成严格每 60 秒定时保存。模拟器和未更新的 on-policy 窗口不恢复，重新从新回合采样。

旧检查点缺少真实参数更新计数时，程序用完整 `updates.jsonl` 的批大小和已保存的 epoch/minibatch 配置推算，并与 Adam 保存的每参数 step 计数交叉核对。证据不完整或不一致时，累计 `optimizer_steps` 记为 `null`，只明确记录续训之后真正发生的更新，不伪造历史总数。

本机脚本在训练前和运行中检查可用内存、CPU 负载和可取得的系统热状态；配套 guard 可暂停本任务进程组。Linux 上这不代替显卡温度监测，服务器操作者还应检查实际 GPU 负载、显存和温度。没有温度读数不能称测得具体摄氏温度。

## 轻量正确性测试

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python policy_lab/test_policy.py
```

测试仅使用 CPU 的小张量与测试环境，不调用 GPU、Box2D 或 Craftium。覆盖单个动作在正/负优势下的概率方向、两种不同长度回合的 REINFORCE 平均梯度、GAE 真实终止/截断/窗口边界、采集时先看真实末帧再 reset、PPO 固定旧概率与裁剪，以及 CNN 和灰度叠帧形状。真实游戏与加速器的训练效果仍需单独实测。
