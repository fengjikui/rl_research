# 从一只站不稳的狗开始理解 PPO

我们先想一个简单目标：在电脑里放一只宇树 Go2，给它“向前走”的速度指令，希望它走得稳、尽量跟上指令，并且不要靠疯狂甩腿得分。我们不先规定每条腿每一刻该怎么摆，而是让它反复尝试，再根据结果调整动作选择。

下面跟踪官方 **Isaac Lab v2.3.2 的 Go2 平地任务**。先读前六节建立整体认识，再去第七节点开源码。不需要先装仿真器才能读懂。

## 1 训练最初在发生什么

一开始，网络还不会走路。它给四条腿一些尝试性的指令，机器狗可能晃动、停住或身体碰地。仿真器负责计算接触、重力、电机和身体运动，告诉我们发生了什么。

学习程序把这种尝试保存下来：刚才看到了什么、下了什么指令、结果如何。之后用 PPO 更新网络，再尝试下一批动作。训练想要改善的是长期累计奖励；能否表现为好看的步态，需要看奖励设计与实际录像，不能只看 loss 下降。

```text
身体状态与运动指令
        ↓
策略网络给出连续动作分布 → 采样 12 个动作数值
        ↓
把动作换成关节位置目标 → 电机模型与物理仿真执行
        ↓
新身体状态、奖励、是否需要重置
        ↓
收集一小段经历 → 计算优势与价值目标 → PPO 更新
```

这里每一环都有现成实现。我们主要读清楚环与环之间传什么数据，而不是先重写整套系统。

## 2 网络看到的不是整只机器狗的照片

这款平地任务使用数值观察。你可以把它想成每只狗的一张“当前身体检查表”：身体移动多快、转动多快、重力朝哪个方向、希望它往哪里走、关节当前偏离默认姿态多少、关节动多快、上一次给了什么动作。

| 观察项 | 对单只 Go2 的维数 | 直观含义 |
|---|---:|---|
| 身体线速度 | 3 | 向前、侧向、竖直的运动速度 |
| 身体角速度 | 3 | 身体绕三个轴转动的速度 |
| 身体坐标系中的重力方向 | 3 | 身体是否倾斜、倒向哪边 |
| 速度指令 | 3 | 希望前进、侧移、转向多快 |
| 关节相对位置 | 12 | 四条腿各三个关节，离默认姿态多少 |
| 关节相对速度 | 12 | 这些关节正在怎样运动 |
| 上一次动作 | 12 | 给策略提供上一拍控制的信息 |

按本教程固定版本的拼接项手算，平地观察是 **48 维**；粗糙地形配置还会加入地形高度扫描。实际运行仍应核对环境打印的观察项与维度，不能把 48 这个数字当作所有 Go2 任务的永久接口。[观察配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L116)、[平地任务去掉高度扫描](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py)。

若并行放 64 只狗，这张表就是 `[N=64,48]`。每一行对应一只狗，不是 64 个时间步。平地 Actor 是三层 128 维隐藏层的 MLP，输出每个关节动作分布的均值。Critic 是另一路 MLP，为每只狗输出一个状态价值。[PPO 与网络配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py)。

在这一任务里，动作是连续数值。因此 Actor 不输出“向左、停留、向右”三个概率，而是给出 12 维高斯分布。该默认配置的标准差也是可学习参数，训练时从分布里采样，推理时通常使用均值。12 个维度的 log 概率相加，得到这一个完整动作向量的 log 概率；PPO 比较的也是同一个已采样向量的新旧概率密度。[Actor–Critic 实现](https://github.com/leggedrobotics/rsl_rl/blob/c49573e4d1eb6662d7b047fe935522c23665f6f3/rsl_rl/modules/actor_critic.py)。

## 3 输出的 12 个数怎样让腿动起来

每条腿有三个受控关节，四条腿共 12 个。因此一批动作是 `[N,12]`。但这些数并不是直接施加到关节上的力矩。

本任务使用位置目标控制，简化成：

\[
q_{target}=q_{default}+0.25\,a.
\]

`a` 是策略动作，`q_default` 是该关节默认角度。Go2 任务把动作缩放系数设为 0.25，并使用默认姿态作为偏移。例如某关节默认角度是 0.8 弧度，这次动作是 0.4，那么给控制器的位置目标就是 `0.8 + 0.25×0.4 = 0.9` 弧度。这里的动作值不应自行理解成角度或力矩，单位要沿着处理代码确认。[Go2 动作缩放](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py)、[动作处理](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)。

接下来，电机模型根据目标与实际位置、速度的差异计算作用。用熟悉的近似写法表示，就是位置误差乘刚度，再减去速度误差对应的阻尼项；实际还受电机与力矩限制。网络负责提出目标，控制器与物理引擎负责把目标变成运动。不能把“策略输出关节位置目标”说成“策略直接输出最终身体姿态”。[Go2 资产与电机参数](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_assets/isaaclab_assets/robots/unitree.py#L140)。

一次策略决策后，仿真还会推进几小步。本版本物理步长为 0.005 秒，每次动作推进 4 个物理步，因此策略控制间隔为 0.02 秒，即每秒 50 次决策。这个 `decimation=4` 是控制与物理执行频率的关系，不是 batch 大小，也不是叠四帧。[环境时间配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L305)。

## 4 它怎么知道自己走得好不好

先看“跟上速度指令”：要求向前 1 米/秒，实际只有 0.8 米/秒，会得到一定的跟踪奖励；如果越来越接近要求，这部分通常会更高。但如果只奖励向前速度，策略可能找到很不稳、很费力的动作。

因此环境把多项评价合起来，包括线速度跟踪、转向跟踪，以及竖直晃动、身体翻滚、关节力矩、关节加速度、动作跳变、姿态偏离等惩罚或约束。它们的权重决定任务更看重什么。平地任务还覆盖了部分 Go2 粗糙地形设置，读源码时要沿继承链看到最终生效值。

源码先计算每项函数，再乘权重和本次策略时间间隔：

\[
r_t=\sum_j w_j f_j(s_t,a_t,s_{t+1})\,\Delta t.
\]

例如某项跟踪函数值为 0.85、权重为 1.5，控制间隔为 0.02 秒，那么这项本步贡献约 `0.85×1.5×0.02 = 0.0255`。它还要与其余项目相加，才是送给 PPO 的本步总奖励。乘时间间隔有助于保持不同控制频率下奖励累计的时间尺度。[奖励项配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L231)、[实际求和](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab/isaaclab/managers/reward_manager.py#L129)。

到这里先记住：**奖励不是一张人工标注的“正确关节角度表”，而是执行动作后的评价。** 同一个 PPO 优化器，面对不同奖励权重，就可能学出不同风格的控制策略。

日志里的某些 `Episode_Reward/...` 指标还会按时长归一化。它们不能直接与游戏的“完整一局原始分数”混为一谈；比较实验要固定指标定义、指令、地形与时间限制。

## 5 不必等整局结束才学

默认回合最长 20 秒，约 1000 个策略决策。可是 PPO 每轮不必把这 20 秒全部跑完：固定配置每只狗先收集 **24 步**，约半秒，再用 Critic 估计窗口末端往后的价值。

64 只狗各收集 24 步，有 `24×64=1536` 条状态转移。观察可以写成 `[T=24,N=64,48]`，动作是 `[24,64,12]`，奖励和价值对应每只狗每一步。整理给 MLP 更新时，将前两维合并成 `[1536,...]`。原论文或官方默认里常用更多环境，但从 64 个开始是为了先确认安装与数据流，不代表已经足够得到好步态。

保存的数据包括当时的观察、实际采样动作、奖励、结束信息、旧动作 log 概率和旧价值。采样时不需要保留整个物理仿真的计算图。

然后计算两个量：

1. **优势**：这次动作之后，结果比当时预计的好还是差。GAE 把几步 TD 误差组合起来，减少只看一步或只看最终回报的各自问题。
2. **价值训练目标**：Critic 这一轮应该拟合的回报估计。它也固定下来，不在每个 PPO epoch 里随便改写。

身体真正碰地导致失败，与只是到达时间上限，是两类边界。前者不再有这一局的后续价值；后者是否 bootstrap 要看任务被定义成有限时域还是持续任务。这里必须检查包装器和算法如何传递超时信息，不能仅拿一个 `done` 到处乘零。

在固定的 RSL-RL 3.1.2 中，包装器把超时单独放进 `extras['time_outs']`；PPO 在入库前给超时奖励补入一项 `gamma × 已保存价值`，随后 storage 用合并后的结束标记切断回合。这是该版本的超时校正实现，不能把它描述成“显式取得真实最后下一状态并精确计算它的 V”。阅读我们自己的游戏代码时，会看到它选择保留真实末帧、在 reset 前求下一状态价值；两套实现不要混讲。[包装器](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)、[超时校正](https://github.com/leggedrobotics/rsl_rl/blob/c49573e4d1eb6662d7b047fe935522c23665f6f3/rsl_rl/algorithms/ppo.py#L159)。

## 6 PPO 真正更新的还是神经网络参数

有了一批固定的观察和实际动作，就重新计算当前策略在这些观察下，产生同一个动作向量的概率密度。新旧 log 概率之差取指数，得到概率比：

\[
\rho_t=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)).
\]

正优势倾向于提高该连续动作的概率密度，负优势倾向于降低；PPO 的裁剪目标限制继续朝某些方向优化的激励。连续分布中单个精确向量的概率与密度不同，代码比较的是密度比。Critic 同时拟合回报目标，熵项鼓励保留探索。它们组成一个标量损失，之后仍然是 `zero_grad → backward → 梯度裁剪 → optimizer.step`。

固定配置会对这一批数据做 5 个学习 epoch，每个 epoch 分为 4 个 minibatch。旧 log 概率不会每次跟着更新成新策略，物理仿真也不用在反向传播时重新执行。学习完成后，丢掉这批 on-policy 窗口，再用更新后的网络采新数据。[RSL-RL PPO.update](https://github.com/leggedrobotics/rsl_rl/blob/c49573e4d1eb6662d7b047fe935522c23665f6f3/rsl_rl/algorithms/ppo.py#L194)。

因此，这一轮采集窗口会触发 `5×4=20` 次 `optimizer.step()`；一轮 rollout 更新不等于只改一次参数。

这就是与游戏 PPO 最紧密的对应：输入和动作变了，奖励设计变了，但旧概率、优势、价值目标、裁剪和参数更新之间的关系仍然成立。训练是否成功，要看指定速度下的实际行走、跌倒率、跟踪误差等指标。

## 7 按什么顺序打开源码

下面每个链接都固定到本教程核对的 commit。先在网页看，不需要整仓库所有文件一起读。

| 阅读顺序 | 文件 | 只找这件事 |
|---|---|---|
| 1 | [Go2 任务注册](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py) | 任务字符串对应哪个环境和 PPO 配置 |
| 2 | [平地](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py)与[Go2 粗糙地形父配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py) | 平地改了什么、动作缩放是多少 |
| 3 | [通用速度任务配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) | 观察拼接、奖励、终止、控制时间 |
| 4 | [Go2 的 PPO 配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py) | 每只狗收几步、MLP 多大、每批更新几轮 |
| 5 | [训练入口](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/scripts/reinforcement_learning/rsl_rl/train.py) | 创建环境、包装、创建 runner、保存配置和开始 learn |
| 6 | [OnPolicyRunner](https://github.com/leggedrobotics/rsl_rl/blob/c49573e4d1eb6662d7b047fe935522c23665f6f3/rsl_rl/runners/on_policy_runner.py) | 采集 T 步与更新的外层循环 |
| 7 | [RolloutStorage](https://github.com/leggedrobotics/rsl_rl/blob/c49573e4d1eb6662d7b047fe935522c23665f6f3/rsl_rl/storage/rollout_storage.py) | GAE 和 flatten/minibatch 怎样实现 |
| 8 | [PPO](https://github.com/leggedrobotics/rsl_rl/blob/c49573e4d1eb6662d7b047fe935522c23665f6f3/rsl_rl/algorithms/ppo.py) | 实际策略损失、价值损失、KL 学习率调整、梯度更新 |

如果你能用自己的话串起“身体检查表 → 动作分布 → 目标关节位置 → 物理执行 → 奖励 → 优势 → PPO”，这一部分的主干就讲清楚了。先去看仿真录像，再逐个研究更复杂地形、随机化和仿真迁移。
