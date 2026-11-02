# 策略梯度的发展路径

这条学习路线服务于三个目标：读懂大语言模型的 PPO 和 GRPO，理解机器狗的策略训练，再回到 AlphaZero 的网络、搜索和自我对弈。你已经掌握 DQN，可以从“网络输出动作价值”转到“网络直接输出动作分布”。

建议学习顺序是：**REINFORCE → baseline 与优势 → Actor–Critic → GAE → TRPO 的直觉 → PPO → LLM PPO 与 GRPO**。随后再学蒙特卡洛树搜索和自我对弈，重读 AlphaZero。历史线与教学线有所不同：Actor–Critic 的思想早于 REINFORCE，并非每个算法都由上一项直接演变而来。

## 一 先校准年份和贡献

| 时间 | 工作 | 解决的关键问题 | 我们需要掌握的程度 |
|---|---|---|---|
| 1983 | Barto、Sutton、Anderson 的自适应控制工作 | 动作选择者与评价者分工，是早期 Actor–Critic 的代表 | 知道思想很早已有 |
| 1992 | Williams，REINFORCE | 用采样反馈构造期望奖励的梯度方向 | 亲自推导和写代码 |
| 1999 | Sutton 等，Policy Gradient Methods with Function Approximation | 策略梯度定理，以及在相应条件下函数近似与价值估计如何配合 | 理解为何不必对未知环境求导 |
| 1999，2003 | Konda 与 Tsitsiklis 的 Actor–Critic 分析 | 用不同时间尺度分析 Actor 与 TD Critic 的学习 | 理解分工；暂不深究收敛证明 |
| 2001 | Kakade，Natural Policy Gradient | 参数空间同样大小的步子，对策略分布的影响可能很不同 | 为 TRPO 建立直觉 |
| 2015 | TRPO | 在改进策略的同时约束策略变化 | 理解概率比、KL 和信赖域 |
| 2015 预印本，2016 ICLR | GAE | 在优势估计中折中偏差和方差 | 会从轨迹计算优势 |
| 2016 | A3C | 多个环境和异步学习者提高深度 Actor–Critic 的训练效率 | 知道并行采集与更新方式 |
| 2017 | PPO | 用更容易实现的目标反复利用一批新采集的数据 | 重点掌握损失和训练循环 |
| 2022 | InstructGPT | 将人类偏好奖励与 PPO 结合，用于语言模型后训练 | 理解数据、奖励、价值模型和 KL |
| 2024 | DeepSeekMath，GRPO | 同一问题采多份回答，用组内奖励比较替代额外价值模型 | 重点掌握优势来源和 token 损失 |
| 2025 | DeepSeek-R1 | 展示大规模强化学习促进推理能力的训练路径 | 区分应用突破与 GRPO 的首次提出 |

1999 年论文属于 NIPS 1999，论文集有时按 2000 年引用，所以你看到两个年份并不奇怪。1992 年的 REINFORCE 已有数学推导；不能理解成“1999 年之前只是猜算法，之后才第一次证明策略梯度”。1999 年的工作进一步给出重要定理和有条件的收敛分析，也不是对任意现代深度网络训练作出全局最优保证。[1992 原文](https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf)、[1999 会议记录](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)、[Actor–Critic 原文](https://papers.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)。

早期 Actor–Critic 见 [Barto 等 1983](https://ieeexplore.ieee.org/document/6313077/)，自然策略梯度见 [Kakade 2001](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)。这里列的是与本课程相关的代表节点，不声称是所有思想的最早出处。

## 二 从 DQN 转过来 这次网络直接决定概率

还用接球：动作是左移、停留、右移。

DQN 输出 `[Q左, Q停, Q右]`，例如 `[2.1, 0.4, 1.2]`。它们有累计回报的含义，不是概率。我们再借助 ε-greedy 把估值变成实际动作。

策略网络输出 logits，经 softmax 得到 `[0.6, 0.1, 0.3]`。这次数字就是动作概率，训练时通常从这个分布采样。模型通过改参数直接改变动作概率。探索可由策略自身的随机性提供，也可以加入熵奖励等机制，不必照搬 DQN 的 ε-greedy。

logits 是网络尚未归一化的原始分数，softmax 将它们转换成总和为 1 的概率。代码里的 `Categorical` 就是离散动作的概率分布对象：`sample()` 按概率抽动作，`log_prob(action)` 返回这个动作概率的自然对数。

```python
logits = actor(states)                 # [B, A]，不是 Q 值
dist = torch.distributions.Categorical(logits=logits)
actions = dist.sample()                # [B]，动作编号
log_prob = dist.log_prob(actions)      # [B]，实际采样动作的对数概率
```

共同点仍是前向、损失、反传、参数更新。区别在于输出的含义、训练数据与损失。DQN 是价值方法，不因它也通过反向传播求梯度就成为策略梯度方法。

## 三 REINFORCE 奖励怎样改变动作概率

先考虑只有一步的任务：观察状态，选动作，得到奖励。希望最大化的是平均奖励：

\[
J(\theta)=\sum_a\pi_\theta(a\mid s)r(a).
\]

假设奖励不显式依赖模型参数，利用 \(\nabla\pi=\pi\nabla\log\pi\)：

\[
\nabla J(\theta)
=\sum_a\pi_\theta(a\mid s)r(a)\nabla\log\pi_\theta(a\mid s)
=\mathbb E_{a\sim\pi_\theta}[r(a)\nabla\log\pi_\theta(a\mid s)].
\]

因此，采样一个动作、拿到奖励，就可以构造一个随机梯度估计。这里的“按奖励提高概率”是从期望目标推导来的，不是先编一条经验规则再希望它有效。[策略搜索推导](https://underactuated.csail.mit.edu/rl_policy_search.html)。

多步任务把动作换成轨迹。为避免一开始就纠结折扣分布，先考虑有限长度、无折扣的情形，令 \(G_t\) 是从动作 \(a_t\) 开始直到结束的累计奖励：

\[
\nabla J(\theta)
=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_t G_t\nabla\log\pi_\theta(a_t\mid s_t)\right].
\]

轨迹概率包括策略的动作概率与环境的状态转移概率。当环境动力学不依赖 θ 时，对轨迹对数概率求导，只留下策略的项。**我们需要环境给反馈，但不需要环境可微，也不需要把梯度穿过采样出的离散动作和环境。**

有限长度的折扣目标若定义为 \(J=\mathbb E[\sum_t\gamma^t r_t]\)，而 \(G_t=\sum_{k\ge t}\gamma^{k-t}r_k\)，严格对应的求和还带外层 \(\gamma^t\)。实用实现也常采用不同的状态采样权重或近似；初学时不要混着定义又声称公式完全相同。[1999 原文](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)。

优化器默认最小化，因此可以写：

```python
# 简化版：T 步、N 个完整等长回合；按 [T,N,...] 顺序展平，时间维在前。
# 此处不加熵或 baseline。新游戏采集器也采用这个时间在前的排列方式。
logits = actor(saved_states)               # [T*N, A]，重新前向，有梯度
dist = Categorical(logits=logits)
logp = dist.log_prob(saved_actions)        # [T*N]
loss = -(logp * returns.detach()).reshape(T, N).sum(0).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

固定一个正回报，最小化负的 `回报 × log概率`，会推动相应采样动作的 log 概率增大。在最简单的单步 softmax 模型里，这一点可直接算出来。复杂共享网络里，各样本和状态的梯度会共同作用，所以不能保证每个动作概率都按单条样本直觉变化，更不能保证每次更新后真实分数上涨。

`returns` 在这一轮是权重，停止梯度；可微的是当前策略对已采样动作算出的 log 概率。变长回合需要明确按回合求和还是按 token 求平均，否则可能改变不同长度样本的权重。

### 用一个参数看见概率真的被推高

先把游戏缩成一次“左或右”的决策，仅为看清更新，不替代后面的真实游戏。令右移概率为 `sigmoid(θ)`，θ 初始为 0，左右概率都是 0.5。这次实际采样到右移，得到 +1 分：

\[
L=-\log\sigma(\theta),\qquad
\frac{\partial L}{\partial\theta}=\sigma(\theta)-1=-0.5.
\]

用学习率 0.1 的 SGD，θ 更新为 `0 - 0.1×(-0.5) = 0.05`，右移概率变为约 0.5125。我们没有把“正确动作是右移”作为标签，而是用实际得分给采样动作的 log 概率加权。单次正反馈并不证明右移真的最好；多次采样平均的梯度才对应期望目标。后面的 baseline 则让“好与坏”相对于预期来判断。

## 四 为什么需要 baseline 和优势

假设走左边平均能拿 9 分，走右边平均能拿 2 分，两边都是正奖励。只说“正奖励就鼓励”，还不足以表达“右边其实比当前平均表现差”。

没有 baseline 的 REINFORCE 也能在反复采样后学出偏好，因为动作概率要共同归一化，长期平均的更新会区分 9 分与 2 分。baseline 的主要作用是减少估计的波动，让一次样本更直接表达“高于还是低于预期”，并非凭空改变正确的平均更新方向。

引入只依赖状态的 baseline：

\[
\hat A_t=G_t-b(s_t),\qquad
L_{actor}=-\sum_t\operatorname{stopgrad}(\hat A_t)\log\pi_\theta(a_t\mid s_t).
\]

例如 baseline 是 5，9 分对应 +4，2 分对应 −3。动作的训练信号变成“比预期好还是差”。状态 baseline 在给定状态下满足：

\[
\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a\mid s)]
=b(s)\nabla\sum_a\pi(a\mid s)=0.
\]

在适当条件下，它不改变梯度期望，合适的 baseline 可以降低方差。不是任意 baseline 都一定降低方差；优势估计也不是简单的奖励改名。

## 五 Actor–Critic 用学习出的评价替代部分等待

让 Critic 学 \(V_\phi(s)\)：当前策略从这个状态继续的预期回报。注意是当前策略的价值，不是自动等于 DQN 所追求的最优动作价值。

Actor 给动作分布，Critic 给状态价值。它们可以共享骨干再接两个头，也可以用两个独立网络；Actor–Critic 不是特指“两张完全独立的网络”。

一个常见的一步估计是：

\[
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t).
\]

这里 `d_t` 表示任务真正终止。Actor 可用停止梯度的 δ 作为优势估计；Critic 则拟合停止梯度的 TD 目标。Critic 准确时，TD 残差能提供有用的优势信号；Critic 不准时会带来偏差，不是多了评价者就必然更好。

```python
with torch.no_grad():
    target = reward + gamma * (1 - terminated) * next_value  # [B]
    advantage = target - old_value                         # [B]
actor_loss = -(logp * advantage).mean()                    # []
critic_loss = F.mse_loss(value, target)                    # []
```

只添加一个用完整回报训练的价值 baseline，也可先作为 REINFORCE 的改进来理解；进入经典 TD Actor–Critic 时，进一步用价值估计来 bootstrap，不必每一步都等到最终结果。1999 年的分析已经把此类分工放入严格框架，但不能直接推出任意深度 Actor–Critic 的收敛保证。

## 六 GAE 把看一步和看很多步连接起来

一步 TD 通常方差较低，但更依赖 Critic 是否准确；完整蒙特卡洛回报更多依靠实际结果，但可能波动很大。GAE 对连续多个 TD 残差加权：

\[
\hat A_t^{GAE}=\delta_t+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots.
\]

λ 控制“更多相信近处的 bootstrap”还是“把更后面的实际反馈也带回来”。λ=0 对应一步 TD；完整回合且正确处理终止时，λ=1 对应折扣回报减价值 baseline。有限采样窗口末端仍可能依赖最后状态的价值。

真正终止要停止 bootstrap；只是采样窗口结束通常仍需估最后状态价值。时间上限截断与自动 reset 还要防止把下一局拼进上一局。代码常需要分别处理“是否 bootstrap”和“是否继续沿当前轨迹累积 GAE”两个边界。[GAE 原论文](https://arxiv.org/abs/1506.02438)。

## 七 从 TRPO 到 PPO 为什么要管住更新

即使梯度方向有道理，一步迈得太大，也可能把策略改坏。神经网络参数的欧氏距离，还不直接等于策略概率分布的变化。

自然策略梯度考虑策略分布的几何；TRPO 用 KL 信赖域来控制更新。实际 TRPO 使用近似，不能把理论上的单调改进结论不加条件搬到每次神经网络更新上。[TRPO 2015](https://proceedings.mlr.press/v37/schulman15.html)。

PPO 保留“采一批数据，再做若干次更新”的工作方式，用容易实现的概率比目标：

\[
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)},
\]

\[
L^{clip}(\theta)=\mathbb E_t\left[
\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
\]

这里写的是最大化目标，代码最小化时取负。比较的是**同一状态、同一已采样动作**的新旧概率，不是奖励比，也不是比较两次各自最优的动作。

正优势时，概率提高到一定程度后，目标不再继续奖励这一方向；负优势时，概率降低到一定程度后同理。PPO clip 并不硬性保证所有概率比都留在区间内，也不是梯度范数裁剪，不能把我们 DQN 的 `clip_grad_norm_` 与它混为一谈。[PPO 2017](https://arxiv.org/abs/1707.06347)。

一批 PPO 数据通常保存观察、动作、奖励、终止/截断信息、旧 log-prob 和旧价值；计算优势与价值目标后固定它们，每轮更新重新计算当前策略概率。更新若干轮之后采新数据，不能不加校正地照搬 DQN 的任意历史回放。

## 八 两个应用分支 同一个优化器面对不同任务

### 机器狗的 PPO

策略的输入可以包含关节位置与速度、机身角速度、重力方向、运动指令和历史动作。输出常见为连续动作分布的均值与标准差，而非三个离散按钮；动作可能是关节位置目标或力矩等，需看具体控制接口。

奖励可结合速度跟踪、姿态稳定、能耗、动作平滑和跌倒惩罚。大量并行物理仿真提供轨迹，训练循环仍可使用 PPO、GAE 和价值网络。难点还包括仿真速度、奖励设计、课程、动力学随机化以及仿真到实体的迁移。它们不是换一条 PPO 公式就自动解决的。

代表案例 [Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978) 于 2021 年发布预印本、2022 年 CoRL 论文集发表，采用大规模并行仿真和 PPO 训练 ANYmal。论文中的分钟级结果属于其硬件、仿真与任务配置，不是任意机器狗和电脑的训练时间承诺。

### 语言模型的 PPO

状态是 prompt 加已生成 token，动作是下一个 token，策略是词表上的概率分布。回答末尾的奖励可能来自人类偏好训练出的奖励模型，也可能来自其他评价器。

典型 RLHF PPO 系统里，Actor、Critic、Reward Model、Reference Model 承担不同角色；有时部分角色共享参数或结构。旧策略是生成本批样本的策略，参考策略通常用于约束偏离，两者不能混为一谈。模型生成答案后，算奖励和优势，再更新各 token 的概率；不需要对评分程序本身反向传播。[InstructGPT 2022](https://arxiv.org/abs/2203.02155)。

### GRPO 为同一道题建立组内比较

DeepSeekMath 于 **2024 年 2 月 5 日**发布，提出 GRPO。2025 年的 DeepSeek-R1 让其推理训练路线广为人知，不能据此把 GRPO 的提出年份记成 2025。

同一道题用旧策略生成 G 份回答，得到奖励 \(r_1,\dots,r_G\)。在结果奖励版本中，核心优势估计可概括为：

\[
\hat A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}
{\operatorname{std}(r_1,\dots,r_G)+\varepsilon_{num}}.
\]

\(\varepsilon_{num}\) 是实际实现常用的数值稳定项，不是 PPO 裁剪 ε。组内全同分时，该项优势通常为零，不代表模型已经掌握任务。

同一回答内的有效 token 使用这个结果优势，结合 token 级的新旧概率比与裁剪项训练；原始配方还带参考策略的 KL 项。GRPO 免去的是额外学习的价值模型，不是免去奖励评价，也不是必须只用规则奖励。它还要为每道题采多份答案，节省 Critic 内存不等于生成成本变小。

组内标准化、长度归一化、奖励形式和 KL 的处理在后续变体中都有变化。组均值又包含本样本本身，所以不要把它与上一节“与采样动作独立的状态 baseline”不加区别地当成同一个无偏性证明。[DeepSeekMath 第 4 节](https://arxiv.org/html/2402.03300v3#S4)、[DeepSeek-R1 原始预印本](https://arxiv.org/abs/2501.12948v1)。

## 九 还应知道的分支

**连续控制不只有 PPO。** DPG（2014）与之后的 DDPG、TD3 属于确定性策略梯度及 Actor–Critic 路线；SAC（2018）是离策略最大熵 Actor–Critic，常配经验回放。它们对机器人控制很重要，但不必先全部推完才能读懂 PPO。先把随机策略主线学通，再比较样本复用、稳定性和连续动作的差别。[DPG](https://proceedings.mlr.press/v32/silver14.html)、[SAC](https://arxiv.org/abs/1801.01290)。

**DPO（2023）是偏好优化的另一条重要路线。** 常见离线 DPO 根据正负回答对构造直接优化目标，不能简单称为“PPO 后面的策略梯度算法”；它和基于在线采样奖励的 PPO/GRPO，在数据来源和目标上有区别。[DPO](https://arxiv.org/abs/2305.18290)。

**AlphaZero 不是 PPO 加 MCTS。** 它用网络指导搜索，搜索访问次数形成策略目标 π，自我对弈结果形成价值目标 z，再训练网络拟合这两个目标。理解 Actor 与 Critic 有助于理解两种输出的作用，但 AlphaZero 的策略头主要拟合搜索目标，而非直接照搬 REINFORCE 或 PPO 损失。[AlphaZero 原文](https://arxiv.org/html/1712.01815v1)。

## 十 接下来如何一步一步学

| 次序 | 我们只解决的一个问题 | 可以对照的现有文件 |
|---|---|---|
| 1 | 一个得分为什么能改变动作概率，环境为什么不用可微 | 本文第三节；先从一步任务推导 |
| 2 | 整局最后才得分，为什么前面的动作也能得到训练信号 | REINFORCE 的轨迹概率与回报 |
| 3 | 都是正分，为什么有些动作仍应少做 | baseline 与优势 |
| 4 | Critic 怎么训练，又怎么帮助 Actor | [第 03 课](03_策略梯度与ActorCritic.md) |
| 5 | 一步 TD、整局回报与 GAE 怎么连接 | 同课 `gae()` |
| 6 | 为什么概率比、裁剪和多轮更新构成 PPO | [第 04 课](04_PPO逐行拆解.md) |
| 7 | 机器狗动作与 LLM token 怎样放进同一框架 | 本文第八节 |
| 8 | GRPO 为什么要同题多采样，替代了什么 | [第 06 课](06_微型语言模型SFT_DPO_GRPO.md)与[第 07 课](07_预训练语言模型实战.md) |
| 9 | 搜索如何产生比网络自身更好的动作目标 | [AlphaZero 整体导读](B_AlphaZero整体导读.md)与[五子棋导读](C_用五子棋聊懂AlphaZero.md) |

每一步都用同样的方式检查理解：你先说明输入输出、采样动作、训练信号和更新方向；再拿一个具体数字或一小段代码核验。先讲通机制，再扩到完整工程。
