# 03｜不预测每个动作的 Q，直接训练动作概率

代码：[03_policy_gradient.py](../code/03_policy_gradient.py)。环境、输入表示和 64 维骨干保持不变。

## REINFORCE 的模型与数据

骨干输出 `[B,64]`，Actor 头输出 `[B,3]` logits，再构造 `Categorical`。本课为与后续比较而实例化了同一个 ActorCritic 模型；REINFORCE 模式的 Critic 头不参与损失、不被训练。

```bash
python tutorials/code/03_policy_gradient.py --algo reinforce --out tutorials/runs/my_reinforce
```

每次并行运行 64 个完整回合，每回合 5 步，所以 rollout 形状是 `[T=5,B=64,...]`。采样在 `no_grad()` 中进行：它记录动作和观测，不保留环境执行期间的计算图。

采样完成后重新前向：

```python
logits, _ = model(saved_states)
dist = Categorical(logits=logits)
loss = -(dist.log_prob(saved_actions) * returns).mean()
```

因为只有终局奖励、无折扣，每条轨迹的五个动作都使用同一个回报。这是每条轨迹梯度求和后再平均的常数缩放版本；本环境回合等长，所以按 token／时间步平均只改变固定比例。可变长度任务中不能忽略这种差别。

## 为什么要 Critic

同样得 1 分，在本来就很容易成功的状态和非常困难的状态，含义不同。Critic 估计从状态继续的回报，让 Actor 比较“比预期更好还是更差”。

```text
共享骨干 h [B,64]
├ Actor: Linear(64,3) → 动作 logits
└ Critic: Linear(64,1) → V(s)
```

```bash
python tutorials/code/03_policy_gradient.py --algo ac --out tutorials/runs/my_ac
```

此模式使用 GAE Actor–Critic，一批数据更新一次；不是对某个完整 A3C／A2C 工程实现的复刻。TD 误差和 GAE：

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\qquad
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}.
\]

代码 `gae()` 从后往前累积，默认 γ=0.99、λ=0.95。Critic 回归目标是 `returns = advantages + rollout_values`，这些量均在 no-grad 下计算。

## 两个损失如何一起反传

\[
L=L_{actor}+0.5L_{value}-0.01H(\pi).
\]

Actor 使用停止梯度的优势；Critic 用均方误差拟合目标；熵项鼓励保留探索。共享骨干会收到两种损失的梯度，两个输出头承担不同任务。Critic 不是“给 Actor 输出一个可微奖励，让梯度穿过去”。

REINFORCE 分支使用 γ=1、无价值 baseline，AC 分支使用折扣 GAE。因此现有比较展示的是完整配方差别。若要单独研究 Critic 的效果，须统一 γ、优势估计方式、更新次数、熵与其余超参数。

## 必须理解的边界

游戏真正结束：后续 V 置零。只是 rollout 收集暂时结束：通常还需估计最后状态 V。代码提供的测试分别检查这两个情形，防止自动 reset 的下一局价值进入上一局。

Actor loss 的绝对数值不等于任务质量；出现负 loss 不代表错误。观察成功率、熵、Critic 误差和梯度尺度，才能解释训练状态。

**练习：** λ 从 0 改到 1；关掉熵项；将共享骨干换成两个独立 MLP。分别说明偏差／方差、探索和计算量的变化。不要用一个 seed 给算法下总排名。

阅读：[GAE](https://arxiv.org/abs/1506.02438)、[异步 Actor–Critic](https://arxiv.org/abs/1602.01783)。更完整推导见 [已有专题第 7–8 节](../../docs/06_为什么奖励能通过梯度改进行为.md)。
