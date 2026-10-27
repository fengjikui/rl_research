# 02｜DQN：让神经网络学会接球

## 先玩明白环境

代码：[common.py](../code/common.py) 中的 `Catch` 和 [02_dqn.py](../code/02_dqn.py)。画面为 6×5，黄色球每步下落一格，底部挡板每步可左移、不动或右移。第五步接到得 +1，未接到得 −1，回合结束；没有距离奖励。

球和挡板用两个独立通道表示，观测是 `[B,2,6,5]` 的 0/1 张量。这是符号化网格图，不是从真实游戏截图中识别物体。开始位置随机，所有球都可在五步内接到。动作空间为三个离散动作。

```bash
python tutorials/code/02_dqn.py --encoder mlp --out tutorials/runs/my_dqn
```

## 模型结构

```text
[B,2,6,5] → Flatten → [B,60]
→ Linear(60,64) → Tanh → Linear(64,64) → Tanh
→ Linear(64,3) → Q(s, left/stay/right)
```

这三个 Q 值不经过 softmax。训练时 ε-greedy 随机探索，评估时取最大 Q 对应的动作。网络输入是当前网格，不包含正确动作或最终得分。

## 一批训练数据怎么来

每执行一步保存 `(s, a, r, s_next, done)`。经验池最多存 10000 条转移。满 128 条后，每四个环境步抽 64 条做一次更新。抽样打散相邻帧相关性，也允许旧数据重复利用。

张量流：`states [64,2,6,5] → Q [64,3] → gather(actions) [64]`。只直接监督当时实际执行的动作 Q 值；其他动作会通过共享网络参数间接受影响。

```python
q = model(states).gather(1, actions[:, None]).squeeze(1)
with torch.no_grad():
    target = rewards + gamma * (1-dones) * target_model(next_states).max(-1).values
loss = smooth_l1_loss(q, target)
```

这里是回归损失。预测值来自在线网络，目标来自定期复制的目标网络。目标停止梯度，避免一次更新同时追逐会立刻移动的自举目标。即便如此，非线性函数逼近、bootstrap 与 off-policy 数据的组合仍可能不稳定。

## 训练循环中不能漏的四件事

1. 探索率从较高值衰减到 0.05，否则一开始很难发现有用转移。
2. 每 200 环境步复制在线网络到目标网络；不是每次梯度更新都复制。
3. `done` 屏蔽终局后的价值。本环境自动 reset 返回新一局观测，只有因为这里确实是终局且乘了 0，才可以安全地不用其后续价值。
4. 评估没有 ε 探索；随机、训练前与训练后策略使用固定评估环境种子与预算。

## 看什么结果

打开输出目录的 `random.gif`、`trained.gif` 和 `curve.png`。回放来自实际执行动作；曲线是固定种子 500 局接球成功率，不是 TD loss，也不是训练中的探索回报。

loss 降低可能只是更好拟合自己的错误目标，所以任务指标不可省略。500 局是这个有限状态游戏的重复抽样，不是 500 个新任务，更不能推断对新规则的泛化。

## 修改实验

- 只关闭经验回放，观察相关性与复用改变，但保持环境步预算。
- 只把目标同步周期改为 1 或 2000，观察稳定性；不要同时改网络宽度。
- 换 `--encoder cnn`，输入信息完全相同，但参数量不同，结果见第 05 课。
- 扩大网格时同步检查可达性；若最大位移不足，部分球根本无法接到，100% 就不再是合理目标。

原始 DQN：[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)。实现对照可读 [PyTorch 官方 DQN 教程](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)。本课环境与代码为独立教学实现，没有复现 Atari 的结果。
