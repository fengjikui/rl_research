# 04｜在刚才的代码上加入 PPO

```bash
python tutorials/code/03_policy_gradient.py --algo ppo --out tutorials/runs/my_ppo
```

## 结构不变，变的是经验复用与目标

仍然是同一个 ActorCritic。rollout 时记录 `old_logp`，更新时计算 `new_logp`。两者对应同一状态下同一个已采样动作。

```python
ratio = (new_logp - old_logp).exp()
unclipped = ratio * advantages
clipped = ratio.clamp(0.8, 1.2) * advantages
actor_loss = -torch.minimum(unclipped, clipped).mean()
```

ρ 是新旧动作概率比，不是奖励比。第一次更新前新旧参数相同，ρ=1，未裁剪项的梯度接回熟悉的 `A * grad(log π)`。后续更新继续使用同一份旧概率和优势。

## 裁剪到底在限制什么

当 A>0，提高动作概率会增加目标；超过上边界后，裁剪使这一方向的目标不再继续增加。当 A<0，降低动作概率是改善方向；低于下边界后对应方向变平。

不是“超出区间的样本全部没有梯度”。正优势但概率被大幅降低、负优势但概率被大幅提高，仍然需要纠正。`check_invariants.py` 中有正负优势四种情况的梯度核验。

PPO clip 也不是严格 KL 上限或每次更新必定提高真实回报的保证。这是实用代理目标；模型共享参数、状态分布改变、多次更新都会使实际行为更复杂。

## 本课完整更新安排

1. 64 个环境各采 5 步，共 320 条 transition。
2. 无梯度计算 GAE、价值目标并冻结 old log-prob。
3. 对整批数据做 4 次优化，每次重新计算当前 log-prob。
4. 下一轮再采新数据。

没有 minibatch 切分、advantage 标准化、value clipping、KL early stop；它是便于逐行阅读的小型 PPO，不代表生产配方。REINFORCE／AC 只有一次更新，所以比较环境步相同不等于比较计算量相同。

## 一个最容易写错的地方

不能在每个 epoch 末尾把 `old_logp` 更新成刚算的 `new_logp`，否则你会改变本批数据的概率比基准。也不能只缓存 rollout 时的带梯度 log-prob 然后无限复用同一个图；这里每次对当前模型重新前向。

## 与语言模型相接

把动作从三个按钮换成一个词表上的 token，log-prob 的计算方式仍然成立。变化在于动作序列变长、奖励可能只在回答末尾出现、采样更昂贵、可能添加参考策略 KL，以及 Advantage 的来源不同。

`old policy` 用来表示生成本批数据的策略；`reference policy` 常用来限制偏离某个固定模型。两者不能因为都叫“旧模型”就混为一谈。

**练习：** epoch 从 4 改为 1、16，比较相同环境步下成功率及训练成本。再记录 `mean(abs(ratio-1))`，理解重复使用一批数据时策略如何远离采样分布。没有新增观察时，不要仅因为 loss 下降就无限增加 epoch。

阅读：[PPO 原论文](https://arxiv.org/abs/1707.06347)。
