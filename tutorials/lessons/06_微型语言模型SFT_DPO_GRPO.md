# 06｜同一个微型语言模型，三种训练方式

代码：[05_tiny_lm.py](../code/05_tiny_lm.py)。不下载权重，在 CPU 上从随机初始化开始。它是字符级因果语言模型，具备真实的自回归概率与梯度训练，但不是预训练大语言模型。

## 任务与数据

词表为 `0 1 2 3 >`，输入 `12>`，输出应为 `21`。总共 16 个输入组合，全部用于训练与机制评估。每次输出固定两个 token，不含 EOS，因此本课不用处理变长 mask。完整字符串与模型输出都保存到 JSON。

```bash
python tutorials/code/05_tiny_lm.py --algo sft --out tutorials/runs/my_tiny_sft
python tutorials/code/05_tiny_lm.py --algo dpo --out tutorials/runs/my_tiny_dpo
python tutorials/code/05_tiny_lm.py --algo grpo --out tutorials/runs/my_tiny_grpo
```

三个实验默认 seed 相同、SFT 热身 5 步相同，然后重置优化器；所以比较阶段的初始参数和优化器状态相同。热身让格式与采样支持不至于完全随机，但没有先把任务训到满分。

## 模型到底长什么样

```text
token ids [B,L]
→ token embedding(5,32) + position embedding(8,32)
→ 1 个带因果 mask 的 Transformer block
   4 个 attention heads；FFN 宽度 64；dropout=0
→ LayerNorm → Linear(32,5)
→ logits [B,L,5]
```

底层使用 PyTorch `TransformerEncoderLayer`，但加了上三角因果 mask，所以未来 token 不可见；模块名字不等于实际信息流。它是简化的 decoder-only 因果语言模型，没有 encoder–decoder cross-attention。

推理时输入 `12>`，先生成 `2`，再输入 `12>2` 生成 `1`。训练计算两个答案 token 的概率时，可输入 `12>2` 一次前向，分别取对应位置的 logits。因果 mask 保证预测第一个答案 token 时没有看到第二个；测试中专门检查未来 token 修改不影响早期 logits。

## 三种损失逐一读

**SFT：** 已知正确答案，直接最大化两个正确 token 的条件概率。

```python
loss = -completion_logp(model, prompt, correct_answer).mean()
```

**DPO：** 正确反转作为 chosen，把答案第一位改错作为 rejected；这是程序化偏好，不是人类标注。固定热身后的参考模型，定义序列 log-prob 差：

```python
margin = log_pi_chosen - log_pi_rejected
ref_margin = log_ref_chosen - log_ref_rejected
loss = -logsigmoid(beta * (margin - ref_margin)).mean()
```

chosen 的优势是相对 rejected 的；DPO 目标本身不直接保证绝对正确序列概率达到 1。当前负例很简单，后续可加入不同错误类型检验偏好覆盖问题。

**GRPO：** 每个 prompt 采样 16 个两 token 答案，完整正确得 1，否则得 0。组内标准化获得 A；所有答案 token 使用所属序列的 A，配合 PPO 式概率比裁剪。

```python
advantage = (reward - group_mean) / (group_std + 1e-4)
ratio = exp(current_logp - old_logp)
loss = -minimum(ratio*A, clip(ratio)*A).mean()
```

一批采样做 3 次更新；old log-prob 和 A 固定。β=0，没有参考 KL，这是有意简化的 GRPO 配方，不是逐项复刻 DeepSeekMath。DPO 仍使用固定 reference。

## 为什么同时画两条指标

`greedy_success` 枚举 16 道题，每次取最大概率 token；`exact_success_probability` 计算正确完整答案的概率乘积并对题目平均。后者是温度 1、自回归采样下的精确成功概率，不是蒙特卡洛估计。

两个模型都可能贪心全对，但采样稳定性不同。另一方面，GRPO 用更多候选与梯度更新，SFT 有直接正确标签，DPO 有程序化偏好，所以不能用相同横轴步数声称三者等预算公平竞赛。

当同组全对或全错，组内优势为零。训练后期全对是好事，初期全错则可能是探索失败；`last_informative_group_fraction` 要结合得分解释。

## 本课最重要的边界与练习

这是训练支持集上的机制演示。它说明相同网络可以被不同目标训练，不说明 RL 创造了通用推理能力。下一课才划分不同的输入用于留出验证。

练习：把反转长度加到 3，并修改生成长度、位置数、对齐切片与评估；设置训练与测试组合分离；尝试无热身；改变 G 观察全零组和生成成本；给 DPO 加多种困难负例。不要只改奖励函数而忘记同步评估。

阅读：[Transformer](https://arxiv.org/abs/1706.03762)、[DPO](https://arxiv.org/abs/2305.18290)、[DeepSeekMath／GRPO](https://arxiv.org/abs/2402.03300)。
