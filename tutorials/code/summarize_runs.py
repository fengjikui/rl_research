"""Create a static scientific figure and a traceable report from saved runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = {p.parent.name: json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/results.json')}
    panels = [(['dqn_mlp', 'dqn_cnn'], 'success', 'DQN: same grid input', 'Environment steps'),
              (['reinforce', 'ac', 'ppo'], 'success', 'Policy optimization on Catch', 'Environment steps'),
              (['memory_mlp', 'memory_gru'], 'success', 'Delayed cue: information matters', 'Training updates'),
              (['tiny_sft', 'tiny_dpo', 'tiny_grpo'], 'exact_success_probability',
               'Tiny LM: probability of correct sequence', 'Updates (different compute per method)')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (names, metric, title, xlabel) in zip(axes.flat, panels):
        for name in names:
            hist = runs[name]['history']
            ax.plot([h['step'] for h in hist], [h[metric] for h in hist], label=name, linewidth=2)
        ax.set(title=title, xlabel=xlabel, ylabel='Success / probability', ylim=(-.03, 1.06))
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Measured CPU teaching runs, seed=0 | Toy tasks, not algorithm rankings', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(ROOT/'runs/overview.png', dpi=160)
    plt.close(fig)
    lines = ['# 本机实验记录', '',
             '由 `code/summarize_runs.py` 从各实验的 `results.json` 自动汇总。每项训练 seed=0；未做多 seed 算法排名。全部使用 CPU。运行时长是脚本记录值，不包含安装依赖，也不是其他机器的耗时保证。', '',
             '![实测曲线](runs/overview.png)', '',
             '| 实验 | 训练前 | 训练后 | 参数量 | 记录时长／秒 | 原始记录 |',
             '|---|---:|---:|---:|---:|---|']
    for name in ['bandit', 'q_table', 'dqn_mlp', 'dqn_cnn', 'reinforce', 'ac', 'ppo',
                 'memory_mlp', 'memory_gru', 'tiny_sft', 'tiny_dpo', 'tiny_grpo', 'tool_agent']:
        r = runs[name]; m = r['metrics']
        if name == 'bandit':
            before, after = m['initial_expected_reward'], m['final_expected_reward']
        elif name == 'q_table':
            before, after = None, None
        elif name == 'tool_agent':
            before, after = m['initial']['success'], m['final']['success']
        else: before, after = m['initial_success'], m['final_success']
        b = f'{before:.3f}' if before is not None else '未测'
        a = f'{after:.3f}' if after is not None else '到达终点'
        lines.append(f"| {name} | {b} | {a} | {r.get('parameters', '不适用')} | {r['runtime']['seconds']} | [JSON](runs/{name}/results.json) |")
    lines += ['', '## 指标含义与限制', '',
        '- bandit：真实分布下的期望奖励；不是单次奖励。Q 表输出贪心路径 0→1→2→3→4→5，没有训练前路径指标。',
        '- Catch：固定评估种子的 500 局贪心成功率；随机策略为 0.190。来自同一个有限游戏的重复初态抽样，不是 500 个未见任务。DQN 与策略梯度的环境步和优化预算不同。',
        '- Memory：两种线索均衡评估。MLP 只看到相同的最后一帧；GRU 读取历史。因此它验证信息与记忆，不能当成等信息网络容量竞赛。',
        '- Tiny LM：训练后贪心正确率在 16 个训练 prompt 上枚举。图右下是正确序列的精确采样概率，指标与表内贪心正确率不同；没有留出泛化结论。SFT、DPO、GRPO 的标签与采样／计算预算不同。',
        '- Tool Agent：只枚举两个隐藏状态，至多两次调用；成功率由 0.5 到 1.0，检查工具比例由 0 到 1。不代表真实长程任务能力。',
        '- 每条曲线保留训练中的波动，不只展示最好检查点。本报告使用最终检查点。', '',
        '## 实际回放与下一步', '',
        '[随机接球回放](runs/dqn_mlp/random.gif) · [DQN 训练后回放](runs/dqn_mlp/trained.gif) · [PPO 回放](runs/ppo/trained.gif)', '',
        '本表保留初版小型实验。后续预训练 Qwen 权重上的实际 GRPO 训练另见 [手写 GRPO 实战](../grpo_lab/README.md)，不以这里的微型 LM 结果替代它。', '',
        '边界检查：`check_invariants.py` 核验终止／bootstrap、因果 mask、EOS mask、数据拆分与 PPO 裁剪梯度。可选的 `check_pretrained_stack.py` 在随机微型 Qwen2 上验证生成概率与重算概率一致，以及 LoRA 可反传并更新；它不下载预训练权重。', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    print('Wrote tutorials/RESULTS.md and tutorials/runs/overview.png')


if __name__ == '__main__': main()
