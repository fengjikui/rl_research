"""Fixed-budget Catch representation comparison; serial CPU, shared DQN update.

No task rules or shared lesson files are modified. All seeds and checkpoints
are reported, including regressions; this is not a parameter-matched ablation.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tutorials/code'))
from common import Catch, QNet
from dqn_core import update

VARIANTS = ['coordinates_mlp', 'dual_grid_mlp', 'dual_grid_cnn', 'signed_grid_cnn']
NAMES = ['Coordinates + MLP', 'Two channels + MLP', 'Two channels + CNN', 'Signed channel + CNN']
CONFIG = dict(environment_steps=12000, seeds=list(range(5)), gamma=.99,
    learning_rate=.001, batch_size=64, replay_capacity=10000, learning_starts=128,
    train_every=4, target_sync_every=200, epsilon='max(.05, 1-step/(12000*.65))',
    evaluation_steps=[0,250,*range(500,12001,500)], optimizer='Adam', loss='SmoothL1',
    torch_threads=1, gradient_clip_norm=5., evaluation='all 25 initial positions, greedy',
    selection='fixed final step; no early stopping or seed selection')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def encode(observations, variant):
    # Input [B,2,6,5]; all representations retain the full position information.
    if variant == 'coordinates_mlp':
        size = observations.shape[0]
        ball = observations[:,0].reshape(size,-1).argmax(-1)
        paddle = observations[:,1].reshape(size,-1).argmax(-1)
        return np.stack([ball//5/5., ball%5/4., paddle%5/4.],axis=1).astype(np.float32)
    if variant == 'signed_grid_cnn':
        signed = observations[:,0]-observations[:,1]
        signed[(observations[:,0]>0)&(observations[:,1]>0)] = 2.
        return signed[:,None].astype(np.float32)
    return observations.copy()


def make_model(variant):
    if variant == 'coordinates_mlp':
        return nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
    if variant == 'dual_grid_mlp': return QNet('mlp')
    if variant == 'dual_grid_cnn': return QNet('cnn')
    return nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),
        nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.Flatten(),
        nn.Linear(480,64),nn.ReLU(),nn.Linear(64,3))


def verify_encodings():
    observations=[]
    for row in range(6):
        for ball in range(5):
            for paddle in range(5):
                state=np.zeros((2,6,5),dtype=np.float32)
                state[0,row,ball]=1;state[1,5,paddle]=1;observations.append(state)
    observations=np.asarray(observations)
    result={}
    for variant in VARIANTS:
        encoded=encode(observations,variant)
        unique=len({state.tobytes() for state in encoded})
        assert unique==150, (variant,unique)
        result[variant]=dict(encoded_shape=list(encoded.shape[1:]),distinct_states=unique,
            checked_states=150,terminal_overlap_preserved=True)
    return result


@torch.no_grad()
def evaluate(model,variant):
    env=Catch(25,seed=10000)
    env.ball[:]=np.repeat(np.arange(5),5);env.player[:]=np.tile(np.arange(5),5)
    rows=[dict(ball=int(b),paddle=int(p),actions=[]) for b,p in zip(env.ball,env.player)]
    for _ in range(5):
        actions=model(torch.as_tensor(encode(env.obs(),variant))).argmax(-1).numpy()
        for row,action in zip(rows,actions):row['actions'].append(int(action))
        _,reward,_,_=env.step(actions)
    for row,r in zip(rows,reward):row['success']=bool(r>0)
    return rows


def train_one(variant,seed,out):
    out.mkdir(parents=True,exist_ok=False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=make_model(variant)
    target=deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'])
    env=Catch(1,seed);replay=deque(maxlen=CONFIG['replay_capacity'])
    history=[];updates=0;started=time.monotonic();initial_positions=[]
    write_json(out/'model_config.json',dict(variant=variant,seed=seed,
        architecture=str(model),parameters=sum(p.numel() for p in model.parameters()),
        input_shape=list(encode(env.obs(),variant).shape[1:])))

    def checkpoint(step):
        records=evaluate(model,variant)
        item=dict(step=step,updates=updates,n=25,correct=sum(r['success'] for r in records))
        item['success']=item['correct']/25
        history.append(item)
        with (out/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(**item,records=records))+'\n')
        write_json(out/'learning_curve.json',history)

    checkpoint(0)
    for step in range(1,12001):
        raw=env.obs()
        if env.row[0]==0:initial_positions.append([int(env.ball[0]),int(env.player[0])])
        state=encode(raw,variant)
        epsilon=max(.05,1-step/(12000*.65))
        with torch.no_grad():
            action=random.randrange(3) if random.random()<epsilon else int(model(torch.as_tensor(state)).argmax(-1)[0])
        following,reward,done,terminal=env.step([action])
        next_raw=terminal if done[0] else following
        next_state=encode(next_raw,variant)
        replay.append((state[0].copy(),action,float(reward[0]),next_state[0].copy(),bool(done[0])))
        if len(replay)>=128 and step%4==0:
            update(model,target,optimizer,random.sample(replay,64),gamma=.99,
                   audit_dir=out/'first_batch' if updates==0 else None,
                   action_names=['LEFT','STAY','RIGHT'])
            updates+=1
        if step%200==0:target.load_state_dict(model.state_dict())
        if step in CONFIG['evaluation_steps']:checkpoint(step)
    torch.save(model.state_dict(),out/'model.pt')
    restored=make_model(variant)
    restored.load_state_dict(torch.load(out/'model.pt',weights_only=True))
    assert evaluate(restored,variant)==evaluate(model,variant)
    auc=float(np.trapezoid([h['success'] for h in history],[h['step'] for h in history])/12000)
    thresholds={str(p):next((h['step'] for h in history if h['success']>=p),None) for p in [.8,.95,1.]}
    metrics=dict(variant=variant,seed=seed,final_success=history[-1]['success'],
        final_correct=history[-1]['correct'],initial_success=history[0]['success'],
        optimizer_updates=updates,seconds=time.monotonic()-started,normalized_auc=auc,
        first_observed_threshold_step=thresholds,checkpoint_reload_identical=True,
        initial_positions_sha256=hashlib.sha256(json.dumps(initial_positions).encode()).hexdigest())
    write_json(out/'initial_positions.json',initial_positions)
    write_json(out/'metrics.json',metrics)
    print('RUN',json.dumps(metrics),flush=True)
    return metrics


def summarize(out,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aggregates=[]
    figure,ax=plt.subplots(figsize=(9,5))
    individual,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for index,(variant,name) in enumerate(zip(VARIANTS,NAMES)):
        curves=[json.loads((out/variant/f'seed_{seed}/learning_curve.json').read_text()) for seed in range(5)]
        steps=np.array([p['step'] for p in curves[0]])
        values=np.array([[p['success'] for p in curve] for curve in curves])
        mean=values.mean(0);std=values.std(0,ddof=1)
        ax.plot(steps,mean,label=name)
        ax.fill_between(steps,np.clip(mean-std,0,1),np.clip(mean+std,0,1),alpha=.12)
        panel=axes.flat[index]
        for seed,value in enumerate(values):panel.plot(steps,value,alpha=.65,lw=1,label=f'seed {seed}')
        panel.plot(steps,mean,color='black',lw=2,label='mean');panel.set_title(name)
        panel.grid(alpha=.2);panel.set_ylim(0,1.03)
        subset=[m for m in metrics if m['variant']==variant]
        config=json.loads((out/variant/'seed_0/model_config.json').read_text())
        aggregates.append(dict(variant=variant,name=name,parameters=config['parameters'],
            final_success_mean=float(mean[-1]),final_success_sample_std=float(std[-1]),
            final_success_per_seed=[m['final_success'] for m in subset],
            normalized_auc_mean=float(np.mean([m['normalized_auc'] for m in subset])),
            mean_success_by_step=[dict(step=int(s),mean=float(m),sample_std=float(d)) for s,m,d in zip(steps,mean,std)]))
    ax.set(xlabel='Environment steps (same budget)',ylabel='Success on all 25 initial positions',ylim=(0,1.03))
    ax.grid(alpha=.2);ax.legend();figure.tight_layout();figure.savefig(out/'learning_curves.png',dpi=150)
    for panel in axes[-1]:panel.set_xlabel('Environment steps')
    for panel in axes[:,0]:panel.set_ylabel('Success rate')
    axes.flat[0].legend(fontsize=8,ncol=3)
    individual.tight_layout();individual.savefig(out/'per_seed_curves.png',dpi=150)
    plt.close('all')
    write_json(out/'summary.json',dict(config=CONFIG,runs=metrics,aggregates=aggregates))
    write_report(out,aggregates)


def write_report(out,aggregates):
    rows=['# 接球 DQN：坐标、网格与卷积网络的真实对照','',
        '日期：2026-11-01。四组各 5 个训练种子，共 20 次独立训练；每次固定 12,000 个环境步、2,969 次梯度更新，串行 CPU 单线程。没有按结果挑选种子、延长个别运行或提前停止。','',
        '## 主结果','',
        '| 表示与模型 | 参数量 | 最终成功率均值 ± 种子间样本标准差 | seed 0–4 最终成功率 | 曲线面积均值 |',
        '|---|---:|---:|---|---:|']
    labels=['坐标 MLP','双通道网格 MLP','双通道网格 CNN','单通道有符号网格 CNN']
    for label,item in zip(labels,aggregates):
        rates=' / '.join(f'{v:.0%}' for v in item['final_success_per_seed'])
        rows.append(f'| {label} | {item["parameters"]:,} | {item["final_success_mean"]:.1%} ± {item["final_success_sample_std"]:.1%} | {rates} | {item["normalized_auc_mean"]:.4f} |')
    rows+=['','曲线面积为在观测检查点间做梯形积分，再除以 12,000 步，取值约 0–1，较高表示相同采样预算下更早、更持续地成功。它不是泛化成绩或统计显著性。','',
        '![均值与种子间标准差](learning_curves.png)','',
        '阴影为五个种子间的样本标准差，已裁剪到 0–1 显示，并非置信区间。折线仅连接离散检查点，不表示未测时刻的实际轨迹。','',
        '![全部种子，不隐藏回落](per_seed_curves.png)','',
        '## 实验到底控制了什么','',
        '- 物理规则、奖励、动作、回合终止保持原 Catch 不变：球每步下落一行，五步后接到 +1、否则 −1；无中间距离奖励。',
        '- 同一训练种子使用相同环境初态序列，并保存逐回合位置表与哈希。不同策略的动作及经历自然不同；不能要求策略改变时整条轨迹不变。',
        '- 使用同一份 dqn_core.update，Adam lr=0.001、gamma=0.99、batch=64、经验池=10,000、warmup=128、每四步更新、每 200 环境步同步目标网络。epsilon 按同一固定 12,000 步日程下降至 0.05。',
        '- 在 0、250、500 步及之后每 500 步，贪心评估全部 25 个球列×挡板列初态。每次评估记录各局五个动作和结果，不只保存均值。',
        '- 坐标为球行/5、球列/4、挡板列/4；未手工提供相对距离或正确动作。单通道为球 +1、挡板 −1、同格 +2。编码检查枚举 150 个位置状态，确认每种表示都保留不同状态，不因终局重叠抵消。','',
        '## 如何解释与不能推出什么','',
        '坐标数值确实可以表达位置规律，MLP 可以拟合其中的关系；二维网格则给 CNN 提供了局部连接、权重共享的空间归纳偏置。实际学习快慢必须看上面的曲线，不能仅凭“有空间结构”或“输入更小”断言某组更好。',
        '这不是严格的参数量匹配实验：坐标与网格 MLP 第一层输入大小不同；CNN 和 MLP 的参数量、激活函数、结构也不同。因此不能把不同组的全部差异单独归因于表示形式或卷积。双通道 MLP 与 CNN 是相同表示下的模型配方比较；双通道和有符号 CNN 是接近的结构、但并非完全相同参数量的编码比较。',
        '四组输入都是已由环境提供对象位置的编码。坐标直接显式提供位置，网格也已经标出了对象，并不是原始 Atari 视觉识别任务。这项有限状态任务上的表现不能推广为所有游戏、视觉问题或新物理规则的结论。',
        '首次达到某阈值仅指离散检查点中第一次观察到，后面可能回落；不叫已经收敛。不同种子的原始阈值与曲线都保留。五个种子提供一定重复证据，但不能证明某架构普遍占优。','',
        '## 原始数据与验证','',
        '- [全部配置、逐种子指标与逐阶段聚合](summary.json)。',
        '- [编码信息保留检查](encoding_checks.json) · [运行后独立核验](verification.json)。',
        '- 每组目录内的 seed_0 到 seed_4，learning_curve.json 保存全部检查点；evaluations.jsonl 保存每个检查点完整 25 局的初态、动作和结果；metrics.json 保存末步成绩、首次阈值、重载检查与初态哈希。',
        '- 每次运行 model_config.json 保存完整结构与参数量；first_batch 保存真实输入 NPZ、预测、标签和逐项损失。model.pt 只在本机保留，按仓库规则忽略。',
        '- [资源日志](resources.jsonl) · [资源采样摘要](resource_summary.json) · [冻结配置](config.json)。采样值不等于连续峰值，热状态不是摄氏温度。','',
        '## 复现','',
        '在仓库根目录使用新输出目录，并通过现有 guard 监控；本程序拒绝覆盖已有目录。','',
        '```bash',
        '.venv/bin/python grpo_lab/guard.py --log /tmp/my_representation_resources.jsonl --max-seconds 900 -- .venv/bin/python dqn_lab/representation_ablation.py --out dqn_lab/runs/my_representation',
        '```','',
        '本次没有改变主对话正在阅读的环境、训练循环或 DQN 核心代码，也没有提交或推送。','']
    (out/'REPORT.md').write_text('\n'.join(rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=verify_encodings();write_json(args.out/'encoding_checks.json',checks)
    write_json(args.out/'config.json',dict(**CONFIG,variants=VARIANTS,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),ROOT/'dqn_lab/dqn_core.py',ROOT/'tutorials/code/common.py']}))
    metrics=[]
    for variant in VARIANTS:
        for seed in range(5):
            metrics.append(train_one(variant,seed,args.out/variant/f'seed_{seed}'))
            time.sleep(.5)
    for seed in range(5):
        assert len({m['initial_positions_sha256'] for m in metrics if m['seed']==seed})==1
    summarize(args.out,metrics)
    print('ALL_RUNS_COMPLETE',str(args.out),flush=True)


if __name__=='__main__':main()
