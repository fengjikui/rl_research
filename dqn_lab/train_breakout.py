"""Readable, bounded Atari DQN training on MPS/CPU with auditable checkpoints.

The Bellman label, gather, Smooth L1 and gradient update are in dqn_core.update.
This file adds the real game, device, replay, evaluation and experiment records.
"""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

from atari_evaluation import evaluate_stage
from atari_replay import FrameReplay
from breakout import NatureQNet, make_env
from dqn_core import update


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return value


def atomic_torch(path, value):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(cpu_copy(value),temporary)
    temporary.replace(path)


def report(directory, stages, status):
    rows = ['# 本机 GPU 打砖块 DQN 实验', '', f'状态：{status}', '',
        '固定 3 个评估种子；评估采用贪心策略，丢球后自动发球。训练分数和评估分数分开记录。',
        '单局上限以 evaluation.json 为准；达到上限的分数不应当作完整回合得分。', '',
        '| 阶段 | 交互步 | 各局原始分数 | 均分 | 有截断 | 录像 |',
        '|---|---:|---|---:|---|---|']
    for s in stages:
        scores = ', '.join(f'{e["score"]:g}' for e in s['evaluation']['episodes'])
        media = '旧模型与录像已按留存规则精简；原始分数仍保留' if s.get('artifacts_pruned') else f'[MP4]({s["stage"]}/play.mp4) · [GIF]({s["stage"]}/preview.gif)'
        rows.append(f'| {s["stage"]} | {s["step"]} | {scores} | '
            f'{s["evaluation"]["mean_score"]:.2f} | {s["evaluation"]["any_capped"]} | '
            f'{media} |')
    rows += ['', '每阶段 model.pt 是当时实际评估的权重；evaluation.json 与 episode_*.jsonl 保存原始数据。',
        'latest.pt 保存在线/目标网络、优化器、随机数状态、步数和回放计数。replay/ 为最新回放池。',
        '续训保留学习状态和回放数据，但重置环境开启新一局；不承诺位级连续复现。',
        '阶段 model.pt 用于推理；只有 latest.pt 与最新 replay/ 配套用于继续训练。',
        '这是小预算教学实验，使用 Adam、较小回放池和完整游戏终止，不是 2015 年全部配方的复现。',
        '不得由 loss 下降或单局好运断言已学会；应查看所有固定种子的逐阶段分数。']
    (directory/'REPORT.md').write_text('\n'.join(rows)+'\n')
    if stages:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        x = [s['step'] for s in stages]
        y = [s['evaluation']['mean_score'] for s in stages]
        std = [s['evaluation']['std_score'] for s in stages]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.errorbar(x,y,yerr=std,fmt='o-',capsize=4,label='Fixed-seed mean +/- SD')
        n = len(stages[0]['evaluation']['episodes'])
        ax.set(xlabel='Training environment decisions',ylabel='Original game score',
               title=f'Breakout DQN evaluation ({n} fixed seeds)')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(directory/'evaluation_curve.png',dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--stage-seconds',type=float,default=300)
    parser.add_argument('--buffer-size',type=int,default=100000)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--learning-starts',type=int,default=5000)
    parser.add_argument('--epsilon-decay-steps',type=int,default=200000)
    parser.add_argument('--eval-max-steps',type=int,default=4500)
    parser.add_argument('--eval-episodes',type=int,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.seconds<=0 or args.stage_seconds<=0 or args.eval_max_steps<1 or args.eval_episodes<1:
        parser.error('Time and evaluation limits must be positive')
    if args.buffer_size < max(args.learning_starts,args.batch_size)+4:
        parser.error('Replay capacity must exceed warmup/batch size by at least four')
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; refusing to silently fall back to CPU')
    if shutil.which('ffmpeg') is None: raise RuntimeError('ffmpeg is required for real videos')
    if args.resume:
        if not (args.out/'latest.pt').is_file(): raise RuntimeError('No latest.pt to resume')
    else:
        args.out.mkdir(parents=True,exist_ok=False)
        if shutil.disk_usage(args.out).free < 6*2**30: raise RuntimeError('Need 6 GiB free disk space')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 1. 环境与网络：[4,84,84] 的四帧灰度图 -> 三层卷积 -> 4 个动作的 Q 值。
    env = make_env()
    observation, info = env.reset(seed=args.seed)
    names = env.unwrapped.get_action_meanings()
    online = NatureQNet(len(names)).to(device)
    target = deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(),lr=1e-4)
    replay = FrameReplay(args.out/'replay',args.buffer_size,args.resume)
    step = updates = completed_episodes = 0
    stages = []
    if args.resume:
        saved = torch.load(args.out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['replay_count'] != replay.count:
            raise RuntimeError('Checkpoint and replay metadata counts differ')
        for key in ('buffer_size','batch_size','epsilon_decay_steps','learning_starts','seed'):
            if saved['config'][key] != getattr(args,key): raise RuntimeError(f'Resume config mismatch: {key}')
        online.load_state_dict(saved['online']); target.load_state_dict(saved['target'])
        optimizer.load_state_dict(saved['optimizer'])
        step, updates, completed_episodes = saved['step'],saved['updates'],saved['completed_episodes']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device=='mps' and saved.get('mps_rng') is not None:
            torch.mps.set_rng_state(saved['mps_rng'])
        stages = json.loads((args.out/'stages.json').read_text())
    config = {**vars(args),'out':str(args.out),'device':str(device),'action_names':names,
        'parameters':sum(p.numel() for p in online.parameters()),'input_shape':[4,84,84],
        'torch':torch.__version__,'gamma':.99,'learning_rate':1e-4,'update_every':4,
        'target_copy_every':1000,'reward_for_training':'sign(raw_reward)',
        'termination':'full game terminated; bootstrap truncation',
        'serving':'after reset and lost life, auto FIRE; actual action is recorded',
        'eval_seeds':list(range(10000,10000+args.eval_episodes)),
        'budget':'30-minute run window by default, including intermediate evaluations; baseline/final evaluation extra',
        'resume_environment':'reset to new episode; model/optimizer/replay/RNG retained'}
    atomic_json(args.out/'config.json',config)
    sources = args.out/'source'; sources.mkdir(exist_ok=True)
    for filename in ('train_breakout.py','atari_replay.py','atari_evaluation.py','dqn_core.py','breakout.py'):
        shutil.copy2(Path(__file__).parent/filename,sources/filename)
    started_at = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    overall_start = time.monotonic()
    window_start = None
    episode_score = 0.0; episode_steps = 0
    lives = int(info.get('lives',env.unwrapped.ale.lives()))
    new_episode = True; serve_pending = False; stop_requested = False
    last_loss = None

    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)

    def status(state):
        elapsed = time.monotonic()-window_start if window_start else 0.0
        value = dict(status=state,pid=os.getpid(),started_at=started_at,device=str(device),
            step=step,updates=updates,completed_episodes=completed_episodes,
            elapsed_window_seconds=elapsed,budget_seconds=args.seconds,last_loss=last_loss,
            epsilon=max(.1,1-.9*step/args.epsilon_decay_steps),
            replay_count=replay.count,replay_size=len(replay),
            latest_stage=stages[-1]['stage'] if stages else None,
            mps_allocated_mib=torch.mps.current_allocated_memory()/2**20 if args.device=='mps' else None,
            mps_driver_allocated_mib=torch.mps.driver_allocated_memory()/2**20 if args.device=='mps' else None)
        atomic_json(args.out/'status.json',value)

    def save_latest():
        replay.flush()
        atomic_torch(args.out/'latest.pt',dict(online=online.state_dict(),target=target.state_dict(),
            optimizer=optimizer.state_dict(),step=step,updates=updates,completed_episodes=completed_episodes,
            replay_count=replay.count,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if args.device=='mps' else None,config=config))

    def checkpoint(label):
        stage_name = f'{label}_s{step:09d}'
        directory = args.out/stage_name
        directory.mkdir()
        atomic_torch(directory/'model.pt',online.state_dict())
        digest = hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest()
        status('evaluating')
        evaluation = evaluate_stage(online,directory,config['eval_seeds'],args.eval_max_steps)
        stage = dict(stage=stage_name,step=step,updates=updates,
            elapsed_window_seconds=time.monotonic()-window_start if window_start else 0.0,
            model_sha256=digest,evaluation=evaluation)
        stages.append(stage)
        atomic_json(args.out/'stages.json',stages)
        save_latest(); report(args.out,stages,'训练中')
        print('EVALUATION',json.dumps(stage),flush=True)

    try:
        # 2. 首先保存并评估尚未训练的模型；后续使用相同开局种子。
        status('initializing')
        checkpoint('before' if not args.resume else 'resumed')
        window_start = time.monotonic()
        next_stage = args.stage_seconds
        last_save = last_status = window_start
        status('training')
        with (args.out/'updates.jsonl').open('a') as update_log, (args.out/'episodes.jsonl').open('a') as episode_log:
            while time.monotonic()-window_start < args.seconds and not stop_requested:
                # 3. epsilon-greedy 采集：探索与训练时的 gather 各司其职。
                epsilon = max(.1,1-.9*step/args.epsilon_decay_steps)
                if serve_pending:
                    action = 1
                elif random.random()<epsilon:
                    action = random.randrange(len(names))
                else:
                    with torch.no_grad():
                        action = int(online(torch.as_tensor(np.asarray(observation),device=device)[None]).argmax(-1).item())
                following,reward,terminated,truncated,info = env.step(action)
                replay.append(observation,action,float(np.sign(reward)),following,bool(terminated),new_episode)
                new_episode = False
                episode_score += float(reward); episode_steps += 1
                next_lives = int(info.get('lives',env.unwrapped.ale.lives()))
                serve_pending = next_lives<lives and not (terminated or truncated)
                lives = next_lives; observation = following; step += 1
                if terminated or truncated:
                    completed_episodes += 1
                    episode_log.write(json.dumps(dict(episode=completed_episodes,step=step,
                        score=episode_score,episode_steps=episode_steps,epsilon=epsilon,
                        terminated=bool(terminated),truncated=bool(truncated)))+'\n')
                    episode_score = 0.0; episode_steps = 0
                    observation,info = env.reset()
                    lives = int(info.get('lives',env.unwrapped.ale.lives()))
                    new_episode = True; serve_pending = False

                # 4. 原来的 DQN 更新函数：只增加设备转移，标签和损失公式不变。
                if step>=args.learning_starts and len(replay)>=args.batch_size and step%4==0:
                    last_loss = update(online,target,optimizer,replay.sample(args.batch_size),.99,
                        audit_dir=args.out/'first_batch' if updates==0 else None,action_names=names)
                    if not np.isfinite(last_loss): raise RuntimeError('Non-finite training loss')
                    updates += 1
                    update_log.write(json.dumps(dict(step=step,update=updates,loss=last_loss,epsilon=epsilon))+'\n')
                if step%1000==0: target.load_state_dict(online.state_dict())
                now = time.monotonic()
                if now-last_status>=5:
                    status('training'); update_log.flush(); episode_log.flush(); last_status=now
                if now-last_save>=60:
                    save_latest(); last_save=now
                if now-window_start>=next_stage and next_stage<args.seconds:
                    checkpoint(f'minute_{int(next_stage/60):02d}')
                    next_stage += args.stage_seconds
                    last_save=time.monotonic()

        save_latest()
        if not stop_requested:
            checkpoint('final')
        state = 'stopped' if stop_requested else 'completed'
        status(state); report(args.out,stages,state)
        atomic_json(args.out/'result.json',dict(status=state,step=step,updates=updates,
            elapsed_total_seconds=time.monotonic()-overall_start,stages=stages,
            unfinished_training_episode=dict(score=episode_score,steps=episode_steps)))
        print('FINISHED',state,'steps',step,'updates',updates,flush=True)
    except BaseException as error:
        status('failed')
        atomic_json(args.out/'error.json',dict(type=type(error).__name__,message=str(error),step=step,updates=updates))
        report(args.out,stages,f'失败：{type(error).__name__}: {error}')
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
