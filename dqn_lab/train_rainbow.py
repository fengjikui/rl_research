"""Handwritten, guarded-friendly Rainbow training for real Atari Breakout.

Run under grpo_lab/guard.py. A stage model is for inference; latest.pt together
with replay/ is the resumable state. No files from the DQN run are overwritten.
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
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

from atari_evaluation import evaluate_stage
from breakout import make_env
from rainbow_core import RainbowQNet, update
from rainbow_replay import RainbowReplay
from train_breakout import atomic_json, atomic_torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grpo_lab.guard import snapshot, pressure


def report(directory, stages, state):
    rows=['# 本机 GPU Rainbow 打砖块实验','',f'状态：{state}','',
        '六组件：Double DQN、Dueling、C51、优先经验回放、多步回报和 NoisyNet。',
        '完整六组件教学实现；超参数与游戏终止设置不完全复刻原论文。',
        '阶段评估关闭 NoisyNet，使用固定种子贪心策略；游戏分数为原始奖励之和。',
        '三个固定种子可能产生同一轨迹，不能当作三份独立泛化证据。','',
        '| 阶段 | 交互步 | 各局原始分数 | 均分 | 有截断 | 录像 |',
        '|---|---:|---|---:|---|---|']
    for stage in stages:
        value=stage['evaluation'];name=stage['stage']
        scores=', '.join(f'{episode["score"]:g}' for episode in value['episodes'])
        media=('旧模型与录像已精简；原始分数保留' if stage.get('artifacts_pruned') else
               f'[MP4]({name}/play.mp4) · [GIF]({name}/preview.gif)')
        rows.append(f'| {name} | {stage["step"]} | {scores} | {value["mean_score"]:.2f} | '
            f'{value["any_capped"]} | {media} |')
    heldout=directory/'latest_heldout.json'
    if heldout.exists():
        value=json.loads(heldout.read_text())
        rows.extend(['','## 独立随机复测','',
            f'模型阶段：{value["stage"]}；交互步：{value["step"]}；epsilon={value["epsilon"]}。',
            f'{len(value["episodes"])} 局均分 {value["mean_score"]:.2f}，标准差 {value["std_score"]:.2f}。',
            f'逐局分数：{[x["score"] for x in value["episodes"]]}；是否存在达到上限的局：{value["any_capped"]}。',
            f'[原始复测]({value["stage"]}/heldout/evaluation.json) · [首局全程录像]({value["stage"]}/heldout/play.mp4)'])
    rows.extend(['','每阶段 model.pt 是实际评估权重，episode_*.jsonl 保存每一步动作和原始奖励。',
        'latest.pt 配合 replay/ 保存两个网络、优化器、帧数据、PER 优先级、计数和随机数状态。',
        '单独阶段权重用于推理，不能与最新回放配对续训；续训会重置模拟器，标记采集边界并保留 bootstrap。',
        'first_batch/ 保存 C51 投影、Double 选择、IS 权重、KL 优先级和实际标量损失。',
        '单局步数上限见 evaluation.json；capped 分数是有限时长得分，不能称完整游戏终局成绩。'])
    (directory/'REPORT.md').write_text('\n'.join(rows)+'\n')
    if stages:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        ax.errorbar([x['step'] for x in stages],[x['evaluation']['mean_score'] for x in stages],
            yerr=[x['evaluation']['std_score'] for x in stages],fmt='o-',capsize=3,label='Fixed-seed greedy mean +/- SD')
        ax.set(xlabel='Training environment decisions',ylabel='Original game score',title='Rainbow Breakout evaluation')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(directory/'evaluation_curve.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--stage-seconds',type=float,default=300)
    p.add_argument('--buffer-size',type=int,default=100000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-starts',type=int,default=5000)
    p.add_argument('--n-step',type=int,default=3)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--per-alpha',type=float,default=.5)
    p.add_argument('--beta-steps',type=int,default=10000000)
    p.add_argument('--learning-rate',type=float,default=6.25e-5)
    p.add_argument('--target-every',type=int,default=8000)
    p.add_argument('--eval-max-steps',type=int,default=4500)
    p.add_argument('--eval-episodes',type=int,default=3)
    p.add_argument('--heldout-episodes',type=int,default=10)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.seconds,args.stage_seconds,args.batch_size,args.n_step,args.eval_max_steps,
           args.eval_episodes,args.heldout_episodes,args.target_every,args.beta_steps)<=0:
        p.error('Budgets, intervals, batch and evaluation sizes must be positive')
    if not 0<args.gamma<1 or not 0<=args.per_alpha<=1:p.error('Invalid gamma/PER alpha')
    if args.buffer_size<max(args.learning_starts,args.batch_size)+args.n_step+4:
        p.error('Replay capacity must exceed warmup/batch plus n-step and history')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU fallback')
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg is required')
    resource=snapshot()
    if pressure(resource,2):raise RuntimeError(f'Refusing high-load start under resource pressure: {resource}')
    if args.resume:
        if not (args.out/'latest.pt').is_file():raise RuntimeError('No latest.pt to resume')
    else:args.out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(args.out).free<6*2**30:raise RuntimeError('Need 6 GiB free disk')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    env=make_env();observation,info=env.reset(seed=args.seed)
    actions=env.unwrapped.get_action_meanings()
    online=RainbowQNet(len(actions)).to(device)
    target=deepcopy(online).requires_grad_(False)
    optimizer=torch.optim.Adam(online.parameters(),lr=args.learning_rate,eps=1.5e-4)
    replay=RainbowReplay(args.out/'replay',args.buffer_size,args.n_step,args.gamma,args.per_alpha,args.resume)
    step=updates=completed_episodes=0;stages=[]
    if args.resume:
        saved=torch.load(args.out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['replay_count']!=replay.count:raise RuntimeError('Checkpoint/replay count mismatch; refusing inconsistent resume')
        for name in ('buffer_size','batch_size','learning_starts','n_step','gamma','per_alpha',
                     'beta_steps','learning_rate','target_every','seed'):
            if saved['config'][name]!=getattr(args,name):raise RuntimeError(f'Resume config mismatch: {name}')
        online.load_state_dict(saved['online']);target.load_state_dict(saved['target'])
        optimizer.load_state_dict(saved['optimizer'])
        step,updates,completed_episodes=saved['step'],saved['updates'],saved['completed_episodes']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        if args.device=='mps' and saved.get('mps_rng') is not None:torch.mps.set_rng_state(saved['mps_rng'])
        stages=json.loads((args.out/'stages.json').read_text())
    config={**vars(args),'out':str(args.out),'algorithm':'Rainbow (six components)',
        'device':str(device),'input_shape':[4,84,84],'action_names':actions,
        'parameters':sum(x.numel() for x in online.parameters()),'torch':torch.__version__,
        'atoms':51,'v_min':-10.,'v_max':10.,'update_every':4,'gradient_clip':10.,
        'exploration':'uniform warmup; learned factorized NoisyNet afterwards; no epsilon train schedule',
        'beta_schedule':'0.4 -> 1.0 over beta_steps environment decisions',
        'reward_for_training':'sign(raw_reward); evaluation sums original rewards',
        'termination':'full-game terminal zeroes bootstrap; truncation/resume cuts history but bootstraps',
        'serving':'auto FIRE on reset and after lost life; actual action recorded',
        'evaluation':'3 fixed-seed greedy episodes at stages, final heldout epsilon=0.05 episodes',
        'eval_seeds':list(range(10000,10000+args.eval_episodes)),
        'budget':'seconds includes intermediate evaluation; baseline, final, final heldout are additional',
        'resume_environment':'new episode; model/target/optimizer/PER/replay/RNG restored',
        'reproduction_scope':'complete six-component teaching implementation; not exact original Rainbow recipe'}
    atomic_json(args.out/'config.json',config)
    sources=args.out/'source';sources.mkdir(exist_ok=True)
    for name in ('train_rainbow.py','rainbow_core.py','rainbow_replay.py','atari_evaluation.py',
                 'breakout.py','train_breakout.py'):
        shutil.copy2(Path(__file__).parent/name,sources/name)
    started=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    overall_start=time.monotonic();window_start=None;stop_requested=False;last_loss=None
    episode_score=0.;episode_steps=0;new_episode=True;serve_pending=False
    lives=int(info.get('lives',env.unwrapped.ale.lives()))

    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

    def beta():return min(1.,.4+.6*step/args.beta_steps)

    def status(state):
        value=dict(status=state,algorithm='Rainbow',pid=os.getpid(),started_at=started,device=str(device),
            step=step,updates=updates,completed_episodes=completed_episodes,
            elapsed_window_seconds=time.monotonic()-window_start if window_start else 0.,
            budget_seconds=args.seconds,last_loss=last_loss,beta=beta(),replay_count=replay.count,
            replay_size=min(replay.count,replay.capacity),sampleable_transitions=len(replay),
            latest_stage=stages[-1]['stage'] if stages else None,
            mps_allocated_mib=torch.mps.current_allocated_memory()/2**20 if args.device=='mps' else None,
            mps_driver_allocated_mib=torch.mps.driver_allocated_memory()/2**20 if args.device=='mps' else None)
        atomic_json(args.out/'status.json',value)

    def save_latest():
        replay.flush()
        atomic_torch(args.out/'latest.pt',dict(algorithm='Rainbow',online=online.state_dict(),
            target=target.state_dict(),optimizer=optimizer.state_dict(),step=step,updates=updates,
            completed_episodes=completed_episodes,replay_count=replay.count,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if args.device=='mps' else None,config=config))

    def checkpoint(label):
        name=f'{label}_s{step:09d}'
        directory=args.out/name
        if directory.exists():
            name+=datetime.datetime.now().strftime('_%Y%m%d%H%M%S');directory=args.out/name
        directory.mkdir()
        atomic_torch(directory/'model.pt',online.state_dict())
        status('evaluating')
        evaluation=evaluate_stage(online,directory,config['eval_seeds'],args.eval_max_steps)
        value=dict(stage=name,step=step,updates=updates,
            elapsed_window_seconds=time.monotonic()-window_start if window_start else 0.,
            model_sha256=hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),evaluation=evaluation)
        stages.append(value);atomic_json(args.out/'stages.json',stages)
        save_latest();report(args.out,stages,'Rainbow 训练中')
        print('EVALUATION',json.dumps(value),flush=True)

    try:
        status('initializing');checkpoint('resumed' if args.resume else 'before')
        window_start=time.monotonic();next_stage=args.stage_seconds
        last_save=last_status=window_start;status('training')
        with (args.out/'updates.jsonl').open('a') as update_log,(args.out/'episodes.jsonl').open('a') as episode_log:
            while time.monotonic()-window_start<args.seconds and not stop_requested:
                if serve_pending:action=1
                elif step<args.learning_starts:action=random.randrange(len(actions))
                else:
                    # NoisyNet supplies exploration. Refresh every four decisions,
                    # matching the learner cadence, rather than adding epsilon.
                    if step%4==0:online.reset_noise()
                    with torch.no_grad():
                        action=int(online(torch.as_tensor(np.asarray(observation),device=device)[None]).argmax(-1).item())
                following,reward,terminated,truncated,info=env.step(action)
                replay.append(observation,action,float(np.sign(reward)),following,
                    bool(terminated),bool(truncated),new_episode)
                new_episode=False;step+=1;episode_score+=float(reward);episode_steps+=1
                next_lives=int(info.get('lives',env.unwrapped.ale.lives()))
                serve_pending=next_lives<lives and not (terminated or truncated)
                lives=next_lives;observation=following
                if terminated or truncated:
                    completed_episodes+=1
                    episode_log.write(json.dumps(dict(episode=completed_episodes,step=step,
                        score=episode_score,episode_steps=episode_steps,terminated=bool(terminated),
                        truncated=bool(truncated)))+'\n')
                    episode_score=0.;episode_steps=0;observation,info=env.reset()
                    lives=int(info.get('lives',env.unwrapped.ale.lives()));new_episode=True;serve_pending=False
                if step>=args.learning_starts and len(replay)>=args.batch_size and step%4==0:
                    last_loss=update(online,target,optimizer,replay.sample(args.batch_size,beta()),replay,
                        audit_dir=args.out/'first_batch' if updates==0 else None)
                    if not np.isfinite(last_loss):raise RuntimeError('Non-finite Rainbow loss')
                    updates+=1;update_log.write(json.dumps(dict(step=step,update=updates,loss=last_loss,beta=beta()))+'\n')
                if step%args.target_every==0:target.load_state_dict(online.state_dict())
                now=time.monotonic()
                if now-last_status>=5:
                    status('training');update_log.flush();episode_log.flush();last_status=now
                if now-last_save>=60:save_latest();last_save=now
                if now-window_start>=next_stage and next_stage<args.seconds:
                    checkpoint(f'minute_{int(next_stage/60):02d}');next_stage+=args.stage_seconds
                    last_save=time.monotonic()
        save_latest()
        if not stop_requested:
            checkpoint('final');status('evaluating_heldout')
            heldout=args.out/stages[-1]['stage']/'heldout'
            independent=evaluate_stage(online,heldout,range(30000,30000+args.heldout_episodes),
                args.eval_max_steps,epsilon=.05)
            atomic_json(args.out/'latest_heldout.json',dict(stage=stages[-1]['stage'],step=step,**independent))
        state='stopped' if stop_requested else 'completed'
        status(state);report(args.out,stages,f'Rainbow {state}')
        atomic_json(args.out/'result.json',dict(status=state,algorithm='Rainbow',step=step,updates=updates,
            elapsed_total_seconds=time.monotonic()-overall_start,stages=stages,
            unfinished_training_episode=dict(score=episode_score,steps=episode_steps)))
        print('FINISHED',state,'steps',step,'updates',updates,flush=True)
    except BaseException as error:
        status('failed');atomic_json(args.out/'error.json',dict(type=type(error).__name__,message=str(error),step=step,updates=updates))
        raise
    finally:env.close()


if __name__=='__main__':main()
