"""Handwritten visual REINFORCE -> Actor-Critic -> PPO for existing games.

A teaching implementation, not a claim that CarRacing or Craftium is solved.
Use the explicit device backend and run sustained local jobs under a guard.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from envs import make_visual_env
from evaluation import evaluate
from policy_core import VisualActorCritic,optimize
from rollouts import Collector
from runtime import (select_device,setup,atomic_json,atomic_torch,rng_state,restore_rng,
                     local_resource_snapshot,resource_pressure)


def report(out,config,stages,state):
    rows=[f'# 视觉策略梯度实验 {config["algo"].upper()}','',f'环境：{config["env"]}；状态：{state}。',
        '所有得分为环境原始奖励之和；相同固定种子和相同评估策略用于阶段比较。',
        '达到评估步数上限的记录明确标为 capped，不能称完整回合最终成绩。','',
        '| 阶段 | 采集决策 | 均分 | 标准差 | 有截断 | 录像 |','|---|---:|---:|---:|---|---|']
    for stage in stages:
        value=stage['evaluation'];name=stage['stage']
        rows.append(f'| {name} | {stage["step"]} | {value["mean_score"]:.2f} | {value["std_score"]:.2f} | '
            f'{value["any_capped"]} | [MP4]({name}/evaluation/play.mp4) |')
    rows+=['','训练前后 model.pt 均为实际保存的权重；source/、config.json、逐步原始轨迹和 update 日志可用于复核。',
        'latest.pt 用于继续学习；模拟器和未更新的 on-policy 轨迹不恢复，续训从新一局重新采集。',
        '本程序尚不能由代码本身证明游戏成绩提升；以本目录实际评估和录像为准。']
    (out/'REPORT.md').write_text('\n'.join(rows)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algo',choices=['reinforce','ac','ppo'],default='ppo')
    p.add_argument('--env',default='CarRacing-v3')
    p.add_argument('--env-kwargs',default='{}',help='JSON object passed to gym.make')
    p.add_argument('--device',choices=['cpu','cuda','npu','mps'],default='cpu')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--stage-seconds',type=float,default=300)
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--rollout-steps',type=int,default=256)
    p.add_argument('--episodes-per-update',type=int,default=2)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--minibatch-size',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--clip',type=float,default=.2)
    p.add_argument('--value-coef',type=float,default=.5)
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--normalize-advantages',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--stack',type=int,default=4)
    p.add_argument('--action-repeat',type=int,default=1)
    p.add_argument('--max-episode-steps',type=int,default=None)
    p.add_argument('--eval-episodes',type=int,default=3)
    p.add_argument('--eval-max-steps',type=int,default=2000)
    p.add_argument('--eval-policy',choices=['greedy','sample'],default='greedy')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.seconds,args.stage_seconds,args.num_envs,args.rollout_steps,args.episodes_per_update,
           args.epochs,args.minibatch_size,args.size,args.stack,args.action_repeat,args.eval_episodes,args.eval_max_steps)<1:
        p.error('Time, batch, image and evaluation sizes must be positive')
    if args.size<36:p.error('CNN needs size >= 36')
    if not 0<args.gamma<=1 or not 0<=args.gae_lambda<=1:p.error('Invalid discount or GAE lambda')
    if args.max_episode_steps is not None and args.max_episode_steps<1:p.error('Invalid episode limit')
    if args.algo=='reinforce' and args.num_envs!=1:p.error('Use --num-envs 1 for the complete-episode REINFORCE lesson')
    try:env_kwargs=json.loads(args.env_kwargs)
    except json.JSONDecodeError as error:p.error(f'Invalid env kwargs: {error}')
    if not isinstance(env_kwargs,dict):p.error('--env-kwargs must be a JSON object')
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg is required for actual evaluation video')
    resource=local_resource_snapshot()
    if resource_pressure(resource):raise RuntimeError(f'Resource pressure at start: {resource}')
    setup(args.seed,args.threads);device=select_device(args.device)
    if args.resume:
        if not (args.out/'latest.pt').is_file():raise RuntimeError('No latest.pt to resume')
    else:args.out.mkdir(parents=True,exist_ok=False)
    factory=lambda:make_visual_env(args.env,env_kwargs,args.size,args.stack,args.action_repeat,args.max_episode_steps)
    envs=[];collector=None;step=updates=completed_base=0;raw_base=0;stages=[]
    config={**vars(args),'out':str(args.out),'env_kwargs':env_kwargs,'device':str(device),
        'torch':torch.__version__,'observation':'HWC uint8 -> grayscale resize -> [stack,H,W]',
        'reward':'raw environment reward, no clipping or normalization',
        'sampler':'synchronous CPU env list, one batched policy forward per vector step; not async',
        'budget':'training window includes mid-stage evaluation; initial/final evaluation extra',
        'reinforce_objective':'complete finite observed episodes; TimeLimit is treated as finite task horizon, no bootstrap; unfinished wall-clock prefix is discarded',
        'ac_ppo_bootstrap':'true termination zeros bootstrap; truncation keeps final-observation bootstrap and cuts GAE recursion',
        'resume':'model/optimizer/RNG/counters restored; env and unlearned rollout reset',
        'eval_seeds':list(range(10000,10000+args.eval_episodes))}
    stop_requested=False;window_start=None;started=datetime.datetime.now().astimezone().isoformat();last_resource=0.;last_metrics=None
    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for _ in range(args.num_envs):envs.append(factory())
        collector=Collector(envs,args.seed)
        action_count=int(envs[0].action_space.n)
        model=VisualActorCritic(action_count,args.stack,args.size).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,eps=1e-5)
        if args.resume:
            saved=torch.load(args.out/'latest.pt',map_location='cpu',weights_only=False)
            for key in ('algo','env','env_kwargs','stack','size','action_repeat','max_episode_steps','gamma','gae_lambda',
                        'learning_rate','num_envs','normalize_advantages','clip','value_coef','entropy_coef'):
                if saved['config'][key]!=config[key]:raise RuntimeError(f'Resume config mismatch: {key}')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'],device)
            step,updates,completed_base,raw_base=saved['step'],saved['updates'],saved['completed_episodes'],saved['raw_steps']
            stages=json.loads((args.out/'stages.json').read_text())
        base_step=step
        config.update(action_count=action_count,parameters=sum(x.numel() for x in model.parameters()),network=str(model))
        atomic_json(args.out/'config.json',config)
        sources=args.out/'source';sources.mkdir(exist_ok=True)
        for name in ('train_visual_policy.py','policy_core.py','rollouts.py','envs.py','evaluation.py','runtime.py'):
            shutil.copy2(Path(__file__).parent/name,sources/name)

        def status(state):
            atomic_json(args.out/'status.json',dict(status=state,pid=os.getpid(),started_at=started,
                algo=args.algo,environment=args.env,device=str(device),step=base_step+collector.decisions,
                raw_steps=raw_base+collector.raw_steps,updates=updates,
                completed_episodes=completed_base+collector.completed_episodes,
                elapsed_window_seconds=time.monotonic()-window_start if window_start else 0.,
                latest_stage=stages[-1]['stage'] if stages else None,last_metrics=last_metrics))

        def should_stop():return stop_requested or (window_start is not None and time.monotonic()-window_start>=args.seconds)

        def monitor():
            nonlocal last_resource
            now=time.monotonic()
            if now-last_resource>=5:
                snapshot=local_resource_snapshot();last_resource=now
                with (args.out/'resources.jsonl').open('a') as log:log.write(json.dumps(snapshot)+'\n')
                while resource_pressure(snapshot) and not should_stop():
                    status('waiting_for_resources');time.sleep(5)
                    snapshot=local_resource_snapshot()
                    with (args.out/'resources.jsonl').open('a') as log:log.write(json.dumps(snapshot)+'\n')
                status('training')
            return should_stop()

        def save_latest():
            atomic_torch(args.out/'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng_state(device),step=base_step+collector.decisions,raw_steps=raw_base+collector.raw_steps,
                updates=updates,completed_episodes=completed_base+collector.completed_episodes,config=config))

        def checkpoint(label):
            name=f'{label}_s{base_step+collector.decisions:09d}'
            directory=args.out/name
            if directory.exists():name+=datetime.datetime.now().strftime('_%Y%m%d%H%M%S');directory=args.out/name
            directory.mkdir();atomic_torch(directory/'model.pt',model.state_dict());status('evaluating')
            evaluation=evaluate(model,directory/'evaluation',args.env,env_kwargs,config['eval_seeds'],device,
                args.size,args.stack,args.action_repeat,args.max_episode_steps,args.eval_max_steps,args.eval_policy)
            stage=dict(stage=name,step=base_step+collector.decisions,updates=updates,
                model_sha256=hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),evaluation=evaluation)
            stages.append(stage);atomic_json(args.out/'stages.json',stages);save_latest();report(args.out,config,stages,'训练中')
            print('EVALUATION',json.dumps(stage),flush=True)

        status('initializing');checkpoint('resumed' if args.resume else 'before')
        window_start=time.monotonic();next_stage=args.stage_seconds;last_save=window_start
        with (args.out/'episodes.jsonl').open('a') as episodes,(args.out/'updates.jsonl').open('a') as update_log:
            def episode_log(row):
                row['step']+=base_step;row['episode']+=completed_base
                episodes.write(json.dumps(row)+'\n');episodes.flush()
            status('training');model.train()
            while not monitor():
                if args.algo=='reinforce':collected=collector.complete_episodes(model,device,args.episodes_per_update,monitor,episode_log,args.gamma)
                else:collected=collector.window(model,device,args.rollout_steps,monitor,episode_log,args.gamma,args.gae_lambda)
                if collected is None:break
                batch,audit=collected
                # End the budget before a new learner update. Any unlearned
                # on-policy window is deliberately discarded rather than replayed.
                if monitor():break
                old_logp_hash=hashlib.sha256(batch['old_logp'].tobytes()).hexdigest()
                if updates==0:
                    first=args.out/'first_batch';first.mkdir(exist_ok=True)
                    np.savez_compressed(first/'rollout.npz',**batch)
                    if args.algo!='reinforce':np.savez_compressed(first/'gae_inputs.npz',**audit)
                    else:atomic_json(first/'monte_carlo_scope.json',audit)
                last_metrics=optimize(model,optimizer,batch,args.algo,device,args.epochs,args.minibatch_size,
                    args.clip,args.value_coef,args.entropy_coef,args.normalize_advantages)
                if hashlib.sha256(batch['old_logp'].tobytes()).hexdigest()!=old_logp_hash:raise RuntimeError('Behavior log probabilities mutated')
                updates+=1
                row=dict(update=updates,step=base_step+collector.decisions,batch_size=len(batch['actions']),
                    behavior_old_logp_sha256=old_logp_hash,**last_metrics)
                update_log.write(json.dumps(row)+'\n');update_log.flush()
                if updates==1:atomic_json(args.out/'first_batch/update.json',row)
                status('training');now=time.monotonic()
                if now-last_save>=60:save_latest();last_save=now
                if now-window_start>=next_stage and next_stage<args.seconds:
                    checkpoint(f'minute_{int(next_stage/60):02d}');next_stage+=args.stage_seconds;last_save=time.monotonic()
        save_latest()
        if not stop_requested:checkpoint('final')
        state='stopped' if stop_requested else 'completed';status(state);report(args.out,config,stages,state)
        atomic_json(args.out/'result.json',dict(status=state,algo=args.algo,step=base_step+collector.decisions,
            updates=updates,completed_episodes=completed_base+collector.completed_episodes,
            unfinished_training_scores=collector.scores.tolist(),stages=stages))
    except BaseException as error:
        if args.out.exists():
            atomic_json(args.out/'error.json',dict(type=type(error).__name__,message=str(error)))
            if callable(locals().get('status')):status('failed')
            else:atomic_json(args.out/'status.json',dict(status='failed',pid=os.getpid(),algo=args.algo,
                environment=args.env,device=str(device),step=step,updates=updates,error=str(error)))
        raise
    finally:
        for env in envs:env.close()


if __name__=='__main__':main()
