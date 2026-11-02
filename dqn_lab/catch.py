"""A short DQN run with every intermediate checkpoint and actual game replay."""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tutorials/code'))
from common import Catch, QNet, setup, tensor, replay_catch, save_run
from dqn_core import update


@torch.no_grad()
def evaluate_all(online):
    # 同时维护 N=25 个游戏，枚举球的 5 列 × 挡板的 5 列。
    env = Catch(25, seed=10000)
    # repeat -> [0,0,0,0,0,1,1,...]，tile -> [0,1,2,3,4,0,1,...]；均为 [25]。
    # 按对应下标配对后，恰好覆盖全部 25 个初态，不是随机抽 25 次。
    env.ball[:] = np.repeat(np.arange(5), 5)
    env.player[:] = np.tile(np.arange(5), 5)
    records = [dict(ball=int(b), paddle=int(p), actions=[]) for b,p in zip(env.ball,env.player)]
    for _ in range(5):
        # env.obs(): [25,2,6,5]；tensor 转 float32，形状不变。
        # online(...): [25,3]；argmax(-1) 沿最后的动作维选索引 -> [25]。
        # .numpy() 仅转换容器；actions 仍为 [25]，每个环境各一个动作。
        actions = online(tensor(env.obs())).argmax(-1).numpy()
        for row, action in zip(records, actions): row['actions'].append(int(action))
        # rewards: [25]，第五步结束后才用它统计每局是否接到球。
        _, rewards, _, _ = env.step(actions)
    for row,reward in zip(records,rewards): row['success'] = bool(reward>0)
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    setup(args.seed)
    started=time.time()
    # 采集阶段 N=1，观察形状 [1,2,6,5]。
    # 这个 N 与后面更新时的 B=64 是不同的维度含义。
    env=Catch(1,args.seed)
    # QNet: [B,2,6,5] -> [B,3]，三个输出是左/停/右的 Q 值，不是概率。
    online=QNet('mlp')
    # 复制结构和初始参数；两套参数独立。目标网络不由 optimizer 更新。
    target_net=deepcopy(online).requires_grad_(False)
    optimizer=torch.optim.Adam(online.parameters(),lr=1e-3)
    # 队列存最多 10,000 条“单步经历”，不是 10,000 个 batch。
    # 超过容量时 deque 自动移除最早记录。
    replay=deque(maxlen=10000)
    history=[]
    updates=0
    milestones={0,250,500,1000,2000,4000,8000,args.steps}

    def checkpoint(step):
        records=evaluate_all(online)
        # success 是 Python 标量，25 个 bool 先求和，再除以 25。
        success=sum(r['success'] for r in records)/25
        history.append(dict(step=step,success=success))
        stage=args.out/f'step_{step:05d}'; stage.mkdir()
        torch.save(online.state_dict(),stage/'model.pt')
        (stage/'evaluation.json').write_text(json.dumps(dict(step=step,n=25,success=success,records=records),indent=2)+'\n')
        replay_catch(online,stage/'play.gif',seed=123,games=4)
        with (args.out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,updates=updates,success=success))+'\n')
        print(f'CATCH step={step} updates={updates} success={success:.0%}',flush=True)

    checkpoint(0)
    for step in range(1,args.steps+1):
        # NumPy float32 [1,2,6,5]：环境数、球/挡板通道、行、列。
        observation=env.obs()
        # epsilon 是标量概率，随全局环境步数降低，下限为 0.05。
        epsilon=max(.05,1-step/(args.steps*.65))
        with torch.no_grad():
            # 随机分支直接得到一个 Python 动作编号 0/1/2。
            # 贪心分支：tensor(observation) [1,2,6,5] -> online 输出 [1,3]
            # -> argmax(-1) [1] -> [0] 取唯一环境的动作 -> int 转 Python 整数。
            # 这里的 [0] 取“第一个环境”，不是“动作 0”。本阶段不更新参数。
            action=random.randrange(3) if random.random()<epsilon else int(online(tensor(observation)).argmax(-1)[0])
        # 单个整数 action 包成 [action]，形状 [1]，对应当前的一个环境。
        # following / terminal_observation: [1,2,6,5]。
        # reward: float32 [1]；done: bool [1]，每个环境各返回一个值。
        following,reward,done,terminal_observation=env.step([action])
        # Save the actual terminal frame rather than the auto-reset observation.
        # 未结束：following 是本局下一画面；结束：following 已是新一局开场。
        # 因此结束时取单独保留的终局画面，再 [0] 去掉环境维 -> [2,6,5]。
        nxt=terminal_observation[0] if done[0] else following[0]
        # 一条记录：两个 [2,6,5] 数组 + 一个 int 动作 + 一个 float 奖励 + 一个 bool。
        # observation[0] 仍保留两个通道；copy 将观察保存为独立快照。
        replay.append((observation[0].copy(),action,float(reward[0]),nxt.copy(),bool(done[0])))
        # 两个条件同时满足：先积累 128 条，之后每 4 个环境步更新一次。
        # 第一次在 128 步，后续 132、136……，不是每 128×4 步。
        if len(replay)>=128 and step%4==0:
            # 从经验池随机抽 64 条记录：B=64，而采集的环境数仍是 N=1。
            # update 中堆叠为 states [64,2,6,5]、actions/rewards/terminated [64]。
            # 同一条历史经历可以在不同批次再次使用，标签届时重新计算。
            update(online,target_net,optimizer,random.sample(replay,64),gamma=.99,
                   audit_dir=args.out/'first_batch' if updates==0 else None,
                   action_names=['LEFT','STAY','RIGHT'])
            updates+=1
        # 每 200 个环境步复制参数值，网络结构与张量形状不变。
        # 这是同步目标网络，不是再次反向传播，也不是让两份网络共享参数。
        if step%200==0:target_net.load_state_dict(online.state_dict())
        if step in milestones:checkpoint(step)
    config=argparse.Namespace(steps=args.steps,seed=args.seed,out=str(args.out))
    save_run(args.out,config,dict(initial_success=history[0]['success'],
        final_success=history[-1]['success'],evaluation_episodes=25,
        scope='all 25 initial ball/paddle combinations in this finite game',
        optimizer_updates=updates,algorithm='DQN; no imitation labels'),history,online,started)


if __name__=='__main__':main()
