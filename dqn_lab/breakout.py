"""Real Atari Breakout + Nature-shaped CNN + readable, handwritten DQN.

Short runs verify the data/gradient pipeline; they do not reproduce Atari scores.
This uses Adam/Huber, a small replay pool, full-game termination and no sticky
actions; those choices differ from the complete 2015 experiment protocol.
"""
import argparse
from collections import deque
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import random
import time

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn

from dqn_core import update


class NatureQNet(nn.Module):
    def __init__(self,actions):
        super().__init__()
        # B 是本批观察数，A=actions；本文 Breakout A=4。
        # 输入 [B,4,84,84] 的 4 是四个时刻的灰度帧，不是四种颜色。
        # 每层形状：
        # Conv(8,stride=4): [B,4,84,84] -> [B,32,20,20]
        # Conv(4,stride=2):             -> [B,64,9,9]
        # Conv(3,stride=1):             -> [B,64,7,7]
        # Flatten -> [B,3136]；Linear -> [B,512]；末层 Linear -> [B,A]。
        # 卷积均无 padding；ReLU 不改变形状，最终不使用 softmax。
        self.net=nn.Sequential(
            nn.Conv2d(4,32,kernel_size=8,stride=4),nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512,actions),  # 输出头：[B,512] -> [B,A]，此处 A=4。
        )

    def forward(self,frames):
        # frames 原为 uint8 [B,4,84,84]，像素范围 0..255。
        # .float()/255. -> float32、约 0..1，形状不变；输出 [B,A]。
        return self.net(frames.float()/255.)


class FireOnReset(gym.Wrapper):
    """Launch a ball once per reset; all later actions belong to the policy."""
    def reset(self,**kwargs):
        observation,info=self.env.reset(**kwargs)
        observation,_,terminated,truncated,info=self.env.step(1)
        if terminated or truncated: observation,info=self.env.reset()
        return observation,info


def make_env():
    gym.register_envs(ale_py)
    # 原始单环境观察是 uint8 [210,160,3]，即 HWC；此处没有 batch 维。
    env=gym.make('ALE/Breakout-v5',render_mode='rgb_array',frameskip=1,
                 repeat_action_probability=0.,full_action_space=False)
    # 灰度+缩放 -> [84,84]。frame_skip=4 是执行频率设置，不是通道数量。
    env=gym.wrappers.AtariPreprocessing(env,noop_max=30,frame_skip=4,
        screen_size=84,terminal_on_life_loss=False,grayscale_obs=True)
    env=FireOnReset(env)
    # 连续四帧堆成 [4,84,84]，由 HWC 改成适合 CNN 的历史帧/高/宽布局。
    # 与接球不同，此单环境接口不自动增加 N=1 的最前维。
    return gym.wrappers.FrameStackObservation(env,stack_size=4)


@torch.no_grad()
def evaluate(online,path,steps=200):
    env=make_env()
    observation,info=env.reset(seed=50000)
    frames=[];score=0.;records=[];terminated=False;truncated=False
    for step in range(steps):
        # np.asarray(observation): [4,84,84]；as_tensor 不改变形状。
        # [None] 在最前面增加 batch 维 -> [1,4,84,84]。
        # online 输出 [1,A]；[0] 去掉唯一 batch 行 -> [A] 个动作 Q 值。
        values=online(torch.as_tensor(np.asarray(observation))[None])[0]
        # 此时 values 是一维 [A]，argmax() 返回标量动作索引，再转 Python int。
        action=int(values.argmax())
        # 原生单环境用整数 action；reward 是标量，两个结束标记也是标量 bool。
        # observation 为 [4,84,84]，不同于接球的 [N,2,6,5] 接口。
        observation,reward,terminated,truncated,info=env.step(action)
        score+=reward
        records.append(dict(step=step+1,action=action,reward=float(reward),q_values=values.tolist()))
        if step<120:
            frame=Image.fromarray(env.render()).resize((320,420))
            ImageDraw.Draw(frame).text((8,8),f'step={step+1} score={score:g}',fill='white')
            frames.append(frame)
        if terminated or truncated:break
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=65,loop=0)
    env.close()
    return dict(score=score,steps=len(records),terminated=terminated,truncated=truncated,
        evaluation_is_capped=not (terminated or truncated),records=records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--steps',required=True,type=int)
    parser.add_argument('--learning-starts',type=int,default=1000)
    parser.add_argument('--buffer-size',type=int,default=5000)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--eval-steps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.batch_size>args.buffer_size:parser.error('batch exceeds replay capacity')
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    env=make_env(); observation,info=env.reset(seed=args.seed)
    names=env.unwrapped.get_action_meanings()
    # 观察不含 batch 维，实际送入网络前再手动增加。
    assert np.asarray(observation).shape==(4,84,84)
    online=NatureQNet(env.action_space.n)
    target_net=deepcopy(online).requires_grad_(False)
    optimizer=torch.optim.Adam(online.parameters(),lr=1e-4)
    replay=deque(maxlen=args.buffer_size)
    config={**vars(args),'out':str(args.out),'action_names':names,
        'parameters':sum(p.numel() for p in online.parameters()),'input_shape':[4,84,84],
        'network':str(online),'torch':torch.__version__,
        'gymnasium':importlib.metadata.version('gymnasium'),'ale_py':importlib.metadata.version('ale-py'),
        'reward_for_learning':'sign(raw_reward)','termination':'full game; bootstrap time-limit truncations',
        'protocol':'teaching implementation, not an exact reproduction of 2015 scores'}
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for i in range(4):Image.fromarray(np.asarray(observation)[i]).save(args.out/f'input_frame_{i}.png')
    before=evaluate(online,args.out/'before.gif',args.eval_steps)
    (args.out/'before_evaluation.json').write_text(json.dumps(before,indent=2)+'\n')
    updates=0;episode_score=0.;episode_scores=[];losses=[]
    for step in range(1,args.steps+1):
        epsilon=max(.1,1-step/100000.)
        with torch.no_grad():
            # 随机分支产生一个 int；贪心分支为
            # [4,84,84] -> [1,4,84,84] -> [1,A] -> argmax [1] -> int。
            action=random.randrange(len(names)) if random.random()<epsilon else int(online(torch.as_tensor(np.asarray(observation))[None]).argmax(-1)[0])
        following,reward,terminated,truncated,info=env.step(action)
        # 一条经验保存两个 uint8 [4,84,84] 数组与三个标量。
        # sign(reward) 将训练奖励裁剪到 -1/0/+1；评测仍累计原始分数。
        # 此环境不自动重置，所以 following 是真实下一观察/终局观察。
        # 只存 terminated 作为标签 mask；时间限制 truncated 仍允许 bootstrap。
        replay.append((np.asarray(observation).copy(),action,float(np.sign(reward)),
                       np.asarray(following).copy(),bool(terminated)))
        episode_score+=reward;observation=following
        if terminated or truncated:
            episode_scores.append(float(episode_score));episode_score=0.
            observation,info=env.reset()
        if step>=args.learning_starts and len(replay)>=args.batch_size and step%4==0:
            # 抽到 B 条后，更新函数堆成 uint8 states [B,4,84,84]。
            # 网络内部才转 float32 并归一化；其余动作/奖励/终止标记为 [B]。
            # 最终在线预测、标签、逐样本损失都是 [B]，mean 后的 loss 为标量 []。
            loss=update(online,target_net,optimizer,random.sample(replay,args.batch_size),.99,
                audit_dir=args.out/'first_batch' if updates==0 else None,action_names=names)
            updates+=1;losses.append(loss)
            with (args.out/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,update=updates,loss=loss,
                    epsilon=epsilon,finished_episode_scores=episode_scores[-10:]))+'\n')
        if step%1000==0:target_net.load_state_dict(online.state_dict())
        if step%64==0 or step==args.steps:
            print(f'BREAKOUT step={step} updates={updates} episode_score={episode_score:g}',flush=True)
    after=evaluate(online,args.out/'after.gif',args.eval_steps)
    (args.out/'after_evaluation.json').write_text(json.dumps(after,indent=2)+'\n')
    torch.save(online.state_dict(),args.out/'model.pt')
    result=dict(environment='ALE/Breakout-v5',training_steps=args.steps,optimizer_updates=updates,
        before_capped_score=before['score'],after_capped_score=after['score'],
        eval_step_budget=args.eval_steps,finished_training_episode_scores=episode_scores,
        last_unfinished_training_episode_score=float(episode_score),
        first_loss=losses[0] if losses else None,last_loss=losses[-1] if losses else None,
        elapsed_seconds=time.time()-started,
        evidence_scope='Short pipeline check; not evidence of learning to solve Breakout.')
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    env.close();print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
