"""Small shared pieces. Training algorithms remain in their lesson files."""
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # tiny CPU batches: thread overhead otherwise dominates


def tensor(x):
    # 仅把数组转换成 float32 张量；不自动增加 batch 维，也不改变形状。
    # 接球采集 [1,2,6,5] -> [1,2,6,5]；训练 [B,2,6,5] -> [B,2,6,5]。
    return torch.as_tensor(x, dtype=torch.float32)


class Catch:
    """Vectorized 6x5 catch game. Actions: left=0, stay=1, right=2.

    Ball falls one row each step. Each episode ends after exactly five actions.
    step auto-resets finished slots; terminal observation is returned separately.
    This is a finite game termination, not an artificial time-limit truncation.
    """
    def __init__(self, n=1, seed=0):
        # N=n 表示同时维护多少个独立游戏；它是环境接口的设计，不是 DQN 规定。
        self.n = n
        self.rng = np.random.default_rng(seed)
        # ball、player、row 都是 [N] 整数数组：每个下标对应一个游戏。
        # ball 是球所在列，player 是挡板所在列，row 是球当前所在行。
        self.ball = self.rng.integers(0, 5, n)
        self.player = self.rng.integers(0, 5, n)
        self.row = np.zeros(n, dtype=int)

    def obs(self):
        # [N,C,H,W]=[N,2,6,5]，NumPy float32，元素为 0/1。
        # C=2 是两个对象通道：0 标记球，1 标记挡板；不是 RGB 颜色通道。
        x = np.zeros((self.n, 2, 6, 5), dtype=np.float32)
        # np.arange(N)、row、ball 都为 [N]。高级索引逐个游戏配对：
        # x[i,0,row[i],ball[i]]=1，而不是生成 N×N 个坐标组合。
        x[np.arange(self.n), 0, self.row, self.ball] = 1
        # 同理，每个游戏在第 1 通道的最后一行（行号 5）标记挡板。
        x[np.arange(self.n), 1, 5, self.player] = 1
        return x

    def step(self, actions):
        # 输入需含 N 个动作。单环境用 [action]，N 环境用长度 N 的列表/数组。
        # reshape(N) 规范为一维 [N]；不表示把一个动作广播给所有环境。
        actions = np.asarray(actions, dtype=int).reshape(self.n)
        if not np.isin(actions, [0, 1, 2]).all():
            raise ValueError('Catch actions must be 0, 1, or 2')
        # [N]+[N]-标量 -> [N]；动作 0/1/2 对应列位移 -1/0/+1。
        # clip 将每个挡板限制在 0..4 列，不改变 [N] 形状。
        self.player = np.clip(self.player + actions - 1, 0, 4)
        self.row += 1
        # 逐元素比较 -> bool [N]；第五步是真正终止，不是训练侧时间截断。
        done = self.row == 5
        # 两层 where 均按元素计算 -> [N]：未结束 0，结束时接到 +1，否则 -1。
        reward = np.where(done, np.where(self.player == self.ball, 1., -1.), 0.)
        # 在重置之前拍下 [N,2,6,5]，保留这一局执行动作后的真实画面。
        final_obs = self.obs()
        # done.sum() 把 [N] 个布尔值累加为标量，count 是本步结束的游戏数量。
        count = int(done.sum())
        # 用 bool [N] 选择已结束游戏，选中的形状为 [count]；只重置这些游戏。
        self.ball[done] = self.rng.integers(0, 5, count)
        self.player[done] = self.rng.integers(0, 5, count)
        self.row[done] = 0
        # 依次返回 [N,2,6,5]、float32 [N]、bool [N]、[N,2,6,5]。
        # 若某局结束，第一个观察对应新一局，final_obs 对应刚结束的旧一局。
        return self.obs(), reward.astype(np.float32), done, final_obs


class Backbone(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        if kind == 'mlp':
            # 输入 [B,2,6,5]；Flatten 默认从第 1 维起展平，保留 B：
            # [B,2,6,5] -> [B,60] -> [B,64] -> [B,64]。
            # Tanh 逐元素运算，不改变形状。B 可以是采集的 1，也可以是训练的 64。
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(60, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh())
        else:
            # 接球 CNN 的形状流（不是 Atari 的四帧 CNN）：
            # [B,2,6,5] -> [B,8,6,5] -> [B,16,6,5]
            # -> Flatten [B,480] -> Linear [B,64]。
            # 两层卷积 stride=1、padding=1、kernel=3，故高宽都保持 6×5。
            self.net = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(480, 64), nn.ReLU())

    def forward(self, x):
        # 无论 MLP 还是 CNN，都将 [B,2,6,5] 编成 [B,64] 特征。
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        self.features = Backbone(kind)
        # Linear 的输入特征数是 64，输出特征数是 3；参数权重为 [3,64]。
        # 输出的三个数对应 LEFT/STAY/RIGHT 的动作价值，不经过 softmax。
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        # x [B,2,6,5] -> features [B,64] -> head [B,3]。
        # 三个动作共享特征层，更新一个动作的误差也可能间接影响其他输出。
        return self.head(self.features(x))


class ActorCritic(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        self.features = Backbone(kind)
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        # 后续 Actor-Critic 课程使用，非当前 DQN：h [B,64]。
        # actor(h) [B,3] 是动作 logits；critic(h) [B,1] 是状态价值。
        # squeeze(-1) 只去掉价值末尾的单维，返回 [B]，保留 batch 维。
        h = self.features(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@torch.no_grad()
def evaluate_catch(model=None, seed=10000, n=500):
    # N=n 个环境组成评测批次，观察 [N,2,6,5]，不是 n 次梯度更新。
    env = Catch(n, seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(5):
        if model is None:
            # [N]，每个游戏随机选择一个 0/1/2 动作。
            actions = rng.integers(0, 3, n)
        else:
            # QNet 返回 [N,3]；ActorCritic 返回 ([N,3],[N])。
            output = model(tensor(env.obs()))
            # 通用变量名 logits：DQN 情况下实际是 Q 值；都可按最大值选动作。
            logits = output[0] if isinstance(output, tuple) else output
            # 沿最后一维（动作列）取最大值索引：[N,3] -> [N]。
            actions = logits.argmax(-1).numpy()
        _, reward, _, _ = env.step(actions)
    # reward [N] -> 成功标记 bool [N] -> mean 标量 -> Python float。
    return float((reward > 0).mean())


def replay_catch(model, path, seed=123, games=4):
    """Render an actual policy rollout, not an illustrative trajectory."""
    from PIL import Image, ImageDraw
    env = Catch(1, seed)
    rng = np.random.default_rng(seed + 1)
    frames = []

    def frame(obs, label):
        im = Image.new('RGB', (300, 410), '#101b2c')
        d = ImageDraw.Draw(im)
        d.text((15, 12), label, fill='white')
        for y in range(6):
            for x in range(5):
                box = (x*54+15, y*54+45, x*54+64, y*54+94)
                d.rectangle(box, outline='#34445c')
                if obs[0, y, x]: d.ellipse(box, fill='#ffd166')
                if obs[1, y, x]: d.rectangle((box[0], box[3]-12, box[2], box[3]), fill='#40d6bd')
        frames.append(im)

    with torch.no_grad():
        for i in range(games):
            frame(env.obs()[0], f'Game {i+1} / {games}')
            for t in range(5):
                if model is None:
                    action = rng.integers(0, 3, 1)
                else:
                    output = model(tensor(env.obs()))
                    logits = output[0] if isinstance(output, tuple) else output
                    action = logits.argmax(-1).numpy()
                _, reward, done, final = env.step(action)
                label = f'Action: {int(action[0])}   Step: {t+1}'
                if done[0]: label += '   CAUGHT' if reward[0] > 0 else '   MISSED'
                frame(final[0], label)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0)


def save_run(out, args, metrics, history, model=None, started=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(config=vars(args), metrics=metrics, history=history,
                  runtime=dict(python=platform.python_version(), torch=torch.__version__,
                               platform=platform.platform(), device='cpu',
                               seconds=round(time.time()-started, 3) if started else None))
    if model is not None:
        record['parameters'] = sum(p.numel() for p in model.parameters())
        record['architecture'] = str(model)
        torch.save(model.state_dict(), out / 'model.pt')
    (out / 'results.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    if history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in history[0]:
            if key == 'step': continue
            ax.plot([h['step'] for h in history], [h[key] for h in history], label=key)
        ax.set_xlabel('Training step (see lesson for units)')
        ax.set_ylabel('Metric')
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / 'curve.png', dpi=150)
        plt.close(fig)
    print(json.dumps({'output': str(out), **metrics}, ensure_ascii=False), flush=True)
