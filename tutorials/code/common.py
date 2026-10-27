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
    return torch.as_tensor(x, dtype=torch.float32)


class Catch:
    """Vectorized 6x5 catch game. Actions: left=0, stay=1, right=2.

    Ball falls one row each step. Each episode ends after exactly five actions.
    step auto-resets finished slots; terminal observation is returned separately.
    This is a finite game termination, not an artificial time-limit truncation.
    """
    def __init__(self, n=1, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.ball = self.rng.integers(0, 5, n)
        self.player = self.rng.integers(0, 5, n)
        self.row = np.zeros(n, dtype=int)

    def obs(self):
        x = np.zeros((self.n, 2, 6, 5), dtype=np.float32)
        x[np.arange(self.n), 0, self.row, self.ball] = 1
        x[np.arange(self.n), 1, 5, self.player] = 1
        return x

    def step(self, actions):
        actions = np.asarray(actions, dtype=int).reshape(self.n)
        if not np.isin(actions, [0, 1, 2]).all():
            raise ValueError('Catch actions must be 0, 1, or 2')
        self.player = np.clip(self.player + actions - 1, 0, 4)
        self.row += 1
        done = self.row == 5
        reward = np.where(done, np.where(self.player == self.ball, 1., -1.), 0.)
        final_obs = self.obs()
        count = int(done.sum())
        self.ball[done] = self.rng.integers(0, 5, count)
        self.player[done] = self.rng.integers(0, 5, count)
        self.row[done] = 0
        return self.obs(), reward.astype(np.float32), done, final_obs


class Backbone(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        if kind == 'mlp':
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(60, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh())
        else:
            self.net = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(480, 64), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        self.features = Backbone(kind)
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        return self.head(self.features(x))


class ActorCritic(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        self.features = Backbone(kind)
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.features(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@torch.no_grad()
def evaluate_catch(model=None, seed=10000, n=500):
    env = Catch(n, seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(5):
        if model is None:
            actions = rng.integers(0, 3, n)
        else:
            output = model(tensor(env.obs()))
            logits = output[0] if isinstance(output, tuple) else output
            actions = logits.argmax(-1).numpy()
        _, reward, _, _ = env.step(actions)
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
