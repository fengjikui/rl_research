"""Bounded Atari DQN throughput probe; timing evidence, not learning evidence.

Run under grpo_lab.guard. Each device gets an independent, short serial run.
Includes ALE stepping, replay sampling, transfers, inference and B=32 updates.
The small replay pool and short duration do not model a full training run.
"""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from breakout import NatureQNet, make_env


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def probe(device, seconds, batch_size):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = make_env()
    observation, _ = env.reset(seed=0)
    online = NatureQNet(env.action_space.n).to(device)
    target = deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    replay = deque(maxlen=512)

    def collect():
        nonlocal observation
        if random.random() < 0.5:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(online(torch.as_tensor(
                    np.asarray(observation), device=device)[None]).argmax(-1).item())
        following, reward, terminated, truncated, _ = env.step(action)
        replay.append((np.asarray(observation).copy(), action,
                       float(np.sign(reward)), np.asarray(following).copy(), terminated))
        observation = following
        if terminated or truncated:
            observation, _ = env.reset()

    def train():
        states, actions, rewards, following, done = map(
            np.asarray, zip(*random.sample(replay, batch_size)))
        states = torch.as_tensor(states, device=device)
        following = torch.as_tensor(following, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.float32, device=device)
        prediction = online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            label = rewards + 0.99 * (1 - done) * target(following).max(1).values
        loss = F.smooth_l1_loss(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach().item())
        if not np.isfinite(value):
            raise RuntimeError('Non-finite benchmark loss')
        return value

    try:
        # Exclude initialization and first-device kernel warmup from timed interval.
        for _ in range(128):
            collect()
        for _ in range(8):
            train()
        synchronize(device)
        started = time.perf_counter()
        steps = updates = 0
        losses = []
        while time.perf_counter() - started < seconds:
            collect()
            steps += 1
            if steps % 4 == 0:
                losses.append(train())
                updates += 1
            if steps % 1000 == 0:
                target.load_state_dict(online.state_dict())
        synchronize(device)
        elapsed = time.perf_counter() - started
        rate = steps / elapsed
        return dict(device=device, elapsed_seconds=elapsed, environment_steps=steps,
                    gradient_updates=updates, steps_per_second=rate,
                    raw_frames_per_second_approximately=4 * rate,
                    hours_for_1m_steps=1e6 / rate / 3600,
                    hours_for_10m_steps=1e7 / rate / 3600,
                    days_for_50m_steps=5e7 / rate / 86400,
                    final_loss=losses[-1] if losses else None)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    results = []
    for device in ['cpu', 'mps']:
        if device == 'mps' and not torch.backends.mps.is_available():
            continue
        print('BENCHMARK', device, flush=True)
        result = probe(device, args.seconds, args.batch_size)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        scope='Short throughput probe, not evidence of task improvement; '
              'long-run estimates exclude evaluation, checkpointing and large-replay costs.',
        batch_size=args.batch_size, replay_capacity=512, update_every=4,
        epsilon=0.5, torch=torch.__version__, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
