"""Frame-based, disk-backed replay; reconstruct four-frame states when sampling."""
import json
from pathlib import Path
import random

import numpy as np


class FrameReplay:
    def __init__(self, directory, capacity=100000, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.count = 0
        self.start = 0
        # 每条只存当前/下一观察的最后一帧，各 [84,84] uint8。
        # 100,000 条约 1.41 GB；直接存两份四帧栈则约 5.64 GB。
        specs = {'frames': (np.uint8, (capacity,84,84)),
                 'following': (np.uint8, (capacity,84,84)),
                 'actions': (np.int64, (capacity,)),
                 'rewards': (np.float32, (capacity,)),
                 'terminated': (np.bool_, (capacity,)),
                 'starts': (np.int64, (capacity,))}
        for name, (dtype, shape) in specs.items():
            array = np.lib.format.open_memmap(self.directory/f'{name}.npy',
                mode='r+' if resume else 'w+', dtype=dtype, shape=shape)
            if array.shape != shape:
                raise ValueError(f'Replay shape differs: {name}')
            setattr(self, name, array)
        if resume:
            state = json.loads((self.directory/'metadata.json').read_text())
            if state['capacity'] != capacity:
                raise ValueError('Replay capacity differs from saved run')
            self.count = state['count']

    def __len__(self):
        return min(self.count, self.capacity)

    def append(self, state, action, reward, following, terminated, new_episode):
        # FrameStackObservation 在 reset 时重复同一帧，不能把上一局接到新一局。
        state, following = np.asarray(state), np.asarray(following)
        if new_episode:
            if not np.all(state == state[-1]):
                raise ValueError('Expected reset-padded four-frame stack')
            self.start = self.count
        i = self.count % self.capacity
        self.frames[i], self.following[i] = state[-1], following[-1]
        self.actions[i], self.rewards[i] = action, reward
        self.terminated[i], self.starts[i] = terminated, self.start
        self.count += 1

    def sample(self, batch_size):
        # 排除环形缓冲区最旧的三条，保证重建时需要的历史帧尚未被覆盖。
        oldest = max(0, self.count-self.capacity)
        low = oldest+3 if oldest else 0
        indices = np.array(random.sample(range(low,self.count),batch_size))
        slots = indices % self.capacity
        history = np.maximum(indices[:,None] + np.arange(-3,1), self.starts[slots,None])
        states = np.asarray(self.frames[history % self.capacity])  # [B,4,84,84]
        next_states = np.empty_like(states)
        next_states[:,:3] = states[:,1:]
        next_states[:,3] = self.following[slots]
        # 与原 update 接口一致：B 条 (s,a,r,s',terminated)，无需改变 DQN 公式。
        return list(zip(states, self.actions[slots], self.rewards[slots],
                        next_states, self.terminated[slots]))

    def flush(self):
        for name in ('frames','following','actions','rewards','terminated','starts'):
            getattr(self,name).flush()
        state = dict(capacity=self.capacity,count=self.count)
        temporary = self.directory/'metadata.json.tmp'
        temporary.write_text(json.dumps(state)+'\n')
        temporary.replace(self.directory/'metadata.json')
