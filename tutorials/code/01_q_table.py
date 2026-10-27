"""Q-learning without a neural network: a six-state corridor."""
import argparse
import time
import numpy as np
from common import setup, save_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='tutorials/runs/q_table')
    args = p.parse_args()
    setup(args.seed)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    q = np.zeros((6, 2))
    history = []
    for episode in range(args.episodes):
        state = 0
        for _ in range(40):
            action = int(rng.integers(2)) if rng.random() < .25 else int(q[state].argmax())
            nxt = int(np.clip(state + (1 if action else -1), 0, 5))
            terminated = nxt == 5
            reward = 1. if terminated else -.02
            target = reward + .95 * (not terminated) * q[nxt].max()
            q[state, action] += .2 * (target - q[state, action])
            state = nxt
            if terminated: break
        # The 40-step collector limit is NOT treated as an absorbing state.
        if episode % 50 == 0:
            history.append(dict(step=episode, start_value=float(q[0].max())))
    state, path = 0, [0]
    for _ in range(20):
        state = int(np.clip(state + (1 if q[state].argmax() else -1), 0, 5))
        path.append(state)
        if state == 5: break
    save_run(args.out, args, {'q_table': q.tolist(), 'greedy_path': path,
                             'reached_goal': state == 5}, history, started=started)


if __name__ == '__main__': main()
