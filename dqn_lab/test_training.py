"""Checks for frame history boundaries and the CPU/MPS training interface."""
from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from atari_replay import FrameReplay
from dqn_core import update


class TrainingTests(unittest.TestCase):
    def test_replay_matches_original_stacks_across_resets_and_wraparound(self):
        random.seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            replay = FrameReplay(tmp,19)
            reference = {}
            index = 0
            for length in [1,2,8,3,25,2,14]:
                state = np.full((4,84,84),index,dtype=np.uint8)
                for t in range(length):
                    following = np.concatenate([state[1:],np.full((1,84,84),index+1,dtype=np.uint8)])
                    reference[index] = (state.copy(),following.copy(),t==length-1)
                    replay.append(state,index,float(index),following,t==length-1,t==0)
                    index += 1
                    state = following
                    if len(replay)>=10:
                        for s,a,r,nxt,done in replay.sample(7):
                            expected = reference[int(a)]
                            np.testing.assert_array_equal(s,expected[0])
                            np.testing.assert_array_equal(nxt,expected[1])
                            self.assertEqual(bool(done),expected[2])
                            self.assertEqual(float(r),int(a))
            replay.flush()
            restored = FrameReplay(tmp,19,resume=True)
            self.assertEqual(restored.count,index)
            for s,a,r,nxt,done in restored.sample(7):
                np.testing.assert_array_equal(s,reference[int(a)][0])
                np.testing.assert_array_equal(nxt,reference[int(a)][1])

    def test_cpu_and_mps_keep_labels_constant_and_audit_inputs_intact(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        original = nn.Sequential(nn.Flatten(),nn.Linear(60,3))
        rng = np.random.default_rng(8)
        transitions = [(rng.normal(size=(2,6,5)).astype('float32'),i%3,float(i),
                        rng.normal(size=(2,6,5)).astype('float32'),i==2) for i in range(4)]
        losses=[]
        devices = ['cpu']+(['mps'] if os.environ.get('RL_TEST_CPU_ONLY')!='1'
                           and torch.backends.mps.is_available() else [])
        with tempfile.TemporaryDirectory() as tmp:
            for device in devices:
                online = deepcopy(original).to(device)
                target = deepcopy(online).requires_grad_(False)
                optimizer = torch.optim.Adam(online.parameters(),lr=1e-3)
                losses.append(update(online,target,optimizer,transitions,.99,Path(tmp)/device))
                audit = json.loads((Path(tmp)/device/'loss_batch.json').read_text())
                self.assertFalse(audit['label_requires_grad'])
                self.assertEqual(audit['rows'][2]['regression_label'],2.0)
                for row in audit['rows']:
                    expected=row['reward']+.99*(1-row['terminated'])*row['next_max_q']
                    self.assertAlmostEqual(row['regression_label'],expected,places=5)
                with np.load(Path(tmp)/device/'batch_inputs.npz') as arrays:
                    np.testing.assert_array_equal(arrays['states'][0],transitions[0][0])
                self.assertTrue(all(p.grad is None for p in target.parameters()))
                self.assertTrue(any(not torch.equal(p.detach().cpu(),q)
                                    for p,q in zip(online.parameters(),original.parameters())))
        if len(losses)==2:
            self.assertAlmostEqual(losses[0],losses[1],places=5)


if __name__=='__main__':
    unittest.main()
