"""Meaningful CPU checks for distribution projection, Double-Q and replay edges."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from rainbow_core import NoisyLinear, categorical_projection, update
from rainbow_replay import RainbowReplay, SumTree


class TinyDistribution(nn.Module):
    def __init__(self,probabilities):
        super().__init__()
        self.parameter=nn.Parameter(torch.tensor(probabilities).log())
        self.register_buffer('support',torch.tensor([-1.,0.,1.]))
    def reset_noise(self):pass
    def logits(self,x):return self.parameter[None].expand(len(x),-1,-1)
    def distribution(self,x):return self.logits(x).softmax(-1)
    def forward(self,x):return (self.distribution(x)*self.support).sum(-1)


class RainbowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_projection_conserves_mass_at_integer_bins_and_clipped_endpoints(self):
        support=torch.tensor([-1.,0.,1.])
        probabilities=torch.tensor([[.2,.3,.5]]*5)
        projected=categorical_projection(probabilities,torch.tensor([0.,0.,8.,-8.,.5]),
            torch.tensor([1.,0.,1.,1.,0.]),support)
        expected=torch.tensor([[.2,.3,.5],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.],[0.,.5,.5]])
        torch.testing.assert_close(projected,expected)
        torch.testing.assert_close(projected.sum(-1),torch.ones(5))
        self.assertFalse(projected.requires_grad)

    def test_noisy_eval_is_deterministic_training_is_not(self):
        torch.manual_seed(3);layer=NoisyLinear(4,3);x=torch.ones(2,4)
        a=layer(x);layer.reset_noise();b=layer(x)
        self.assertFalse(torch.equal(a,b))
        layer.eval();a=layer(x);layer.reset_noise();b=layer(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        layer.train();layer(x).sum().backward()
        self.assertIsNotNone(layer.weight_sigma.grad)

    def test_tree_prefix_sampling_and_zero_priority(self):
        tree=SumTree(5)
        for i,value in enumerate([1.,2.,0.,4.,1.]):tree.set(i,value)
        self.assertEqual(tree.active,4);self.assertEqual(tree.total,8.)
        self.assertEqual([tree.find(v) for v in [.5,1.,2.9,3.,6.9,7.9]],[0,1,1,3,3,4])
        tree.set(1,0.);self.assertEqual(tree.active,3)
        self.assertEqual(tree.total,6.)

    def test_n_step_replay_matches_full_stacks_through_terminal_truncation_and_wrap(self):
        np.random.seed(8)
        with tempfile.TemporaryDirectory() as tmp:
            replay=RainbowReplay(tmp,19,n_step=3,gamma=.9)
            reference={};absolute=0
            for length,terminal in [(1,True),(2,False),(8,True),(3,False),(25,True),(2,True),(14,False)]:
                state=np.full((4,84,84),absolute,dtype=np.uint8)
                for t in range(length):
                    following=np.concatenate([state[1:],np.full((1,84,84),absolute+1,dtype=np.uint8)])
                    ended=t==length-1
                    replay.append(state,absolute,float(t+1),following,ended and terminal,ended and not terminal,t==0)
                    reference[absolute]=(state.copy(),following.copy(),float(t+1),ended,ended and terminal)
                    absolute+=1;state=following
                    if len(replay):
                        batch=replay.sample(7,.7)
                        for k,index in enumerate(batch['indices']):
                            index=int(index);reward=0.;discount=1.
                            for offset in range(3):
                                row=reference[index+offset];reward+=discount*row[2];discount*=.9
                                if row[3]:break
                            if row[4]:discount=0.
                            np.testing.assert_array_equal(batch['states'][k],reference[index][0])
                            np.testing.assert_array_equal(batch['next_states'][k],row[1])
                            self.assertAlmostEqual(float(batch['returns'][k]),reward,places=5)
                            self.assertAlmostEqual(float(batch['discounts'][k]),discount,places=5)
                            self.assertEqual(int(batch['actions'][k]),index)
                        self.assertTrue(np.all(batch['weights']>0));self.assertLessEqual(batch['weights'].max(),1.)
                        replay.update_priorities(batch['indices'],np.arange(7)+.1)
            replay.flush();restored=RainbowReplay(tmp,19,n_step=3,gamma=.9,resume=True)
            self.assertEqual(restored.count,replay.count)
            np.testing.assert_allclose(restored.tree.values,replay.tree.values,atol=1e-5)

    def test_resume_boundary_bootstraps_from_saved_last_state_not_new_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay=RainbowReplay(tmp,15,n_step=3,gamma=.9)
            state=np.zeros((4,84,84),dtype=np.uint8)
            for i in range(2):
                following=np.concatenate([state[1:],np.full((1,84,84),i+1,dtype=np.uint8)])
                replay.append(state,0,1.,following,False,False,i==0);state=following
            self.assertEqual(len(replay),0);replay.flush()
            restored=RainbowReplay(tmp,15,n_step=3,gamma=.9,resume=True)
            new=np.full((4,84,84),99,dtype=np.uint8)
            restored.append(new,1,9.,new,False,False,True)
            s,a,r,nxt,d=restored.transition(0)
            self.assertAlmostEqual(r,1.9);self.assertAlmostEqual(d,.81)
            np.testing.assert_array_equal(nxt,state)
            self.assertEqual(len(restored),2)

    def test_double_q_uses_online_action_and_importance_weighted_distribution_loss(self):
        online=TinyDistribution([[.1,.1,.8],[.1,.8,.1]])
        target=TinyDistribution([[.8,.1,.1],[.1,.1,.8]]).requires_grad_(False)
        original=online.parameter.detach().clone()
        batch=dict(states=np.zeros((2,1),np.float32),next_states=np.zeros((2,1),np.float32),
            actions=np.array([0,1]),returns=np.zeros(2,np.float32),discounts=np.ones(2,np.float32),
            weights=np.array([.5,1.],np.float32),indices=np.array([0,1]))
        class Replay:
            def update_priorities(self,indices,values):self.values=values
        replay=Replay()
        with tempfile.TemporaryDirectory() as tmp:
            result=update(online,target,torch.optim.SGD(online.parameters(),lr=.01),batch,replay,Path(tmp))
            with np.load(Path(tmp)/'distributional_labels.npz') as data:
                np.testing.assert_array_equal(data['next_actions'],[0,0])
                np.testing.assert_allclose(data['projected_distribution'],[[.8,.1,.1]]*2,atol=1e-6)
            audit=json.loads((Path(tmp)/'loss_batch.json').read_text())
            expected=np.mean(np.array(audit['per_item_cross_entropy'])*batch['weights'])
            self.assertAlmostEqual(result,expected,places=6)
            self.assertFalse(audit['target_requires_grad'])
            entropy_term=np.sum(np.array([.8,.1,.1])*np.log([.8,.1,.1]))
            expected_priorities=np.maximum(np.array(audit['per_item_cross_entropy'])+entropy_term,0)+1e-6
            np.testing.assert_allclose(replay.values,expected_priorities,atol=1e-6)
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertFalse(torch.equal(online.parameter.detach(),original))
        self.assertTrue(np.all(replay.values>0))


if __name__=='__main__':unittest.main()
