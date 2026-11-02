"""Small CPU checks of learning semantics; no accelerator or game engine needed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from envs import GrayFrameStack
from policy_core import (VisualActorCritic,compute_gae,score_function_loss,
                         ppo_clipped_loss,optimize)
from rollouts import Collector
from runtime import recover_optimizer_steps


class ValueFromPixel(nn.Module):
    def forward(self,obs):
        return torch.zeros(len(obs),2),obs[:,0,0,0].float()


class ResettingVisual(gym.Env):
    def __init__(self):
        self.action_space=gym.spaces.Discrete(2)
        self.observation_space=gym.spaces.Box(0,255,(4,64,64),dtype=np.uint8)
    def reset(self,seed=None,options=None):return np.ones((4,64,64),np.uint8),{}
    def step(self,action):return np.full((4,64,64),5,np.uint8),1.,False,True,{}


class VariableEpisode(ResettingVisual):
    def __init__(self):super().__init__();self.episode=-1
    def reset(self,seed=None,options=None):
        self.episode+=1;self.t=0
        return np.zeros((4,64,64),np.uint8),{}
    def step(self,action):
        self.t+=1
        length=1 if self.episode==0 else 2
        reward=1. if self.episode==0 else float(self.t+1)
        return np.zeros((4,64,64),np.uint8),reward,self.t==length,False,{}


class RawRGB(gym.Env):
    def __init__(self):
        self.action_space=gym.spaces.Discrete(2)
        self.observation_space=gym.spaces.Box(0,255,(12,16,3),dtype=np.uint8)
    def reset(self,seed=None,options=None):self.t=0;return np.zeros((12,16,3),np.uint8),{}
    def step(self,action):
        self.t+=1
        return np.full((12,16,3),self.t,np.uint8),float(self.t),False,self.t==2,{}


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.logits=nn.Parameter(torch.zeros(2));self.value=nn.Parameter(torch.zeros(()))
    def forward(self,x):return self.logits[None].expand(len(x),-1),self.value.expand(len(x))


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_positive_and_negative_advantage_change_sampled_probability_in_opposite_directions(self):
        for advantage,direction in [(1.,1),(-1.,-1)]:
            logits=nn.Parameter(torch.zeros(2));optimizer=torch.optim.SGD([logits],lr=.1)
            logp=Categorical(logits=logits).log_prob(torch.tensor(0))
            loss=score_function_loss(logp[None],torch.tensor([advantage]))
            loss.backward();optimizer.step()
            self.assertGreater(direction*(float(logits.softmax(-1)[0].detach())-.5),0)

    def test_gae_uses_final_frame_at_truncation_and_cuts_recursion(self):
        rewards=torch.tensor([[1.],[3.]])
        values=torch.tensor([[2.],[99.]],requires_grad=True)
        following=torch.tensor([[5.],[1000.]])
        term=torch.tensor([[False],[True]]);trunc=torch.tensor([[True],[False]])
        advantage,returns=compute_gae(rewards,values,following,term,trunc,.9,.8)
        torch.testing.assert_close(advantage,torch.tensor([[3.5],[-96.]]))
        torch.testing.assert_close(returns,torch.tensor([[5.5],[3.]]))
        self.assertFalse(advantage.requires_grad);self.assertFalse(returns.requires_grad)

    def test_gae_window_edge_bootstraps_even_without_game_end(self):
        fields=[torch.tensor([[1.]]),torch.tensor([[2.]]),torch.tensor([[5.]]),
                torch.tensor([[False]]),torch.tensor([[False]])]
        advantage,returns=compute_gae(*fields,.9,.95)
        torch.testing.assert_close(advantage,torch.tensor([[3.5]]))
        torch.testing.assert_close(returns,torch.tensor([[5.5]]))

    def test_collector_bootstraps_old_final_observation_before_reset(self):
        collector=Collector([ResettingVisual()],seed=4)
        batch,raw=collector.window(ValueFromPixel(),'cpu',2,lambda:False,lambda row:None,.9,.8)
        np.testing.assert_array_equal(raw['next_values'],[[5.],[5.]])
        np.testing.assert_allclose(batch['returns'],[5.5,5.5])
        self.assertEqual(collector.completed_episodes,2)

    def test_reinforce_two_different_lengths_average_episode_sums(self):
        collector=Collector([VariableEpisode()],seed=1)
        batch,scope=collector.complete_episodes(ValueFromPixel(),'cpu',2,lambda:False,lambda row:None,gamma=1.)
        np.testing.assert_array_equal(batch['returns'],[1.,5.,3.])
        np.testing.assert_array_equal(batch['policy_weights'],[1.5,1.5,1.5])
        logits=nn.Parameter(torch.zeros(2))
        logp=Categorical(logits=logits).log_prob(torch.zeros(3,dtype=torch.long))
        weighted=torch.tensor(batch['returns']*batch['policy_weights'])
        score_function_loss(logp,weighted).backward()
        # -(sum G for episode 1 + sum G for episode 2)/2 * grad log p(a=0).
        self.assertAlmostEqual(float(logits.grad[0]),-2.25)
        self.assertEqual(scope['complete_episodes_in_batch'],2)

    def test_ppo_old_probabilities_and_advantages_are_constants(self):
        new=torch.tensor([np.log(.7)],requires_grad=True)
        old=torch.tensor([np.log(.5)],requires_grad=True)
        advantage=torch.tensor([1.],requires_grad=True)
        loss=ppo_clipped_loss(new,old,advantage,.2)
        self.assertAlmostEqual(float(loss.detach()),-1.2,places=6)
        loss.backward()
        self.assertIsNone(old.grad);self.assertIsNone(advantage.grad)
        self.assertAlmostEqual(float(new.grad),0.)  # positive advantage at clipped plateau
        negative=ppo_clipped_loss(torch.tensor([np.log(.7)]),torch.tensor([np.log(.5)]),torch.tensor([-1.]),.2)
        self.assertAlmostEqual(float(negative),1.4,places=6)

    def test_ppo_multiple_epochs_do_not_overwrite_behavior_logp(self):
        model=TinyPolicy();original=model.logits.detach().clone()
        batch=dict(states=np.ones((4,1),np.uint8),actions=np.zeros(4,np.int64),
            old_logp=np.full(4,np.log(.5),np.float32),advantages=np.ones(4,np.float32),returns=np.zeros(4,np.float32))
        snapshot=batch['old_logp'].copy()
        optimize(model,torch.optim.SGD(model.parameters(),lr=.05),batch,'ppo','cpu',
            epochs=3,minibatch_size=2,value_coef=0,entropy_coef=0,normalize_advantages=False)
        np.testing.assert_array_equal(batch['old_logp'],snapshot)
        self.assertFalse(torch.equal(model.logits.detach(),original))

    def test_count_actual_optimizer_steps_separately_from_collected_batches(self):
        for algorithm,expected in [('ppo',9),('ac',1),('reinforce',1)]:
            model=TinyPolicy();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            calls=[];original_step=optimizer.step
            def counted_step(*args,**kwargs):
                calls.append(1)
                return original_step(*args,**kwargs)
            optimizer.step=counted_step
            batch=dict(states=np.zeros((5,1),np.uint8),actions=np.zeros(5,np.int64),
                old_logp=np.full(5,np.log(.5),np.float32),advantages=np.ones(5,np.float32),
                returns=np.ones(5,np.float32))
            result=optimize(model,optimizer,batch,algorithm,'cpu',epochs=3,minibatch_size=2,
                            normalize_advantages=False)
            self.assertEqual(len(calls),expected)
            self.assertEqual(result['optimizer_steps'],expected)

    def test_legacy_step_count_requires_log_config_and_matching_adam_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'updates.jsonl'
            path.write_text('\n'.join(json.dumps(dict(update=i,batch_size=b))
                                      for i,b in [(1,5),(2,4)]))
            saved=dict(updates=2,config=dict(algo='ppo',epochs=3,minibatch_size=2),
                       optimizer={'state':{0:{'step':torch.tensor(15.)}}})
            value,source=recover_optimizer_steps(saved,path)
            self.assertEqual(value,15)  # 3*ceil(5/2) + 3*ceil(4/2)
            self.assertIn('cross-checked',source)
            saved['optimizer']['state'][0]['step']=torch.tensor(14.)
            self.assertIsNone(recover_optimizer_steps(saved,path)[0])
            path.unlink()
            self.assertIsNone(recover_optimizer_steps(saved,path)[0])
            self.assertEqual(recover_optimizer_steps({'optimizer_steps':21},path)[0],21)

    def test_cnn_shapes_and_rgb_frame_repeat_boundary(self):
        model=VisualActorCritic(actions=5,channels=4,size=64)
        logits,values=model(torch.zeros(2,4,64,64,dtype=torch.uint8))
        self.assertEqual(tuple(logits.shape),(2,5));self.assertEqual(tuple(values.shape),(2,))
        env=GrayFrameStack(RawRGB(),64,4,action_repeat=3)
        obs,_=env.reset();self.assertEqual(obs.dtype,np.uint8)
        obs,reward,term,trunc,info=env.step(0)
        self.assertEqual(reward,3.);self.assertFalse(term);self.assertTrue(trunc)
        self.assertEqual(info['policy_lab_raw_steps'],2)
        self.assertTrue(np.all(obs[:-1]==0));self.assertTrue(np.all(obs[-1]==2))
        env.close()


if __name__=='__main__':unittest.main()
