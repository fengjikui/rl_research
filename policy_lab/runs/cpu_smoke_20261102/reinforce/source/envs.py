"""Existing Gymnasium visual games -> explicit discrete actions and frame stacks."""
from collections import deque
import importlib
import json

import gymnasium as gym
import numpy as np
from PIL import Image


class GrayFrameStack(gym.Wrapper):
    def __init__(self,env,size=64,stack=4,action_repeat=1):
        super().__init__(env)
        shape=getattr(env.observation_space,'shape',None)
        if shape is None or len(shape)!=3 or shape[-1] not in (1,3,4):
            raise ValueError(f'Expected HWC visual observation, got {env.observation_space}')
        if env.observation_space.dtype!=np.uint8:raise ValueError('Expected uint8 images, no implicit scaling assumptions')
        if not isinstance(env.action_space,gym.spaces.Discrete):
            raise ValueError(f'Expected Discrete actions, got {env.action_space}')
        if min(size,stack,action_repeat)<1:raise ValueError('Image/stack/repeat must be positive')
        self.size,self.stack,self.action_repeat=size,stack,action_repeat
        self.frames=deque(maxlen=stack)
        self.observation_space=gym.spaces.Box(0,255,(stack,size,size),dtype=np.uint8)

    def _gray(self,observation):
        image=np.asarray(observation)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1] not in (1,3,4):
            raise ValueError(f'Invalid actual HWC observation: {image.shape}, {image.dtype}')
        source=Image.fromarray(image[:,:,0]) if image.shape[-1]==1 else Image.fromarray(image[:,:,:3]).convert('L')
        return np.asarray(source.resize((self.size,self.size),Image.Resampling.BILINEAR)).copy()

    def reset(self,**kwargs):
        observation,info=self.env.reset(**kwargs)
        frame=self._gray(observation);self.frames.clear()
        for _ in range(self.stack):self.frames.append(frame.copy())
        return np.stack(self.frames),info

    def step(self,action):
        reward=0.
        for repeats in range(1,self.action_repeat+1):
            observation,value,terminated,truncated,info=self.env.step(int(action));reward+=float(value)
            if terminated or truncated:break
        self.frames.append(self._gray(observation))
        info=dict(info);info['policy_lab_raw_steps']=repeats
        # The actual terminal/truncated observation is returned before reset.
        return np.stack(self.frames),reward,bool(terminated),bool(truncated),info


def make_visual_env(env_id,env_kwargs=None,size=64,stack=4,action_repeat=1,max_episode_steps=None):
    if env_id.startswith('Craftium/'):
        try:importlib.import_module('craftium')
        except ImportError as error:raise RuntimeError('Craftium is not installed; see policy_lab/README.md') from error
    kwargs=dict(env_kwargs or {})
    if 'render_mode' in kwargs and kwargs['render_mode']!='rgb_array':raise ValueError('Use rgb_array for reproducible video')
    kwargs['render_mode']='rgb_array'
    if env_id.startswith('CarRacing'):
        if kwargs.get('continuous',False):raise ValueError('This script requires CarRacing continuous=False')
        kwargs['continuous']=False
    if max_episode_steps is not None:kwargs['max_episode_steps']=max_episode_steps
    env=gym.make(env_id,**kwargs)
    try:return GrayFrameStack(env,size,stack,action_repeat)
    except BaseException:
        env.close();raise
