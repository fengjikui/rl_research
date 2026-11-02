"""Disk-backed PER; reconstruct four-frame observations and n-step targets.

Stores two uint8 frames per transition instead of two full frame stacks.
True termination zeroes the bootstrap. Truncation cuts the sampled trajectory
while retaining the actual final observation as a bootstrap state.
"""
import json
from pathlib import Path
import random

import numpy as np


class SumTree:
    def __init__(self,capacity):
        self.size=1
        while self.size<capacity:self.size*=2
        self.values=np.zeros(2*self.size,dtype=np.float64)
        self.active=0

    @property
    def total(self):return float(self.values[1])

    def set(self,index,value):
        node=index+self.size
        old=self.values[node]
        self.active+=int(value>0)-int(old>0)
        self.values[node]=value
        node//=2
        while node:
            self.values[node]=self.values[2*node]+self.values[2*node+1]
            node//=2

    def find(self,mass):
        node=1
        mass=min(mass,np.nextafter(self.total,0.))
        while node<self.size:
            left=2*node
            if mass<self.values[left]:node=left
            else:mass-=self.values[left];node=left+1
        return node-self.size


class RainbowReplay:
    def __init__(self,directory,capacity=100000,n_step=3,gamma=.99,alpha=.5,resume=False):
        if capacity<n_step+5:raise ValueError('Capacity too small for frame history and n-step')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.capacity,self.n_step,self.gamma,self.alpha=capacity,n_step,gamma,alpha
        self.count,self.start,self.max_priority=0,0,1.
        specs=dict(frames=(np.uint8,(capacity,84,84)),following=(np.uint8,(capacity,84,84)),
            actions=(np.int64,(capacity,)),rewards=(np.float32,(capacity,)),
            terminated=(np.bool_,(capacity,)),boundaries=(np.bool_,(capacity,)),
            starts=(np.int64,(capacity,)),priorities=(np.float32,(capacity,)))
        for name,(dtype,shape) in specs.items():
            array=np.lib.format.open_memmap(self.directory/f'{name}.npy',
                mode='r+' if resume else 'w+',dtype=dtype,shape=shape)
            if array.shape!=shape:raise ValueError(f'Replay shape mismatch: {name}')
            setattr(self,name,array)
        self.tree=SumTree(capacity)
        if resume:
            metadata=json.loads((self.directory/'metadata.json').read_text())
            for key in ('capacity','n_step','gamma','alpha'):
                if metadata[key]!=getattr(self,key):raise ValueError(f'Replay configuration mismatch: {key}')
            self.count=metadata['count'];self.start=metadata['start'];self.max_priority=metadata['max_priority']
            # Only scalar priorities are restored; ~O(capacity) tree reconstruction.
            oldest=max(0,self.count-capacity)
            for index in range(oldest,self.count):
                if self._valid(index):self.tree.set(index%capacity,float(self.priorities[index%capacity])**alpha)

    def __len__(self):return self.tree.active

    def _valid(self,index):
        oldest=max(0,self.count-self.capacity)
        if not (oldest<=index<self.count):return False
        # The oldest three retained transitions may depend on overwritten frames.
        if oldest and index<oldest+3:return False
        for offset in range(self.n_step):
            i=index+offset
            if i>=self.count:return False
            if self.boundaries[i%self.capacity]:return True
        return True

    def _activate_recent(self):
        for index in range(max(0,self.count-self.n_step),self.count):
            slot=index%self.capacity
            self.tree.set(slot,float(self.priorities[slot])**self.alpha if self._valid(index) else 0.)

    def append(self,state,action,reward,following,terminated,truncated,new_episode):
        state,following=np.asarray(state),np.asarray(following)
        if new_episode:
            if not np.all(state==state[-1]):raise ValueError('Reset stack must be padded with its initial frame')
            if self.count and not self.boundaries[(self.count-1)%self.capacity]:
                # Resuming resets the emulator: mark this collection boundary,
                # bootstrap from the saved last next-state, never join two games.
                self.boundaries[(self.count-1)%self.capacity]=True
                self._activate_recent()
            self.start=self.count
        slot=self.count%self.capacity
        self.tree.set(slot,0.)
        self.frames[slot],self.following[slot]=state[-1],following[-1]
        self.actions[slot],self.rewards[slot]=action,reward
        self.terminated[slot]=terminated;self.boundaries[slot]=terminated or truncated
        self.starts[slot]=self.start;self.priorities[slot]=self.max_priority
        self.count+=1
        oldest=max(0,self.count-self.capacity)
        if oldest:
            for index in range(oldest,oldest+3):self.tree.set(index%self.capacity,0.)
        self._activate_recent()

    def _state(self,index):
        start=int(self.starts[index%self.capacity])
        history=np.maximum(np.arange(index-3,index+1),start)
        return np.asarray(self.frames[history%self.capacity])

    def transition(self,index):
        if not self._valid(index):raise ValueError(f'Incomplete or overwritten transition: {index}')
        total=0.;discount=1.;last=index
        for offset in range(self.n_step):
            last=index+offset;slot=last%self.capacity
            total+=discount*float(self.rewards[slot]);discount*=self.gamma
            if self.boundaries[slot]:break
        if self.terminated[last%self.capacity]:discount=0.
        state_last=self._state(last)
        following=np.concatenate([state_last[1:],self.following[last%self.capacity][None]],axis=0)
        return self._state(index),int(self.actions[index%self.capacity]),total,following,discount

    def sample(self,batch_size,beta):
        if not len(self):raise ValueError('No complete transitions available')
        masses=(np.arange(batch_size)+np.random.random(batch_size))*self.tree.total/batch_size
        slots=np.array([self.tree.find(float(mass)) for mass in masses])
        oldest=max(0,self.count-self.capacity)
        # Convert a circular slot to its current absolute transition index.
        indices=oldest+(slots-oldest)%self.capacity
        probabilities=self.tree.values[self.tree.size+slots]/self.tree.total
        weights=(len(self)*probabilities)**(-beta)
        weights/=weights.max()
        transitions=[self.transition(int(index)) for index in indices]
        states,actions,returns,next_states,discounts=map(np.asarray,zip(*transitions))
        return dict(states=states,actions=actions,returns=returns.astype(np.float32),
            next_states=next_states,discounts=discounts.astype(np.float32),indices=indices,
            probabilities=probabilities,weights=weights.astype(np.float32))

    def update_priorities(self,indices,priorities):
        for index,priority in zip(indices,priorities):
            priority=max(float(priority),1e-6)
            if not np.isfinite(priority):raise ValueError('Non-finite PER priority')
            slot=int(index)%self.capacity
            self.priorities[slot]=priority;self.max_priority=max(self.max_priority,priority)
            self.tree.set(slot,priority**self.alpha if self._valid(int(index)) else 0.)

    def flush(self):
        for name in ('frames','following','actions','rewards','terminated','boundaries','starts','priorities'):
            getattr(self,name).flush()
        value=dict(capacity=self.capacity,n_step=self.n_step,gamma=self.gamma,alpha=self.alpha,
            count=self.count,start=self.start,max_priority=self.max_priority)
        temporary=self.directory/'metadata.json.tmp';temporary.write_text(json.dumps(value)+'\n')
        temporary.replace(self.directory/'metadata.json')
