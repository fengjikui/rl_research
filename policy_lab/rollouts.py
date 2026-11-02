"""On-policy data collection: real observations/actions, never replayed old data."""
import time

import numpy as np
import torch
from torch.distributions import Categorical

from policy_core import compute_gae,monte_carlo_returns


class Collector:
    def __init__(self,envs,seed):
        self.envs=envs
        self.observations=[env.reset(seed=seed+i)[0] for i,env in enumerate(envs)]
        self.scores=np.zeros(len(envs));self.lengths=np.zeros(len(envs),dtype=np.int64)
        self.decisions=0;self.raw_steps=0;self.completed_episodes=0

    def _step(self,actions,episode_log):
        following=[];rewards=[];terminated=[];truncated=[]
        for i,(env,action) in enumerate(zip(self.envs,actions)):
            obs,reward,term,trunc,info=env.step(int(action))
            following.append(obs);rewards.append(reward);terminated.append(term);truncated.append(trunc)
            self.decisions+=1;self.raw_steps+=int(info.get('policy_lab_raw_steps',1))
            self.scores[i]+=reward;self.lengths[i]+=1
            if term or trunc:
                self.completed_episodes+=1
                episode_log(dict(episode=self.completed_episodes,env_index=i,step=self.decisions,
                    score=float(self.scores[i]),episode_steps=int(self.lengths[i]),
                    terminated=bool(term),truncated=bool(trunc)))
                self.scores[i]=0.;self.lengths[i]=0
        return np.stack(following),np.asarray(rewards,np.float32),np.asarray(terminated,bool),np.asarray(truncated,bool)

    def _reset_finished(self,following,terminated,truncated):
        self.observations=[env.reset()[0] if term or trunc else following[i]
            for i,(env,term,trunc) in enumerate(zip(self.envs,terminated,truncated))]

    @torch.no_grad()
    def window(self,model,device,horizon,should_stop,episode_log,gamma=.99,lam=.95):
        records={name:[] for name in ('states','actions','rewards','values','next_values','terminated','truncated','old_logp')}
        for _ in range(horizon):
            if should_stop():break
            states=np.stack(self.observations)  # [N,C,H,W], current live game states
            logits,values=model(torch.as_tensor(states,device=device))
            distribution=Categorical(logits=logits);actions=distribution.sample()
            following,rewards,terminated,truncated=self._step(actions.cpu().numpy(),episode_log)
            # This is deliberately BEFORE reset. Truncations bootstrap the final
            # frame of the old episode, not the opening frame of the new one.
            _,next_values=model(torch.as_tensor(following,device=device))
            values_by_field=dict(states=states,actions=actions.cpu().numpy(),rewards=rewards,
                values=values.cpu().numpy(),next_values=next_values.cpu().numpy(),
                terminated=terminated,truncated=truncated,old_logp=distribution.log_prob(actions).cpu().numpy())
            for key,value in values_by_field.items():records[key].append(value.copy())
            self._reset_finished(following,terminated,truncated)
        if not records['states']:return None
        arrays={key:np.stack(value) for key,value in records.items()}  # [T,N,...]
        advantages,returns=compute_gae(*[torch.as_tensor(arrays[key]) for key in
            ('rewards','values','next_values','terminated','truncated')],gamma,lam)
        batch={key:arrays[key].reshape((-1,*arrays[key].shape[2:])) for key in ('states','actions','old_logp')}
        batch.update(advantages=advantages.numpy().reshape(-1),returns=returns.numpy().reshape(-1))
        return batch,arrays

    @torch.no_grad()
    def complete_episodes(self,model,device,count,should_stop,episode_log,gamma=.99):
        if len(self.envs)!=1:raise ValueError('REINFORCE deliberately uses a single environment')
        completed=[]
        for _ in range(count):
            records={key:[] for key in ('states','actions','rewards','old_logp')}
            finished=False
            while not should_stop():
                state=np.asarray(self.observations[0])
                logits,_=model(torch.as_tensor(state,device=device)[None])
                distribution=Categorical(logits=logits);action=distribution.sample()
                following,rewards,terminated,truncated=self._step(action.cpu().numpy(),episode_log)
                records['states'].append(state.copy());records['actions'].append(int(action.item()))
                records['rewards'].append(float(rewards[0]));records['old_logp'].append(float(distribution.log_prob(action).item()))
                self._reset_finished(following,terminated,truncated)
                if terminated[0] or truncated[0]:finished=True;break
            if finished:
                value={key:np.asarray(rows) for key,rows in records.items()}
                value['returns']=monte_carlo_returns(value['rewards'],gamma)
                value['advantages']=value['returns'].copy()  # pure Monte Carlo, no critic baseline
                # For J=E[sum gamma^t r_t], the exact episodic score-function
                # estimator has gamma^t outside the return-to-go as well.
                value['policy_weights']=np.power(gamma,np.arange(len(value['rewards']))).astype(np.float32)
                completed.append(value)
            else:
                # A wall-clock interruption is not a completed Monte Carlo sample.
                # Keep no invented zero tail and perform no update from this prefix.
                break
        if not completed:return None
        batch={key:np.concatenate([value[key] for value in completed]) for key in completed[0]}
        batch.pop('rewards')
        # optimize() takes a sample mean. Rescale it so REINFORCE estimates
        # the mean of episode sums, rather than dividing by random total length.
        batch['policy_weights']*=len(batch['actions'])/len(completed)
        return batch,dict(complete_episodes_in_batch=len(completed),incomplete_prefix_discarded=not finished)
