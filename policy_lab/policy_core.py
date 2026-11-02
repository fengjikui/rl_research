"""Readable policy-gradient math, independent of any game or RL trainer."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class VisualActorCritic(nn.Module):
    def __init__(self, actions, channels=4, size=64):
        super().__init__()
        # Input [B,4,64,64] = four consecutive grayscale observations.
        # Three unpadded convolutions give [B,32,15,15], [B,64,6,6], [B,64,4,4].
        self.encoder=nn.Sequential(nn.Conv2d(channels,32,8,4),nn.ReLU(),
            nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad():features=self.encoder(torch.zeros(1,channels,size,size)).shape[-1]
        self.features=nn.Sequential(nn.Linear(features,256),nn.ReLU())
        self.actor=nn.Linear(256,actions)  # logits [B,A], not Q values
        self.critic=nn.Linear(256,1)       # state value [B,1] -> [B]
        nn.init.normal_(self.actor.weight,std=.01);nn.init.zeros_(self.actor.bias)

    def forward(self, observations):
        x=self.features(self.encoder(observations.float()/255.))
        return self.actor(x),self.critic(x).squeeze(-1)


@torch.no_grad()
def compute_gae(rewards,values,next_values,terminated,truncated,gamma=.99,lam=.95):
    """All fields [T,N]. next_values are from each *actual* post-step observation.

    True termination masks bootstrap; either end flag cuts the recursion. At a
    rollout-window edge running=0, but the last delta still bootstraps next V.
    This handles different episodes in one window without leaking across resets.
    """
    if not (rewards.shape==values.shape==next_values.shape==terminated.shape==truncated.shape):
        raise ValueError('GAE fields must have identical [T,N] shapes')
    advantages=torch.zeros_like(values)
    running=torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        bootstrap=(~terminated[t].bool()).to(values.dtype)
        continue_episode=(~(terminated[t].bool()|truncated[t].bool())).to(values.dtype)
        delta=rewards[t]+gamma*bootstrap*next_values[t]-values[t]
        running=delta+gamma*lam*continue_episode*running
        advantages[t]=running
    return advantages.detach(),(advantages+values).detach()


def monte_carlo_returns(rewards,gamma=.99):
    """Finite complete episode: no learned bootstrap, no critic/baseline."""
    result=np.empty(len(rewards),dtype=np.float32);running=0.
    for i in reversed(range(len(rewards))):
        running=float(rewards[i])+gamma*running;result[i]=running
    return result


def score_function_loss(log_probabilities,advantages):
    # [B] -> scalar []. Advantages/returns are fixed coefficients, never targets
    # to backpropagate through. Positive advantage raises the sampled action's p.
    return -(log_probabilities*advantages.detach()).mean()


def ppo_clipped_loss(log_probabilities,old_log_probabilities,advantages,clip=.2):
    # Old rollout probabilities stay fixed through every learner epoch.
    ratio=(log_probabilities-old_log_probabilities.detach()).exp()
    return -torch.minimum(ratio*advantages.detach(),
        ratio.clamp(1-clip,1+clip)*advantages.detach()).mean()


def optimize(model,optimizer,batch,algo,device,epochs=4,minibatch_size=64,clip=.2,
             value_coef=.5,entropy_coef=.01,normalize_advantages=True,max_grad_norm=.5):
    """On-policy update; CPU uint8 rollout is copied to the learner in mini-batches.

    REINFORCE/AC use one accumulated full-batch gradient and one optimizer step.
    PPO does multiple shuffled mini-batch updates against the same frozen data.
    """
    old_logp=torch.as_tensor(batch['old_logp'],dtype=torch.float32,device=device).detach().clone()
    old_snapshot=old_logp.clone()
    advantages=torch.as_tensor(batch['advantages'],dtype=torch.float32,device=device).detach().clone()
    returns=torch.as_tensor(batch['returns'],dtype=torch.float32,device=device).detach()
    policy_weights=torch.as_tensor(batch.get('policy_weights',np.ones(len(advantages))),dtype=torch.float32,device=device)
    if normalize_advantages and algo!='reinforce' and len(advantages)>1:
        advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    actions=torch.as_tensor(batch['actions'],dtype=torch.long,device=device)
    count=len(actions);logs=[];optimizer_steps=0
    for epoch in range(epochs if algo=='ppo' else 1):
        indices=np.random.permutation(count) if algo=='ppo' else np.arange(count)
        if algo!='ppo':optimizer.zero_grad(set_to_none=True)
        for start in range(0,count,minibatch_size):
            selection=indices[start:start+minibatch_size]
            ii=torch.as_tensor(selection,dtype=torch.long,device=device)
            states=torch.as_tensor(batch['states'][selection],device=device)  # [b,C,H,W] uint8
            logits,values=model(states)
            distribution=Categorical(logits=logits);logp=distribution.log_prob(actions[ii])
            if algo=='ppo':actor=ppo_clipped_loss(logp,old_logp[ii],advantages[ii],clip)
            else:actor=score_function_loss(logp,advantages[ii]*policy_weights[ii])
            critic=F.mse_loss(values,returns[ii]) if algo!='reinforce' else values.new_zeros(())
            entropy=distribution.entropy().mean()
            loss=actor+value_coef*critic-entropy_coef*entropy
            if not torch.isfinite(loss):raise RuntimeError('Non-finite policy loss')
            if algo=='ppo':optimizer.zero_grad(set_to_none=True)
            # For REINFORCE/AC, exactly one full-batch mean gradient is accumulated
            # without retaining the enormous graph for an entire visual episode.
            (loss if algo=='ppo' else loss*(len(selection)/count)).backward()
            if algo=='ppo':
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
                optimizer.step()
                optimizer_steps+=1
            else:norm=values.new_zeros(())
            with torch.no_grad():
                log_ratio=logp-old_logp[ii];ratio=log_ratio.exp()
                logs.append(dict(actor_loss=float(actor.detach()),critic_loss=float(critic.detach()),
                    entropy=float(entropy.detach()),loss=float(loss.detach()),
                    approx_kl=float((ratio-1-log_ratio).mean()),
                    clip_fraction=float(((ratio-1).abs()>clip).float().mean()),
                    samples=len(selection),epoch=epoch,gradient_norm=float(norm)))
        if algo!='ppo':
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm);optimizer.step()
            optimizer_steps+=1
            logs[-1]['gradient_norm']=float(norm)
    # No write to this data is allowed: it is the behavior policy, not a moving target.
    if not torch.equal(old_snapshot,old_logp):raise RuntimeError('PPO old probabilities were modified')
    total=sum(row['samples'] for row in logs)
    result={key:sum(row[key]*row['samples'] for row in logs)/total
            for key in ('actor_loss','critic_loss','entropy','loss','approx_kl','clip_fraction')}
    result['optimizer_steps']=optimizer_steps  # actual optimizer.step() calls in this batch
    return result
