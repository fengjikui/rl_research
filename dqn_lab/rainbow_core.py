"""Rainbow's six ingredients: the network and distributional Double-Q update.

PER and n-step returns live in rainbow_replay.py. No high-level RL trainer is used.
"""
import math
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorized Gaussian parameter noise; eval() uses the learned means only."""
    def __init__(self, inputs, outputs, sigma0=.5):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(outputs, inputs))
        self.weight_sigma = nn.Parameter(torch.full((outputs, inputs), sigma0/math.sqrt(inputs)))
        self.bias_mu = nn.Parameter(torch.empty(outputs))
        self.bias_sigma = nn.Parameter(torch.full((outputs,), sigma0/math.sqrt(outputs)))
        self.register_buffer('weight_epsilon', torch.zeros(outputs, inputs))
        self.register_buffer('bias_epsilon', torch.zeros(outputs))
        bound = 1/math.sqrt(inputs)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        self.reset_noise()

    @torch.no_grad()
    def reset_noise(self):
        def scaled(size):
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign()*x.abs().sqrt()
        incoming, outgoing = scaled(self.weight_mu.shape[1]), scaled(self.weight_mu.shape[0])
        self.weight_epsilon.copy_(outgoing[:, None]*incoming[None, :])
        self.bias_epsilon.copy_(outgoing)

    def forward(self, x):
        weight, bias = self.weight_mu, self.bias_mu
        if self.training:
            weight = weight+self.weight_sigma*self.weight_epsilon
            bias = bias+self.bias_sigma*self.bias_epsilon
        return F.linear(x, weight, bias)


class RainbowQNet(nn.Module):
    def __init__(self, actions, atoms=51, v_min=-10., v_max=10.):
        super().__init__()
        self.actions, self.atoms = actions, atoms
        self.register_buffer('support', torch.linspace(v_min, v_max, atoms))
        # [B,4,84,84] -> [B,32,20,20] -> [B,64,9,9] -> [B,64,7,7]
        self.features = nn.Sequential(nn.Conv2d(4,32,8,4),nn.ReLU(),
            nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        # Dueling: a value distribution and action advantages for every atom.
        self.value = nn.Sequential(NoisyLinear(3136,512),nn.ReLU(),NoisyLinear(512,atoms))
        self.advantage = nn.Sequential(NoisyLinear(3136,512),nn.ReLU(),NoisyLinear(512,actions*atoms))

    def reset_noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):layer.reset_noise()

    def logits(self, frames):
        x = self.features(frames.float()/255.)  # [B,3136]
        value = self.value(x).view(-1,1,self.atoms)  # [B,1,Z]
        advantage = self.advantage(x).view(-1,self.actions,self.atoms)  # [B,A,Z]
        return value+advantage-advantage.mean(dim=1,keepdim=True)

    def distribution(self, frames):
        return self.logits(frames).softmax(dim=-1)  # [B,A,Z], sums to 1 over Z

    def forward(self, frames):
        # Evaluation code and action selection still see ordinary Q values [B,A].
        return (self.distribution(frames)*self.support).sum(dim=-1)


@torch.no_grad()
def categorical_projection(probabilities, returns, discounts, support):
    """Project R_n + gamma^n Z onto the fixed C51 support; labels have no grad."""
    # probabilities [B,Z]; returns/discounts [B]; support [Z]. Discount is zero
    # for true termination, but remains gamma^h for a time-limit truncation.
    batch, atoms = probabilities.shape
    vmin, vmax = support[0], support[-1]
    spacing = (vmax-vmin)/(atoms-1)
    target = (returns[:,None]+discounts[:,None]*support[None,:]).clamp(vmin,vmax)
    positions = (target-vmin)/spacing
    lower = positions.floor().long().clamp(0,atoms-1)
    upper = positions.ceil().long().clamp(0,atoms-1)
    # Integer positions must retain all probability, including clipped endpoints.
    lower_weight = upper.to(positions.dtype)-positions+(lower==upper).to(positions.dtype)
    upper_weight = positions-lower.to(positions.dtype)
    result = torch.zeros_like(probabilities)
    result.scatter_add_(1,lower,probabilities*lower_weight)
    result.scatter_add_(1,upper,probabilities*upper_weight)
    return result


def update(online, target, optimizer, batch, replay, audit_dir=None):
    device = next(online.parameters()).device
    states = torch.as_tensor(batch['states'],device=device)  # [B,4,84,84] uint8
    following = torch.as_tensor(batch['next_states'],device=device)
    actions = torch.as_tensor(batch['actions'],device=device,dtype=torch.long)  # [B]
    returns = torch.as_tensor(batch['returns'],device=device,dtype=torch.float32)
    discounts = torch.as_tensor(batch['discounts'],device=device,dtype=torch.float32)
    weights = torch.as_tensor(batch['weights'],device=device,dtype=torch.float32)
    online.train(); target.train()
    online.reset_noise(); target.reset_noise()
    with torch.no_grad():
        # Double DQN: online chooses the next action; target evaluates its atoms.
        next_actions = online(following).argmax(dim=1)  # [B]
        next_distribution = target.distribution(following)[torch.arange(len(actions),device=device),next_actions]
        label = categorical_projection(next_distribution,returns,discounts,online.support)  # [B,Z]
    log_probabilities = online.logits(states).log_softmax(dim=-1)
    chosen = log_probabilities[torch.arange(len(actions),device=device),actions]  # [B,Z]
    per_item = -(label*chosen).sum(dim=-1)  # [B], categorical cross entropy
    loss = (per_item*weights).mean()  # scalar [], PER importance correction
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(online.parameters(),10.)
    optimizer.step()
    # Original Rainbow uses distributional KL divergence as PER priority.
    # Entropy is constant w.r.t. prediction, so optimization still uses CE.
    label_entropy_term = (label*label.clamp_min(1e-12).log()).sum(dim=-1)
    priorities = (per_item.detach()+label_entropy_term).clamp_min(0).cpu().numpy()+1e-6
    replay.update_priorities(batch['indices'],priorities)
    if audit_dir is not None:
        audit_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(audit_dir/'batch_inputs.npz',**batch)
        np.savez_compressed(audit_dir/'distributional_labels.npz',
            next_distribution=next_distribution.cpu().numpy(),
            projected_distribution=label.cpu().numpy(),support=online.support.cpu().numpy(),
            chosen_log_probabilities=chosen.detach().cpu().numpy(),next_actions=next_actions.cpu().numpy())
        (audit_dir/'loss_batch.json').write_text(json.dumps(dict(
            loss=float(loss.detach()),per_item_cross_entropy=per_item.detach().cpu().tolist(),
            importance_weights=weights.cpu().tolist(),per_item_kl_priority=priorities.tolist(),
            target_requires_grad=label.requires_grad,
            input_shape=list(states.shape),label_shape=list(label.shape),
            gradient_norm_before_clipping=float(grad_norm)),indent=2)+'\n')
    return float(loss.detach())
