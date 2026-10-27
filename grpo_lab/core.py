"""GRPO primitives shared by the game and pretrained language model.

No critic, no PPO/GRPO Trainer. beta=0: no reference-policy KL in this lab.
"""
import torch


@torch.no_grad()
def group_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError('Complete groups of at least two trajectories are required')
    grouped = rewards.reshape(-1, group_size)
    std = grouped.std(-1, unbiased=False, keepdim=True)
    advantage = (grouped-grouped.mean(-1, keepdim=True))/(std+1e-4)
    return advantage.flatten(), float((std > 0).float().mean())


def clipped_loss(new_logp, old_logp, advantage, mask, clip=.2):
    ratio = (new_logp-old_logp.detach()).exp()
    a = advantage.detach()[:, None]
    objective = torch.minimum(ratio*a, ratio.clamp(1-clip, 1+clip)*a)
    # Mean over actual actions in each sequence, then mean over sequences.
    loss = -((objective*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
    return loss
