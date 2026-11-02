"""Token-level PPO mathematics. Rewards are data, not differentiable labels.

All response-shaped fields have [B,T]; prompt tokens are never policy actions.
The finite task ends on the first EOS or on the declared generation budget.
"""
import torch


def response_mask(tokens, eos_id):
    """Include first EOS; exclude later padding, including when PAD == EOS."""
    eos = tokens.eq(eos_id).long()
    return (eos.cumsum(-1) - eos).eq(0)


def masked_mean(values, mask):
    """A mean over valid response tokens; longer responses get more weight."""
    return (values * mask).sum() / mask.sum().clamp_min(1)


def token_statistics(model, prompt, prompt_mask, response, mask):
    # Concatenated ids [B,P+T]; attention excludes prompt padding and post-EOS pad.
    ids = torch.cat([prompt, response], dim=1)
    attention = torch.cat([prompt_mask, mask.long()], dim=1)
    logits, all_values = model(ids, attention)
    # Hidden state at P-1 sees the entire prompt and predicts response token 0.
    # States must precede the action: no off-by-one or future-token value leakage.
    start = prompt.shape[1] - 1
    logits = logits[:, start:start + response.shape[1]].float()  # [B,T,V]
    values = all_values[:, start:start + response.shape[1]].float()  # [B,T]
    logp = logits.log_softmax(-1).gather(-1, response[..., None]).squeeze(-1)
    entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    return logp, values, entropy


@torch.no_grad()
def build_rewards(scores, old_logp, reference_logp, mask, beta):
    """Signed sampled KL cost per token, plus ONE terminal task score.

    log(pi_old/pi_ref) can be negative on an individual sampled token. Its
    expectation under pi_old is KL(pi_old||pi_ref), which is nonnegative.
    """
    mask = mask.bool()
    if not mask.any(-1).all():
        raise ValueError('Every completion must have at least one action')
    if not torch.equal(mask, torch.arange(mask.shape[1], device=mask.device)[None] < mask.sum(-1)[:, None]):
        raise ValueError('Response masks must be contiguous prefixes')
    kl_sample = (old_logp - reference_logp).detach() * mask
    rewards = -beta * kl_sample
    last = mask.sum(-1) - 1
    rewards[torch.arange(len(scores), device=mask.device), last] += scores.detach()
    return rewards.detach(), kl_sample.detach()


@torch.no_grad()
def finite_episode_gae(rewards, old_values, mask, gamma=1., lam=.95):
    """Complete finite response episodes: EOS AND max-token cap have V_next=0.

    A generation cap is deliberately part of this task's horizon, not a claim
    that an arbitrary interrupted long-running task has no future value.
    """
    mask = mask.bool()
    advantages = torch.zeros_like(old_values)
    running = torch.zeros_like(old_values[:, 0])
    for t in reversed(range(mask.shape[1])):
        continues = mask[:, t + 1] if t + 1 < mask.shape[1] else torch.zeros_like(mask[:, t])
        next_value = old_values[:, t + 1] if t + 1 < mask.shape[1] else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value * continues - old_values[:, t]
        running = (delta + gamma * lam * continues * running) * mask[:, t]
        advantages[:, t] = running
    returns = (advantages + old_values) * mask
    return advantages.detach(), returns.detach()


def normalize_advantages(advantages, mask):
    selected = advantages[mask.bool()]
    return ((advantages - selected.mean()) / (selected.std(unbiased=False) + 1e-8)) * mask


def ppo_loss(logp, values, entropy, old_logp, old_values, advantages, returns,
             mask, clip=.2, value_clip=.2, value_coef=.5, entropy_coef=0.):
    """PPO old policy / advantages / returns stay frozen for ALL learner epochs."""
    log_ratio = logp - old_logp.detach()
    ratio = log_ratio.exp()
    advantages = advantages.detach()
    actor = -masked_mean(torch.minimum(ratio * advantages,
                                      ratio.clamp(1 - clip, 1 + clip) * advantages), mask)
    clipped_values = old_values.detach() + (values - old_values.detach()).clamp(-value_clip, value_clip)
    target = returns.detach()
    value = .5 * masked_mean(torch.maximum((values - target).square(),
                                          (clipped_values - target).square()), mask)
    ent = masked_mean(entropy, mask)
    total = actor + value_coef * value - entropy_coef * ent
    with torch.no_grad():
        approx_kl = masked_mean(ratio - 1 - log_ratio, mask)
        clip_fraction = masked_mean((ratio - 1).abs().gt(clip).float(), mask)
    return total, dict(actor_loss=float(actor.detach()), value_loss=float(value.detach()),
                       entropy=float(ent.detach()), approx_old_new_kl=float(approx_kl),
                       clip_fraction=float(clip_fraction))
