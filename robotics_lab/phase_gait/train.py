"""Train or continue Go2 and G1 locomotion policies with Isaac Lab and PPO."""

import argparse
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--rear-stand", action="store_true")
mode.add_argument("--foot-reference", action="store_true")
mode.add_argument("--posture-refinement", action="store_true")
mode.add_argument("--g1-flat", action="store_true")
mode.add_argument("--g1-stairs", action="store_true")
parser.add_argument("--g1-rough-network", action="store_true")
stair_rewards = parser.add_mutually_exclusive_group()
stair_rewards.add_argument("--g1-progress-reward", action="store_true")
stair_rewards.add_argument("--g1-step-practice", action="store_true")
parser.add_argument("--exploration-std", type=float)
parser.add_argument("--fixed-lr", type=float)
parser.add_argument("--height-reward-width", type=float, default=0.05)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.height_reward_width <= 0.0:
    parser.error("--height-reward-width must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from g1_steps import configure_g1
from rear_stand import configure_rear_stand
from task import configure

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


def warm_start(runner, path, extra_inputs=2, reset_optimizer=False):
    """Preserve the old policy by initializing new input weights to zero."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint["model_state_dict"]
    expected = runner.alg.policy.state_dict()
    same_shape = all(saved[key].shape == value.shape for key, value in expected.items())
    if same_shape:
        runner.load(str(path), load_optimizer=not reset_optimizer)
        return
    for key, value in expected.items():
        weight = saved[key]
        if weight.shape == value.shape:
            continue
        if (
            key not in ("actor.0.weight", "critic.0.weight")
            or weight.ndim != 2
            or weight.shape[0] != value.shape[0]
            or weight.shape[1] + extra_inputs != value.shape[1]
        ):
            raise ValueError(f"Unexpected checkpoint shape for {key}: {weight.shape} -> {value.shape}")
        extension = torch.zeros((weight.shape[0], extra_inputs), dtype=weight.dtype)
        saved[key] = torch.cat((weight, extension), dim=1)
    runner.alg.policy.load_state_dict(saved)


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    is_g1 = args.g1_flat or args.g1_stairs
    task = "Isaac-Velocity-Flat-G1-v0" if is_g1 else TASK
    env_cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.log_dir = str(args.output)
    if is_g1:
        configure_g1(
            env_cfg, steps=args.g1_stairs, progress_reward=args.g1_progress_reward,
            step_practice=args.g1_step_practice,
        )
    elif args.rear_stand:
        configure_rear_stand(env_cfg)
    else:
        configure(
            env_cfg, with_phase=True, with_path=args.foot_reference or args.posture_refinement,
            with_posture=args.posture_refinement, height_width=args.height_reward_width,
        )
    agent_cfg = load_cfg_from_registry(
        "Isaac-Velocity-Rough-G1-v0" if is_g1 and args.g1_rough_network else task, "rsl_rl_cfg_entry_point"
    )
    if args.rear_stand:
        agent_cfg.policy.init_noise_std = 0.3
    if args.fixed_lr is not None:
        agent_cfg.algorithm.learning_rate = args.fixed_lr
        agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = 50
    agent_cfg.device = env_cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(args.output), device=agent_cfg.device)
        if args.checkpoint is not None:
            warm_start(
                runner, args.checkpoint, extra_inputs=187 if is_g1 else 2,
                reset_optimizer=args.exploration_std is not None or args.fixed_lr is not None,
            )
        if args.exploration_std is not None:
            with torch.no_grad():
                runner.alg.policy.std.fill_(args.exploration_std)
        dump_yaml(str(args.output / "env.yaml"), env_cfg)
        dump_yaml(str(args.output / "agent.yaml"), agent_cfg)
        print(f"Source checkpoint: {args.checkpoint}")
        started = time.monotonic()
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(args.output / "final.pt"))
        print(f"Training seconds: {time.monotonic() - started:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
