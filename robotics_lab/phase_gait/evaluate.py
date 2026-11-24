"""Measure Go2 and G1 locomotion in simulation and optionally record an MP4."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--baseline", action="store_true")
mode.add_argument("--rear-stand", action="store_true")
mode.add_argument("--g1-flat", action="store_true")
mode.add_argument("--g1-stairs", action="store_true")
parser.add_argument("--g1-rough-network", action="store_true")
parser.add_argument("--video", action="store_true")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=43)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from g1_steps import configure_g1, evaluate_g1
from rear_stand import configure_rear_stand, evaluate_rear_stand
from task import CONTACT_THRESHOLD_N, FEET, FREQUENCY_HZ, configure, force_magnitude, target_contact

TASK = "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
WARMUP_STEPS = 100


def evaluate(env, policy):
    task = env.unwrapped
    robot = task.scene["robot"]
    contact_cfg = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
    contact_cfg.resolve(task.scene)
    foot_ids, _ = robot.find_bodies(FEET, preserve_order=True)
    checks = torch.zeros(14, device=task.device)
    trace = []
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args.steps):
            obs, _, dones, _ = env.step(policy(obs))
            checks[0] += task.termination_manager.get_term("base_contact").sum()
            if step < WARMUP_STEPS:
                continue
            valid = ~dones.bool()
            actual = force_magnitude(task, contact_cfg) > CONTACT_THRESHOLD_N
            target = target_contact(task) >= 0.5
            speed = robot.data.root_lin_vel_b[:, :2]
            error = torch.linalg.vector_norm(speed - speed.new_tensor([0.75, 0.0]), dim=-1)
            tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
            support = valid[:, None]
            contacting = actual & support
            foot_xy_speed = robot.data.body_link_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
            checks += torch.stack((
                torch.zeros_like(checks[0]),
                valid.sum(),
                speed[:, 0].masked_fill(~valid, 0).sum(),
                error.masked_fill(~valid, 0).sum(),
                tilt.square().masked_fill(~valid, 0).sum(),
                (target & support).sum(),
                (target & actual & support).sum(),
                (~target & support).sum(),
                (~target & ~actual & support).sum(),
                ((actual[:, 0] & actual[:, 3] & ~actual[:, 1] & ~actual[:, 2]
                  | actual[:, 1] & actual[:, 2] & ~actual[:, 0] & ~actual[:, 3]) & valid).sum(),
                (foot_xy_speed * contacting).sum(),
                contacting.sum(),
                robot.data.root_pos_w[:, 2].masked_fill(~valid, 0).sum(),
                robot.data.root_pos_w[:, 2].square().masked_fill(~valid, 0).sum(),
            ))
            trace.append(torch.cat((target[0], actual[0], valid[:1])))
    (
        falls, n, speed, error, tilt_sq, stance_count, stance, swing_count, swing,
        diagonal, slip, contact_count, height, height_sq,
    ) = checks.tolist()
    height_mean = height / n
    result = {
        "seed": args.seed,
        "num_envs": task.num_envs,
        "steps": args.steps,
        "warmup_steps": WARMUP_STEPS,
        "command_mps": 0.75,
        "frequency_hz": FREQUENCY_HZ,
        "body_contact_terminations": int(falls),
        "mean_forward_mps": speed / n,
        "planar_error_mps": error / n,
        "tilt_rms_deg": (tilt_sq / n) ** 0.5,
        "mean_base_height_m": height_mean,
        "base_height_std_m": max(0.0, height_sq / n - height_mean**2) ** 0.5,
        "stance_accuracy": stance / stance_count,
        "swing_accuracy": swing / swing_count,
        "balanced_contact_accuracy": (stance / stance_count + swing / swing_count) / 2.0,
        "exact_diagonal_fraction": diagonal / n,
        "stance_slip_proxy_mps": slip / contact_count if contact_count else None,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "contacts.json").write_text(json.dumps({
        "feet": FEET, "dt": task.step_dt, "rows": torch.stack(trace).int().cpu().tolist()
    }) + "\n")
    print(json.dumps(result, indent=2))


def main():
    if args.steps <= WARMUP_STEPS:
        raise ValueError("Evaluation must run beyond the warmup.")
    args.output.mkdir(parents=True, exist_ok=False)
    is_g1 = args.g1_flat or args.g1_stairs
    task = "Isaac-Velocity-Flat-G1-Play-v0" if is_g1 else TASK
    env_cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.log_dir = str(args.output)
    if is_g1:
        configure_g1(env_cfg, steps=args.g1_stairs, evaluation=True)
    elif args.rear_stand:
        configure_rear_stand(env_cfg, evaluation=True)
        env_cfg.episode_length_s = (args.steps + 1) * env_cfg.sim.dt * env_cfg.decimation
    else:
        configure(env_cfg, with_phase=not args.baseline, evaluation=True)
    cfg = load_cfg_from_registry(
        "Isaac-Velocity-Rough-G1-v0" if is_g1 and args.g1_rough_network else task, "rsl_rl_cfg_entry_point"
    )
    cfg.device = env_cfg.sim.device
    raw_env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        raw_env = gym.wrappers.RecordVideo(
            raw_env, str(args.output / "video"), name_prefix="go2",
            step_trigger=lambda step: step == 0, video_length=args.steps, disable_logger=True
        )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=cfg.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env_cfg.sim.device)
        if is_g1 or args.rear_stand:
            if is_g1:
                result = evaluate_g1(env, policy, args.steps, args.g1_stairs)
            else:
                result = evaluate_rear_stand(env, policy, args.steps)
            result["seed"] = args.seed
            (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        else:
            evaluate(env, policy)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
