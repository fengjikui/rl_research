"""Run the Go2 rear-leg PPO curriculum or measure a frozen checkpoint."""

import argparse
import json
from pathlib import Path
import signal
import statistics
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--duration", type=float, default=18_000)
parser.add_argument("--elapsed", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--start", choices=("upright", "floor"), default="upright")
parser.add_argument("--target", type=float, default=0.20)
parser.add_argument("--trial-steps", type=int, default=1000)
parser.add_argument("--video", action="store_true")
training_mode = parser.add_mutually_exclusive_group()
training_mode.add_argument("--motion-practice", action="store_true")
training_mode.add_argument("--route-practice", action="store_true")
parser.add_argument("--swing-practice", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.evaluate and not args.checkpoint:
    parser.error("Evaluation requires --checkpoint.")
if args.video and (not args.evaluate or args.num_envs != 1):
    parser.error("Video requires single-environment evaluation.")
if args.swing_practice and not args.route_practice:
    parser.error("Swing practice requires the fixed route.")
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from rear_walk import configure_rear_walk, horizontal_velocity, motion_axes, upright_quality
from task import FEET, force_magnitude


TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


class TrainingFinished(Exception):
    pass


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


class RearRunner(OnPolicyRunner):
    def setup_curriculum(self, elapsed):
        self.started = time.monotonic() - elapsed
        self.last_checkpoint = elapsed - 60
        self.speed_fraction = 0.0
        self.rise_fraction = 0.0
        self.stop_requested = False
        self.completed_updates = 0

    def save_latest(self, status):
        target = args.output / "latest.pt"
        temp = args.output / "latest.tmp"
        self.save(str(temp), infos=status)
        temp.replace(target)
        write_json(args.output / "status.json", status)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        self.completed_updates += 1
        elapsed = time.monotonic() - self.started
        episode_length = statistics.median(locs["lenbuffer"]) if locs["lenbuffer"] else 0.0
        length_s = episode_length * self.env.unwrapped.step_dt
        train = self.env.unwrapped

        # Expand only while the policy can recover a useful fraction of full episodes.
        if self.completed_updates % 20 == 0:
            if length_s >= 7.0:
                speed_ceiling = min(1.0, max(0.0, (elapsed - 420.0) / 6500.0))
                rise_ceiling = min(1.0, max(0.0, (elapsed - 1200.0) / 8200.0))
                self.speed_fraction = max(self.speed_fraction, min(speed_ceiling, self.speed_fraction + 0.018))
                self.rise_fraction = max(self.rise_fraction, min(rise_ceiling, self.rise_fraction + 0.018))
            train.rear_rise_difficulty = self.rise_fraction
            command = train.command_manager.get_term("base_velocity")
            speed = self.speed_fraction
            command.cfg.ranges.lin_vel_x = (-0.55 * speed, 0.80 * speed)
            lateral = (0.12 if args.route_practice else 0.23) * speed
            turning = 0.0 if args.route_practice else 0.7 * speed
            command.cfg.ranges.lin_vel_y = (-lateral, lateral)
            command.cfg.ranges.ang_vel_z = (-turning, turning)

        finished = elapsed >= args.duration or self.stop_requested
        if elapsed - self.last_checkpoint >= 300 or finished or self.completed_updates == 1:
            status = {
                "elapsed_training_seconds": elapsed,
                "iteration": locs["it"],
                "session_updates": self.completed_updates,
                "session_samples": self.tot_timesteps,
                "median_episode_seconds": length_s,
                "speed_fraction": self.speed_fraction,
                "rise_fraction": self.rise_fraction,
                "finished": finished,
            }
            self.save_latest(status)
            self.last_checkpoint = elapsed
            print("Progress snapshot", json.dumps(status), flush=True)
        if finished:
            raise TrainingFinished


def evaluate(env, policy):
    task = env.unwrapped
    data = task.scene["robot"].data
    contacts = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
    contacts.resolve(task.scene)
    obs = env.get_observations()
    direction, _ = motion_axes(task)
    direction = direction.clone()
    origin = data.root_pos_w[:, :2].clone()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=task.device)
    saw_upright = torch.zeros_like(alive)
    upright_steps = torch.zeros(env.num_envs, device=task.device)
    rear_only_steps = torch.zeros_like(upright_steps)
    left_stance_steps = torch.zeros_like(upright_steps)
    right_stance_steps = torch.zeros_like(upright_steps)
    live_steps = torch.zeros_like(upright_steps)
    tracking_steps = torch.zeros_like(upright_steps)
    squared_error = torch.zeros((env.num_envs, 3), device=task.device)
    progress = torch.zeros_like(upright_steps)
    with torch.inference_mode():
        for index in range(args.trial_steps):
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()
            standing = alive & (upright_quality(task) > 0.94)
            contact = force_magnitude(task, contacts) > 5.0
            rear_only = standing & ~contact[:, :2].any(dim=1) & contact[:, 2:].any(dim=1)
            saw_upright |= rear_only
            live_steps += alive
            upright_steps += standing
            rear_only_steps += rear_only
            left_stance_steps += rear_only & contact[:, 2] & ~contact[:, 3]
            right_stance_steps += rear_only & contact[:, 3] & ~contact[:, 2]
            progress = torch.where(alive, ((data.root_pos_w[:, :2] - origin) * direction).sum(dim=1), progress)
            if index >= 100:
                velocity = horizontal_velocity(task)
                error = torch.cat((velocity, data.root_ang_vel_w[:, 2:3]), dim=1)
                error[:, 0] -= args.target
                squared_error += (error.square() * alive[:, None])
                tracking_steps += alive
    required = 0.70 * args.trial_steps
    alternating = (left_stance_steps > 0.025 * args.trial_steps) & (right_stance_steps > 0.025 * args.trial_steps)
    direction_sign = -1.0 if args.target < 0.0 else 1.0
    completed = alive & alternating & (rear_only_steps >= required) & (direction_sign * progress >= 0.5)
    if args.target == 0.0:
        completed.zero_()
    errors = (squared_error.sum(dim=0) / tracking_steps.sum().clamp_min(1)).sqrt().tolist()
    denom = env.num_envs * args.trial_steps
    return {
        "seed": args.seed,
        "start": args.start,
        "target_forward_mps": args.target,
        "environments": env.num_envs,
        "steps": args.trial_steps,
        "seconds": args.trial_steps * task.step_dt,
        "survived": int(alive.sum().item()),
        "reached_upright_hind_support": int(saw_upright.sum().item()),
        "completed_hind_walk": int(completed.sum().item()),
        "mean_survival_seconds": (live_steps.mean() * task.step_dt).item(),
        "upright_fraction": upright_steps.sum().item() / denom,
        "hind_only_support_fraction": rear_only_steps.sum().item() / denom,
        "left_only_stance_fraction": left_stance_steps.sum().item() / denom,
        "right_only_stance_fraction": right_stance_steps.sum().item() / denom,
        "mean_forward_displacement_m": progress.mean().item(),
        "linear_rmse_xy_mps": errors[:2] if tracking_steps.sum().item() else None,
        "yaw_rmse_rad_s": errors[2] if tracking_steps.sum().item() else None,
    }


def load_policy(runner, path):
    if args.route_practice:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved = checkpoint["model_state_dict"]
        expected = runner.alg.policy.state_dict()
        if saved["actor.0.weight"].shape != expected["actor.0.weight"].shape:
            for key in ("actor.0.weight", "critic.0.weight"):
                weight = saved[key]
                if expected[key].shape != (weight.shape[0], weight.shape[1] + 2):
                    raise ValueError(f"Unexpected heading observation shape for {key}")
                saved[key] = torch.cat((weight, torch.zeros_like(weight[:, :2])), dim=1)
            runner.alg.policy.load_state_dict(saved)
            runner.current_learning_iteration = checkpoint.get("iter", 0)
            return checkpoint.get("infos")
    return runner.load(str(path), load_optimizer=not (args.evaluate or args.motion_practice or args.route_practice))


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.log_dir = str(args.output)
    trial_seconds = (args.trial_steps + 1) * env_cfg.sim.dt * env_cfg.decimation
    configure_rear_walk(
        env_cfg, start=args.start if args.evaluate else "curriculum", evaluation=args.evaluate,
        command=(args.target, 0.0, 0.0) if args.evaluate else (0.0, 0.0, 0.0),
        trial_seconds=trial_seconds if args.evaluate else 15.0,
        motion_practice=args.motion_practice and not args.evaluate,
        route_practice=args.route_practice,
        swing_practice=args.swing_practice and not args.evaluate,
    )
    cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    cfg.seed = args.seed
    cfg.device = env_cfg.sim.device
    cfg.policy.actor_hidden_dims = [256, 256, 128]
    cfg.policy.critic_hidden_dims = [256, 256, 128]
    cfg.policy.init_noise_std = 0.32
    cfg.save_interval = 1000
    cfg.algorithm.learning_rate = 0.00035
    cfg.algorithm.entropy_coef = 0.005
    if args.motion_practice or args.route_practice:
        cfg.algorithm.learning_rate = 0.00015
        cfg.algorithm.schedule = "fixed"
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, str(args.output / "video"), name_prefix="go2-hind",
            step_trigger=lambda step: step == 0, video_length=args.trial_steps, disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
    try:
        runner = RearRunner(env, cfg.to_dict(), log_dir=None if args.evaluate else str(args.output), device=cfg.device)
        if args.checkpoint:
            infos = load_policy(runner, args.checkpoint)
        else:
            infos = None
        if args.evaluate:
            metrics = evaluate(env, runner.get_inference_policy(device=cfg.device))
            write_json(args.output / "metrics.json", metrics)
            print(json.dumps(metrics, indent=2), flush=True)
        else:
            dump_yaml(str(args.output / "env.yaml"), env_cfg)
            dump_yaml(str(args.output / "agent.yaml"), cfg)
            runner.setup_curriculum(args.elapsed)
            if infos:
                runner.speed_fraction = infos.get("speed_fraction", 0.0)
                runner.rise_fraction = infos.get("rise_fraction", 0.0)
            signal.signal(signal.SIGTERM, lambda *_: setattr(runner, "stop_requested", True))
            print(f"Training budget seconds: {args.duration:.1f}", flush=True)
            try:
                runner.learn(num_learning_iterations=100_000, init_at_random_ep_len=False)
            except TrainingFinished:
                pass
            runner.save(str(args.output / "final.pt"), infos=json.loads((args.output / "status.json").read_text()))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
