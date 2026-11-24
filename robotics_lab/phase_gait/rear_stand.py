"""Experimental Go2 rear-foot balance environment with near-upright resets."""

import math

import torch
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg

from task import FEET, configure, force_magnitude

PITCH_RAD = math.radians(-85.0)
HEIGHT_M = 0.53


def upright(env):
    gravity = env.scene["robot"].data.projected_gravity_b
    desired = gravity.new_tensor([math.sin(PITCH_RAD), 0.0, -math.cos(PITCH_RAD)])
    return torch.exp(-((gravity - desired).square().sum(dim=-1)) / 0.10)


def holding_position(env):
    robot = env.scene["robot"].data
    error = (robot.root_pos_w[:, 2] - HEIGHT_M).square() / 0.01
    error += robot.root_lin_vel_w[:, :2].square().sum(dim=-1) / 0.25
    return torch.exp(-error)


def rear_support(env, contact_cfg):
    forces = force_magnitude(env, contact_cfg)
    supports = (forces / 25.0).clamp(0.0, 1.0)
    return supports[:, 2:].amin(dim=-1) * (1.0 - supports[:, :2].amax(dim=-1))


def tipped(env):
    robot = env.scene["robot"].data
    return (robot.projected_gravity_b[:, 0] > -0.55) | (robot.root_pos_w[:, 2] < 0.30)


def configure_rear_stand(cfg, evaluation=False):
    configure(cfg, with_phase=False, evaluation=evaluation)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.episode_length_s = 12.0 if evaluation else 6.0
    cfg.scene.robot.init_state.pos = (0.0, 0.0, HEIGHT_M)
    cfg.scene.robot.init_state.rot = (math.cos(PITCH_RAD * 0.5), 0.0, math.sin(PITCH_RAD * 0.5), 0.0)
    cfg.scene.robot.init_state.joint_pos = {
        ".*L_hip_joint": 0.06,
        ".*R_hip_joint": -0.06,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 2.35,
        "F[L,R]_calf_joint": -1.5,
        "R[L,R]_calf_joint": -1.53,
    }
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.reset_base.params["pose_range"].update(
        roll=(-0.02, 0.02), pitch=(-0.03, 0.03), yaw=(0.0, 0.0)
    )
    cfg.events.reset_base.params["velocity_range"] = {
        key: (-0.015, 0.015) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "flat_orientation_l2", "ang_vel_xy_l2"):
        setattr(cfg.rewards, name, None)
    contact_cfg = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
    cfg.rewards.rear_upright = RewardTermCfg(func=upright, weight=2.0)
    cfg.rewards.rear_hold = RewardTermCfg(func=holding_position, weight=1.2)
    cfg.rewards.rear_support = RewardTermCfg(
        func=rear_support, weight=0.9, params={"contact_cfg": contact_cfg}
    )
    cfg.terminations.rear_tipped = TerminationTermCfg(func=tipped)


def evaluate_rear_stand(env, policy, steps):
    scene = env.unwrapped.scene
    robot = scene["robot"].data
    contact_cfg = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
    contact_cfg.resolve(scene)
    obs = env.get_observations()
    counts = torch.zeros(5, device=env.unwrapped.device)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()
            forces = force_magnitude(env.unwrapped, contact_cfg)
            contact = forces > 5.0
            upright_now = robot.projected_gravity_b[:, 0] < -0.90
            counts += torch.stack((
                alive.sum(),
                (alive & upright_now).sum(),
                (alive & ~contact[:, :2].any(dim=-1)).sum(),
                (alive & contact[:, 2:].all(dim=-1) & ~contact[:, :2].any(dim=-1) & upright_now).sum(),
                (alive & upright_now & (robot.root_lin_vel_w[:, :2].norm(dim=-1) < 0.15)).sum(),
            ))
    live_steps, upright_steps, airborne_steps, supported_steps, steady_steps = counts.tolist()
    return {
        "steps": steps,
        "num_envs": env.num_envs,
        "trial_seconds": steps * env.unwrapped.step_dt,
        "survivors": int(alive.sum().item()),
        "mean_time_before_reset_s": live_steps * env.unwrapped.step_dt / env.num_envs,
        "upright_fraction_of_trial": upright_steps / (steps * env.num_envs),
        "front_clear_fraction_of_trial": airborne_steps / (steps * env.num_envs),
        "rear_support_upright_fraction_of_trial": supported_steps / (steps * env.num_envs),
        "steady_upright_fraction_of_trial": steady_steps / (steps * env.num_envs),
    }
