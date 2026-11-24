"""Flat approach and ascending stairs curriculum for the Isaac Lab Unitree G1."""

import numpy as np
import torch
import trimesh
from isaaclab.managers import RewardTermCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import load_cfg_from_registry

SPAWN_X_M = 0.9
FIRST_STEP_X_M = 1.9
TREAD_M = 0.42
COURSE_LENGTH_M = 8.0
COURSE_WIDTH_M = 4.0
RISERS = 5
EVAL_RISER_M = 0.08
CLIMB_END_M = FIRST_STEP_X_M + (RISERS - 1) * TREAD_M - SPAWN_X_M
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def straight_terrain(difficulty, cfg):
    length, width = cfg.size
    low, high = cfg.step_height_range
    height = low + difficulty * (high - low)
    floor = trimesh.creation.box(extents=(length, width, 0.1))
    floor.apply_translation((length / 2, width / 2, -0.05))
    meshes = [floor]
    if height > 0.0:
        for step in range(RISERS):
            start = FIRST_STEP_X_M + step * TREAD_M
            end = length if step == RISERS - 1 else start + TREAD_M
            top = height * (step + 1)
            block = trimesh.creation.box(extents=(end - start, width, top))
            block.apply_translation(((start + end) / 2, width / 2, top / 2))
            meshes.append(block)
    return meshes, np.array([SPAWN_X_M, width / 2, 0.0])


@configclass
class StraightTerrainCfg(SubTerrainBaseCfg):
    function = straight_terrain
    step_height_range: tuple[float, float] = (0.0, 0.0)


def upright_height(env):
    data = env.scene["robot"].data
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-(height - 0.70).square() / 0.09**2)


def posture_above_terrain(env, gate_on_progress=False):
    data = env.scene["robot"].data
    hit = env.scene.sensors["height_scanner"].data.ray_hits_w
    ground = hit[:, hit.shape[1] // 2, 2]
    ground = torch.where(ground.isfinite(), ground, env.scene.env_origins[:, 2])
    reward = torch.exp(-(data.root_pos_w[:, 2] - ground - 0.70).square() / 0.09**2)
    if gate_on_progress:
        reward = reward * (data.root_lin_vel_w[:, 0] / 0.42).clamp(0.0, 1.0)
    return reward


def forward_route_tracking(env):
    robot = env.scene["robot"].data
    velocity = robot.root_lin_vel_w[:, :2]
    desired = env.command_manager.get_command("base_velocity")[:, 0]
    forward = (velocity[:, 0] / desired.clamp_min(0.1)).clamp(0.0, 1.0)
    normalized_error = ((velocity[:, 0] - desired).square() + velocity[:, 1].square()) / 0.26**2
    return forward * torch.exp(-normalized_error)


def signed_route_progress(env):
    velocity = env.scene["robot"].data.root_lin_vel_w[:, 0]
    desired = env.command_manager.get_command("base_velocity")[:, 0]
    return (velocity / desired.clamp_min(0.1)).clamp(-1.0, 1.0)


def configure_g1(cfg, steps=False, evaluation=False, progress_reward=False, step_practice=False):
    # Add the same height observation in both stages so the flat checkpoint transfers directly.
    rough = load_cfg_from_registry("Isaac-Velocity-Rough-G1-v0", "env_cfg_entry_point")
    cfg.scene.height_scanner = rough.scene.height_scanner
    cfg.observations.policy.height_scan = rough.observations.policy.height_scan
    heights = (0.0, 0.0)
    if steps:
        heights = (EVAL_RISER_M, EVAL_RISER_M) if evaluation or step_practice else (0.045, 0.085)
        cfg.rewards.lin_vel_z_l2.weight = -0.05
        cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
        cfg.rewards.posture_above_terrain = RewardTermCfg(func=posture_above_terrain, weight=0.5)
        if progress_reward:
            cfg.rewards.track_lin_vel_xy_exp = RewardTermCfg(func=forward_route_tracking, weight=2.5)
            cfg.rewards.track_ang_vel_z_exp = None
            cfg.rewards.posture_above_terrain.params["gate_on_progress"] = True
            cfg.rewards.termination_penalty.weight = -40.0
        if step_practice:
            cfg.rewards.posture_above_terrain = None
            cfg.rewards.route_progress = RewardTermCfg(func=signed_route_progress, weight=0.65)
    if not steps:
        cfg.rewards.upright_height = RewardTermCfg(func=upright_height, weight=0.65)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(COURSE_LENGTH_M, COURSE_WIDTH_M), num_rows=1 if evaluation else 5,
        num_cols=1 if evaluation else 8,
        border_width=1.0, curriculum=False, use_cache=False, color_scheme="height",
        sub_terrains={"walkway": StraightTerrainCfg(step_height_range=heights)},
    )
    cfg.curriculum.terrain_levels = None
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.1, 0.1) if not evaluation else (0.0, 0.0),
        "y": (-0.1, 0.1) if not evaluation else (0.0, 0.0),
        "yaw": (-0.12, 0.12) if not evaluation else (0.0, 0.0),
    }
    cfg.events.base_external_force_torque = None
    commands = cfg.commands.base_velocity
    commands.heading_command = True
    commands.rel_heading_envs = 1.0
    commands.rel_standing_envs = 0.0
    commands.ranges.heading = (0.0, 0.0)
    training_range = (0.37, 0.50) if steps else (0.35, 0.68)
    commands.ranges.lin_vel_x = (0.42, 0.42) if evaluation or step_practice else training_range
    commands.ranges.lin_vel_y = (0.0, 0.0)
    commands.ranges.ang_vel_z = (-0.5, 0.5)
    commands.debug_vis = False
    if evaluation:
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (-2.1, -3.1, 1.15) if steps else (1.0, -3.05, 1.55)
        cfg.viewer.lookat = (0.55, 0.0, -0.35) if steps else (0.25, 0.0, -0.25)
        cfg.viewer.resolution = (960, 540)


def evaluate_g1(env, policy, trial_steps, stairs):
    task = env.unwrapped
    articulation = task.scene["robot"]
    robot = articulation.data
    obs = env.get_observations()
    initial = robot.root_pos_w.clone()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=task.device)
    peak = torch.zeros(env.num_envs, device=task.device)
    speed_sum = torch.zeros(env.num_envs, device=task.device)
    samples = torch.zeros_like(speed_sum)
    last_progress = torch.zeros_like(speed_sum)
    last_rise = torch.zeros_like(speed_sum)
    if stairs:
        sensor = task.scene.sensors["contact_forces"]
        foot_links, _ = articulation.find_bodies(FEET, preserve_order=True)
        foot_sensors, _ = sensor.find_bodies(FEET, preserve_order=True)
        top_contacts = torch.zeros((env.num_envs, len(FEET)), dtype=torch.bool, device=task.device)
        platform_height = EVAL_RISER_M * RISERS
    with torch.inference_mode():
        for index in range(trial_steps):
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()
            relative = robot.root_pos_w - initial
            peak = torch.maximum(peak, relative[:, 2].masked_fill(~alive, 0))
            last_progress = torch.where(alive, relative[:, 0], last_progress)
            last_rise = torch.where(alive, relative[:, 2], last_rise)
            if stairs:
                foot = robot.body_link_pos_w[:, foot_links] - task.scene.env_origins[:, None]
                force_z = sensor.data.net_forces_w[:, foot_sensors, 2]
                top_contacts |= (
                    alive[:, None]
                    & (foot[..., 0] >= CLIMB_END_M + 0.08)
                    & (foot[..., 0] <= COURSE_LENGTH_M - SPAWN_X_M)
                    & (foot[..., 1].abs() < COURSE_WIDTH_M / 2)
                    & (foot[..., 2] >= platform_height - 0.03)
                    & (foot[..., 2] <= platform_height + 0.12)
                    & (force_z > 20.0)
                )
            if index >= 100:
                speed_sum += robot.root_lin_vel_w[:, 0].masked_fill(~alive, 0)
                samples += alive.float()
    result = {
        "num_envs": env.num_envs,
        "seconds": trial_steps * task.step_dt,
        "command_mps": 0.42,
        "survivors": int(alive.sum().item()),
        "mean_forward_speed_mps": (speed_sum.sum() / samples.sum().clamp_min(1)).item(),
        "mean_forward_displacement_m": last_progress.mean().item(),
        "mean_peak_pelvis_rise_m": peak.mean().item(),
    }
    if stairs:
        passed_boundary = alive & (last_progress >= CLIMB_END_M)
        climbed = passed_boundary & (last_rise >= platform_height - 0.13)
        reached_top = passed_boundary & top_contacts.all(dim=1)
        result.update({
            "riser_height_m": EVAL_RISER_M,
            "risers": RISERS,
            "passed_stair_boundary": int(passed_boundary.sum().item()),
            "mean_final_pelvis_rise_m": last_rise.mean().item(),
            "climbed_all_risers": int(climbed.sum().item()),
            "both_feet_supported_on_top": int(reached_top.sum().item()),
            "top_support_env_ids": reached_top.nonzero(as_tuple=True)[0].tolist(),
        })
    return result
