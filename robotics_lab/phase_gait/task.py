"""Phase-conditioned flat-ground Go2 task for Isaac Lab 2.3.2."""

import math

import torch
from isaaclab.envs import mdp
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
FREQUENCY_HZ = 2.0
STANCE_FRACTION = 0.56
CONTACT_THRESHOLD_N = 5.0


def gait_phase(env):
    return env.episode_length_buf * (env.step_dt * FREQUENCY_HZ)


def clock(env):
    angle = gait_phase(env) * (2.0 * math.pi)
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def target_contact(env):
    cosine = (gait_phase(env) * (2.0 * math.pi)).cos()
    paired = torch.stack((cosine, -cosine, -cosine, cosine), dim=-1)
    threshold = math.cos(math.pi * STANCE_FRACTION)
    return ((paired - threshold) / 0.32 + 0.5).clamp(0.0, 1.0)


def force_magnitude(env, contact_cfg):
    sensor = env.scene.sensors[contact_cfg.name]
    return torch.linalg.vector_norm(sensor.data.net_forces_w[:, contact_cfg.body_ids], dim=-1)


def phase_tracking(env, contact_cfg, swing_emphasis=1.0):
    contact_strength = (force_magnitude(env, contact_cfg) / 15.0).clamp(0.0, 1.0)
    desired = target_contact(env)
    weight = 1.0 + (swing_emphasis - 1.0) * (1.0 - desired)
    return 1.0 - ((desired - contact_strength).abs() * weight).sum(dim=-1) / weight.sum(dim=-1)


def swing_clearance(env, feet_cfg):
    heights = env.scene[feet_cfg.name].data.body_link_pos_w[:, feet_cfg.body_ids, 2]
    height_score = ((heights - 0.025) / 0.045).clamp(0.0, 1.0)
    return ((1.0 - target_contact(env)) * height_score).mean(dim=-1)


def toe_path_tracking(env, feet_cfg):
    robot = env.scene[feet_cfg.name].data
    toe = robot.body_link_pos_w[:, feet_cfg.body_ids]
    relative_w = toe - robot.root_pos_w[:, None, :]
    yaw = yaw_quat(robot.root_quat_w)[:, None, :].expand(-1, len(FEET), -1)
    relative = quat_apply_inverse(yaw.reshape(-1, 4), relative_w.reshape(-1, 3)).reshape_as(relative_w)
    offset = toe.new_tensor([0.0, 0.5, 0.5, 0.0])
    phase = (gait_phase(env)[:, None] + offset + STANCE_FRACTION / 2.0).remainder(1.0)
    stance_progress = (phase / STANCE_FRACTION).clamp(0.0, 1.0)
    swing_progress = ((phase - STANCE_FRACTION) / (1.0 - STANCE_FRACTION)).clamp(0.0, 1.0)
    amplitude = env.command_manager.get_command("base_velocity")[:, :1] * (STANCE_FRACTION / (2.0 * FREQUENCY_HZ))
    stance_x = 1.0 - 2.0 * stance_progress
    swing_x = -(math.pi * swing_progress).cos()
    nominal_x = toe.new_tensor([0.175, 0.175, -0.245, -0.245])
    desired_x = nominal_x + amplitude * torch.where(phase < STANCE_FRACTION, stance_x, swing_x)
    desired_z = 0.024 + 0.065 * (math.pi * swing_progress).sin()
    error_x = (relative[:, :, 0] - desired_x).square() / 0.09**2
    error_z = (toe[:, :, 2] - desired_z).square() / 0.045**2
    return torch.exp(-error_x - error_z).mean(dim=-1)


def nominal_height(env, width=0.05):
    height = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-(height - 0.30).square() / width**2)


def configure(env_cfg, with_phase, evaluation=False, with_path=False, with_posture=False, height_width=0.05):
    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.75, 0.75) if evaluation else (0.55, 0.95)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.debug_vis = False

    if evaluation:
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (0.45, -2.35, 0.85)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.12)
        env_cfg.viewer.resolution = (960, 540)

    if with_phase:
        env_cfg.observations.policy.gait_clock = ObservationTermCfg(func=clock)
        feet_cfg = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)
        contact_cfg = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
        env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
        env_cfg.rewards.feet_air_time = None
        env_cfg.rewards.phase_tracking = RewardTermCfg(
            func=phase_tracking, weight=1.5, params={"contact_cfg": contact_cfg}
        )
        env_cfg.rewards.swing_clearance = RewardTermCfg(
            func=swing_clearance, weight=0.25, params={"feet_cfg": feet_cfg}
        )

        if with_path:
            env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.2
            env_cfg.rewards.phase_tracking.weight = 2.5
            env_cfg.rewards.swing_clearance = None
            env_cfg.rewards.toe_path = RewardTermCfg(
                func=toe_path_tracking, weight=2.0, params={"feet_cfg": feet_cfg}
            )

        if with_posture:
            env_cfg.rewards.phase_tracking.params["swing_emphasis"] = 1.35
            env_cfg.rewards.nominal_height = RewardTermCfg(
                func=nominal_height, weight=1.2, params={"width": height_width}
            )
            shanks = SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"])
            env_cfg.rewards.undesired_contacts = RewardTermCfg(
                func=mdp.undesired_contacts, weight=-0.45, params={"sensor_cfg": shanks, "threshold": 5.0}
            )
            env_cfg.rewards.dof_pos_limits.weight = -0.25
