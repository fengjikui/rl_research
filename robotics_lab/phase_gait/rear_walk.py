"""Go2 rear-leg standing, stepping, and rise curriculum for flat-ground simulation."""

import math

import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from rear_stand import HEIGHT_M, PITCH_RAD, configure_rear_stand
from task import FEET, force_magnitude

HIND_THIGH_UPRIGHT = 2.35
HIND_THIGH_FLOOR = 1.02
CALF_ANGLE = -1.53
THIGH_LENGTH_M = 0.213
HIP_OFFSET_M = 0.1934
FOOT_CENTER_M = 0.028
STEP_HZ = 2.0


def uniform(shape, low, high, device):
    return torch.rand(shape, device=device) * (high - low) + low


def reset_rear_pose(env, env_ids, start, fixed_route=False):
    robot = env.scene["robot"]
    if not hasattr(env, "rear_thigh_joints"):
        env.rear_thigh_joints, _ = robot.find_joints(["RL_thigh_joint", "RR_thigh_joint"])
    n = len(env_ids)
    device = env.device
    if start == "floor":
        pitch_fraction = torch.zeros(n, device=device)
    else:
        pitch_fraction = uniform(n, 0.96, 1.0, device)
        if start == "curriculum":
            difficulty = getattr(env, "rear_rise_difficulty", 0.0)
            practice_rise = torch.rand(n, device=device) < 0.65 * difficulty
            from_floor = torch.rand(n, device=device) < 0.60 * difficulty
            partial_fraction = 1.0 - difficulty * uniform(n, 0.2, 1.0, device)
            hardest = torch.where(from_floor, torch.full_like(partial_fraction, 1.0 - difficulty), partial_fraction)
            pitch_fraction = torch.where(practice_rise, hardest, pitch_fraction)

    pitch = PITCH_RAD * pitch_fraction
    root = robot.data.default_root_state[env_ids].clone()
    q = robot.data.default_joint_pos[env_ids].clone()
    neutral = HIND_THIGH_FLOOR + (HIND_THIGH_UPRIGHT - HIND_THIGH_FLOOR) * pitch_fraction
    q[:, env.rear_thigh_joints] = neutral[:, None]
    q += uniform(q.shape, -0.025, 0.025, device)

    # Compute height from the two rear leg chains to start close to physical contact.
    calf = neutral + CALF_ANGLE
    foot_x = -HIP_OFFSET_M - THIGH_LENGTH_M * (neutral.sin() + calf.sin())
    foot_z = -THIGH_LENGTH_M * (neutral.cos() + calf.cos())
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, :2] += uniform((n, 2), -0.12, 0.12, device)
    root[:, 2] += FOOT_CENTER_M + pitch.sin() * foot_x - pitch.cos() * foot_z
    roll = uniform(n, -0.025, 0.025, device)
    yaw = uniform(n, -0.12, 0.12, device)
    root[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    if fixed_route:
        if not hasattr(env, "rear_route_forward"):
            env.rear_route_forward = torch.zeros((env.num_envs, 2), device=device)
            env.rear_route_forward[:, 0] = 1.0
        env.rear_route_forward[env_ids] = torch.stack((yaw.cos(), yaw.sin()), dim=1)
    root[:, 7:13] = uniform((n, 6), -0.015, 0.015, device)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

def heading_axes(data):
    # The body x-axis points up. The negative body z-axis points forward in the plane.
    body_forward = quat_apply(data.root_quat_w, data.root_pos_w.new_tensor([0.0, 0.0, -1.0]).expand_as(data.root_pos_w))
    body_x = quat_apply(data.root_quat_w, data.root_pos_w.new_tensor([1.0, 0.0, 0.0]).expand_as(data.root_pos_w))
    direction = torch.where((body_forward[:, 2].abs() < 0.9)[:, None], body_forward, body_x)
    forward = torch.nn.functional.normalize(direction[:, :2], dim=-1)
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    return forward, left


def horizontal_velocity(env):
    data = env.scene["robot"].data
    forward, left = motion_axes(env)
    speed = data.root_lin_vel_w[:, :2]
    return torch.stack(((speed * forward).sum(dim=1), (speed * left).sum(dim=1)), dim=1)


def motion_axes(env):
    if hasattr(env, "rear_route_forward"):
        forward = env.rear_route_forward
        return forward, torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    return heading_axes(env.scene["robot"].data)


def heading_error(env):
    actual, _ = heading_axes(env.scene["robot"].data)
    target, left = motion_axes(env)
    return torch.stack(((actual * target).sum(dim=1) - 1.0, (actual * left).sum(dim=1)), dim=1)


def route_heading_quality(env):
    quality = (1.0 + 0.5 * heading_error(env)[:, 0]).clamp(0.0, 1.0)
    return upright_quality(env).square() * quality.square()


def motion_phase(env):
    t = env.episode_length_buf * (env.step_dt * 2.0 * math.pi * STEP_HZ)
    return torch.stack((t.sin(), t.cos()), dim=1)


def upright_quality(env):
    gravity = env.scene["robot"].data.projected_gravity_b
    target = gravity.new_tensor([math.sin(PITCH_RAD), 0.0, -math.cos(PITCH_RAD)])
    return (1.0 - 0.5 * (gravity - target).square().sum(dim=1)).clamp_min(0.0)


def height_quality(env):
    z = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((z - HEIGHT_M) / 0.14).square())


def motion_quality(env):
    data = env.scene["robot"].data
    command = env.command_manager.get_command("base_velocity")
    velocity = horizontal_velocity(env)
    tracking = torch.exp(-(velocity - command[:, :2]).square().sum(dim=1) / 0.27**2)
    yaw_tracking = torch.exp(-(data.root_ang_vel_w[:, 2] - command[:, 2]).square() / 0.45**2)
    balance = torch.exp(-data.root_lin_vel_w[:, 2].square() / 0.32**2)
    return upright_quality(env).pow(4) * (tracking + 0.20 * yaw_tracking + 0.20 * balance)


def support_quality(env, contact_cfg, feet_cfg):
    data = env.scene["robot"].data
    forces = force_magnitude(env, contact_cfg)
    rear_load = (forces[:, 2:] / 25.0).clamp(0.0, 1.0)
    support = rear_load.amax(dim=1)
    front_clear = torch.exp(-forces[:, :2].sum(dim=1) / 25.0)
    rear_pos = data.body_link_pos_w[:, feet_cfg.body_ids[2:], :2]
    weight = rear_load + 0.04
    center = (rear_pos * weight[:, :, None]).sum(dim=1) / weight.sum(dim=1, keepdim=True)
    com_offset = data.root_com_pos_w[:, :2] - center
    forward, left = motion_axes(env)
    com_error = torch.stack(((com_offset * forward).sum(dim=1), (com_offset * left).sum(dim=1)), dim=1)
    velocity_error = env.command_manager.get_command("base_velocity")[:, :2] - horizontal_velocity(env)
    balance = torch.exp(-(com_error - 0.13 * velocity_error).square().sum(dim=1) / 0.15**2)
    return upright_quality(env).square() * front_clear * support * (0.7 + 0.3 * balance)


def gait_contact(env):
    cosine = motion_phase(env)[:, 1]
    return (0.5 + torch.stack((cosine, -cosine), dim=1) / 0.65).clamp(0.0, 1.0)


def stepping_quality(env, contact_cfg, swing_emphasis=1.0):
    cmd = env.command_manager.get_command("base_velocity")
    moving = ((cmd[:, :2].norm(dim=1) + 0.20 * cmd[:, 2].abs() - 0.04) / 0.15).clamp(0.0, 1.0)
    expected = gait_contact(env)
    actual = (force_magnitude(env, contact_cfg)[:, 2:] / 18.0).clamp(0.0, 1.0)
    weight = 1.0 + (swing_emphasis - 1.0) * (1.0 - expected)
    score = 1.0 - ((expected - actual).abs() * weight).sum(dim=1) / weight.sum(dim=1)
    return moving * upright_quality(env).pow(4) * score


def signed_motion(env):
    command = env.command_manager.get_command("base_velocity")[:, :2]
    velocity = horizontal_velocity(env)
    speed = command.norm(dim=1)
    moving = ((speed - 0.04) / 0.14).clamp(0.0, 1.0)
    ratio = (velocity * command).sum(dim=1) / (speed.square() + 0.012)
    return moving * upright_quality(env).pow(4) * ratio.clamp(-1.0, 1.0)


def yaw_error_cost(env):
    command = env.command_manager.get_command("base_velocity")[:, 2]
    yaw = env.scene["robot"].data.root_ang_vel_w[:, 2]
    return (yaw - command).abs().clamp_max(2.0)


def route_yaw_cost(env):
    yaw = env.scene["robot"].data.root_ang_vel_w[:, 2]
    return yaw.square().clamp_max(36.0)


def rear_foot_path(env, feet_cfg):
    data = env.scene["robot"].data
    command = env.command_manager.get_command("base_velocity")[:, :2]
    speed = command.norm(dim=1)
    moving = ((speed - 0.035) / 0.14).clamp(0.0, 1.0)
    foot = data.body_link_pos_w[:, feet_cfg.body_ids[2:]]
    rel = foot[:, :, :2] - data.root_pos_w[:, None, :2]
    forward, left = motion_axes(env)
    axis = torch.nn.functional.normalize(command[:, :1] * forward + command[:, 1:] * left, dim=1)
    displacement = (rel * axis[:, None, :]).sum(dim=2)
    duty = 0.57
    gait = env.episode_length_buf[:, None] * (env.step_dt * STEP_HZ)
    phase = (gait + foot.new_tensor([0.0, 0.5]) + duty / 2).remainder(1.0)
    stance = (phase / duty).clamp(0.0, 1.0)
    swing = ((phase - duty) / (1.0 - duty)).clamp(0.0, 1.0)
    path = torch.where(phase < duty, 1.0 - 2.0 * stance, -(math.pi * swing).cos())
    target_x = path * (speed[:, None] * duty / (2 * STEP_HZ)).clamp_max(0.12)
    floor = env.scene.env_origins[:, 2:3]
    target_z = floor + 0.027 + 0.065 * (math.pi * swing).sin()
    error = ((displacement - target_x) / 0.087).square() + ((foot[:, :, 2] - target_z) / 0.040).square()
    return moving * upright_quality(env).pow(4) * torch.exp(-error).mean(dim=1)


def lift_rear_swing_foot(env, contact_cfg, feet_cfg):
    height = env.scene["robot"].data.body_link_pos_w[:, feet_cfg.body_ids[2:], 2]
    height = height - env.scene.env_origins[:, 2:3]
    clearance = ((height - 0.024) / 0.052).clamp(0.0, 1.0)
    unloaded = (1.0 - force_magnitude(env, contact_cfg)[:, 2:] / 12.0).clamp(0.0, 1.0)
    speed = env.command_manager.get_command("base_velocity")[:, :2].norm(dim=1)
    moving = ((speed - 0.04) / 0.14).clamp(0.0, 1.0)
    return moving * upright_quality(env).pow(4) * ((1.0 - gait_contact(env)) * clearance * unloaded).sum(dim=1)


def rear_stance_slip(env, contact_cfg, feet_cfg):
    velocity = env.scene["robot"].data.body_link_lin_vel_w[:, feet_cfg.body_ids[2:], :2].norm(dim=2)
    load = (force_magnitude(env, contact_cfg)[:, 2:] / 25.0).clamp(0.0, 1.0)
    return (velocity.clamp_max(1.2) * load).mean(dim=1)


def tipped_after_grace(env):
    data = env.scene["robot"].data
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height < 0.14) | ((env.episode_length_buf * env.step_dt > 2.0) & (upright_quality(env) < 0.58))


def configure_rear_walk(
    cfg, start, evaluation=False, command=(0.0, 0.0, 0.0), trial_seconds=20.0,
    motion_practice=False, route_practice=False, swing_practice=False,
):
    configure_rear_stand(cfg, evaluation=evaluation)
    cfg.episode_length_s = trial_seconds
    cfg.observations.policy.rear_planar_velocity = ObservationTermCfg(func=horizontal_velocity)
    cfg.observations.policy.rear_clock = ObservationTermCfg(func=motion_phase)
    cfg.observations.policy.enable_corruption = not evaluation
    cfg.actions.joint_pos.scale = 0.55
    cfg.events.reset_base.func = reset_rear_pose
    cfg.events.reset_base.params = {"start": start, "fixed_route": route_practice}
    cfg.events.reset_joints = None
    cfg.commands.base_velocity.resampling_time_range = (trial_seconds, trial_seconds) if evaluation else (5.0, 9.0)
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = (command[0], command[0])
    ranges.lin_vel_y = (command[1], command[1])
    ranges.ang_vel_z = (command[2], command[2])

    feet = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)
    contacts = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
    legs = SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"])
    cfg.rewards.rear_upright = RewardTermCfg(func=upright_quality, weight=3.0)
    cfg.rewards.rear_hold = RewardTermCfg(func=height_quality, weight=1.6)
    cfg.rewards.rear_motion = RewardTermCfg(func=motion_quality, weight=2.3)
    cfg.rewards.rear_support = RewardTermCfg(
        func=support_quality, weight=1.4, params={"contact_cfg": contacts, "feet_cfg": feet}
    )
    cfg.rewards.rear_phase = RewardTermCfg(func=stepping_quality, weight=0.6, params={"contact_cfg": contacts})
    cfg.rewards.undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts, weight=-0.55, params={"sensor_cfg": legs, "threshold": 5.0}
    )
    cfg.rewards.lin_vel_z_l2.weight = -0.35
    cfg.terminations.rear_tipped = TerminationTermCfg(func=tipped_after_grace)
    if motion_practice or route_practice:
        cfg.rewards.signed_motion = RewardTermCfg(func=signed_motion, weight=1.4)
        cfg.rewards.yaw_error_cost = RewardTermCfg(func=yaw_error_cost, weight=-0.65)
        cfg.rewards.rear_foot_path = RewardTermCfg(func=rear_foot_path, weight=1.25, params={"feet_cfg": feet})
    if route_practice:
        cfg.observations.policy.rear_heading_error = ObservationTermCfg(func=heading_error)
        cfg.rewards.route_heading = RewardTermCfg(func=route_heading_quality, weight=1.7)
        cfg.rewards.yaw_error_cost = RewardTermCfg(func=route_yaw_cost, weight=-0.22)
    if swing_practice:
        cfg.rewards.rear_phase.weight = 1.1
        cfg.rewards.rear_phase.params["swing_emphasis"] = 2.0
        limb_params = {"contact_cfg": contacts, "feet_cfg": feet}
        cfg.rewards.rear_lift = RewardTermCfg(func=lift_rear_swing_foot, weight=1.0, params=limb_params)
        cfg.rewards.rear_slip = RewardTermCfg(func=rear_stance_slip, weight=-0.18, params=limb_params)
    if evaluation:
        cfg.viewer.eye = (-1.05, -2.35, 1.05)
        cfg.viewer.lookat = (0.15, 0.0, -0.12)
