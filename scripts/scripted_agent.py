# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Heuristic Expert Agent with Differential IK for Franka Pick-and-Place (Milestone 1: Cube)."""

import argparse
import contextlib
import math
import sys
from enum import IntEnum

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
try:
    from isaaclab.app import add_launcher_args, launch_simulation
except ImportError:
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation

try:
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
except ImportError:
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul

# Import custom task to register environment with Gymnasium
import franka_multi_pick_place.tasks  # noqa: F401


def to_torch(data):
    """Helper to unwrap Warp buffers to PyTorch."""
    if hasattr(data, "torch"):
        return data.torch
    return data

class CubeTaskState(IntEnum):
    """Finite State Machine states for Milestone 1 (Cube only)."""
    HOVER_CUBE = 0
    DESCEND_CUBE = 1
    GRASP_CUBE = 2
    LIFT_CUBE = 3
    CARRY_TO_BIN = 4
    RELEASE_CUBE = 5
    DONE = 6


# Add CLI arguments
parser = argparse.ArgumentParser(description="Scripted IK Agent for Franka multi pick-and-place environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1).")
parser.add_argument("--task", type=str, default="Franka-Multi-Pick-Place-Direct-v0", help="Task name.")
parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode.")
parser.add_argument("--viser_port", type=int, default=8080, help="Viser port (default: 8080).")

add_launcher_args(parser)
parser.set_defaults(task="Franka-Multi-Pick-Place-Direct-v0")
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Run Scripted Heuristic IK Expert on Franka-Multi-Pick-Place-Direct-v0."""

    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    # Configure Viser visualizer if requested
    if getattr(args_cli, "visualizer", None) == "viser" or getattr(args_cli, "viz", None) == "viser":
        with contextlib.suppress(ImportError):
            from isaaclab_visualizers.viser import ViserVisualizerCfg
            env_cfg.sim.visualizer_cfgs = [ViserVisualizerCfg(port=args_cli.viser_port)]

    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # Create Gym environment
        env = gym.make(args_cli.task, cfg=env_cfg)
        direct_env = env.unwrapped
        device = direct_env.device
        num_envs = direct_env.num_envs

        print("=" * 60)
        print(f"[INFO]: Initializing Scripted IK Agent for {num_envs} envs on {device}")
        # 1. Setup Differential IK Controller in pose mode (6-DOF tracking: pos + quat)
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=num_envs, device=device)

        # Cache robot constants
        robot = direct_env.robot
        arm_joint_indices = direct_env.arm_joint_indices
        arm_action_scale = to_torch(direct_env.arm_action_scale)  # shape: [7]
        default_joint_pos = to_torch(direct_env.robot_default_joint_pos)[:, arm_joint_indices]  # [num_envs, 7]
        ee_body_idx = direct_env.ee_body_idx
        # Joint 7 spins about the fingertip axis once the hand points down.
        wrist_arm_idx = [robot.joint_names[i] for i in arm_joint_indices].index("panda_joint7")

        # Reset environment first so physics states and buffers are fully initialized
        obs, _ = env.reset()
        sim = direct_env.sim
        step = 0

        # Read initial hand pose after reset
        body_pos_w = to_torch(robot.data.body_pos_w)
        body_quat_w = to_torch(robot.data.body_quat_w)

        current_target_pos = (body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins).clone()
        current_target_quat = body_quat_w[:, ee_body_idx].clone()

        # FSM State & dwell counters per environment
        states = torch.zeros(num_envs, dtype=torch.long, device=device)
        dwell_counters = torch.zeros(num_envs, dtype=torch.long, device=device)

        print("[INFO]: Starting Scripted Agent stepping loop...")
        while True:
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break

            # -------------------------------------------------------------
            # 2. Read Ground Truth Positions from Simulator
            # -------------------------------------------------------------
            body_pos_w = to_torch(robot.data.body_pos_w)
            body_quat_w = to_torch(robot.data.body_quat_w)
            cube_pos_w = to_torch(direct_env.cube.data.root_pos_w)
            cube_quat_w = to_torch(direct_env.cube.data.root_quat_w)
            bin_pos_w = to_torch(direct_env.bin.data.root_pos_w)

            ee_pos = body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins
            ee_quat = body_quat_w[:, ee_body_idx]
            cube_pos = cube_pos_w - direct_env.scene.env_origins
            bin_pos = bin_pos_w - direct_env.scene.env_origins

            # Gripper command buffer (+1.0 = open, -1.0 = closed)
            gripper_cmds = torch.ones((num_envs, 1), device=device)
            desired_pos = ee_pos.clone()

            # -------------------------------------------------------------
            # 3. Finite State Machine Logic (Hover implemented, others pass)
            # -------------------------------------------------------------
            for e in range(num_envs):
                st = states[e].item()

                if st == CubeTaskState.HOVER_CUBE:
                    # Hover 20 cm above the cube (fingertip clearance ~10 cm)
                    desired_pos[e] = cube_pos[e].clone()
                    desired_pos[e, 2] += 0.20
                    gripper_cmds[e] = 1.0

                    dist = torch.norm(ee_pos[e] - desired_pos[e])
                    if dist < 0.03:
                        dwell_counters[e] += 1
                        if dwell_counters[e] % 30 == 1:
                            finger_axis = quat_apply(
                                ee_quat[e],
                                torch.tensor([0.0, 0.0, 1.0], device=device, dtype=ee_quat.dtype),
                            )
                            print(
                                f"[Env {e}] Stable hover on top of cube | Z={ee_pos[e, 2]:.3f}m "
                                f"(dist: {dist*100:.1f}cm) finger axis=({finger_axis[0]:.2f}, "
                                f"{finger_axis[1]:.2f}, {finger_axis[2]:.2f})"
                            )

                elif st == CubeTaskState.DESCEND_CUBE:
                    pass

                elif st == CubeTaskState.GRASP_CUBE:
                    pass

                elif st == CubeTaskState.LIFT_CUBE:
                    pass

                elif st == CubeTaskState.CARRY_TO_BIN:
                    pass

                elif st == CubeTaskState.RELEASE_CUBE:
                    pass

                elif st == CubeTaskState.DONE:
                    pass

            # -------------------------------------------------------------
            # 4. Smooth Trajectory & Orientation Interpolation
            # -------------------------------------------------------------
            max_step_m = 0.008  # ~0.48 m/s speed limit
            pos_err = desired_pos - current_target_pos
            current_target_pos += torch.clamp(pos_err, -max_step_m, max_step_m)

            # panda_hand +Z points out through the fingertips, and the fingers open
            # along +Y. A half turn about X, (w, x, y, z) = (0, 1, 0, 0), points the
            # fingertips down and lines the jaws up with the world axes. Yaw is the
            # cube's own yaw, not the direction from the base to the cube, so the
            # jaws stay parallel to the cube faces.
            # q = q_z(cube_yaw) * q_x(pi) = (0, cos(yaw/2), sin(yaw/2), 0).
            _, _, yaw = euler_xyz_from_quat(cube_quat_w)
            zeros = torch.zeros_like(yaw)
            down_quat = quat_from_euler_xyz(torch.full_like(yaw, math.pi), zeros, zeros)
            yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)
            target_down_quat = quat_mul(yaw_quat, down_quat)

            # Shortest-path sign alignment and smooth NLERP
            dot = torch.sum(current_target_quat * target_down_quat, dim=-1, keepdim=True)
            aligned_down_quat = torch.where(dot < 0.0, -target_down_quat, target_down_quat)
            current_target_quat = (1.0 - 0.04) * current_target_quat + 0.04 * aligned_down_quat
            current_target_quat = current_target_quat / torch.norm(current_target_quat, dim=-1, keepdim=True)

            # -------------------------------------------------------------
            # 5. Inverse Kinematics (IK) Calculation via Newton View & DifferentialIKController
            # -------------------------------------------------------------
            target_pose_7d = torch.cat([current_target_pos, current_target_quat], dim=-1)
            ik_controller.set_command(target_pose_7d)

            # Newton geometric Jacobian for fixed-base articulation (fixed-root excluded):
            jacobi_ee_idx = ee_body_idx - 1
            jacobian = to_torch(robot.data.body_link_jacobian_w)[:, jacobi_ee_idx, :, arm_joint_indices]

            current_arm_q = to_torch(robot.data.joint_pos)[:, arm_joint_indices]

            q_des = ik_controller.compute(ee_pos, ee_quat, jacobian, current_arm_q)

            # The arm IK holds the hover and the fingers-down pitch. Jaw alignment
            # is a pure spin about the fingertip axis, so it goes on joint 7 only.
            # With the fingers down, increasing joint 7 decreases the jaw's yaw.
            hand_y = torch.zeros(num_envs, 3, device=device, dtype=ee_quat.dtype)
            hand_y[:, 1] = 1.0
            hand_z = torch.zeros(num_envs, 3, device=device, dtype=ee_quat.dtype)
            hand_z[:, 2] = 1.0
            cube_x = torch.zeros(num_envs, 3, device=device, dtype=ee_quat.dtype)
            cube_x[:, 0] = 1.0
            opening = quat_apply(ee_quat, hand_y)
            finger_z = quat_apply(ee_quat, hand_z)[:, 2]
            cube_x_w = quat_apply(cube_quat_w, cube_x)
            open_yaw = torch.atan2(opening[:, 1], opening[:, 0])
            cube_yaw = torch.atan2(cube_x_w[:, 1], cube_x_w[:, 0])
            delta = torch.atan2(torch.sin(open_yaw - cube_yaw), torch.cos(open_yaw - cube_yaw))
            quarter = math.pi / 2.0
            nearest = torch.round(delta / quarter) * quarter
            yaw_err = torch.atan2(torch.sin(nearest - delta), torch.cos(nearest - delta))
            fingers_down = finger_z < -0.5
            q_des = q_des.clone()
            q_des[:, wrist_arm_idx] = torch.where(
                fingers_down,
                current_arm_q[:, wrist_arm_idx] - yaw_err,
                q_des[:, wrist_arm_idx],
            )

            # -------------------------------------------------------------
            # 6. Convert Joint Angles into Normalized 8D Action [-1, 1]
            # -------------------------------------------------------------
            arm_action = (q_des - default_joint_pos) / arm_action_scale.unsqueeze(0)
            arm_action = torch.clamp(arm_action, -1.0, 1.0)

            # Combine arm actions (0-6) and gripper action (7)
            actions = torch.cat([arm_action, gripper_cmds], dim=-1)

            # Step environment
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(actions)

            # Auto-reset protection: re-sync target pose if episode resets
            if terminated.any() or truncated.any():
                body_pos_w = to_torch(robot.data.body_pos_w)
                body_quat_w = to_torch(robot.data.body_quat_w)
                current_target_pos = (body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins).clone()
                current_target_quat = body_quat_w[:, ee_body_idx].clone()
                states[:] = CubeTaskState.HOVER_CUBE
                dwell_counters[:] = 0

            step += 1
            if not sim.visualizers and step >= args_cli.max_steps:
                break

        print("[INFO]: Closing simulation.")
        env.close()


if __name__ == "__main__":
    main()
