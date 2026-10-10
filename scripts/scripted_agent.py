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
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

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
    ALIGN_CUBE = 1
    DESCEND_CUBE = 2
    GRASP_CUBE = 3
    LIFT_CUBE = 4
    CARRY_TO_BIN = 5
    RELEASE_CUBE = 6
    DONE = 7


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
        # Pose command is position plus a quaternion. Hover keeps the current
        # orientation. ALIGN_CUBE replaces it with an Euler angle converted below.
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
            # Read Ground Truth Positions from Simulator
            # -------------------------------------------------------------
            body_pos_w = to_torch(robot.data.body_pos_w)
            body_quat_w = to_torch(robot.data.body_quat_w)
            cube_pos_w = to_torch(direct_env.cube.data.root_pos_w)

            ee_pos = body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins
            ee_quat = body_quat_w[:, ee_body_idx]
            cube_pos = cube_pos_w - direct_env.scene.env_origins
            hand_z = torch.zeros(num_envs, 3, device=device, dtype=ee_quat.dtype)
            hand_z[:, 2] = 1.0
            finger_z = quat_apply(ee_quat, hand_z)[:, 2]
            # Hand Y is the jaw opening. In the world it must lie on X or Y.
            jaw_y = torch.zeros(num_envs, 3, device=device, dtype=ee_quat.dtype)
            jaw_y[:, 1] = 1.0
            jaw_axis = quat_apply(ee_quat, jaw_y)
            jaws_aligned = (jaw_axis[:, 0].abs() > 0.95) | (jaw_axis[:, 1].abs() > 0.95)
            grasp_offset = to_torch(direct_env.grasp_frame_offset).reshape(1, 3).to(device=device, dtype=ee_quat.dtype)
            tip_offset_w = quat_apply(ee_quat, grasp_offset.expand(num_envs, 3))
            fingertip_pos = ee_pos + tip_offset_w
            hover_pos = cube_pos.clone()
            hover_pos[:, 2] += 0.20
            # Shift the hand so the fingertips, not the palm origin, sit on the cube's X and Y.
            align_pos = hover_pos.clone()
            align_pos[:, :2] = cube_pos[:, :2] - tip_offset_w[:, :2]
            descend_pos = cube_pos - tip_offset_w

            # Gripper command buffer (+1.0 = open, -1.0 = closed)
            gripper_cmds = torch.ones((num_envs, 1), device=device)

            # -------------------------------------------------------------
            # Finite State Machine Logic (Hover through descend, later states pass)
            # -------------------------------------------------------------
            for e in range(num_envs):
                st = states[e].item()

                if st == CubeTaskState.HOVER_CUBE:
                    # Hover 20 cm above the cube (fingertip clearance ~10 cm)
                    gripper_cmds[e] = 1.0

                    dist = torch.norm(ee_pos[e] - hover_pos[e])
                    if dist < 0.03:
                        dwell_counters[e] += 1
                        if dwell_counters[e] == 30:
                            states[e] = CubeTaskState.ALIGN_CUBE
                            dwell_counters[e] = 0
                            print(f"[Env {e}] Hover settled, aligning gripper")

                elif st == CubeTaskState.ALIGN_CUBE:
                    gripper_cmds[e] = 1.0

                    tip_xy = torch.norm(fingertip_pos[e, :2] - cube_pos[e, :2])
                    if tip_xy < 0.01 and finger_z[e] < -0.95 and jaws_aligned[e]:
                        dwell_counters[e] += 1
                        if dwell_counters[e] == 20:
                            states[e] = CubeTaskState.DESCEND_CUBE
                            dwell_counters[e] = 0
                            print(f"[Env {e}] Gripper aligned, jaws on cube faces")
                    else:
                        dwell_counters[e] = 0

                elif st == CubeTaskState.DESCEND_CUBE:
                    gripper_cmds[e] = 1.0

                    dist = torch.norm(ee_pos[e] - descend_pos[e])
                    if dist < 0.02 and finger_z[e] < -0.95 and jaws_aligned[e]:
                        dwell_counters[e] += 1
                        if dwell_counters[e] == 15:
                            states[e] = CubeTaskState.GRASP_CUBE
                            dwell_counters[e] = 0
                            print(f"[Env {e}] Fingertips at the cube")
                    else:
                        dwell_counters[e] = 0

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

            max_step_m = 0.008  # ~0.48 m/s speed limit
            roll = torch.full((num_envs,), math.radians(180.0), device=device)
            pitch = torch.zeros(num_envs, device=device)
            yaw = torch.zeros(num_envs, device=device)
            down_quat = quat_from_euler_xyz(roll, pitch, yaw)
            dot = torch.sum(current_target_quat * down_quat, dim=-1, keepdim=True)
            down_quat = torch.where(dot < 0.0, -down_quat, down_quat)

            hovering = states == CubeTaskState.HOVER_CUBE
            if hovering.any():
                pos_err = hover_pos - current_target_pos
                current_target_pos[hovering] += torch.clamp(pos_err[hovering], -max_step_m, max_step_m)
                # Track the measured orientation so the pose IK only has to reach the point.
                current_target_quat[hovering] = ee_quat[hovering]

            aligning = states == CubeTaskState.ALIGN_CUBE
            if aligning.any():
                pos_err = align_pos - current_target_pos
                current_target_pos[aligning] += torch.clamp(pos_err[aligning], -max_step_m, max_step_m)
                blended = (1.0 - 0.04) * current_target_quat + 0.04 * down_quat
                blended = blended / torch.norm(blended, dim=-1, keepdim=True)
                current_target_quat[aligning] = blended[aligning]

            descending = states == CubeTaskState.DESCEND_CUBE
            if descending.any():
                pos_err = descend_pos - current_target_pos
                current_target_pos[descending] += torch.clamp(pos_err[descending], -max_step_m, max_step_m)
                current_target_quat[descending] = down_quat[descending]

            # -------------------------------------------------------------
            # Pose IK. The quaternion is only the converted Euler angle.
            # -------------------------------------------------------------
            ik_controller.set_command(torch.cat([current_target_pos, current_target_quat], dim=-1))

            # Newton geometric Jacobian for fixed-base articulation (fixed-root excluded):
            jacobi_ee_idx = ee_body_idx - 1
            jacobian = to_torch(robot.data.body_link_jacobian_w)[:, jacobi_ee_idx, :, arm_joint_indices]

            current_arm_q = to_torch(robot.data.joint_pos)[:, arm_joint_indices]
            q_des = ik_controller.compute(ee_pos, ee_quat, jacobian, current_arm_q)

            # -------------------------------------------------------------
            # Convert Joint Angles into Normalized 8D Action [-1, 1]
            # -------------------------------------------------------------
            arm_action = (q_des - default_joint_pos) / arm_action_scale.unsqueeze(0)
            arm_action = torch.clamp(arm_action, -1.0, 1.0)

            # Combine arm actions (0-6) and gripper action (7)
            actions = torch.cat([arm_action, gripper_cmds], dim=-1)

            # Step environment
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(actions)

            # Auto-reset protection: re-sync the position target if the episode resets
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
