# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Heuristic Expert Agent with Differential IK for Franka Pick-and-Place (Milestone 1: Cube)."""

import argparse
import contextlib
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

# Import custom task to register environment with Gymnasium
import franka_multi_pick_place.tasks  # noqa: F401


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
        print("=" * 60)

        # 1. Setup Differential IK Controller (Damped Least Squares)
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=num_envs, device=device)

        # Cache robot constants
        robot = direct_env.robot
        arm_joint_indices = direct_env.arm_joint_indices
        arm_action_scale = direct_env.arm_action_scale  # shape: [7]
        default_joint_pos = direct_env.robot_default_joint_pos[:, arm_joint_indices]  # [num_envs, 7]
        ee_body_idx = direct_env.ee_body_idx

        # Fixed gripper pointing straight down (read from default home pose)
        ee_quat_down = robot.data.default_body_quat_w[:, ee_body_idx].clone()

        # FSM State & dwell counters per environment
        states = torch.zeros(num_envs, dtype=torch.long, device=device)
        dwell_counters = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Reset environment
        obs, _ = env.reset()
        sim = direct_env.sim
        step = 0

        # Current interpolated target position (starts at initial hand position)
        current_target_pos = (robot.data.body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins).clone()

        print("[INFO]: Starting Scripted Agent stepping loop...")
        while True:
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break

            # -------------------------------------------------------------
            # 2. Read Ground Truth Positions from Simulator
            # -------------------------------------------------------------
            ee_pos = robot.data.body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins
            ee_quat = robot.data.body_quat_w[:, ee_body_idx]
            cube_pos = direct_env.cube.data.root_pos_w - direct_env.scene.env_origins
            bin_pos = direct_env.bin.data.root_pos_w - direct_env.scene.env_origins

            # Gripper command buffer (+1.0 = open, -1.0 = closed)
            gripper_cmds = torch.ones((num_envs, 1), device=device)
            desired_pos = ee_pos.clone()

            # -------------------------------------------------------------
            # 3. Finite State Machine Logic
            # -------------------------------------------------------------
            for e in range(num_envs):
                st = states[e].item()

                if st == CubeTaskState.HOVER_CUBE:
                    # Hover 12 cm above the cube with gripper open
                    desired_pos[e] = cube_pos[e].clone()
                    desired_pos[e, 2] += 0.12
                    gripper_cmds[e] = 1.0

                    if torch.norm(ee_pos[e] - desired_pos[e]) < 0.02:
                        states[e] = CubeTaskState.DESCEND_CUBE
                        print(f"[Env {e}] Arrived at Hover -> DESCENDING")

                elif st == CubeTaskState.DESCEND_CUBE:
                    # Lower down to grasp height (z ≈ 0.035 m)
                    desired_pos[e] = cube_pos[e].clone()
                    desired_pos[e, 2] = 0.035
                    gripper_cmds[e] = 1.0

                    if torch.norm(ee_pos[e] - desired_pos[e]) < 0.015:
                        states[e] = CubeTaskState.GRASP_CUBE
                        dwell_counters[e] = 0
                        print(f"[Env {e}] At grasp height -> CLOSING GRIPPER")

                elif st == CubeTaskState.GRASP_CUBE:
                    # Hold position and clamp gripper for 20 steps
                    desired_pos[e] = cube_pos[e].clone()
                    desired_pos[e, 2] = 0.035
                    gripper_cmds[e] = -1.0
                    dwell_counters[e] += 1

                    if dwell_counters[e] >= 20:
                        states[e] = CubeTaskState.LIFT_CUBE
                        print(f"[Env {e}] Grasp tight -> LIFTING")

                elif st == CubeTaskState.LIFT_CUBE:
                    # Lift high to clear the bin lip
                    desired_pos[e] = cube_pos[e].clone()
                    desired_pos[e, 2] = 0.32
                    gripper_cmds[e] = -1.0

                    if ee_pos[e, 2] >= 0.28:
                        states[e] = CubeTaskState.CARRY_TO_BIN
                        print(f"[Env {e}] Lifted -> CARRYING TO BIN")

                elif st == CubeTaskState.CARRY_TO_BIN:
                    # Travel horizontally to bin center
                    desired_pos[e] = bin_pos[e].clone()
                    desired_pos[e, 2] = 0.32
                    gripper_cmds[e] = -1.0

                    dist_to_bin = torch.norm(ee_pos[e, :2] - bin_pos[e, :2])
                    if dist_to_bin < 0.04:
                        states[e] = CubeTaskState.RELEASE_CUBE
                        dwell_counters[e] = 0
                        print(f"[Env {e}] Above bin -> RELEASING CUBE")

                elif st == CubeTaskState.RELEASE_CUBE:
                    # Open gripper and hold for 20 steps to let cube fall
                    desired_pos[e] = bin_pos[e].clone()
                    desired_pos[e, 2] = 0.32
                    gripper_cmds[e] = 1.0
                    dwell_counters[e] += 1

                    if dwell_counters[e] >= 20:
                        states[e] = CubeTaskState.DONE
                        print(f"[Env {e}] SUCCESS! Cube successfully dropped in bin.")

                elif st == CubeTaskState.DONE:
                    # Hold above bin
                    desired_pos[e] = bin_pos[e].clone()
                    desired_pos[e, 2] = 0.32
                    gripper_cmds[e] = 1.0

            # -------------------------------------------------------------
            # 4. Smooth Trajectory Interpolation (No violent jerking)
            # -------------------------------------------------------------
            max_step_m = 0.008  # ~0.48 m/s speed limit
            pos_err = desired_pos - current_target_pos
            current_target_pos += torch.clamp(pos_err, -max_step_m, max_step_m)

            # -------------------------------------------------------------
            # 5. Inverse Kinematics (IK) Calculation
            # -------------------------------------------------------------
            target_pose_7d = torch.cat([current_target_pos, ee_quat_down], dim=-1)
            ik_controller.set_command(target_pose_7d)

            # Retrieve Jacobian for panda_hand
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm_joint_indices]
            current_arm_q = robot.data.joint_pos[:, arm_joint_indices]

            # Solve desired joint angles
            q_des = ik_controller.compute(ee_pos, ee_quat, jacobian, current_arm_q)

            # -------------------------------------------------------------
            # 6. Convert Joint Angles into Normalized 8D Action [-1, 1]
            # -------------------------------------------------------------
            # Formula: a_i = (q_des - q_default) / action_scale
            arm_action = (q_des - default_joint_pos) / arm_action_scale.unsqueeze(0)
            arm_action = torch.clamp(arm_action, -1.0, 1.0)

            # Combine arm actions (0-6) and gripper action (7)
            actions = torch.cat([arm_action, gripper_cmds], dim=-1)

            # Step environment
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(actions)

            step += 1
            if not sim.visualizers and step >= args_cli.max_steps:
                break

        print("[INFO]: Closing simulation.")
        env.close()


if __name__ == "__main__":
    main()
