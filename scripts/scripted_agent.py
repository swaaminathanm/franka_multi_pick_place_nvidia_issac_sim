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
        # 1. Setup Differential IK Controller in position mode (smoothly track target xyz)
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
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

        # Read initial hand pose after reset (hand is in ready posture pointing down)
        body_pos_w = to_torch(robot.data.body_pos_w)
        body_quat_w = to_torch(robot.data.body_quat_w)

        ee_quat_down = body_quat_w[:, ee_body_idx].clone()
        current_target_pos = (body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins).clone()

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
            bin_pos_w = to_torch(direct_env.bin.data.root_pos_w)

            ee_pos = body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins
            ee_quat = body_quat_w[:, ee_body_idx]
            cube_pos = cube_pos_w - direct_env.scene.env_origins
            bin_pos = bin_pos_w - direct_env.scene.env_origins

            # -------------------------------------------------------------
            # Target: Directly on top of the cube (+10 cm above cube)
            # -------------------------------------------------------------
            target_hover_pos = cube_pos.clone()
            target_hover_pos[:, 2] += 0.10

            # Smooth interpolation towards the hover target (no dynamic jerking)
            max_step_m = 0.008  # ~0.48 m/s speed limit
            pos_err = target_hover_pos - current_target_pos
            current_target_pos += torch.clamp(pos_err, -max_step_m, max_step_m)

            # Set position command for Differential IK
            ik_controller.set_command(current_target_pos, ee_quat=ee_quat)

            # Newton geometric Jacobian for fixed-base Franka (fixed-root excluded):
            jacobi_ee_idx = ee_body_idx - 1
            jacobian = to_torch(robot.data.body_link_jacobian_w)[:, jacobi_ee_idx, :, arm_joint_indices]
            current_arm_q = to_torch(robot.data.joint_pos)[:, arm_joint_indices]

            # Solve joint targets via Newton Jacobian & DifferentialIK
            q_des = ik_controller.compute(ee_pos, ee_quat, jacobian, current_arm_q)

            # Convert to normalized 8D action in [-1, 1]
            arm_action = (q_des - default_joint_pos) / arm_action_scale.unsqueeze(0)
            arm_action = torch.clamp(arm_action, -1.0, 1.0)

            # Gripper remains open (+1.0)
            gripper_cmds = torch.ones((num_envs, 1), device=device)
            actions = torch.cat([arm_action, gripper_cmds], dim=-1)

            # Step environment
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(actions)

            # On episode reset, re-sync target position to the robot's hand position
            if terminated.any() or truncated.any():
                body_pos_w = to_torch(robot.data.body_pos_w)
                current_target_pos = (body_pos_w[:, ee_body_idx] - direct_env.scene.env_origins).clone()

            # Progress printout every 20 steps
            dist_to_hover = torch.norm(ee_pos[0] - target_hover_pos[0]).item()
            if step % 20 == 0 or dist_to_hover < 0.02:
                print(f"[Step {step:4d}] Hovering on cube | EE: ({ee_pos[0,0]:.3f}, {ee_pos[0,1]:.3f}, {ee_pos[0,2]:.3f}) | Target: ({target_hover_pos[0,0]:.3f}, {target_hover_pos[0,1]:.3f}, {target_hover_pos[0,2]:.3f}) | Dist: {dist_to_hover*100:.1f} cm")

            step += 1
            if not sim.visualizers and step >= args_cli.max_steps:
                break

        print("[INFO]: Closing simulation.")
        env.close()


if __name__ == "__main__":
    main()
