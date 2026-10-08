# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the Franka multi-object pick-and-place environment with a zero action agent."""

import argparse
import contextlib
import sys

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

# Add argparse arguments
parser = argparse.ArgumentParser(description="Zero action agent for Franka multi pick-and-place environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Franka-Multi-Pick-Place-Direct-v0",
    help="Name of the task (default: Franka-Multi-Pick-Place-Direct-v0).",
)
parser.add_argument("--max_steps", type=int, default=100, help="Number of steps to run without a visualizer.")

# Append Isaac Lab AppLauncher CLI arguments
add_launcher_args(parser)
# Default to kit visualizer if available, or override via --viz viser / --viz none
parser.set_defaults(task="Franka-Multi-Pick-Place-Direct-v0")
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# Import custom task to register environment with Gymnasium
import franka_multi_pick_place.tasks  # noqa: F401


def main():
    """Run Zero actions agent on Franka-Multi-Pick-Place-Direct-v0."""

    torch.manual_seed(42)

    # Parse configuration via Hydra (supports preset overrides)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        # Override configuration with CLI arguments
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # Create Gym environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        print("=" * 60)
        print(f"[INFO]: Task Name: {args_cli.task}")
        print(f"[INFO]: Number of Envs: {env.unwrapped.num_envs}")
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print("=" * 60)

        # Reset environment (triggers randomized initial poses for cube, cable, bin)
        obs, _ = env.reset()

        sim = env.unwrapped.sim
        # Action is strictly zeros: 7 arm joint deltas = 0, gripper finger command = 0
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step = 0

        print("[INFO]: Starting Zero Agent stepping loop...")
        while True:
            if sim.visualizers:
                # If a visualizer (Kit or Viser) is active, keep running until the user closes it
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break

            with torch.inference_mode():
                # Step environment with zero action
                obs, rew, terminated, truncated, info = env.step(actions)

            step += 1
            if step % 50 == 0:
                print(f"[INFO]: Step {step} completed successfully.")

            # Headless smoke-test exit condition
            if not sim.visualizers and step >= args_cli.max_steps:
                print(f"[INFO]: Completed {args_cli.max_steps} headless smoke-test steps.")
                break

        # Cleanup and close simulation
        print("[INFO]: Closing simulation environment.")
        env.close()


if __name__ == "__main__":
    main()
