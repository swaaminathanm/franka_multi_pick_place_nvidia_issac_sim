# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments registered by franka_multi_pick_place.
"""

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
# parse the arguments
args_cli = parser.parse_args()

import gymnasium as gym
from prettytable import PrettyTable

import franka_multi_pick_place.tasks  # noqa: F401


def main():
    """Print all environments registered in `franka_multi_pick_place` extension."""
    task_specs = [
        spec
        for spec in gym.registry.values()
        if "Franka-Multi-Pick-Place" in spec.id and (args_cli.keyword is None or args_cli.keyword in spec.id)
    ]

    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Multi Pick and Place Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    for index, spec in enumerate(task_specs):
        cfg_entry = spec.kwargs.get("env_cfg_entry_point", "N/A") if hasattr(spec, "kwargs") and spec.kwargs else "N/A"
        table.add_row([index + 1, spec.id, spec.entry_point, cfg_entry])

    print(table)


if __name__ == "__main__":
    main()
