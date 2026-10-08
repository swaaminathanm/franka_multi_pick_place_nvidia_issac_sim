# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hollow bin: five kinematic boxes created and moved together.

``create_hollow_bin`` builds the boxes around one table contact point.
``place_bin`` moves every box when that point changes at reset.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

BIN_COLOR = (0.10, 0.45, 0.85)

# Inner opening is 0.24 x 0.24 m, larger than the 0.20 m cable.
# Walls are 0.02 m thick, so the floor is 0.28 m and each wall center
# sits 0.13 m out from the table contact point (0.12 m opening + 0.01 m half wall).
# name, full size (x, y, z), center relative to the table contact point
BIN_PARTS = (
    ("bin", (0.28, 0.28, 0.01), (0.0, 0.0, 0.005)),
    ("bin_wall_left", (0.28, 0.02, 0.08), (0.0, -0.13, 0.05)),
    ("bin_wall_right", (0.28, 0.02, 0.08), (0.0, 0.13, 0.05)),
    ("bin_wall_front", (0.02, 0.24, 0.08), (0.13, 0.0, 0.05)),
    ("bin_wall_back", (0.02, 0.24, 0.08), (-0.13, 0.0, 0.05)),
)


def bin_side(prim_name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> RigidObjectCfg:
    """One side of the bin: the floor or a wall."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(pos), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BIN_COLOR),
        ),
    )


def create_hollow_bin(origin: tuple[float, float, float]) -> dict[str, RigidObjectCfg]:
    """Build the floor and four walls around one point on the table.

    ``origin`` is the point under the middle of the bin. Each piece's center is
    that point plus the offset stored in ``BIN_PARTS``.
    """
    ox, oy, oz = origin
    return {
        name: bin_side(name, size, (ox + local[0], oy + local[1], oz + local[2]))
        for name, size, local in BIN_PARTS
    }


def spawned_bin_parts(scene):
    """Return the live floor and walls, in the same order as ``BIN_PARTS``."""
    return [scene[name] for name, _size, _local in BIN_PARTS]


def place_bin(parts, default_poses, env_origins, env_ids, x, y) -> None:
    """Move the whole bin so its table contact point is ``(x, y)``.

    ``parts`` and ``default_poses`` follow the order of ``BIN_PARTS``.
    ``x`` and ``y`` are one position. Each piece adds its own center from ``BIN_PARTS``.
    """
    origins = env_origins[env_ids]
    for part, default_pose, (_name, _size, local) in zip(parts, default_poses, BIN_PARTS):
        pose = default_pose[env_ids].clone()
        pose[:, 0] = x + local[0]
        pose[:, 1] = y + local[1]
        pose[:, 2] = local[2]
        pose[:, :3] += origins
        velocity = torch.zeros_like(pose[:, :6])
        part.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        part.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


HOLLOW_BIN = create_hollow_bin((0.35, 0.30, 0.0))
