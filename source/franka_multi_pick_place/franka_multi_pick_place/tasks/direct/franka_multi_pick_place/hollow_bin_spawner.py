# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hollow bin made of five Isaac Lab box primitives.

Viser draws a USDA cube or mesh under this rigid body as a collapsed pyramid.
CuboidCfg is a box Newton already understands, so each wall keeps its size.
"""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass


@clone
def spawn_hollow_bin(
    prim_path: str,
    cfg: HollowBinCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn one kinematic body with a floor and four walls.

    The ``@clone`` decorator expands ``{ENV_REGEX_NS}/bin`` once per environment.
    """
    from pxr import Gf, UsdGeom, UsdPhysics  # noqa: PLC0415
    import omni.usd  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()

    xform_ops: list[str] = []
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform_ops.append("xformOp:translate")
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))
        xform_ops.append("xformOp:orient")
    if xform_ops:
        xform.GetXformOpOrderAttr().Set(xform_ops)

    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.GetKinematicEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(cfg.mass)

    half_x = cfg.outer_size_x / 2.0
    half_y = cfg.outer_size_y / 2.0
    floor_thickness = cfg.floor_thickness
    wall_thickness = cfg.wall_thickness
    wall_height = cfg.wall_height
    inner_y = cfg.outer_size_y - 2.0 * wall_thickness

    # size is the full box. The third number in the center is the box center, not a corner.
    pieces = [
        ("floor", (cfg.outer_size_x, cfg.outer_size_y, floor_thickness), (0.0, 0.0, floor_thickness / 2.0), cfg.floor_color),
        ("wall_left", (cfg.outer_size_x, wall_thickness, wall_height), (0.0, -(half_y - wall_thickness / 2.0), floor_thickness + wall_height / 2.0), cfg.wall_color),
        ("wall_right", (cfg.outer_size_x, wall_thickness, wall_height), (0.0, +(half_y - wall_thickness / 2.0), floor_thickness + wall_height / 2.0), cfg.wall_color),
        ("wall_front", (wall_thickness, inner_y, wall_height), (+(half_x - wall_thickness / 2.0), 0.0, floor_thickness + wall_height / 2.0), cfg.wall_color),
        ("wall_back", (wall_thickness, inner_y, wall_height), (-(half_x - wall_thickness / 2.0), 0.0, floor_thickness + wall_height / 2.0), cfg.wall_color),
    ]

    for name, size, center, color in pieces:
        child = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        child.func(f"{prim_path}/{name}", child, translation=center)

    return prim


@configclass
class HollowBinCfg(SpawnerCfg):
    """Five-box hollow bin. Defaults match assets/bin.usda: 20 cm footprint, 8 cm walls."""

    func: Callable = spawn_hollow_bin

    outer_size_x: float = 0.20
    outer_size_y: float = 0.20
    wall_height: float = 0.08
    wall_thickness: float = 0.02
    floor_thickness: float = 0.01
    mass: float = 0.50
    wall_color: tuple = (0.10, 0.45, 0.85)
    floor_color: tuple = (0.10, 0.45, 0.85)
