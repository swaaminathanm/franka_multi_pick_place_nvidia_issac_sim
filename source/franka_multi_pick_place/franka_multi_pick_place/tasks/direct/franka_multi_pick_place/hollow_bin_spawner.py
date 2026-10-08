# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural hollow bin spawner using Isaac Lab native CuboidCfg primitives.

Replaces the USDA-based bin with 5 analytical box colliders constructed entirely
from sim_utils.CuboidCfg calls. This guarantees correct Viser visualization
(each piece is a first-class USD prim with a visual material), zero network
dependency, and optimal Newton/PhysX performance via primitive box colliders.

Layout (all dimensions in metres, Z-up):
  - bottom floor plate : outer_size_x × outer_size_y × floor_thickness
  - wall_left  (-Y side): outer_size_x × wall_thickness × wall_height
  - wall_right (+Y side): outer_size_x × wall_thickness × wall_height
  - wall_front (+X side): wall_thickness × inner_y × wall_height
  - wall_back  (-X side): wall_thickness × inner_y × wall_height
where inner_y = outer_size_y - 2 * wall_thickness
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
    """Spawn a hollow bin as 5 native CuboidCfg box primitives under a kinematic Xform.

    The ``@clone`` decorator resolves regex prim paths (e.g. ``{ENV_REGEX_NS}/bin``)
    and calls this function once per environment.  Each call creates:
      - A parent ``UsdGeom.Xform`` at *prim_path* with ``PhysicsRigidBodyAPI``
        (kinematic) and ``PhysicsMassAPI``.
      - Five child ``UsdGeom.Cube`` prims (via ``sim_utils.CuboidCfg.func``)
        with independent visual materials so Viser renders each piece in 3-D.

    Args:
        prim_path: Absolute USD prim path for the bin root (already env-resolved).
        cfg: HollowBinCfg instance carrying all dimension and material settings.
        translation: World-space (x, y, z) translation for the bin root, metres.
        orientation: Quaternion (w, x, y, z) orientation for the bin root.

    Returns:
        The root ``Usd.Prim`` of the spawned bin.
    """
    # Heavy USD/Isaac imports are inside the function body so that py_compile
    # succeeds locally (these modules only exist inside Isaac Sim on the VM).
    from pxr import Gf, UsdGeom, UsdPhysics  # noqa: PLC0415

    try:
        import isaacsim.core.utils.stage as _stage_utils  # Isaac Sim 6.x
    except ImportError:
        import omni.isaac.core.utils.stage as _stage_utils  # Isaac Sim 4.x fallback

    stage = _stage_utils.get_current_stage()

    # ------------------------------------------------------------------ #
    # 1.  Root Xform – receives the world transform, rigid-body, and mass #
    # ------------------------------------------------------------------ #
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()

    xform_ops: list[str] = []
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform_ops.append("xformOp:translate")
    if orientation is not None:
        # orientation convention: (w, x, y, z)
        xform.AddOrientOp().Set(
            Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3])
        )
        xform_ops.append("xformOp:orient")
    if xform_ops:
        xform.GetXformOpOrderAttr().Set(xform_ops)

    # Apply kinematic rigid-body API to the parent Xform
    rba = UsdPhysics.RigidBodyAPI.Apply(prim)
    rba.GetKinematicEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(cfg.mass)

    # ------------------------------------------------------------------ #
    # 2.  Derived geometry values (all in metres)                         #
    # ------------------------------------------------------------------ #
    ox = cfg.outer_size_x / 2.0          # outer half-extent along X
    oy = cfg.outer_size_y / 2.0          # outer half-extent along Y
    ft = cfg.floor_thickness             # floor slab height
    wt = cfg.wall_thickness              # wall rim thickness
    wh = cfg.wall_height                 # wall height above the floor slab
    inner_y = cfg.outer_size_y - 2 * wt  # clear inner width for front/back walls

    # (sub_name, size_xyz, local_centre_xyz, diffuse_rgb)
    pieces: list[tuple[str, tuple, tuple, tuple]] = [
        (
            "bottom",
            (cfg.outer_size_x, cfg.outer_size_y, ft),
            (0.0, 0.0, ft / 2.0),
            cfg.floor_color,
        ),
        (
            "wall_left",
            (cfg.outer_size_x, wt, wh),
            (0.0, -(oy - wt / 2.0), ft + wh / 2.0),
            cfg.wall_color,
        ),
        (
            "wall_right",
            (cfg.outer_size_x, wt, wh),
            (0.0, +(oy - wt / 2.0), ft + wh / 2.0),
            cfg.wall_color,
        ),
        (
            "wall_front",
            (wt, inner_y, wh),
            (+(ox - wt / 2.0), 0.0, ft + wh / 2.0),
            cfg.wall_color,
        ),
        (
            "wall_back",
            (wt, inner_y, wh),
            (-(ox - wt / 2.0), 0.0, ft + wh / 2.0),
            cfg.wall_color,
        ),
    ]

    # ------------------------------------------------------------------ #
    # 3.  Spawn each piece as a native Isaac Lab CuboidCfg primitive      #
    # ------------------------------------------------------------------ #
    for sub_name, size, centre, color in pieces:
        child_path = f"{prim_path}/{sub_name}"
        child_cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        # CuboidCfg.func == spawn_cuboid (already @clone-decorated; resolves no
        # regex here since child_path is an absolute, fully-resolved path).
        child_cfg.func(child_path, child_cfg, translation=centre)

    return prim


@configclass
class HollowBinCfg(SpawnerCfg):
    """Procedural hollow bin built from 5 native Isaac Lab CuboidCfg primitives.

    Geometry (default values):
    - Outer footprint : 20 cm × 20 cm
    - Floor plate     :  8 mm thick
    - Wall rim        :  8 cm tall, 1.6 cm thick
    - Inner cavity    : 16.8 cm × 16.8 cm × 8.0 cm deep
    - Total height    :  8.8 cm

    No external USDA/USD file is required.  All prims are created at runtime
    via :func:`spawn_hollow_bin` using Isaac Lab's own :class:`CuboidCfg`.
    """

    func: Callable = spawn_hollow_bin

    # ---- outer footprint ----
    outer_size_x: float = 0.20
    """Outer length of the bin along the X axis (metres)."""

    outer_size_y: float = 0.20
    """Outer width of the bin along the Y axis (metres)."""

    # ---- wall geometry ----
    wall_height: float = 0.08
    """Height of the four walls above the floor plate (metres)."""

    wall_thickness: float = 0.016
    """Thickness of each wall rim (metres)."""

    floor_thickness: float = 0.008
    """Thickness of the bottom floor plate (metres)."""

    # ---- physics ----
    mass: float = 0.50
    """Total mass of the bin (kg). Bin is kinematic so this is informational."""

    # ---- visual materials (RGB 0-1) ----
    wall_color: tuple = (0.08, 0.30, 0.65)
    """Diffuse colour of the four wall prims (dark navy blue)."""

    floor_color: tuple = (0.28, 0.62, 0.90)
    """Diffuse colour of the bottom floor prim (lighter sky blue)."""
