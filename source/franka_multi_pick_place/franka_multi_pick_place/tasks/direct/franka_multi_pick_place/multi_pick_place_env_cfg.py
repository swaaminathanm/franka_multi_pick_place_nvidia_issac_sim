# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


@configclass
class FrankaMultiNewtonContactCfg:
    """Global Newton contact tuning for the Franka multi-object pick-and-place task."""

    enabled: bool = True
    ke: float | None = 100_000.0
    kd: float | None = 1_000.0
    kf: float | None = 3_000.0
    mu: float | None = 1.0
    contact_margin: float | None = 0.002
    geom_solimp: tuple[float, float, float, float, float] | None = (0.97, 0.995, 0.0015, 0.5, 2.0)
    solimp_friction: tuple[float, float, float, float, float] | None = (0.97, 0.995, 0.0015, 0.5, 2.0)
    solref_friction: tuple[float, float] | None = (0.008, 2.0)


@configclass
class FrankaMultiPickPlaceEnvCfg(DirectRLEnvCfg):
    """Configuration for Franka multi-object pick-and-place environment."""

    # Environment stepping parameters
    decimation: int = 2
    episode_length_s: float = 10.0

    # Spaces definition
    action_space: int = 8
    observation_space: int = 62
    state_space: int = 0

    # Newton physics solver configuration
    solver_cfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=4000,
        nconmax=2000,
        impratio=100.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=20,
        ls_iterations=100,
        ccd_iterations=80,
        ls_parallel=True,
        use_mujoco_contacts=False,
    )

    newton_cfg = NewtonCfg(solver_cfg=solver_cfg, num_substeps=5, debug_mode=False)

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=newton_cfg,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Robot Articulation
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.spawn.rigid_props.max_depenetration_velocity = 5.0
    robot_cfg.actuators["panda_shoulder"].velocity_limit_sim = 2.175
    robot_cfg.actuators["panda_shoulder"].stiffness = 400.0
    robot_cfg.actuators["panda_shoulder"].damping = 80.0
    robot_cfg.actuators["panda_shoulder"].armature = 0.3
    robot_cfg.actuators["panda_forearm"].velocity_limit_sim = 2.61
    robot_cfg.actuators["panda_forearm"].stiffness = 400.0
    robot_cfg.actuators["panda_forearm"].damping = 80.0
    robot_cfg.actuators["panda_forearm"].armature = 0.11
    robot_cfg.actuators["panda_hand"].effort_limit_sim = 100.0
    robot_cfg.actuators["panda_hand"].velocity_limit_sim = 0.04
    robot_cfg.actuators["panda_hand"].stiffness = 7_500.0
    robot_cfg.actuators["panda_hand"].damping = 220.0
    robot_cfg.actuators["panda_hand"].friction = 0.2
    robot_cfg.actuators["panda_hand"].armature = 0.15

    # 4cm Rigid Cube
    cube_size: float = 0.04
    cube_density: float = 400.0
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, -0.25, 0.5 * 0.04],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                linear_damping=0.1,
                angular_damping=0.2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=cube_density),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.2,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
        ),
    )

    # Target Bin / Collection Tray (Receptacle container)
    bin_size_x: float = 0.16
    bin_size_y: float = 0.16
    bin_height: float = 0.05
    bin_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.35, 0.30, 0.025],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Kitting_Tray/kitting_tray.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)),
        ),
    )

    # Flexible Cable Payload Specifications (0.38m length, 19 segments, radius 0.005m)
    cable_length: float = 0.38
    cable_segments: int = 19
    cable_radius: float = 0.005
    cable_density: float = 100.0
    cable_bend_stiffness: float = 5.0e-4
    cable_stretch_stiffness: float = 1.0e6
    cable_contact_ke: float = 1.0e4
    cable_contact_kd: float = 1.0e-1

    # Scene Interactive configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Action scaling for Franka 7-DoF arm
    action_scale: tuple[float, ...] = (0.45, 1.60, 0.70, 2.70, 0.45, 0.80, 0.30)
    newton_contact: FrankaMultiNewtonContactCfg = FrankaMultiNewtonContactCfg()

    # Reset Spatial Randomization Bounds
    cube_reset_pos_x_range: tuple[float, float] = (0.35, 0.55)
    cube_reset_pos_y_range: tuple[float, float] = (-0.35, -0.12)

    cable_reset_pos_x_range: tuple[float, float] = (0.38, 0.58)
    cable_reset_pos_y_range: tuple[float, float] = (-0.08, 0.12)
    cable_reset_yaw_range: tuple[float, float] = (-1.5708, 1.5708)

    bin_reset_pos_x_range: tuple[float, float] = (0.30, 0.50)
    bin_reset_pos_y_range: tuple[float, float] = (0.22, 0.40)

    # Safety distance between spawned items to prevent overlap
    min_separation_distance: float = 0.12
    reset_arm_noise: float = 0.05

    # Termination bounds
    object_drop_height: float = -0.05
