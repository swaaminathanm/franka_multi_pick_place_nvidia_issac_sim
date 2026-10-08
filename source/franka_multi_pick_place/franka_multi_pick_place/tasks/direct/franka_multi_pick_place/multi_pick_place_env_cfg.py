# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from .hollow_bin_spawner import HollowBinCfg


@configclass
class FrankaMultiNewtonContactCfg:
    """Global Newton contact tuning for the Franka multi-object pick-and-place task."""

    enabled: bool = True
    ke: float | None = 100_000.0
    kd: float | None = 1_000.0
    kf: float | None = 3_000.0
    mu: float | None = 1.0
    contact_margin: float | None = 0.001
    geom_solimp: tuple[float, float, float, float, float] | None = (0.97, 0.995, 0.0015, 0.5, 2.0)
    solimp_friction: tuple[float, float, float, float, float] | None = (0.97, 0.995, 0.0015, 0.5, 2.0)
    solref_friction: tuple[float, float] | None = (0.008, 2.0)


def _create_panda_robot_cfg() -> ArticulationCfg:
    """Create and configure the Franka Panda robot articulation."""
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
    robot.spawn.rigid_props.disable_gravity = True
    robot.spawn.rigid_props.max_depenetration_velocity = 5.0
    robot.actuators = {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=400.0,
            damping=80.0,
            armature=0.3,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=400.0,
            damping=80.0,
            armature=0.11,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            stiffness=7_500.0,
            damping=220.0,
            friction=0.2,
            armature=0.15,
        ),
    }
    for key, act_cfg in robot.actuators.items():
        v_lim = 2.175 if key == "panda_shoulder" else (2.61 if key == "panda_forearm" else 0.04)
        e_lim = 87.0 if key == "panda_shoulder" else (12.0 if key == "panda_forearm" else 100.0)
        for v_attr in ("joint_velocity_limit", "velocity_limit_sim", "velocity_limit"):
            if hasattr(act_cfg, v_attr):
                setattr(act_cfg, v_attr, v_lim)
        for e_attr in ("joint_effort_limit", "effort_limit_sim", "effort_limit"):
            if hasattr(act_cfg, e_attr):
                setattr(act_cfg, e_attr, e_lim)
    return robot


@configclass
class FrankaMultiSceneCfg(InteractiveSceneCfg):
    """Declarative scene configuration for Franka multi-object pick-and-place.

    Assets declared here are automatically instantiated and cloned across all
    environments by Isaac Lab's InteractiveScene without requiring manual cloner code.
    """

    # Ground plane (fixed in world)
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Workstation table (replicated per environment under {ENV_REGEX_NS})
    # Clean procedural workstation tabletop eliminating the overhead L-shaped camera mount rod.
    # Placed in front of the Franka Panda base (x in [0.125, 0.975]) to prevent collision with robot base.
    # To change the table color, adjust diffuse_color below:
    #   Light Studio Gray: (0.75, 0.75, 0.78)
    #   Birch Wood:        (0.80, 0.72, 0.58)
    #   Matte White:       (0.90, 0.90, 0.92)
    #   Slate / Charcoal:  (0.25, 0.25, 0.28)
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.85, 1.0, 1.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, -0.525),
        ),
    )

    # Lighting
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # Robot Articulation (Franka Panda)
    robot: ArticulationCfg = _create_panda_robot_cfg()

    # 4cm Rigid Cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, -0.25, 0.02],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                linear_damping=0.5,
                angular_damping=1.0,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
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

    # Flexible Cable Payload (Flat industrial rubber cable proxy for rigid kinematics/Newton solver)
    # Flat rectangular profile (20 cm x 3.5 cm x 1.5 cm) that sits completely flat on the table without rolling
    cable: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cable",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.48, 0.0, 0.0075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.035, 0.015),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                linear_damping=1.0,
                angular_damping=5.0,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1200.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
        ),
    )

    # Five CuboidCfg boxes. Viser collapses the USDA mesh into the pyramid in the screenshot.
    bin: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.35, 0.30, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=HollowBinCfg(
            outer_size_x=0.20,
            outer_size_y=0.20,
            wall_height=0.08,
            wall_thickness=0.02,
            floor_thickness=0.01,
            mass=0.50,
        ),
    )


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

    # Payload dimension constants
    cube_size: float = 0.04
    cube_density: float = 400.0

    cable_length: float = 0.20
    cable_width: float = 0.035
    cable_thickness: float = 0.015
    cable_radius: float = 0.015  # backward compatibility
    cable_segments: int = 19
    cable_density: float = 1200.0
    cable_bend_stiffness: float = 5.0e-4
    cable_stretch_stiffness: float = 1.0e6
    cable_contact_ke: float = 1.0e4
    cable_contact_kd: float = 1.0e-1

    bin_size_x: float = 0.16
    bin_size_y: float = 0.16
    bin_height: float = 0.08

    # Scene Interactive configuration (Declarative multi-env scene)
    scene: FrankaMultiSceneCfg = FrankaMultiSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Action scaling for Franka 7-DoF arm
    action_scale: tuple[float, ...] = (0.45, 1.60, 0.70, 2.70, 0.45, 0.80, 0.30)
    newton_contact: FrankaMultiNewtonContactCfg = FrankaMultiNewtonContactCfg()

    # Reset Spatial Randomization Bounds (Distinct, guaranteed non-overlapping workspace zones)
    # Right quadrant: Cube
    cube_reset_pos_x_range: tuple[float, float] = (0.38, 0.55)
    cube_reset_pos_y_range: tuple[float, float] = (-0.32, -0.18)

    # Center quadrant: Cable
    cable_reset_pos_x_range: tuple[float, float] = (0.40, 0.55)
    cable_reset_pos_y_range: tuple[float, float] = (-0.05, 0.05)
    cable_reset_yaw_range: tuple[float, float] = (-0.35, 0.35)

    # Left quadrant: Target Bin
    bin_reset_pos_x_range: tuple[float, float] = (0.35, 0.50)
    bin_reset_pos_y_range: tuple[float, float] = (0.24, 0.36)

    # Safety distance between spawned items to prevent overlap
    min_separation_distance: float = 0.15
    reset_arm_noise: float = 0.05

    # Termination bounds
    object_drop_height: float = -0.05
