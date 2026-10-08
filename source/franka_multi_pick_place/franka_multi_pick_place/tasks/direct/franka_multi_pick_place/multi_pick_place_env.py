# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from .multi_pick_place_env_cfg import FrankaMultiPickPlaceEnvCfg

# Backward-compatibility fallback stub if anything calls clone_environments
if not hasattr(InteractiveScene, "clone_environments"):
    InteractiveScene.clone_environments = lambda self, *args, **kwargs: None


class FrankaMultiPickPlaceEnv(DirectRLEnv):
    """Franka Multi Pick-and-Place Environment (Cable & Cube into Bin)."""

    cfg: FrankaMultiPickPlaceEnvCfg

    def __init__(self, cfg: FrankaMultiPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint limits for action clamping
        joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[0]
        self.robot_dof_lower_limits = joint_pos_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_pos_limits[:, 1].to(self.device)
        self.arm_joint_indices, _ = self.robot.find_joints("panda_joint[1-7]")
        self.arm_action_scale = self._resolve_arm_action_scale()
        self.arm_dof_lower_limits = self.robot_dof_lower_limits[self.arm_joint_indices]
        self.arm_dof_upper_limits = self.robot_dof_upper_limits[self.arm_joint_indices]
        self.arm_dof_velocity_limits = self.robot.data.joint_vel_limits.torch[0, self.arm_joint_indices].to(self.device)

        # Store default poses for relative observations and resets
        self.robot_default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self.cube_default_root_pose = self.cube.data.default_root_pose.torch.clone()
        self.cube_default_root_vel = self.cube.data.default_root_vel.torch.clone()
        self.cable_default_root_pose = self.cable.data.default_root_pose.torch.clone()
        self.cable_default_root_vel = self.cable.data.default_root_vel.torch.clone()
        self.bin_default_root_pose = self.bin.data.default_root_pose.torch.clone()

        # Buffers for actions and targets
        self.robot_dof_targets = self.robot_default_joint_pos.clone()
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.previous_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device
        )

        # End-effector and finger body indices
        self.ee_body_idx = self.robot.body_names.index("panda_hand")
        self.lf_body_idx = self.robot.body_names.index("panda_leftfinger")
        self.rf_body_idx = self.robot.body_names.index("panda_rightfinger")

        # Finger joint parameters
        self.finger_joint_indices = [
            self.robot.joint_names.index(name)
            for name in self.robot.joint_names
            if "panda_finger_joint" in name
        ]
        self.finger_joint_ids = torch.tensor(self.finger_joint_indices, device=self.device, dtype=torch.long)
        self.finger_dof_lower_limits = self.robot_dof_lower_limits[self.finger_joint_ids]
        self.finger_dof_upper_limits = self.robot_dof_upper_limits[self.finger_joint_ids]

        finger_lower_limit = float(self.finger_dof_lower_limits.max().item())
        finger_upper_limit = float(self.finger_dof_upper_limits.min().item())
        finger_limit_margin = min(5.0e-4, 0.25 * max(finger_upper_limit - finger_lower_limit, 0.0))
        safe_finger_lower_limit = finger_lower_limit + finger_limit_margin
        safe_finger_upper_limit = finger_upper_limit - finger_limit_margin

        self.gripper_open_pos = float(
            torch.clamp(torch.tensor(0.04, device=self.device), safe_finger_lower_limit, safe_finger_upper_limit).item()
        )
        self.gripper_close_pos = float(
            torch.clamp(torch.tensor(0.0, device=self.device), safe_finger_lower_limit, safe_finger_upper_limit).item()
        )
        self.robot_default_joint_pos[:, self.finger_joint_indices] = self.gripper_open_pos
        self._gripper_span = max(self.gripper_open_pos - self.gripper_close_pos, 1.0e-6)

        # Grasp frame offsets
        self.grasp_frame_offset = torch.tensor(
            (0.0, 0.0, 0.1034),
            device=self.device,
            dtype=self.robot_default_joint_pos.dtype,
        )
        self.finger_contact_offset = torch.tensor(
            (0.0, 0.0, 0.046),
            device=self.device,
            dtype=self.robot_default_joint_pos.dtype,
        )

    def _setup_scene(self):
        """Bind scene entities and register physics callbacks."""
        self._register_newton_contact_callback()

        # Retrieve scene assets (automatically instantiated and cloned across all environments by InteractiveScene)
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.cable = self.scene["cable"]
        self.bin = self.scene["bin"]

    def close(self):
        """Cleanup environment and deregister Newton callbacks."""
        handle = getattr(self, "_newton_model_init_handle", None)
        if handle is not None:
            handle.deregister()
            self._newton_model_init_handle = None
        super().close()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions into arm joint position targets and finger targets."""
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Map arm actions [-1, 1] to target joint position deltas
        arm_targets = self.robot_default_joint_pos[:, self.arm_joint_indices] + self.arm_action_scale.unsqueeze(
            0
        ) * self.actions[:, : len(self.arm_joint_indices)]
        self.robot_dof_targets[:, self.arm_joint_indices] = torch.clamp(
            arm_targets,
            self.arm_dof_lower_limits.unsqueeze(0),
            self.arm_dof_upper_limits.unsqueeze(0),
        )

        # Gripper finger target command: action index 7
        finger_cmd = 0.5 * (self.actions[:, 7] + 1.0)
        finger_target = self.gripper_close_pos + finger_cmd * (self.gripper_open_pos - self.gripper_close_pos)
        self.robot_dof_targets[:, self.finger_joint_indices] = finger_target.unsqueeze(-1).expand(
            -1, len(self.finger_joint_indices)
        )

    def _apply_action(self) -> None:
        """Stabilize robot state and command position targets to actuators."""
        self._stabilize_robot_state()
        self.robot.set_joint_position_target_index(target=self.robot_dof_targets)

    def _stabilize_robot_state(self) -> None:
        """No-op: PD actuators and joint limits govern state natively.
        Mid-step simulation writes corrupt Newton CUDA state buffers, causing NaN and disappearing links.
        """
        pass

    def _resolve_arm_action_scale(self) -> torch.Tensor:
        """Return per-joint arm action scales as a length-7 tensor."""
        arm_action_scale = torch.as_tensor(
            self.cfg.action_scale,
            device=self.device,
            dtype=self.robot_dof_lower_limits.dtype,
        )
        if arm_action_scale.ndim == 0:
            arm_action_scale = arm_action_scale.repeat(len(self.arm_joint_indices))
        else:
            arm_action_scale = arm_action_scale.flatten()
        return arm_action_scale

    def _enforce_arm_joint_velocity_limits(self) -> None:
        """Clamp arm joint speeds within configured Franka limits."""
        joint_vel = self.robot.data.joint_vel.torch
        arm_joint_vel = joint_vel[:, self.arm_joint_indices]
        clamped_arm_joint_vel = torch.clamp(
            arm_joint_vel,
            -self.arm_dof_velocity_limits.unsqueeze(0),
            self.arm_dof_velocity_limits.unsqueeze(0),
        )
        too_fast = torch.any(torch.abs(clamped_arm_joint_vel - arm_joint_vel) > 1.0e-6, dim=-1)
        if not torch.any(too_fast):
            return

        env_ids = too_fast.nonzero(as_tuple=False).squeeze(-1)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=clamped_arm_joint_vel[env_ids],
            joint_ids=self.arm_joint_indices,
            env_ids=env_ids,
        )

    def _enforce_finger_joint_limits(self) -> None:
        """Clamp finger joints within valid range if Newton drifts past mechanical stops."""
        joint_pos = self.robot.data.joint_pos.torch
        finger_joint_pos = joint_pos[:, self.finger_joint_ids]
        clamped_finger_joint_pos = torch.clamp(
            finger_joint_pos,
            self.finger_dof_lower_limits.unsqueeze(0),
            self.finger_dof_upper_limits.unsqueeze(0),
        )
        out_of_bounds = torch.any(torch.abs(clamped_finger_joint_pos - finger_joint_pos) > 1.0e-6, dim=-1)
        if not torch.any(out_of_bounds):
            return

        env_ids = out_of_bounds.nonzero(as_tuple=False).squeeze(-1)
        zero_finger_vel = torch.zeros(
            (env_ids.numel(), len(self.finger_joint_indices)),
            dtype=joint_pos.dtype,
            device=self.device,
        )
        self.robot.write_joint_position_to_sim_index(
            position=clamped_finger_joint_pos[env_ids],
            joint_ids=self.finger_joint_indices,
            env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=zero_finger_vel,
            joint_ids=self.finger_joint_indices,
            env_ids=env_ids,
        )

    def _get_observations(self) -> dict:
        """Assemble the 62-dimensional observation vector."""
        self._stabilize_robot_state()

        joint_pos = self.robot.data.joint_pos.torch
        joint_vel = self.robot.data.joint_vel.torch

        ee_pos = self.robot.data.body_pos_w.torch[:, self.ee_body_idx] - self.scene.env_origins
        ee_quat = self.robot.data.body_quat_w.torch[:, self.ee_body_idx]

        cube_pos = self.cube.data.root_pos_w.torch - self.scene.env_origins
        cube_quat = self.cube.data.root_quat_w.torch

        cable_pos = self.cable.data.root_pos_w.torch - self.scene.env_origins
        cable_quat = self.cable.data.root_quat_w.torch

        bin_pos = self.bin.data.root_pos_w.torch - self.scene.env_origins

        obs = torch.cat(
            [
                joint_pos - self.robot_default_joint_pos,  # 9
                joint_vel,                                 # 9
                ee_pos,                                    # 3
                ee_quat,                                   # 4
                cube_pos,                                  # 3
                cube_pos - ee_pos,                         # 3
                cube_quat,                                 # 4
                cable_pos,                                 # 3
                cable_pos - ee_pos,                        # 3
                cable_quat,                                # 4
                bin_pos,                                   # 3
                cube_pos - bin_pos,                        # 3
                cable_pos - bin_pos,                       # 3
                self.previous_actions,                     # 8
            ],
            dim=-1,
        )  # Total: 62 dimensions

        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        obs = torch.clamp(obs, -100.0, 100.0)
        self.previous_actions[:] = self.actions
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Baseline zero reward tensor (to be shaped in Step 7)."""
        return torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute episode termination (object dropped) and truncation (timeout)."""
        cube_pos_z = (self.cube.data.root_pos_w.torch - self.scene.env_origins)[:, 2]
        cable_pos_z = (self.cable.data.root_pos_w.torch - self.scene.env_origins)[:, 2]

        cube_dropped = cube_pos_z < self.cfg.object_drop_height
        cable_dropped = cable_pos_z < self.cfg.object_drop_height
        terminated = cube_dropped | cable_dropped

        # TODO (Step 7 - Task Success Termination):
        # Check if both cube and cable are successfully placed inside the bin:
        #   cube_pos = self.cube.data.root_pos_w.torch - self.scene.env_origins
        #   cable_pos = self.cable.data.root_pos_w.torch - self.scene.env_origins
        #   bin_pos = self.bin.data.root_pos_w.torch - self.scene.env_origins
        #   cube_in_bin = (
        #       (torch.abs(cube_pos[:, 0] - bin_pos[:, 0]) < 0.5 * self.cfg.bin_size_x)
        #       & (torch.abs(cube_pos[:, 1] - bin_pos[:, 1]) < 0.5 * self.cfg.bin_size_y)
        #       & (cube_pos[:, 2] > bin_pos[:, 2])
        #       & (cube_pos[:, 2] < bin_pos[:, 2] + self.cfg.bin_height)
        #   )
        #   cable_in_bin = (
        #       (torch.abs(cable_pos[:, 0] - bin_pos[:, 0]) < 0.5 * self.cfg.bin_size_x)
        #       & (torch.abs(cable_pos[:, 1] - bin_pos[:, 1]) < 0.5 * self.cfg.bin_size_y)
        #       & (cable_pos[:, 2] > bin_pos[:, 2])
        #       & (cable_pos[:, 2] < bin_pos[:, 2] + self.cfg.bin_height)
        #   )
        #   task_success = cube_in_bin & cable_in_bin
        #   terminated = terminated | task_success

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Randomize positions of robot, cube, cable, and bin without spatial overlap."""
        if env_ids is None:
            env_ids_tensor = wp.to_torch(self.robot._ALL_INDICES).to(dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids_tensor)

        n = len(env_ids_tensor)

        # 1. Reset Franka Panda robot joints (fixed-base articulation remains anchored at env origin)

        joint_pos = self.robot_default_joint_pos[env_ids_tensor].clone()
        arm_noise = sample_uniform(
            -self.cfg.reset_arm_noise,
            self.cfg.reset_arm_noise,
            (n, len(self.arm_joint_indices)),
            self.device,
        )
        joint_pos[:, self.arm_joint_indices] = torch.clamp(
            joint_pos[:, self.arm_joint_indices] + arm_noise,
            self.arm_dof_lower_limits.unsqueeze(0),
            self.arm_dof_upper_limits.unsqueeze(0),
        )
        joint_pos[:, self.finger_joint_indices] = self.gripper_open_pos
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_tensor] = joint_pos
        self.actions[env_ids_tensor] = 0.0
        self.previous_actions[env_ids_tensor] = 0.0

        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids_tensor)

        # 2. Non-overlapping Rejection Sampling for Cube, Cable, and Bin
        cube_pose = self.cube_default_root_pose[env_ids_tensor].clone()
        cable_pose = self.cable_default_root_pose[env_ids_tensor].clone()

        # Sample Cube positions in right quadrant
        cube_x = sample_uniform(self.cfg.cube_reset_pos_x_range[0], self.cfg.cube_reset_pos_x_range[1], (n,), self.device)
        cube_y = sample_uniform(self.cfg.cube_reset_pos_y_range[0], self.cfg.cube_reset_pos_y_range[1], (n,), self.device)
        cube_z = torch.full((n,), 0.5 * self.cfg.cube_size, device=self.device)

        # Sample Bin positions in left quadrant
        bin_x = sample_uniform(self.cfg.bin_reset_pos_x_range[0], self.cfg.bin_reset_pos_x_range[1], (n,), self.device)
        bin_y = sample_uniform(self.cfg.bin_reset_pos_y_range[0], self.cfg.bin_reset_pos_y_range[1], (n,), self.device)

        # Sample Cable positions in center quadrant flush on table
        cable_x = sample_uniform(self.cfg.cable_reset_pos_x_range[0], self.cfg.cable_reset_pos_x_range[1], (n,), self.device)
        cable_y = sample_uniform(self.cfg.cable_reset_pos_y_range[0], self.cfg.cable_reset_pos_y_range[1], (n,), self.device)
        cable_z = torch.full((n,), 0.5 * getattr(self.cfg, "cable_thickness", 0.015), device=self.device)

        # Multi-object spatial separation checks to guarantee zero contact at reset
        dist_cube_cable = torch.hypot(cube_x - cable_x, cube_y - cable_y)
        too_close_cube_cable = dist_cube_cable < self.cfg.min_separation_distance
        cable_x = torch.where(too_close_cube_cable, cable_x + 0.08, cable_x)

        dist_cable_bin = torch.hypot(bin_x - cable_x, bin_y - cable_y)
        too_close_cable_bin = dist_cable_bin < self.cfg.min_separation_distance
        cable_x = torch.where(too_close_cable_bin, cable_x + 0.08, cable_x)

        # Randomize Cable yaw orientation
        cable_yaw = sample_uniform(
            self.cfg.cable_reset_yaw_range[0], self.cfg.cable_reset_yaw_range[1], (n,), self.device
        )
        cable_quat = quat_from_euler_xyz(
            torch.zeros((n,), device=self.device),
            torch.zeros((n,), device=self.device),
            cable_yaw,
        )

        # Set Cube pose and velocity
        cube_pose[:, 0] = cube_x
        cube_pose[:, 1] = cube_y
        cube_pose[:, 2] = cube_z
        cube_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        cube_vel = torch.zeros_like(cube_pose[:, :6])
        self.cube.write_root_pose_to_sim_index(root_pose=cube_pose, env_ids=env_ids_tensor)
        self.cube.write_root_velocity_to_sim_index(root_velocity=cube_vel, env_ids=env_ids_tensor)

        # Set Cable pose and velocity
        cable_pose[:, 0] = cable_x
        cable_pose[:, 1] = cable_y
        cable_pose[:, 2] = cable_z
        cable_pose[:, 3:7] = cable_quat
        cable_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        cable_vel = torch.zeros_like(cable_pose[:, :6])
        self.cable.write_root_pose_to_sim_index(root_pose=cable_pose, env_ids=env_ids_tensor)
        self.cable.write_root_velocity_to_sim_index(root_velocity=cable_vel, env_ids=env_ids_tensor)

        bin_pose = self.bin_default_root_pose[env_ids_tensor].clone()
        bin_pose[:, 0] = bin_x
        bin_pose[:, 1] = bin_y
        bin_pose[:, 2] = self.cfg.bin_root_z
        bin_pose[:, 3:7] = torch.tensor((0.707, 0, 0.707, 0.0), device=self.device)
        bin_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        bin_vel = torch.zeros_like(bin_pose[:, :6])
        self.bin.write_root_pose_to_sim_index(root_pose=bin_pose, env_ids=env_ids_tensor)
        self.bin.write_root_velocity_to_sim_index(root_velocity=bin_vel, env_ids=env_ids_tensor)

    # --------------------------------------------------------------------------
    # Newton Physics & MuJoCo Contact Callbacks
    # --------------------------------------------------------------------------
    def _register_newton_contact_callback(self) -> None:
        """Register a Newton-only callback to apply tuned contact properties."""
        self._newton_model_init_handle = None
        physics_mgr_cls = self.sim.physics_manager
        if physics_mgr_cls.__name__ != "NewtonManager":
            return

        self._newton_model_init_handle = physics_mgr_cls.register_callback(
            self._apply_newton_contact_tuning,
            PhysicsEvent.MODEL_INIT,
            order=100,
            name=f"{self.__class__.__name__}_newton_contact_tuning",
        )

    def _apply_newton_contact_tuning(self, _event) -> None:
        """Apply global Newton and MuJoCo contact defaults before model finalization."""
        if not self.cfg.newton_contact.enabled:
            return

        physics_mgr_cls = self.sim.physics_manager
        builder = getattr(physics_mgr_cls, "_builder", None)
        if builder is None:
            return

        shape_cfg = builder.default_shape_cfg
        for attr_name in ("ke", "kd", "kf", "contact_margin"):
            value = getattr(self.cfg.newton_contact, attr_name)
            if value is not None:
                setattr(shape_cfg, attr_name, value)
        if self.cfg.newton_contact.mu is not None and hasattr(shape_cfg, "mu"):
            shape_cfg.mu = self.cfg.newton_contact.mu
        self._apply_mujoco_solver_contact_tuning(builder)

    def _apply_mujoco_solver_contact_tuning(self, builder) -> None:
        """Apply MuJoCo contact profile for the Newton pipeline."""
        from newton import solvers

        solvers.SolverMuJoCo.register_custom_attributes(builder)
        for key, value in (
            ("mujoco:geom_solimp", self.cfg.newton_contact.geom_solimp),
            ("mujoco:solimpfriction", self.cfg.newton_contact.solimp_friction),
            ("mujoco:solreffriction", self.cfg.newton_contact.solref_friction),
        ):
            self._set_mujoco_custom_attribute_default(builder, key, value)

    @staticmethod
    def _set_mujoco_custom_attribute_default(builder, key: str, value) -> None:
        """Set a MuJoCo custom-attribute default if a task override is provided."""
        if value is None:
            return
        builder.custom_attributes[key].default = list(value) if isinstance(value, tuple) else value
