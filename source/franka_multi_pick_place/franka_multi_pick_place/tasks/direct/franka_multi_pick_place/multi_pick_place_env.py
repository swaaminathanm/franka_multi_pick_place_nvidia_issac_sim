# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from .multi_pick_place_env_cfg import FrankaMultiPickPlaceEnvCfg


class FrankaMultiPickPlaceEnv(DirectRLEnv):
    """Franka Multi Pick and Place Environment."""

    cfg: FrankaMultiPickPlaceEnvCfg

    def __init__(self, cfg: FrankaMultiPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
