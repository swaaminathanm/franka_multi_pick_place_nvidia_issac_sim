# Implementation Plan: Franka Multi Pick-and-Place (Cable & Cube into Bin)

## Overview
Develop a Newton-accelerated, vectorized reinforcement learning environment in Isaac Lab for a Franka Panda robot manipulating multiple objects: picking up a flexible cable and a rigid cube, and placing them into a collection bin. The environment includes randomized initial positions at reset, evaluated via a Zero Agent and a Random Agent.

---

## Phase Breakdown & Progress

- [x] **Step 1: Project Scaffolding & Packaging Setup**
  - [x] Create build configuration ([`pyproject.toml`](../source/franka_multi_pick_place/pyproject.toml)) and installation script ([`setup.py`](../source/franka_multi_pick_place/setup.py)).
  - [x] Define Omniverse extension manifest ([`config/extension.toml`](../source/franka_multi_pick_place/config/extension.toml)).
  - [x] Configure modular subpackages under `source/franka_multi_pick_place/tasks/direct/franka_multi_pick_place`.
  - [x] Register Gym environment ID `Franka-Multi-Pick-Place-Direct-v0`.
  - [x] Create verification script ([`scripts/list_envs.py`](../scripts/list_envs.py)).
  - [x] Document VM & Vast.ai setup with `uv` and Viser in [`README.md`](../README.md).
  - [x] Initialize Git repository and push initial commit to GitHub (`swaaminathanm/franka_multi_pick_place_nvidia_issac_sim`).

- [x] **Step 2: Environment Configuration Definition**
  - [x] Create [`multi_pick_place_env_cfg.py`](../source/franka_multi_pick_place/franka_multi_pick_place/tasks/direct/franka_multi_pick_place/multi_pick_place_env_cfg.py).
  - [x] Configure Newton solver parameters (`MJWarpSolverCfg` with implicit fast integrator, 20 iterations, 100 line-search iterations, elliptic friction cone, parallel line-search, and continuous collision detection).
  - [x] Tune contact parameters (`ke=100_000`, `kd=1_000`, `kf=3_000`, `mu=1.0`).
  - [x] Define Franka Panda articulation settings and joint actuator gains ($k_p$, $k_d$, velocity limits).
  - [x] Define 4 cm Rigid Cube properties (density, friction, visual preview).
  - [x] Define Target Bin receptacle container on table.
  - [x] Define Flexible Cable payload parameters (length $0.38\,\text{m}$, 19 segments, bend/stretch stiffness).
  - [x] Specify domain randomization boundaries for Cube, Cable, and Bin with minimum spatial separation threshold ($0.12\,\text{m}$).
  - [x] Specify observation space (62 dims) and action space (8 dims).

- [x] **Step 3: Core Environment Simulation & Randomized Reset (`multi_pick_place_env.py`)**
  - [x] Implement `FrankaMultiPickPlaceEnv` inheriting from `isaaclab.envs.DirectRLEnv`.
  - [x] Implement `_setup_scene`:
    - [x] Spawn SeattleLabTable / Ground plane.
    - [x] Spawn Franka Panda Articulation.
    - [x] Spawn Rigid Cube.
    - [x] Spawn Target Bin receptacle.
    - [x] Spawn / initialize Cable payload.
    - [x] Register Newton contact callbacks and custom material properties.
  - [x] Allocate action, joint target, and previous action tensors.
  - [x] Implement `_pre_physics_step`:
    - [x] Action clamping to $[-1, 1]$.
    - [x] Arm joint position delta integration with `action_scale` and soft limits.
    - [x] Gripper open/close position command mapping.
  - [x] Implement `_apply_action`:
    - [x] Stabilization logic (clamp Newton joint drift).
    - [x] Set robot position targets.
  - [x] Implement `_reset_idx`:
    - [x] Non-overlapping rejection sampling for $(x, y, \theta)$ of Cube, Cable, and Bin across separate table zones.
    - [x] Reset robot joints to default ready configuration with optional noise ($\sigma = 0.05\,\text{rad}$).
    - [x] Reset velocities to zero.
    - [x] Update simulation root states.
  - [x] Implement `_get_observations`:
    - [x] Robot joint positions and velocities (18).
    - [x] End-effector pose and linear velocity (10).
    - [x] Cube pose, linear velocity, and relative vector to end-effector (10).
    - [x] Cable center pose and relative vector to end-effector (7).
    - [x] Bin position and relative offsets: Cube $\to$ Bin, Cable $\to$ Bin (9).
    - [x] Previous action buffer (8).
    - [x] Total: 62-dimensional observation vector.
  - [x] Implement `_get_dones`:
    - [x] Episode truncation ($t \ge 10.0\,\text{s}$).
    - [x] Object out-of-bounds / fall termination ($z < -0.05\,\text{m}$).
  - [x] Implement `_get_rewards`:
    - [x] Baseline zero / placeholder reward for agent testing.

- [x] **Step 4: Zero Action Agent Implementation (`scripts/zero_agent.py`)**
  - [x] Create `scripts/zero_agent.py` using Isaac Lab launcher arguments.
  - [x] Configure CLI args: `--task`, `--num_envs`, `--max_steps`, `--viz` (kit, viser, none).
  - [x] Instantiate `gym.make("Franka-Multi-Pick-Place-Direct-v0")`.
  - [x] Execute zero actions ($\mathbf{0}$) on every step.
  - [x] Verification criteria:
    - [x] Robot maintains home posture steadily under internal gravity compensation.
    - [x] Randomized cube, cable, and bin settle onto the table surface without jitter or interpenetration.
    - [x] Periodic resets occur cleanly without memory leaks or physics drift.

- [ ] **Step 5: Random Action Agent Implementation (`scripts/random_agent.py`)**
  - [ ] Create `scripts/random_agent.py`.
  - [ ] Sample actions uniformly from $\mathcal{U}[-1, 1]^8$ on each policy step.
  - [ ] Verification criteria:
    - [ ] Arm joint limits safely enforced by clamping logic.
    - [ ] Gripper fingers smoothly transition between limits without joint dislocation.
    - [ ] Newton solver maintains numerical stability under erratic inputs (no NaN or physics blowup).
    - [ ] Episodes terminate and reset properly when objects are knocked off the table.

- [ ] **Step 6: Vast.ai VM Pull & Verification**
  - [ ] Commit and push Step 3, 4, 5 code to GitHub.
  - [ ] Run `git pull` inside `/workspace/franka_multi_pick_place_nvidia_issac_sim` on Vast.ai.
  - [ ] Run headless smoke test with zero agent:
    ```bash
    python scripts/zero_agent.py --task=Franka-Multi-Pick-Place-Direct-v0 --num_envs=1 --max_steps=100 --viz none
    ```
  - [ ] Run headless smoke test with random agent:
    ```bash
    python scripts/random_agent.py --task=Franka-Multi-Pick-Place-Direct-v0 --num_envs=1 --max_steps=100 --viz none
    ```
  - [ ] (Optional) Test live 3D web streaming via Viser: `--viz viser`.

- [ ] **Step 7: (Future) Dense Multi-Stage Reward Function & RL Training**
  - [ ] Design reward shaping for sequential multi-task manipulation:
    - Phase 1: Reaching, grasping, and lifting the cable.
    - Phase 2: Carrying cable and releasing into the bin.
    - Phase 3: Transitioning to the cube, grasping, and lifting.
    - Phase 4: Carrying cube and placing into the bin.
  - [ ] Configure RSL-RL PPO training harness and hyperparameter tuning.
