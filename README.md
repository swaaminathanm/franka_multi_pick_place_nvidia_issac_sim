# Franka Multi Pick-and-Place for Isaac Lab & Newton

Standalone external Isaac Lab project containing a Newton-based direct RL task for picking up both a flexible cable and a rigid cube and placing them into a collection bin with a Franka Panda robot arm.

During environment reset, the **cable, cube, and bin** are initialized at non-overlapping randomized positions on the workspace table.

---

## Requirements

- Isaac Lab `v3.0.0-beta2.patch1` (or newer)
- Python 3.12
- `isaaclab-newton`
- `isaaclab-rl` (with RSL-RL)
- Isaac Sim 6.0.1 (when using the Kit visualizer or running headless)
- NVIDIA GPU with CUDA support (e.g. running on Vast.ai)

---

## Installation (on Vast.ai / Linux)

Run from the project root using the Python environment that contains Isaac Lab:

```bash
python -m pip install -e source/franka_multi_pick_place
```

If Isaac Lab is bundled with `./isaaclab.sh`:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/franka_multi_pick_place
```

---

## Verification & Available Tasks

To verify that the task registration is active in Gym:

```bash
python scripts/list_envs.py
```

Registered Task ID:
- `Franka-Multi-Pick-Place-Direct-v0`

---

## Agents

### 1. Zero Action Agent
Executes zero actions ($\mathbf{0}$) to verify robot joint stabilization, collision shapes, and randomized object settling under gravity without divergence:

```bash
python scripts/zero_agent.py --task=Franka-Multi-Pick-Place-Direct-v0 --num_envs=1 --viz none
```

### 2. Random Action Agent
Samples uniformly random actions from $[-1, 1]^8$ to stress-test joint limits, gripper dynamics, and solver stability:

```bash
python scripts/random_agent.py --task=Franka-Multi-Pick-Place-Direct-v0 --num_envs=1 --viz none
```

To run with GUI visualizer (Kit):

```bash
python scripts/zero_agent.py --task=Franka-Multi-Pick-Place-Direct-v0 --num_envs=1 --viz kit
```

---

## Project Structure

```text
newton_franka_multi_pick_and_place/
├── README.md
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   └── random_agent.py
└── source/
    └── franka_multi_pick_place/
        ├── setup.py
        ├── pyproject.toml
        ├── config/
        │   └── extension.toml
        └── franka_multi_pick_place/
            ├── __init__.py
            └── tasks/
                ├── __init__.py
                └── direct/
                    ├── __init__.py
                    └── franka_multi_pick_place/
                        ├── __init__.py
                        ├── multi_pick_place_env_cfg.py
                        ├── multi_pick_place_env.py
                        └── agents/
                            ├── __init__.py
                            └── rsl_rl_ppo_cfg.py
```
