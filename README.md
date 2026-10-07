# Franka Multi Pick-and-Place for Isaac Lab & Newton

Standalone external Isaac Lab project containing a Newton-based direct RL task for picking up both a flexible cable and a rigid cube and placing them into a collection bin with a Franka Panda robot arm.

During environment reset, the **cable, cube, and bin** are initialized at non-overlapping randomized positions on the workspace table.

---

## VM & Vast.ai Setup Guide (Step-by-Step)

Follow these steps to set up a clean Linux/Vast.ai GPU instance from scratch:

### 1. Install `uv` (Fast Python Package Manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Clone and Setup Isaac Lab with Newton & Viser
In Isaac Lab 3.0, `uv` manages the core packages, the Newton physics engine (`isaaclab-newton`), RSL-RL, and the web-based Viser visualizer:

```bash
cd /workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Sync dependencies and build virtual environment with Viser & RSL-RL
uv sync --extra viser --extra rsl-rl
```

### 3. Activate the Isaac Lab Virtual Environment
Whenever you open a new terminal session, activate the Isaac Lab environment:

```bash
source /workspace/IsaacLab/.venv/bin/activate
```
*(Your prompt will change to show `(isaaclab-dev)`)*

> **Tip:** You can make this automatic on login by adding it to your bashrc:
> ```bash
> echo "source /workspace/IsaacLab/.venv/bin/activate" >> ~/.bashrc
> ```

### 4. Clone and Install This Project
```bash
cd /workspace
git clone https://github.com/swaaminathanm/franka_multi_pick_place_nvidia_issac_sim.git
cd franka_multi_pick_place_nvidia_issac_sim

# Install the extension in editable mode using uv pip
uv pip install -e source/franka_multi_pick_place
```

---

## Verification

To verify that the task registration is active in Gym:

```bash
python scripts/list_envs.py
```

Expected output:
```text
+-------+------------------------------------+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| S. No.| Task Name                          | Entry Point                                                                             | Config                                                                                   |
+-------+------------------------------------+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| 1     | Franka-Multi-Pick-Place-Direct-v0  | franka_multi_pick_place.tasks.direct.franka_multi_pick_place:FrankaMultiPickPlaceEnv     | franka_multi_pick_place.tasks.direct.franka_multi_pick_place:FrankaMultiPickPlaceEnvCfg |
+-------+------------------------------------+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
```

---

## Viewing the Simulation via Viser (Web Viewer)

Because cloud VMs lack a physical display, Viser streams 3D WebGL rendering directly to your local browser over port `8080`.

1. **Forward Port 8080 when connecting to Vast.ai:**
   ```bash
   ssh -p <VAST_PORT> -L 8080:localhost:8080 root@<VAST_IP>
   ```
2. **Launch the environment with Viser:**
   ```bash
   python scripts/zero_agent.py --task=Franka-Multi-Pick-Place-Direct-v0 --num_envs=1 --viz viser
   ```
3. Open `http://localhost:8080` in Chrome/Firefox on your local computer to view the 3D scene live.

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

---

## Project Structure

```text
franka_multi_pick_place_nvidia_issac_sim/
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
